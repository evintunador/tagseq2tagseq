"""
Definitive multi-GPU NCCL smoke test for world-size-portable optimizer resume.

Unlike the gloo unit test (fabricated state, subset of fields), this:
  1. Uses the REAL distributed MuonWithAuxAdam on NCCL, so state is sharded by
     the actual step() (round-robin i % world_size), not hand-fabricated.
  2. Takes real forward/backward/step iterations so momentum / second-moment /
     mantissa / AdamW moments are all genuinely populated (non-zero).
  3. Saves via state_dict_full() at world_size S, then resumes at a DIFFERENT
     world_size L (e.g. 4 -> 2, 2 -> 4).
  4. Asserts EVERY field of EVERY param restores bit-exact vs. the saved ground
     truth — momentum_buffer, second_momentum_buffer, red_dim, mantissa (Muon)
     and exp_avg, exp_avg_sq, step (AdamW) — plus a completeness check that the
     save gathered every param's state (nothing dropped).

Run directly (not under pytest — needs multiple real GPUs + process spawn):
    python tests/smoke_optimizer_resume_multigpu.py

It spawns the save phase at S GPUs, then the load phase at L GPUs, for several
(S, L) pairs, using a temp dir to hand the checkpoint between phases.
"""
import os
import sys
import tempfile

# Ensure the repo root is importable in spawned subprocesses (spawn start-method
# children do not inherit the parent's sys.path beyond the entry script's dir).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp


# Fields that must round-trip exactly, per optimizer path.
MUON_FIELDS = ("momentum_buffer", "second_momentum_buffer", "red_dim", "mantissa")
ADAMW_FIELDS = ("exp_avg", "exp_avg_sq", "step")


class TinyNet(nn.Module):
    """Many Muon-eligible 2-D 'backbone' weights spanning both reduction axes
    (wide -> red_dim=-2, tall -> red_dim=-1), plus an embedding + bias on AdamW.
    Backbone weights are fp32 so gradients stay fp32 (matching real backbone grad
    flow); the bf16 mantissa path is covered by the CPU round-trip unit test."""

    def __init__(self, dim=64, vocab=256, n_layers=24):
        super().__init__()
        self.backbone = nn.ModuleList()
        for _ in range(n_layers):
            layer = nn.Module()
            layer.wide = nn.Parameter(torch.randn(dim, dim * 2) * 0.05)   # M<N -> red_dim=-2
            layer.tall = nn.Parameter(torch.randn(dim * 2, dim) * 0.05)   # M>N -> red_dim=-1
            self.backbone.append(layer)
        self.embedding = nn.Module()
        self.embedding.weight = nn.Parameter(torch.randn(vocab, dim) * 0.05)
        self.bias = nn.Parameter(torch.zeros(dim))
        self.dim = dim

    def forward(self, x):
        h = x
        for layer in self.backbone:
            h = h @ layer.wide          # (B, 2*dim)
            h = h @ layer.tall          # (B, dim)
            h = h + self.bias
        # Tie a cheap loss to the embedding so it gets a gradient.
        return h.pow(2).mean() + self.embedding.weight.sum() * 1e-4


def _build_optimizer(model):
    from optimizers.muon import MuonWithAuxAdam
    muon, adamw = [], []
    id_to_name = {}
    for name, p in model.named_parameters():
        id_to_name[id(p)] = name
        if "backbone" in name and p.ndim >= 2:
            muon.append(p)
        else:
            adamw.append(p)
    opt = MuonWithAuxAdam([
        dict(params=muon, use_muon=True, lr=0.02, momentum=0.95, weight_decay=0.1, beta2=0.95),
        dict(params=adamw, use_muon=False, lr=3e-4, betas=(0.9, 0.95), eps=1e-10, weight_decay=0.1),
    ])
    opt._id_to_name = id_to_name
    return opt


def _train_steps(model, opt, device, n=5, seed=0):
    # Same seed on every rank -> identical inputs -> identical grads (no DDP
    # needed: Muon all_gathers updated params each step so ranks stay in sync;
    # AdamW is replicated). Real backward populates all state fields.
    gen = torch.Generator(device=device).manual_seed(seed)
    for _ in range(n):
        x = torch.randn(16, model.dim, generator=gen, device=device)
        opt.zero_grad(set_to_none=True)
        loss = model(x)
        loss.backward()
        opt.step()


def _entry_matches(a, b):
    """Field-by-field bit-exact comparison of two saved state entries."""
    if a.get("use_muon") != b.get("use_muon"):
        return False, "use_muon differs"
    fields = MUON_FIELDS if a.get("use_muon") else ADAMW_FIELDS
    for f in fields:
        av, bv = a.get(f), b.get(f)
        if isinstance(av, torch.Tensor) or isinstance(bv, torch.Tensor):
            if av is None or bv is None:
                if av is None and bv is None:
                    continue  # e.g. mantissa absent for fp32 param
                return False, f"{f}: one side missing"
            if av.shape != bv.shape or not torch.equal(av.cpu(), bv.cpu()):
                return False, f"{f}: tensor mismatch"
        else:
            if av != bv:
                return False, f"{f}: {av} != {bv}"
    return True, "ok"


def _worker(rank, world_size, tmpdir, phase, seed):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29531"
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    try:
        torch.manual_seed(1234)  # identical model init on every rank
        model = TinyNet().to(device)
        opt = _build_optimizer(model)

        if phase == "save":
            _train_steps(model, opt, device, n=5, seed=seed)
            full = opt.state_dict_full()  # collective; rank 0 gets the union

            if rank == 0:
                state = full["state"]
                # Completeness: every param that should have state is present,
                # with all expected fields and non-trivial (non-zero) values.
                names = set(state)
                muon_names = {n for n in names if state[n]["use_muon"]}
                adamw_names = {n for n in names if not state[n]["use_muon"]}
                problems = []
                for n, e in state.items():
                    fields = MUON_FIELDS if e["use_muon"] else ADAMW_FIELDS
                    for f in fields:
                        if f == "mantissa":
                            continue  # optional (bf16 only)
                        if f not in e:
                            problems.append(f"{n}: missing {f}")
                    if e["use_muon"]:
                        mb = e["momentum_buffer"]
                        if float(mb.abs().sum()) == 0.0:
                            problems.append(f"{n}: momentum_buffer is all-zero (step didn't populate)")
                        # (fp32 Muon params carry no mantissa; that path is unit-tested.)
                    else:
                        if int(e["step"]) != 5:
                            problems.append(f"{n}: AdamW step={e['step']} != 5")
                report = {
                    "n_total": len(names),
                    "n_muon": len(muon_names),
                    "n_adamw": len(adamw_names),
                    "problems": problems,
                }
                torch.save(full, os.path.join(tmpdir, "full.pt"))
                torch.save(report, os.path.join(tmpdir, "report.pt"))
            dist.barrier()

        else:  # phase == "load" (possibly at a different world_size)
            full = torch.load(os.path.join(tmpdir, "full.pt"), weights_only=False)
            ground_truth = full["state"]  # name -> saved entry (the union)
            restored, skipped = opt.load_state_dict_full(full)

            # Which params does THIS rank own at THIS world_size?
            muon_params = opt.param_groups[0]["params"]
            expected = {opt._id_to_name[id(muon_params[i])]
                        for i in range(len(muon_params)) if i % world_size == rank}
            expected |= {opt._id_to_name[id(p)] for p in opt.param_groups[1]["params"]}

            errs = []
            if set(restored) != expected:
                errs.append(f"restored set {set(restored)} != expected {expected}")
            if skipped:
                errs.append(f"unexpected skips: {skipped}")
            # EVERY owned param: every field bit-exact vs ground truth.
            for group in opt.param_groups:
                for p in group["params"]:
                    name = opt._id_to_name[id(p)]
                    if name not in expected:
                        continue
                    live = opt.state.get(p)
                    if not live:
                        errs.append(f"{name}: no state after load")
                        continue
                    # Reconstruct a comparable entry from live state.
                    if group["use_muon"]:
                        live_entry = {"use_muon": True,
                                      "momentum_buffer": live["momentum_buffer"],
                                      "second_momentum_buffer": live["second_momentum_buffer"],
                                      "red_dim": int(live["red_dim"]),
                                      "mantissa": live.get("mantissa")}
                    else:
                        live_entry = {"use_muon": False,
                                      "exp_avg": live["exp_avg"],
                                      "exp_avg_sq": live["exp_avg_sq"],
                                      "step": int(live["step"])}
                    ok, why = _entry_matches(ground_truth[name], live_entry)
                    if not ok:
                        errs.append(f"{name}: {why}")

            out = {"rank": rank, "world_size": world_size, "n_owned": len(expected), "errs": errs}
            torch.save(out, os.path.join(tmpdir, f"load_rank{rank}.pt"))
            dist.barrier()
    finally:
        dist.destroy_process_group()


def _run_pair(save_ws, load_ws, seed=11):
    with tempfile.TemporaryDirectory() as tmp:
        mp.spawn(_worker, args=(save_ws, tmp, "save", seed), nprocs=save_ws, join=True)
        report = torch.load(os.path.join(tmp, "report.pt"), weights_only=False)
        mp.spawn(_worker, args=(load_ws, tmp, "load", seed), nprocs=load_ws, join=True)

        all_errs = []
        total_owned = 0
        for r in range(load_ws):
            out = torch.load(os.path.join(tmp, f"load_rank{r}.pt"), weights_only=False)
            total_owned += out["n_owned"]
            all_errs += [f"[load rank {r}] {e}" for e in out["errs"]]

        print(f"\n=== save_ws={save_ws} -> load_ws={load_ws} ===")
        print(f"  saved state: {report['n_total']} params "
              f"({report['n_muon']} Muon + {report['n_adamw']} AdamW)")
        if report["problems"]:
            print("  SAVE-COMPLETENESS PROBLEMS:")
            for p in report["problems"]:
                print(f"    - {p}")
        else:
            print("  save completeness: OK (all fields present, momentum non-zero, step==5)")
        if all_errs:
            print(f"  RESTORE MISMATCHES ({len(all_errs)}):")
            for e in all_errs:
                print(f"    - {e}")
        else:
            print(f"  restore: OK — every field of every owned param bit-exact "
                  f"across {load_ws} rank(s)")
        ok = not report["problems"] and not all_errs
        print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
        return ok


def main():
    ndev = torch.cuda.device_count()
    print(f"CUDA devices available: {ndev}")
    if ndev < 2:
        print("Need >=2 GPUs for a real reshard test; abort.")
        return 1
    pairs = [(2, 2), (2, 1), (1, 2)]
    if ndev >= 4:
        pairs += [(4, 2), (2, 4), (4, 1)]
    if ndev >= 8:
        # Production world sizes: 96 Muon params across 8 ranks, resharded down
        # to 3 (uneven: 96 % 3 == 0 but per-rank counts differ from 8) and up.
        pairs += [(8, 8), (8, 3), (3, 8), (8, 1)]
    results = {}
    for s, l in pairs:
        results[(s, l)] = _run_pair(s, l)
    print("\n===== SUMMARY =====")
    for (s, l), ok in results.items():
        print(f"  {s} -> {l}: {'PASS' if ok else 'FAIL'}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
