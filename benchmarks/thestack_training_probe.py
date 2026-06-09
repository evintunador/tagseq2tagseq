"""1-layer training probe for the thestack cross_doc_link NaN bug.

Runs a real Muon+AdamW training loop over the exact pack sequence rank R of a
world_size W DDP run would see, using real token IDs.  At every step:

  1. Forward pass uses cdb_bim_v18 (production kernel).
  2. Flex runs on the same q/k/v (detached) → attn_err column shows whether
     v18 and flex diverge BEFORE the NaN appears.
  3. All intermediate activations checked for NaN/Inf.
  4. Per-parameter gradient norms reported; NaN/Inf flagged per param.
  5. Optimizer state buffers (momentum, exp_avg, exp_avg_sq) checked after step.

If the kernel is the cause, attn_err will grow or go NaN before loss does.
If Muon state is the cause, state buffers will show NaN after the corrupt step.
If neither, the bug requires either depth (>1 layer) or specific weight dynamics.

Usage:
    CUDA_VISIBLE_DEVICES=2 python benchmarks/thestack_training_probe.py \\
        --parquet   schedules/thestack_bfs/epoch_0/packs.parquet \\
        --dataset   /fss/evin_t/tagseq2tagseq_artifacts/pretokenized_datasets/thestack \\
        --steps     300 --rank 0 --world-size 16
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.thestack_nan_probe import load_bucket_lists, iter_rank_packs, pack_to_mask_inputs
from benchmarks.attention_harness import _to_bhnd, _to_thd
from data.collate import build_packed_batch
from data.dataset import GraphIndex, PretokShardedBackend
from data.epoch_precompute import PackRecord, _record_to_placements
from data.layout import make_layout_policy
from optimizers.muon import SingleDeviceMuonWithAuxAdam
from main import LRCooldownScheduler

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_compiled_flex = torch.compile(flex_attention, dynamic=True)


# ---------------------------------------------------------------------------
# Flex reference runner (same q/k/v, cross_doc or doc_causal BlockMask)
# ---------------------------------------------------------------------------

def _run_flex(q, k, v, mask_inputs, scale: float) -> torch.Tensor:
    bm = mask_inputs.flex_cross_doc_block_mask or mask_inputs.flex_doc_causal_block_mask
    q4, k4, v4 = _to_bhnd(q), _to_bhnd(k), _to_bhnd(v)
    return _to_thd(_compiled_flex(q4, k4, v4, block_mask=bm, scale=scale))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class CrossDocAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.k_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.v_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.out_proj = nn.Linear(model_dim, model_dim, bias=False)
        # Stash for post-forward flex comparison
        self._last_q: Optional[torch.Tensor] = None
        self._last_k: Optional[torch.Tensor] = None
        self._last_v: Optional[torch.Tensor] = None
        self._last_attn_out: Optional[torch.Tensor] = None
        self._last_mask_inputs = None

    def forward(self, x: torch.Tensor, mask_inputs) -> torch.Tensor:
        T, H, Dh = x.shape[0], self.num_heads, self.head_dim

        q = self.q_proj(x).view(T, H, Dh)
        k = self.k_proj(x).view(T, H, Dh)
        v = self.v_proj(x).view(T, H, Dh)

        from kernels.cross_doc_bitmask_bim_v17 import _build_bim_64
        from kernels.cross_doc_bitmask_bim_v12 import _build_bim_128
        from kernels.cross_doc_bitmask_bim_v18 import triton_attn_cross_doc_bitmask_bim_v18

        n_chunks = mask_inputs.q_bitmasks.shape[0]
        dev = mask_inputs.document_ids.device
        bim128 = _build_bim_128(mask_inputs.seq_len, mask_inputs.document_ids,
                                mask_inputs.q_bitmasks, mask_inputs.kv_bitmasks, dev, n_chunks)
        bim64  = _build_bim_64(mask_inputs.seq_len, mask_inputs.document_ids,
                               mask_inputs.q_bitmasks, mask_inputs.kv_bitmasks, dev, n_chunks)
        attn_out = triton_attn_cross_doc_bitmask_bim_v18(
            q, k, v, mask_inputs.document_ids,
            mask_inputs.q_bitmasks, mask_inputs.kv_bitmasks, bim128, bim64, self.scale,
        )  # (T, H, Dh)

        # Stash for flex comparison (detached — no grad through comparison path)
        self._last_q = q.detach()
        self._last_k = k.detach()
        self._last_v = v.detach()
        self._last_attn_out = attn_out.detach()
        self._last_mask_inputs = mask_inputs

        return self.out_proj(attn_out.reshape(T, H * Dh))


class NLayerModel(nn.Module):
    def __init__(self, vocab_size: int, model_dim: int, num_heads: int, num_layers: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, model_dim)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "attn": CrossDocAttention(model_dim, num_heads),
                "norm": nn.LayerNorm(model_dim),
            })
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(model_dim)
        self.lm_head_weight = self.embedding.weight
        self._act: Dict[str, torch.Tensor] = {}

    def forward(self, tokens: torch.Tensor, mask_inputs) -> torch.Tensor:
        x = self.embedding(tokens)
        self._act["embed"] = x.detach()
        for i, layer in enumerate(self.layers):
            attn_out = layer["attn"](layer["norm"](x), mask_inputs)
            x = x + attn_out
            self._act[f"layer{i}_out"] = x.detach()
        x = self.final_norm(x)
        logits = F.linear(x, self.lm_head_weight)
        self._act["logits"] = logits.detach()
        return F.cross_entropy(logits[:-1].reshape(-1, logits.shape[-1]), tokens[1:].reshape(-1))


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

def build_optimizer(model, muon_lr, adamw_lr, momentum):
    muon_params, embed_params, other_params = [], [], []
    seen = set()
    for name, p in model.named_parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        if 'embedding' in name or 'lm_head' in name:
            embed_params.append(p)
        elif p.ndim >= 2:
            muon_params.append(p)
        else:
            other_params.append(p)
    return SingleDeviceMuonWithAuxAdam([
        dict(params=muon_params,  use_muon=True,  lr=muon_lr,  momentum=momentum,
             weight_decay=0.1, beta2=0.95),
        dict(params=embed_params, use_muon=False, lr=adamw_lr, betas=(0.5, 0.95),
             eps=1e-10, weight_decay=0.005),
        dict(params=other_params, use_muon=False, lr=adamw_lr, betas=(0.9, 0.95),
             eps=1e-10, weight_decay=0.005),
    ])


# ---------------------------------------------------------------------------
# Diagnostic helpers
# ---------------------------------------------------------------------------

def _has_nan_inf(t: torch.Tensor) -> Tuple[bool, bool]:
    return bool(torch.isnan(t).any()), bool(torch.isinf(t).any())


def _check_activations(act: Dict[str, torch.Tensor]) -> List[str]:
    flags = []
    for name, t in act.items():
        n, i = _has_nan_inf(t)
        if n: flags.append(f"ACT_{name.upper()}_NAN")
        if i: flags.append(f"ACT_{name.upper()}_INF")
    return flags


def _check_grads(model: nn.Module) -> Tuple[List[str], str, float]:
    """Returns (flag_list, worst_param_name, worst_param_grad_norm)."""
    flags = []
    worst_name, worst_norm = "", 0.0
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        n, i = _has_nan_inf(p.grad)
        if n: flags.append(f"GRAD_{name}_NAN")
        if i: flags.append(f"GRAD_{name}_INF")
        gnorm = p.grad.norm().item()
        if gnorm > worst_norm:
            worst_norm = gnorm
            worst_name = name
    return flags, worst_name, worst_norm


def _check_optimizer_state(optimizer: SingleDeviceMuonWithAuxAdam) -> List[str]:
    flags = []
    for group in optimizer.param_groups:
        for p in group["params"]:
            state = optimizer.state.get(p, {})
            for key in ("momentum_buffer", "exp_avg", "exp_avg_sq", "second_momentum_buffer"):
                buf = state.get(key)
                if buf is None:
                    continue
                n, i = _has_nan_inf(buf)
                if n: flags.append(f"STATE_{key}_NAN")
                if i: flags.append(f"STATE_{key}_INF")
    # Deduplicate (many params, report unique buffer-type flags)
    return list(dict.fromkeys(flags))


# ---------------------------------------------------------------------------
# Token loading
# ---------------------------------------------------------------------------

def load_tokens(pack, graph, backend, layout, device) -> torch.Tensor:
    batch = build_packed_batch(graph, backend, layout, _record_to_placements(pack))
    return batch["tokens"].squeeze(0).to(device)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_probe(
    parquet_path, dataset_dir, n_steps, rank, world_size, max_grants,
    model_dim, num_heads, num_layers, vocab_size, muon_lr, adamw_lr, momentum,
    warmup_steps, norm_clip, n_buckets, epoch_seed, dtype,
):
    print(f"Loading dataset: {dataset_dir}")
    graph = GraphIndex(Path(dataset_dir))
    backend = PretokShardedBackend(graph)
    enc = tiktoken.get_encoding(graph.metadata.get("tokenizer", "gpt2"))
    layout = make_layout_policy("identifier_prefix_eos", encode_fn=enc.encode_ordinary)
    print(f"  {len(graph)} docs")

    print(f"Loading parquet: {parquet_path}")
    bucket_lists = load_bucket_lists(parquet_path)
    print(f"  {sum(len(v) for v in bucket_lists.values())} packs across {len(bucket_lists)} buckets")

    print(f"\nBuilding model: {num_layers}-layer  dim={model_dim}  heads={num_heads}  vocab={vocab_size}  dtype={dtype}")
    model = NLayerModel(vocab_size, model_dim, num_heads, num_layers).to(device=DEVICE, dtype=dtype)
    print(f"  {sum(p.numel() for p in model.parameters()):,} parameters")

    optimizer = build_optimizer(model, muon_lr, adamw_lr, momentum)
    scheduler = LRCooldownScheduler(
        optimizer, total_steps=n_steps, warmup_steps=warmup_steps,
        cooldown_frac=0.0, min_lr_ratio=0.1,
        muon_momentum_warmup_steps=warmup_steps,
    )

    print(f"\nrank={rank}  world_size={world_size}  steps={n_steps}")
    print(f"muon_lr={muon_lr}  adamw_lr={adamw_lr}  warmup={warmup_steps}  clip={norm_clip}\n")

    hdr = (f"  {'step':>5}  {'bkt':>3}  {'kv_blk':>7}  {'T':>6}  {'grants':>6}"
           f"  {'loss':>10}  {'g_norm':>8}  {'attn_err':>10}  flags")
    sep = "  " + "-" * (len(hdr) - 2)
    print(hdr)
    print(sep)

    prev_attn_err = 0.0

    for step, bucket, pack in iter_rank_packs(
        bucket_lists, n_buckets, world_size, rank, 0, n_steps - 1, epoch_seed,
    ):
        mask_inputs, n_grants = pack_to_mask_inputs(pack, max_grants, DEVICE)

        try:
            tokens = load_tokens(pack, graph, backend, layout, DEVICE)
        except Exception as e:
            print(f"  {step:>5}  TOKEN_LOAD_ERROR: {e}", flush=True)
            continue

        T = mask_inputs.seq_len
        if tokens.shape[0] > T:
            tokens = tokens[:T]
        elif tokens.shape[0] < T:
            tokens = F.pad(tokens, (0, T - tokens.shape[0]))

        # ── Forward ──────────────────────────────────────────────────────────
        optimizer.zero_grad(set_to_none=True)
        try:
            loss = model(tokens, mask_inputs)
        except Exception as e:
            print(f"  {step:>5}  FWD_ERROR: {e}", flush=True)
            break

        flags = []

        # Check loss
        l_nan, l_inf = _has_nan_inf(loss)
        if l_nan: flags.append("LOSS_NAN")
        if l_inf: flags.append("LOSS_INF")

        # Check activations
        flags += _check_activations(model._act)

        # ── Flex comparison — max attn_err across all layers ─────────────────
        attn_err = float("nan")
        try:
            max_err = 0.0
            for layer in model.layers:
                attn = layer["attn"]
                q, k, v = attn._last_q, attn._last_k, attn._last_v
                if q is None:
                    continue
                with torch.no_grad():
                    flex_out = _run_flex(q, k, v, attn._last_mask_inputs, attn.scale)
                err = (attn._last_attn_out.float() - flex_out.float()).abs().max().item()
                if math.isnan(err) or math.isinf(err):
                    flags.append("ATTN_ERR_NAN")
                    max_err = err
                    break
                max_err = max(max_err, err)
            attn_err = max_err
            if not (math.isnan(attn_err) or math.isinf(attn_err)) and attn_err > 1.0:
                flags.append(f"ATTN_ERR_HIGH({attn_err:.2e})")
        except Exception:
            flags.append("FLEX_CMP_ERR")
            attn_err = float("nan")

        # Stop before backward if loss is already NaN/Inf
        if l_nan or l_inf:
            _print_step(step, bucket, pack, n_grants, T, loss, float("nan"), attn_err, flags)
            print(sep)
            print(f"\n  Stopped at step {step}: {' '.join(flags)}")
            _print_detail(model, optimizer, None, None)
            break

        # ── Backward ─────────────────────────────────────────────────────────
        try:
            loss.backward()
        except Exception as e:
            print(f"  {step:>5}  BWD_ERROR: {e}", flush=True)
            break

        grad_flags, worst_param, worst_gnorm = _check_grads(model)
        flags += grad_flags

        params_with_grad = [p for p in model.parameters() if p.grad is not None]
        grad_norm = torch.nn.utils.clip_grad_norm_(params_with_grad, norm_clip).item()
        if math.isnan(grad_norm) or math.isinf(grad_norm):
            flags.append("GNORM_NAN")

        # ── Optimizer step ────────────────────────────────────────────────────
        scheduler.step()
        state_flags = _check_optimizer_state(optimizer)
        flags += state_flags

        # ── Report ────────────────────────────────────────────────────────────
        _print_step(step, bucket, pack, n_grants, T, loss, grad_norm, attn_err, flags)

        if flags:
            print(sep)
            print(f"\n  Stopped at step {step}: {' '.join(flags)}")
            _print_detail(model, optimizer, worst_param, worst_gnorm)
            break

        prev_attn_err = attn_err

    else:
        print(sep)
        print(f"\n  Completed {n_steps} steps with no NaN/Inf.")


def _print_step(step, bucket, pack, n_grants, T, loss, grad_norm, attn_err, flags):
    loss_str = f"{loss.item():>10.4f}" if not (math.isnan(loss.item()) or math.isinf(loss.item())) else f"{'NaN/Inf':>10}"
    gnorm_str = f"{grad_norm:>8.4f}" if not (math.isnan(grad_norm) or math.isinf(grad_norm)) else f"{'NaN':>8}"
    aerr_str = f"{attn_err:>10.2e}" if not (math.isnan(attn_err) or math.isinf(attn_err)) else f"{'NaN':>10}"
    flag_str = " ".join(flags) if flags else "-"
    highlight = " ◄" if flags else ""
    print(
        f"  {step:>5}  {bucket:>3}  {pack.kv_block_count:>7}  {T:>6}  {n_grants:>6}"
        f"  {loss_str}  {gnorm_str}  {aerr_str}  {flag_str}{highlight}",
        flush=True,
    )


def _print_detail(model, optimizer, worst_param, worst_gnorm):
    print("\n  === Per-parameter gradient norms ===")
    for name, p in model.named_parameters():
        if p.grad is None:
            print(f"    {name}: no grad")
            continue
        n, i = _has_nan_inf(p.grad)
        tag = " ← NaN" if n else (" ← Inf" if i else "")
        print(f"    {name}: grad_norm={p.grad.norm().item():.4e}{tag}")

    print("\n  === Optimizer state NaN/Inf check ===")
    for group in optimizer.param_groups:
        for p in group["params"]:
            state = optimizer.state.get(p, {})
            for key, buf in state.items():
                if not isinstance(buf, torch.Tensor):
                    continue
                n, i = _has_nan_inf(buf)
                if n or i:
                    tag = "NaN" if n else "Inf"
                    pname = next((nm for nm, pp in model.named_parameters() if pp is p), "?")
                    print(f"    {pname} / {key}: {tag}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--parquet", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--world-size", type=int, default=16)
    p.add_argument("--max-grants", type=int, default=256)
    p.add_argument("--model-dim", type=int, default=1024)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--num-layers", type=int, default=1)
    p.add_argument("--vocab-size", type=int, default=50304)
    p.add_argument("--muon-lr", type=float, default=0.001)
    p.add_argument("--adamw-lr", type=float, default=1e-5)
    p.add_argument("--momentum", type=float, default=0.95)
    p.add_argument("--warmup-steps", type=int, default=300)
    p.add_argument("--norm-clip", type=float, default=1.0)
    p.add_argument("--n-buckets", type=int, default=32)
    p.add_argument("--epoch-seed", type=int, default=0)
    p.add_argument("--dtype", default="float32")
    args = p.parse_args()

    run_probe(
        parquet_path=args.parquet, dataset_dir=args.dataset, n_steps=args.steps,
        rank=args.rank, world_size=args.world_size, max_grants=args.max_grants,
        model_dim=args.model_dim, num_heads=args.num_heads, num_layers=args.num_layers,
        vocab_size=args.vocab_size,
        muon_lr=args.muon_lr, adamw_lr=args.adamw_lr, momentum=args.momentum,
        warmup_steps=args.warmup_steps, norm_clip=args.norm_clip,
        n_buckets=args.n_buckets, epoch_seed=args.epoch_seed,
        dtype=getattr(torch, args.dtype),
    )


if __name__ == "__main__":
    main()
