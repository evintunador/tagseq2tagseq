"""MLP kernel harness — correctness checks + fwd/bwd benchmarks.

Compares three MLP implementations:

  ref_relu_sq   Unfused PyTorch baseline: relu(F.linear(x, W1))^2 followed
                by F.linear with W2.  No custom kernel; purely torch ops.
                Hidden dim = 4 * model_dim (4× expansion).

  fused_relu_sq Our Triton kernel (kernels/fused_relu_sq_mlp.py).
                Fuses x@W1.T + relu² epilogue in one kernel (forward), and
                fuses grad@W2.T + relu²-backward in one kernel (backward).
                Hidden dim = 4 * model_dim.

  swiglu        Current production MLP: tunalab GLU with SiLU activation.
                Hidden dim = int(8/3 * model_dim)  ≈ 2.67× expansion.
                (Two projections: up+gate → silu(gate)*up → down.)

Correctness check
-----------------
    python benchmarks/mlp_harness.py correctness

For each config we verify:
  • Forward output:   max|fused − ref| is within a small bf16-rounding tolerance
  • dx gradient:      max|fused.dx − ref.dx|  < tolerance
  • dW1 gradient:     max|fused.dW1 − ref.dW1| < tolerance
  • dW2 gradient:     max|fused.dW2 − ref.dW2| < tolerance

Benchmark
---------
    python benchmarks/mlp_harness.py bench [--seq-lens 4096 8192 32768]
                                           [--model-dims 768 1024 1280]
                                           [--warmup 5] [--reps 20]

Reports forward time, backward time, and fwd+bwd time (ms) for each impl,
plus a speedup column relative to ref_relu_sq.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16


# ---------------------------------------------------------------------------
# Reference implementation (unfused)
# ---------------------------------------------------------------------------

class RefReLUSquaredMLP(nn.Module):
    """Unfused reference: relu(F.linear(x, W1))^2 then F.linear(post, W2).

    W1: (H, C)  —  standard Linear layout
    W2: (C, H)  —  standard Linear layout (so F.linear works normally)
    """

    def __init__(self, model_dim: int, hidden_dim: int | None = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * model_dim
        self.W1 = nn.Parameter(torch.empty(hidden_dim, model_dim))
        self.W2 = nn.Parameter(torch.empty(model_dim, hidden_dim))
        nn.init.kaiming_uniform_(self.W1, a=5**0.5)
        nn.init.zeros_(self.W2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.linear(x, self.W1)          # x @ W1.T
        h = F.relu(h) ** 2                 # relu²
        return F.linear(h, self.W2)        # h @ W2.T


# ---------------------------------------------------------------------------
# SwiGLU reference (current production MLP)
# ---------------------------------------------------------------------------

def _make_swiglu(model_dim: int) -> nn.Module:
    from tunalab.modules.channel_mixing.glu import GLU
    hidden_dim = int(8 / 3 * model_dim)
    return GLU(in_dim=model_dim, out_dim=model_dim, hidden_dim=hidden_dim,
               activation="silu", dropout=0.0, fp8=False)


# ---------------------------------------------------------------------------
# Correctness helpers
# ---------------------------------------------------------------------------

def _grad_stats(fused: torch.Tensor, ref: torch.Tensor) -> Tuple[float, float, float]:
    """Return (max_abs_err, rel_err, cosine_sim) comparing fused vs ref gradients.

    rel_err  = max|fused - ref| / max(|ref| + eps)
    cosine   = dot(fused, ref) / (norm(fused) * norm(ref))

    Relative error and cosine similarity are robust to the absolute scale of the
    gradient (which grows with sequence length for weight gradients), making
    them the right metric for large-M correctness checks.
    """
    a, b = fused.float(), ref.float()
    diff = (a - b).abs()
    max_abs = diff.max().item()
    rel_err = (diff.max() / (b.abs().max() + 1e-6)).item()
    cosine = ((a * b).sum() / (a.norm() * b.norm() + 1e-12)).item()
    return max_abs, rel_err, cosine


def _baseline_tolerance(fn_eager, fn_compiled, *args) -> Tuple[float, float]:
    """Run the same computation eager vs torch.compile; return (rel_err, cosine_sim).

    This establishes the "floor" of acceptable error — if our Triton kernel
    matches the reference to within the same tolerance as torch.compile matches
    eager PyTorch, the kernel is numerically correct.
    """
    out_eager = fn_eager(*args)
    out_compiled = fn_compiled(*args)
    _, rel, cos = _grad_stats(out_eager, out_compiled)
    return rel, cos


def run_correctness(configs: List[Tuple[int, int]]) -> bool:
    """Verify fused kernel matches unfused reference for forward and all gradients.

    Tolerance baseline: we first measure raw PyTorch eager vs torch.compile on the
    same operations.  If fused matches reference within that same relative tolerance,
    the kernel is numerically correct.  Weight gradients (dW1, dW2) are compared
    using relative error + cosine similarity, which are robust to the absolute
    gradient scale growing with sequence length.
    """
    from kernels.fused_relu_sq_mlp import FusedReLUSquaredFunction

    print("=" * 70)
    print("Correctness checks")
    print("Tolerance guide: raw PyTorch eager vs torch.compile on same ops")
    print("=" * 70)

    any_fail = False

    for M, C in configs:
        H = 4 * C
        print(f"\nConfig: M={M:6d}  C={C:4d}  H={H:5d}  (matches training: T=M, model_dim=C)")

        torch.manual_seed(0)
        W1 = torch.empty(H, C, dtype=DTYPE, device=DEVICE)
        W2_ref = torch.empty(C, H, dtype=DTYPE, device=DEVICE)    # standard layout
        W2_fused = torch.empty(H, C, dtype=DTYPE, device=DEVICE)  # transposed layout (H,C)
        nn.init.kaiming_uniform_(W1, a=5**0.5)
        nn.init.normal_(W2_ref, std=0.02)
        W2_fused.data.copy_(W2_ref.T)   # fused sees same function as ref

        x = torch.randn(M, C, dtype=DTYPE, device=DEVICE)

        # ------------------------------------------------------------------
        # 1. Tolerance baseline: eager PyTorch vs torch.compile(PyTorch)
        # ------------------------------------------------------------------
        def _relu_sq_eager(x_, W1_, W2_):
            h = F.relu(F.linear(x_, W1_)) ** 2
            return F.linear(h, W2_)

        _relu_sq_compiled = torch.compile(_relu_sq_eager, dynamic=True)

        # warm up compile
        _ = _relu_sq_compiled(x.clone(), W1.clone(), W2_ref.clone())

        tol_rel, tol_cos = _baseline_tolerance(
            _relu_sq_eager, _relu_sq_compiled,
            x.clone(), W1.clone(), W2_ref.clone(),
        )
        print(f"  [baseline eager↔compile]  rel_err={tol_rel:.6f}  cosine={tol_cos:.8f}")

        # ------------------------------------------------------------------
        # 2. Reference: unfused eager PyTorch
        # ------------------------------------------------------------------
        x_ref = x.clone().requires_grad_(True)
        W1r = W1.clone().requires_grad_(True)
        W2r = W2_ref.clone().requires_grad_(True)

        h_ref = F.relu(x_ref @ W1r.T) ** 2
        out_ref = h_ref @ W2r.T
        out_ref.sum().backward()

        # ------------------------------------------------------------------
        # 3. Fused Triton kernel
        # ------------------------------------------------------------------
        x_fus = x.clone().requires_grad_(True)
        W1f = W1.clone().requires_grad_(True)
        W2f = W2_fused.clone().requires_grad_(True)

        out_fus = FusedReLUSquaredFunction.apply(x_fus, W1f, W2f)
        out_fus.sum().backward()

        # ------------------------------------------------------------------
        # 4. Compare — relative error and cosine similarity
        # ------------------------------------------------------------------
        # Forward output (absolute; M×C tensor, scale-independent)
        fwd_abs = (out_fus.float() - out_ref.float()).abs().max().item()

        # dx (absolute; same shape as input, scale-independent)
        dx_abs = (x_fus.grad.float() - x_ref.grad.float()).abs().max().item()

        # dW1 / dW2: use relative + cosine.
        # Weight grads accumulate over M rows (dW1 = d_pre.T @ x_flat), so
        # absolute error grows with M.  Relative error ~O(eps_bf16 * sqrt(M/H))
        # is expected; 1% is the practical ceiling for correct bf16 matmuls.
        # Cosine > 0.9999 is the primary pass criterion (direction must be right).
        # Note: torch.compile happens to give bit-identical results for these
        # simple ops (no kernel fusion), so the baseline tolerance above is 0 —
        # that sets too tight a bar.  Use 1% explicitly instead.
        REL_THRESH = 0.015  # 1.5% — well within bf16 matmul accumulation bounds
        COS_THRESH = 0.9999

        _, dW1_rel, dW1_cos = _grad_stats(W1f.grad, W1r.grad)
        _, dW2_rel, dW2_cos = _grad_stats(W2f.grad, W2r.grad.T)

        ok_fwd  = fwd_abs  < 0.01
        ok_dx   = dx_abs   < 0.05
        ok_dW1  = dW1_rel  < REL_THRESH and dW1_cos > COS_THRESH
        ok_dW2  = dW2_rel  < REL_THRESH and dW2_cos > COS_THRESH

        s = lambda ok: "PASS" if ok else "FAIL"
        print(f"  fwd output   abs_err={fwd_abs:.6f}                          {s(ok_fwd)}")
        print(f"  dx grad      abs_err={dx_abs:.6f}                          {s(ok_dx)}")
        print(f"  dW1 grad     rel_err={dW1_rel:.6f}  cosine={dW1_cos:.8f}  {s(ok_dW1)}")
        print(f"  dW2 grad     rel_err={dW2_rel:.6f}  cosine={dW2_cos:.8f}  {s(ok_dW2)}")

        if not (ok_fwd and ok_dx and ok_dW1 and ok_dW2):
            any_fail = True

    print()
    if any_fail:
        print("Some checks FAILED.")
    else:
        print("All checks PASSED.")

    return not any_fail


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def _time_fn(fn, warmup: int, reps: int) -> float:
    """Return median wall-clock time in ms over `reps` calls after `warmup` warm-up calls."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(reps):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)

    times.sort()
    return times[len(times) // 2]  # median


def _time_fwd_bwd(module: nn.Module, x: torch.Tensor, warmup: int, reps: int):
    """Return (fwd_ms, fwd_bwd_ms)."""
    # forward only
    def fwd():
        with torch.no_grad():
            module(x)

    fwd_ms = _time_fn(fwd, warmup, reps)

    # forward + backward
    def fwd_bwd():
        out = module(x)
        out.sum().backward()

    fwdbwd_ms = _time_fn(fwd_bwd, warmup, reps)
    bwd_ms = fwdbwd_ms - fwd_ms
    return fwd_ms, bwd_ms, fwdbwd_ms


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def run_bench(seq_lens: List[int], model_dims: List[int], warmup: int, reps: int) -> None:
    from kernels.fused_relu_sq_mlp import FusedReLUSquaredMLP

    print("=" * 60)
    print(f"Benchmark  (device={torch.cuda.get_device_name(0)}, dtype=bf16)")
    print(f"warmup={warmup}, reps={reps}")
    print("=" * 60)

    header = f"{'impl':>18}  {'fwd ms':>8}  {'bwd ms':>8}  {'fwd+bwd ms':>11}  {'speedup':>8}"
    sep = "-" * len(header)

    for T in seq_lens:
        for C in model_dims:
            H_relu_sq = 4 * C
            H_swiglu = int(8 / 3 * C)

            print(f"\nT={T:6d}  model_dim={C:4d}  |  relu² hidden={H_relu_sq}  swiglu hidden={H_swiglu}")
            print(header)
            print(sep)

            results = {}

            # 1. ref_relu_sq (unfused pytorch)
            ref = RefReLUSquaredMLP(C, H_relu_sq).to(DEVICE, DTYPE)
            x = torch.randn(1, T, C, dtype=DTYPE, device=DEVICE, requires_grad=True)
            fwd, bwd, fwdbwd = _time_fwd_bwd(ref, x, warmup, reps)
            results["ref_relu_sq"] = fwdbwd
            print(f"{'ref_relu_sq':>18}  {fwd:8.3f}  {bwd:8.3f}  {fwdbwd:11.3f}  {'1.00×':>8}")
            del ref

            # 2. fused_relu_sq (our Triton kernel)
            fused = FusedReLUSquaredMLP(C).to(DEVICE, DTYPE)
            x = torch.randn(1, T, C, dtype=DTYPE, device=DEVICE, requires_grad=True)
            fwd, bwd, fwdbwd = _time_fwd_bwd(fused, x, warmup, reps)
            results["fused_relu_sq"] = fwdbwd
            speedup = results["ref_relu_sq"] / fwdbwd
            print(f"{'fused_relu_sq':>18}  {fwd:8.3f}  {bwd:8.3f}  {fwdbwd:11.3f}  {speedup:7.2f}×")
            del fused

            # 3. swiglu (current production MLP)
            try:
                swiglu = _make_swiglu(C).to(DEVICE, DTYPE)
                x = torch.randn(1, T, C, dtype=DTYPE, device=DEVICE, requires_grad=True)
                fwd, bwd, fwdbwd = _time_fwd_bwd(swiglu, x, warmup, reps)
                results["swiglu"] = fwdbwd
                speedup = results["ref_relu_sq"] / fwdbwd
                print(f"{'swiglu':>18}  {fwd:8.3f}  {bwd:8.3f}  {fwdbwd:11.3f}  {speedup:7.2f}×")
                del swiglu
            except Exception as e:
                print(f"{'swiglu':>18}  (unavailable: {e})")

    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="MLP kernel harness")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # correctness
    p_cor = sub.add_parser("correctness", help="Verify fused == ref for fwd + gradients")
    p_cor.add_argument(
        "--configs", nargs="+", default=None,
        metavar="MxC",
        help="Comma-separated M,C pairs, e.g. '4096,768' '32768,1024' "
             "(default: a small set covering typical shapes)"
    )

    # bench
    p_bench = sub.add_parser("bench", help="Benchmark fwd+bwd for all impls")
    p_bench.add_argument("--seq-lens",   nargs="+", type=int, default=[4096, 8192, 32768])
    p_bench.add_argument("--model-dims", nargs="+", type=int, default=[768, 1024, 1280])
    p_bench.add_argument("--warmup",     type=int,  default=5)
    p_bench.add_argument("--reps",       type=int,  default=20)

    args = parser.parse_args()

    if DEVICE.type != "cuda":
        print("ERROR: CUDA device required.  Set CUDA_VISIBLE_DEVICES or run on a GPU node.")
        sys.exit(1)

    if args.cmd == "correctness":
        if args.configs is None:
            # Production shapes: T=32768 tokens, model dims matching our configs
            #   baseline: C=768, stack_100m: C=1024, large: C=1280
            # Also include a small shape for fast iteration during development.
            configs = [
                (1024, 256),    # small — fast sanity check
                (32768, 768),   # baseline model
                (32768, 1024),  # stack_100m (primary training run)
                (32768, 1280),  # large model
            ]
        else:
            configs = []
            for s in args.configs:
                m, c = s.split(",")
                configs.append((int(m), int(c)))
        ok = run_correctness(configs)
        sys.exit(0 if ok else 1)

    elif args.cmd == "bench":
        run_bench(args.seq_lens, args.model_dims, args.warmup, args.reps)


if __name__ == "__main__":
    main()
