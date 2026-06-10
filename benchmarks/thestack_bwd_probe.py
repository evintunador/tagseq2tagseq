"""Backward correctness test for cross-doc BIM kernels on real thestack activations.

Tests each registered kernel variant against FlexAttention as the reference,
using fixtures that cover two distinct failure modes of cdb_bim_v18:

  zero_<N>.pt  — packs from sparse-bucket steps where v18 backward returns
                 all-zero gradients instead of the correct non-zero values.

  nan_0_qkv_pre.pt / nan_0_qkv_nan.pt  — the exact q/k/v tensors from the
                 layer-0 attention at training steps 97 and 98.  Step 97 is the
                 pre-NaN step (clean input, should produce finite gradients).
                 Step 98 is the NaN step (v18 produces NaN in dQ; flex is finite).
                 These tensors were captured from job 41841 with real trained weights.

The zero fixtures use pack-structure masks only (random q/k/v won't reproduce the
zero-backward bug as shown by thestack_nan_probe.py).  The qkv fixtures use the
real trained activations from job 41841 (steps 97–98) which are required to
reproduce the NaN.

Usage:
    CUDA_VISIBLE_DEVICES=1 python benchmarks/thestack_bwd_probe.py
    CUDA_VISIBLE_DEVICES=1 python benchmarks/thestack_bwd_probe.py --impls cdb_bim_v18 cdb_bim_v19
    CUDA_VISIBLE_DEVICES=1 python benchmarks/thestack_bwd_probe.py --head-dim 64
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.attention_harness import (
    FIXTURES_DIR,
    MaskInputs,
    _build_flex_cross_doc_block_mask,
    _clone_requires_grad,
    _compiled_flex,
    _get_bim128,
    _get_bim64,
    _p99,
    _spans_to_cu_seqlens,
    _spans_to_document_ids,
    _to_bhnd,
    _to_thd,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THESTACK_FIXTURES_DIR = Path(__file__).parent.parent / "tests" / "fixtures" / "thestack_packs"


# ---------------------------------------------------------------------------
# Registered kernel impls
# ---------------------------------------------------------------------------

IMPL_REGISTRY: Dict[str, callable] = {}


def _register(name):
    def decorator(fn):
        IMPL_REGISTRY[name] = fn
        return fn
    return decorator


@_register("cdb_bim_v18")
def _impl_v18(q, k, v, mask_inputs: MaskInputs, scale: float) -> torch.Tensor:
    from kernels.cross_doc_bitmask_bim_v18 import triton_attn_cross_doc_bitmask_bim_v18
    bim128 = _get_bim128(mask_inputs)
    bim64  = _get_bim64(mask_inputs)
    return triton_attn_cross_doc_bitmask_bim_v18(
        q, k, v, mask_inputs.document_ids,
        mask_inputs.q_bitmasks, mask_inputs.kv_bitmasks, bim128, bim64, scale,
    )


def _try_register_v19():
    """Register v19 if its module exists."""
    try:
        from kernels.cross_doc_bitmask_bim_v19 import triton_attn_cross_doc_bitmask_bim_v19

        @_register("cdb_bim_v19")
        def _impl_v19(q, k, v, mask_inputs: MaskInputs, scale: float) -> torch.Tensor:
            from kernels.cross_doc_bitmask_bim_v19 import triton_attn_cross_doc_bitmask_bim_v19
            bim128 = _get_bim128(mask_inputs)
            bim64  = _get_bim64(mask_inputs)
            return triton_attn_cross_doc_bitmask_bim_v19(
                q, k, v, mask_inputs.document_ids,
                mask_inputs.q_bitmasks, mask_inputs.kv_bitmasks, bim128, bim64, scale,
            )
    except ImportError:
        pass


_try_register_v19()


# ---------------------------------------------------------------------------
# Fixture loading
# ---------------------------------------------------------------------------

def _load_pack_fixture(path: Path, num_heads: int, head_dim: int,
                       dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor,
                                                     torch.Tensor, MaskInputs]:
    """Load a pack-only fixture (doc_spans + link_to_target) with random q/k/v.

    Mirrors load_fixture_batch from attention_harness but reads from thestack_packs/.
    """
    data = torch.load(str(path), weights_only=False)

    doc_spans_raw = data["doc_spans"]
    actual_T = int(data["seq_len"])
    raw_to_idx = {d["doc_id"]: i for i, d in enumerate(doc_spans_raw)}
    from types import SimpleNamespace
    doc_spans = [
        SimpleNamespace(doc_id=raw_to_idx[d["doc_id"]], start=d["start"], end=d["end"])
        for d in doc_spans_raw
    ]
    link_to_target = {
        int(k): [raw_to_idx[int(v)] for v in vs]
        for k, vs in data["link_to_target"].items()
    }

    cu_seqlens, max_seqlen = _spans_to_cu_seqlens(doc_spans, DEVICE)
    document_ids = _spans_to_document_ids(doc_spans, actual_T, DEVICE)

    n_grants = int(data.get("n_grants", 0))
    max_grants = int(data.get("max_grants", 256))
    n_chunks = max(1, (max_grants + 63) // 64)

    q_bm_list = [torch.zeros(actual_T, dtype=torch.int64, device=DEVICE) for _ in range(n_chunks)]
    kv_bm_list = [torch.zeros(actual_T, dtype=torch.int64, device=DEVICE) for _ in range(n_chunks)]
    grant_idx = 0
    for link_pos, target_ids in sorted(link_to_target.items()):
        for tid in target_ids:
            ls = next((s for s in doc_spans if s.start < link_pos <= s.end), None)
            ts_span = next((s for s in doc_spans if s.doc_id == tid), None)
            if ls is None or ts_span is None:
                continue
            gs, ge = link_pos, min(actual_T, ls.end)
            tss, tse = max(0, ts_span.start), min(actual_T, ts_span.end)
            if gs >= ge or tss >= tse or grant_idx >= max_grants:
                continue
            chunk = grant_idx // 64
            bit_pos = grant_idx % 64
            bit = (1 << bit_pos) if bit_pos < 63 else -(1 << 63)
            q_bm_list[chunk][gs:ge] |= bit
            kv_bm_list[chunk][tss:tse] |= bit
            grant_idx += 1
    q_bitmasks = torch.stack(q_bm_list)
    kv_bitmasks = torch.stack(kv_bm_list)

    from torch.nn.attention.flex_attention import create_block_mask
    def _dc_mod(b, h, qi, ki):
        return (qi >= ki) & (document_ids[qi] == document_ids[ki])
    flex_doc_causal_bm = create_block_mask(
        _dc_mod, B=None, H=None, Q_LEN=actual_T, KV_LEN=actual_T, device=DEVICE,
    )
    flex_cross_doc_bm = None
    if grant_idx > 0:
        flex_cross_doc_bm = _build_flex_cross_doc_block_mask(
            actual_T, document_ids, q_bitmasks, kv_bitmasks, DEVICE,
        )

    bim = bim64 = None
    try:
        from model.graph_traversal.cross_doc_mask import CrossDocLinkMaskCreator
        from kernels.cross_doc_bitmask_bim_v12 import _build_bim_128
        from kernels.cross_doc_bitmask_bim_v17 import _build_bim_64
        c = CrossDocLinkMaskCreator.__new__(CrossDocLinkMaskCreator)
        c.triton_block_size = 64
        c._n_chunks = n_chunks
        bim = CrossDocLinkMaskCreator._build_block_interaction_mask(
            c, actual_T, document_ids, list(q_bitmasks), list(kv_bitmasks), DEVICE,
        )
        bim64 = _build_bim_64(actual_T, document_ids, q_bitmasks, kv_bitmasks, DEVICE, n_chunks)
    except Exception:
        pass

    # Random q/k/v — seeded by fixture filename for reproducibility
    seed = hash(path.stem) & 0xFFFFFFFF
    gen = torch.Generator(device=DEVICE)
    gen.manual_seed(seed)
    q = torch.randn(actual_T, num_heads, head_dim, dtype=dtype, device=DEVICE,
                    generator=gen).requires_grad_(True)
    k = torch.randn(actual_T, num_heads, head_dim, dtype=dtype, device=DEVICE,
                    generator=gen).requires_grad_(True)
    v = torch.randn(actual_T, num_heads, head_dim, dtype=dtype, device=DEVICE,
                    generator=gen).requires_grad_(True)

    mask_inputs = MaskInputs(
        seq_len=actual_T,
        doc_spans=doc_spans,
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
        document_ids=document_ids,
        dense_mask=None,
        q_bitmasks=q_bitmasks,
        kv_bitmasks=kv_bitmasks,
        flex_doc_causal_block_mask=flex_doc_causal_bm,
        flex_cross_doc_block_mask=flex_cross_doc_bm,
        bim=bim,
    )
    # Attach bim64 so impl wrappers can retrieve it via _get_bim64
    mask_inputs._bim64 = bim64
    return q, k, v, mask_inputs


def _load_qkv_fixture(path: Path,
                      pack_fixture_path: Optional[Path] = None,
                      num_heads: int = 8,
                      head_dim: int = 128,
                      dtype: torch.dtype = torch.bfloat16,
                      ) -> Tuple[torch.Tensor, torch.Tensor,
                                 torch.Tensor, MaskInputs]:
    """Load a captured q/k/v fixture (real trained activations).

    If the fixture's block_mask is empty (e.g. captured from a flex-backend run
    where no TritonMaskInputs exist), a ``pack_fixture_path`` must be supplied to
    provide the mask structure. The q/k/v from ``path`` replace the random tensors.
    """
    data = torch.load(str(path), weights_only=False)
    bm_data = data["block_mask"]

    # If block_mask has the triton fields, use them directly.
    if "q_bitmasks" in bm_data:
        q_bitmasks  = bm_data["q_bitmasks"].to(DEVICE)
        kv_bitmasks = bm_data["kv_bitmasks"].to(DEVICE)
        document_ids = bm_data["document_ids"].to(DEVICE)
        bim  = bm_data["bim"]
        bim64 = bm_data["bim64"]
    else:
        # Flex-backend capture: no triton mask fields. Must supply pack_fixture_path.
        if pack_fixture_path is None:
            raise KeyError("q_bitmasks")
        _, _, _, mask_inputs = _load_pack_fixture(pack_fixture_path, num_heads, head_dim, dtype)
        # The capture has T=max_seq_len+1 (padded); the pack fixture has the
        # actual seq_len. Trim to the pack's seq_len to match the mask shape.
        T_pack = mask_inputs.seq_len
        dtype_load = torch.bfloat16
        q_cap = data["q"][:T_pack].to(DEVICE, dtype=dtype_load).requires_grad_(True)
        k_cap = data["k"][:T_pack].to(DEVICE, dtype=dtype_load).requires_grad_(True)
        v_cap = data["v"][:T_pack].to(DEVICE, dtype=dtype_load).requires_grad_(True)
        return q_cap, k_cap, v_cap, mask_inputs

    T = document_ids.shape[0]
    n_chunks = q_bitmasks.shape[0]

    from torch.nn.attention.flex_attention import create_block_mask
    def _dc_mod(b, h, qi, ki):
        return (qi >= ki) & (document_ids[qi] == document_ids[ki])
    flex_doc_causal_bm = create_block_mask(
        _dc_mod, B=None, H=None, Q_LEN=T, KV_LEN=T, device=DEVICE,
    )
    flex_cross_doc_bm = _build_flex_cross_doc_block_mask(
        T, document_ids, q_bitmasks, kv_bitmasks, DEVICE,
    )

    from types import SimpleNamespace
    # Reconstruct doc_spans from document_ids (consecutive runs)
    doc_spans = []
    prev_id = None
    for i in range(T):
        did = document_ids[i].item()
        if did != prev_id:
            if prev_id is not None:
                doc_spans[-1].end = i
            doc_spans.append(SimpleNamespace(doc_id=did, start=i, end=T))
            prev_id = did

    cu_seqlens, max_seqlen = _spans_to_cu_seqlens(doc_spans, DEVICE)

    # Convert q/k/v from fp32 to the test dtype at load time
    dtype = torch.bfloat16
    q = data["q"].to(DEVICE, dtype=dtype).requires_grad_(True)
    k = data["k"].to(DEVICE, dtype=dtype).requires_grad_(True)
    v = data["v"].to(DEVICE, dtype=dtype).requires_grad_(True)

    mask_inputs = MaskInputs(
        seq_len=T,
        doc_spans=doc_spans,
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
        document_ids=document_ids,
        dense_mask=None,
        q_bitmasks=q_bitmasks,
        kv_bitmasks=kv_bitmasks,
        flex_doc_causal_block_mask=flex_doc_causal_bm,
        flex_cross_doc_block_mask=flex_cross_doc_bm,
        bim=bim,
    )
    mask_inputs._bim64 = bim64
    return q, k, v, mask_inputs


# Patch _get_bim64 to also check mask_inputs._bim64 so the registered impls work
_orig_get_bim64 = _get_bim64.__wrapped__ if hasattr(_get_bim64, '__wrapped__') else None

def _patched_get_bim64(mask_inputs):
    cached = getattr(mask_inputs, '_bim64', None)
    if cached is not None:
        return cached
    return _get_bim64(mask_inputs)


# ---------------------------------------------------------------------------
# Per-fixture result
# ---------------------------------------------------------------------------

@dataclass
class FixtureResult:
    fixture: str
    impl: str
    fwd_has_nan: bool = False
    fwd_has_inf: bool = False
    bwd_has_nan: bool = False
    bwd_has_inf: bool = False
    bwd_dq_zero: bool = False   # all-zero dQ when flex is non-zero
    bwd_dq_max_err: float = 0.0
    bwd_dk_max_err: float = 0.0
    bwd_dv_max_err: float = 0.0
    bwd_max_err: float = 0.0
    bwd_p99_err: float = 0.0
    error: Optional[str] = None

    @property
    def pass_(self) -> bool:
        return (not self.error and not self.fwd_has_nan and not self.fwd_has_inf
                and not self.bwd_has_nan and not self.bwd_has_inf
                and not self.bwd_dq_zero and self.bwd_max_err <= 2e-1)


# ---------------------------------------------------------------------------
# Run one (fixture, impl) pair
# ---------------------------------------------------------------------------

def _run_flex_ref(q, k, v, mask_inputs: MaskInputs, scale: float):
    """Flex reference — uses cross_doc BlockMask if available, else doc_causal."""
    bm = mask_inputs.flex_cross_doc_block_mask or mask_inputs.flex_doc_causal_block_mask
    q4, k4, v4 = _to_bhnd(q), _to_bhnd(k), _to_bhnd(v)
    return _to_thd(_compiled_flex(q4, k4, v4, block_mask=bm, scale=scale))


def check_fixture(
    fixture_label: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask_inputs: MaskInputs,
    impl_name: str,
    impl_fn: callable,
    bwd_atol: float = 2e-1,
) -> FixtureResult:
    result = FixtureResult(fixture=fixture_label, impl=impl_name)
    scale = q.shape[-1] ** -0.5

    # Flex reference
    q_ref = _clone_requires_grad(q)
    k_ref = _clone_requires_grad(k)
    v_ref = _clone_requires_grad(v)
    try:
        out_ref = _run_flex_ref(q_ref, k_ref, v_ref, mask_inputs, scale)
        out_ref.backward(torch.ones_like(out_ref))
        ref_dq = q_ref.grad.clone()
        ref_dk = k_ref.grad.clone()
        ref_dv = v_ref.grad.clone()
    except Exception as e:
        result.error = f"flex reference failed: {str(e)[:100]}"
        return result

    # Patch _get_bim64 for this call
    import benchmarks.attention_harness as _harness
    _orig = _harness._get_bim64
    _harness._get_bim64 = _patched_get_bim64

    # Impl under test
    q_i = _clone_requires_grad(q)
    k_i = _clone_requires_grad(k)
    v_i = _clone_requires_grad(v)
    try:
        out_i = impl_fn(q_i, k_i, v_i, mask_inputs, scale)
    except Exception as e:
        _harness._get_bim64 = _orig
        result.error = f"fwd error: {str(e)[:100]}"
        return result
    finally:
        _harness._get_bim64 = _orig

    result.fwd_has_nan = bool(torch.isnan(out_i).any())
    result.fwd_has_inf = bool(torch.isinf(out_i).any())

    _harness._get_bim64 = _patched_get_bim64
    try:
        out_i.backward(torch.ones_like(out_i))
    except Exception as e:
        _harness._get_bim64 = _orig
        result.error = f"bwd error: {str(e)[:100]}"
        return result
    finally:
        _harness._get_bim64 = _orig

    dq_i = q_i.grad
    dk_i = k_i.grad
    dv_i = v_i.grad

    result.bwd_has_nan = bool(
        torch.isnan(dq_i).any() or torch.isnan(dk_i).any() or torch.isnan(dv_i).any()
    )
    result.bwd_has_inf = bool(
        torch.isinf(dq_i).any() or torch.isinf(dk_i).any() or torch.isinf(dv_i).any()
    )

    # Check for all-zero dQ when flex reference is non-zero
    ref_dq_norm = ref_dq.float().norm().item()
    impl_dq_norm = dq_i.float().norm().item()
    result.bwd_dq_zero = (ref_dq_norm > 1e-6 and impl_dq_norm < 1e-12)

    dq_diff = (dq_i.float() - ref_dq.float()).abs()
    dk_diff = (dk_i.float() - ref_dk.float()).abs()
    dv_diff = (dv_i.float() - ref_dv.float()).abs()
    result.bwd_dq_max_err = dq_diff.max().item()
    result.bwd_dk_max_err = dk_diff.max().item()
    result.bwd_dv_max_err = dv_diff.max().item()
    result.bwd_max_err = max(result.bwd_dq_max_err, result.bwd_dk_max_err, result.bwd_dv_max_err)
    result.bwd_p99_err = max(_p99(dq_diff), _p99(dk_diff), _p99(dv_diff))

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--impls", nargs="+", default=None,
                   help="Impl names to test (default: all registered). "
                        "E.g. --impls cdb_bim_v18 cdb_bim_v19")
    p.add_argument("--num-heads", type=int, default=8,
                   help="Number of attention heads (default: 8 — matches thestack_cross_doc.yaml)")
    p.add_argument("--head-dim", type=int, default=128,
                   help="Head dimension (default: 128 — matches thestack_cross_doc.yaml)")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--bwd-atol", type=float, default=2e-1)
    args = p.parse_args()

    dtype = getattr(torch, args.dtype)
    impls_filter = set(args.impls) if args.impls else None
    active_impls = {
        name: fn for name, fn in IMPL_REGISTRY.items()
        if impls_filter is None or name in impls_filter
    }

    if not active_impls:
        print(f"No impls matched. Available: {list(IMPL_REGISTRY)}")
        sys.exit(1)

    if not THESTACK_FIXTURES_DIR.exists():
        print(f"ERROR: thestack fixtures directory not found: {THESTACK_FIXTURES_DIR}")
        print("Run: python scripts/generate_thestack_fixtures.py")
        sys.exit(1)

    sep = "=" * 110

    # ── Zero-backward fixtures (pack structure, random q/k/v) ─────────────
    zero_files = sorted(THESTACK_FIXTURES_DIR.glob("zero_*.pt"))

    # ── Real-weights fixtures: triton-trained (from job 41830/41841) ───────
    # Captured at steps 97 (pre-NaN) and 98 (NaN step) using triton_v18 training.
    triton_qkv_files = [
        ("triton:step97_pre", THESTACK_FIXTURES_DIR / "nan_0_qkv_pre.pt"),
        ("triton:step98_nan", THESTACK_FIXTURES_DIR / "nan_0_qkv_nan.pt"),
    ]

    # ── Real-weights fixtures: flex-trained (from job 41851) ─────────────
    # Captured at the same steps using clean flex training — no contamination
    # from v18's incorrect backward updates. Covers all 10 zero-gradient steps
    # plus steps 97-99. Use these to verify bugs are mask/kernel structural
    # rather than weight-magnitude artifacts.
    flex_qkv_files = [
        (f"flex:step{s}", THESTACK_FIXTURES_DIR / f"flex_step_{s}.pt")
        for s in [64, 67, 80, 83, 84, 85, 88, 89, 90, 97, 98, 99]
    ]

    all_results: List[FixtureResult] = []

    print(f"\n{sep}")
    print(f"  THESTACK BACKWARD PROBE  heads={args.num_heads}  head_dim={args.head_dim}  dtype={dtype}")
    print(f"  Impls: {list(active_impls)}")
    print(sep)

    hdr = (f"  {'fixture':35s}  {'impl':18s}"
           f"  {'fwd_nan':7s}  {'bwd_nan':7s}  {'bwd_zero':8s}"
           f"  {'bwd_dq_max':10s}  {'bwd_dk_max':10s}  {'bwd_dv_max':10s}  {'result':6s}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    # Zero fixtures — random q/k/v to check structural backward correctness
    for path in zero_files:
        label = path.stem
        try:
            q, k, v, mask_inputs = _load_pack_fixture(path, args.num_heads, args.head_dim, dtype)
        except Exception as e:
            print(f"  {label}: load error — {e}")
            continue

        for impl_name, impl_fn in active_impls.items():
            r = check_fixture(label, q, k, v, mask_inputs, impl_name, impl_fn, args.bwd_atol)
            all_results.append(r)
            _print_row(r)

    # Real-weights fixtures (triton-trained: steps 97 pre-NaN, 98 NaN)
    for label, path in triton_qkv_files:
        if not path.exists():
            print(f"  {label}: fixture not found ({path}) — skipping")
            continue
        try:
            q, k, v, mask_inputs = _load_qkv_fixture(path)
        except Exception as e:
            print(f"  {label}: load error — {e}")
            continue
        for impl_name, impl_fn in active_impls.items():
            r = check_fixture(label, q, k, v, mask_inputs, impl_name, impl_fn, args.bwd_atol)
            all_results.append(r)
            _print_row(r)

    # Real-weights fixtures (flex-trained: all zero-steps + steps 97-99)
    # Map each flex capture step to its corresponding pack-structure fixture.
    # zero_0..zero_9 correspond to steps 64,67,80,83,84,85,88,89,90,97 in order.
    _zero_steps = [64, 67, 80, 83, 84, 85, 88, 89, 90, 97]
    _flex_pack_map = {s: THESTACK_FIXTURES_DIR / f"zero_{i}.pt"
                      for i, s in enumerate(_zero_steps)}
    _flex_pack_map[98] = THESTACK_FIXTURES_DIR / "nan_0.pt"
    _flex_pack_map[99] = THESTACK_FIXTURES_DIR / "nan_0.pt"

    for label, path in flex_qkv_files:
        if not path.exists():
            print(f"  {label}: fixture not found ({path}) — skipping")
            continue
        step = int(label.split("step")[1])
        pack_path = _flex_pack_map.get(step)
        try:
            q, k, v, mask_inputs = _load_qkv_fixture(
                path, pack_fixture_path=pack_path,
                num_heads=args.num_heads, head_dim=args.head_dim, dtype=dtype,
            )
        except Exception as e:
            print(f"  {label}: load error — {e}")
            continue
        for impl_name, impl_fn in active_impls.items():
            r = check_fixture(label, q, k, v, mask_inputs, impl_name, impl_fn, args.bwd_atol)
            all_results.append(r)
            _print_row(r)

    print("  " + "-" * (len(hdr) - 2))

    n_pass = sum(1 for r in all_results if r.pass_)
    n_fail = len(all_results) - n_pass
    print(f"\n  {len(all_results)} checks  |  {n_pass} PASS  |  {n_fail} FAIL")

    # Summary of failures
    failures = [r for r in all_results if not r.pass_]
    if failures:
        print(f"\n  Failures:")
        for r in failures:
            reasons = []
            if r.error:       reasons.append(f"error={r.error}")
            if r.fwd_has_nan: reasons.append("fwd_NaN")
            if r.bwd_has_nan: reasons.append("bwd_NaN")
            if r.bwd_dq_zero: reasons.append("bwd_dQ=0 (all-zero gradient)")
            if r.bwd_max_err > 2e-1: reasons.append(f"bwd_max_err={r.bwd_max_err:.2e}")
            print(f"    [{r.impl}] {r.fixture}: {', '.join(reasons)}")

    sys.exit(0 if n_fail == 0 else 1)


def _print_row(r: FixtureResult):
    if r.error:
        print(f"  {r.fixture:35s}  {r.impl:18s}  ERROR: {r.error[:50]}")
        return
    fwd_nan = "NaN" if r.fwd_has_nan else "-"
    bwd_nan = "NaN" if r.bwd_has_nan else ("-" if not r.bwd_has_inf else "Inf")
    bwd_zero = "ZERO" if r.bwd_dq_zero else "-"
    verdict = "PASS" if r.pass_ else "FAIL"
    print(
        f"  {r.fixture:35s}  {r.impl:18s}"
        f"  {fwd_nan:7s}  {bwd_nan:7s}  {bwd_zero:8s}"
        f"  {r.bwd_dq_max_err:10.2e}  {r.bwd_dk_max_err:10.2e}  {r.bwd_dv_max_err:10.2e}"
        f"  {verdict:6s}",
        flush=True,
    )


if __name__ == "__main__":
    main()
