"""Correctness of the production Triton attention kernels.

Two layers of checks, both CUDA-only:
  1. Kernel vs exact reference: cdb_bim_v18 and varlen_bim_v2 against the chunked
     naive PyTorch attention, forward and backward, via benchmarks/attention_harness.py
     (the same checks `attention_harness.py correctness` runs, at a small T).
  2. Model integration: TS2TSAttention(backend='triton') fed by the real mask creators
     matches TS2TSAttention(backend='flex') on the same weights, so the creator ->
     TritonMaskInputs -> kernel dispatch path is exercised end to end.
"""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernels need CUDA")

_HARNESS_PATH = Path(__file__).resolve().parents[2] / "benchmarks" / "attention_harness.py"


@pytest.fixture(scope="module")
def harness():
    spec = importlib.util.spec_from_file_location("attention_harness", _HARNESS_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # its dataclasses resolve annotations via sys.modules
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("mask_name,impl", [
    ("doc_causal", "varlen_bim_v2"),
    ("cross_doc_bitmask", "cdb_bim_v18"),
])
def test_kernel_matches_naive_reference(harness, mask_name, impl):
    seq_len, doc_len = 2048, 128
    mask_type = harness.MaskType(mask_name)
    n_links = seq_len // doc_len if mask_type == harness.MaskType.CROSS_DOC_BITMASK else 0
    q, k, v, mask_inputs = harness.make_synthetic_batch(
        seq_len, doc_len, num_heads=4, head_dim=64, dtype=torch.bfloat16,
        device=torch.device("cuda"), n_links=n_links,
    )
    report = harness.check_correctness(
        mask_type, q, k, v, mask_inputs, impls_filter=["naive_pytorch", impl],
    )
    results = {r.name: r for r in report.results}
    assert impl in results, f"{impl} not run; got {sorted(results)}"
    r = results[impl]
    assert r.error is None, r.error
    assert r.fwd_pass, f"forward max err {r.fwd_max_err:.3e}"
    assert r.bwd_pass, (
        f"backward max err q/k/v {r.bwd_q_max_err:.3e}/{r.bwd_k_max_err:.3e}/{r.bwd_v_max_err:.3e}"
    )


def _spans(seq_len, doc_len):
    from data.collate import DocSpan
    return [
        DocSpan(doc_id=i, normed_identifier=f"d{i}", start=s, end=min(s + doc_len, seq_len),
                truncated=False, outgoing_identifiers=[], raw_identifier=f"d{i}")
        for i, s in enumerate(range(0, seq_len, doc_len))
    ]


def _run(attn, x, mask):
    x = x.detach().clone().requires_grad_(True)
    y = attn(x, mask)
    y.float().pow(2).mean().backward()
    return y.detach().float(), x.grad.detach().float()


@pytest.mark.parametrize("mask_type", ["doc_causal", "cross_doc_link"])
def test_triton_backend_matches_flex(mask_type):
    from model.modules.attention import TS2TSAttention
    from model.graph_traversal.block_mask_creator import (
        create_doc_causal_block_mask, create_doc_causal_triton_mask,
    )
    from model.graph_traversal.cross_doc_mask import CrossDocLinkMaskCreator
    from model.graph_traversal.null_link_detector import NullLinkDetector

    torch.manual_seed(0)
    device = torch.device("cuda")
    seq_len, doc_len, dim, heads = 2048, 256, 256, 4
    tokens = torch.zeros(1, seq_len, dtype=torch.long, device=device)
    spans = _spans(seq_len, doc_len)

    if mask_type == "doc_causal":
        triton_mask = create_doc_causal_triton_mask(tokens, spans)
        flex_mask = create_doc_causal_block_mask(tokens, spans)
    else:
        # Each doc links back to the previous one from its midpoint onward.
        link_to_target = {s.start + doc_len // 2: [s.doc_id - 1] for s in spans[1:]}

        def make(backend):
            creator = CrossDocLinkMaskCreator(NullLinkDetector(), max_grants=64, backend=backend)
            return creator(tokens, spans, link_to_target=link_to_target)

        triton_mask, flex_mask = make("triton"), make("flex")
        assert triton_mask.bim64 is not None

    attn = TS2TSAttention(dim=dim, num_heads=heads, max_seq_len=seq_len).to(device)
    x = torch.randn(1, seq_len, dim, device=device, dtype=torch.bfloat16)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        attn.backend = "triton"
        y_t, gx_t = _run(attn, x, triton_mask)
        attn.backend = "flex"
        y_f, gx_f = _run(attn, x, flex_mask)

    assert torch.isfinite(y_t).all() and torch.isfinite(gx_t).all()
    torch.testing.assert_close(y_t, y_f, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(gx_t, gx_f, atol=5e-2, rtol=5e-2)
