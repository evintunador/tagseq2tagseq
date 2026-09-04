"""
tests/eval/test_link_injection_grid.py — CPU unit tests for the causal 2x2
link-injection eval's pure logic (derangement placebo, serialization, aggregation,
interaction). No model / CUDA required. The model-scoring loop (score_grid) is a thin
wrapper over already-tested eval.scoring primitives and is exercised in the GPU run.
"""
import os
import tempfile

from eval.link_injection_grid import (
    AnnotatedRecord, aggregate_grid, derange_aux, load_records, save_records,
)


def _rec(idx, fired=True, aux=None, is_mc=True, label=0):
    return AnnotatedRecord(
        benchmark="hellaswag", item_index=idx, is_mc=is_mc,
        context_tokens=[1, 2, 3], completions=[[4], [5]], label=label,
        aux_token_lists=(aux if aux is not None else [[100 + idx]]),
        aux_raw_identifiers=[f"Title_{idx}"], target_str=f"Title_{idx}",
        link_opener_prob=0.5, link_fired=fired,
    )


# ─── derangement ────────────────────────────────────────────────────────────────

def test_derange_no_fixed_points():
    recs = [_rec(i) for i in range(6)]
    mapping = derange_aux(recs, seed=0)
    assert set(mapping) == {r.item_index for r in recs}
    for r in recs:
        # placebo aux must differ from the record's own aux (no fixed point)
        assert mapping[r.item_index] != r.aux_token_lists


def test_derange_only_keys_fired_with_aux():
    recs = [_rec(0), _rec(1, fired=False), _rec(2, aux=[[]])]
    mapping = derange_aux(recs, seed=0)
    assert set(mapping) == {0}  # 1 not fired; 2 has only-empty aux


def test_derange_single_fired_falls_back_to_own():
    recs = [_rec(0), _rec(1, fired=False)]
    mapping = derange_aux(recs, seed=0)
    assert mapping == {0: recs[0].aux_token_lists}


def test_derange_deterministic():
    recs = [_rec(i) for i in range(5)]
    assert derange_aux(recs, seed=7) == derange_aux(recs, seed=7)


# ─── serialization ────────────────────────────────────────────────────────────────

def test_records_roundtrip():
    recs = [_rec(0), _rec(1, is_mc=False, label=None)]
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "recs.jsonl")
        save_records(recs, p)
        loaded = load_records(p)
    assert loaded == recs


def test_gold_completion_mc_vs_fill():
    mc = _rec(0, is_mc=True, label=1)
    assert mc.gold_completion() == [5]
    fill = AnnotatedRecord(
        benchmark="lambada", item_index=0, is_mc=False, context_tokens=[1],
        completions=[[9, 9]], label=None, aux_token_lists=[[1]],
        aux_raw_identifiers=["t"], target_str="t", link_opener_prob=0.1, link_fired=True,
    )
    assert fill.gold_completion() == [9, 9]


# ─── aggregation / interaction ────────────────────────────────────────────────────

def _grid(help_per_item):
    """Build a per-checkpoint score dict where the real aux lowers gold NLL by
    `help_per_item[k]` (grant), raw-concat helps half as much, placebo not at all."""
    out = {}
    for k, h in help_per_item.items():
        base = 5.0
        out[k] = {
            "baseline": base,
            "grant": base - h,          # aux helps by h (lower NLL)
            "concat": base - h / 2,     # raw-concat helps half as much
            "invisible": base,          # masked out → equals baseline
            "placebo": base,            # wrong aux → no help
        }
    return out


def test_interaction_positive_when_crossdoc_uses_aux_more():
    # cross-doc ckpt gains 1.0/item from the aux; doc-causal gains only 0.2/item.
    cross = _grid({i: 1.0 for i in range(30)})
    dc = _grid({i: 0.2 for i in range(30)})
    agg = aggregate_grid(cross, dc)

    inter = agg["training_grant_interaction"]
    assert inter["n"] == 30
    assert abs(inter["mean"] - 0.8) < 1e-9      # 1.0 - 0.2
    assert inter["significant"]                  # CI excludes 0

    # aux helps both, but cross more
    assert agg["aux_lift_grant_cross"]["mean"] > agg["aux_lift_grant_doc_causal"]["mean"]
    # mechanism: grant beats raw-concat (concat - grant > 0)
    assert agg["mechanism_cross"]["mean"] > 0
    # placebo separation: real grant beats placebo (placebo - grant > 0)
    assert agg["placebo_sep_cross"]["mean"] > 0
    # invisible sanity ~ 0
    assert abs(agg["invisible_check_cross"]["mean"]) < 1e-9


def test_interaction_not_significant_when_equal():
    cross = _grid({i: 0.5 for i in range(30)})
    dc = _grid({i: 0.5 for i in range(30)})
    agg = aggregate_grid(cross, dc)
    assert abs(agg["training_grant_interaction"]["mean"]) < 1e-9
    assert not agg["training_grant_interaction"]["significant"]


def test_interaction_pairs_only_shared_items():
    cross = _grid({i: 1.0 for i in range(10)})
    dc = _grid({i: 0.2 for i in range(5)})   # only items 0..4 scored
    agg = aggregate_grid(cross, dc)
    assert agg["training_grant_interaction"]["n"] == 5
