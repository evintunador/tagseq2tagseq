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


# ─── gold-aux gradient ────────────────────────────────────────────────────────────

from eval.link_injection_grid import attach_gold_aux, derange_gold_aux


def _rec_gold(idx, gold=True, fired=True):
    r = _rec(idx, fired=fired)
    r.gold_aux_tokens = [900 + idx] if gold else None
    return r


def test_records_roundtrip_with_gold_and_legacy_files():
    recs = [_rec_gold(0), _rec_gold(1, gold=False)]
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "recs.jsonl")
        save_records(recs, p)
        assert load_records(p) == recs
        # A record file written before gold existed (no key) still loads, gold=None.
        import json
        legacy = {k: v for k, v in json.loads(open(p).readline()).items()
                  if k != "gold_aux_tokens"}
        with open(p, "w") as f:
            f.write(json.dumps(legacy) + "\n")
        assert load_records(p)[0].gold_aux_tokens is None


def test_derange_gold_only_keys_fired_gold_records_no_fixed_points():
    recs = [_rec_gold(i) for i in range(5)] + [_rec_gold(5, gold=False), _rec_gold(6, fired=False)]
    mapping = derange_gold_aux(recs, seed=0)
    assert set(mapping) == {0, 1, 2, 3, 4}
    for r in recs[:5]:
        assert mapping[r.item_index] != [r.gold_aux_tokens]
        assert len(mapping[r.item_index]) == 1  # one-element aux list


def test_attach_gold_aux_rejects_non_sciq():
    import pytest
    with pytest.raises(ValueError):
        attach_gold_aux([_rec(0)], "hotpotqa", lambda t: [1])


def test_attach_gold_aux_sciq_by_item_index(monkeypatch):
    import eval.link_injection_grid as g
    fake = [{"support": "alpha beta"}, {"support": ""}, {"support": " gamma "}]

    class _DS:
        def __init__(self, rows): self.rows = rows
        def __iter__(self): return iter(self.rows)
    import types, sys
    fake_mod = types.SimpleNamespace(load_dataset=lambda *a, **k: _DS(fake))
    monkeypatch.setitem(sys.modules, "datasets", fake_mod)
    recs = [_rec(0), _rec(1), _rec(2), _rec(7)]
    n = attach_gold_aux(recs, "sciq", enc=lambda t: [len(t)])
    assert n == 2
    assert recs[0].gold_aux_tokens == [len("alpha beta")]
    assert recs[1].gold_aux_tokens is None          # empty support
    assert recs[2].gold_aux_tokens == [len("gamma")]  # stripped
    assert recs[3].gold_aux_tokens is None          # out of range


def _grid_gold(n, retrieved_help, gold_help):
    """Per-checkpoint scores where retrieved aux helps by retrieved_help and gold aux
    helps by gold_help (both via grant); concat halves it; placebos do nothing."""
    out = {}
    for k in range(n):
        base = 5.0 + 0.01 * k
        out[k] = {
            "baseline": base, "grant": base - retrieved_help, "concat": base - retrieved_help / 2,
            "invisible": base, "placebo": base,
            "grant_gold": base - gold_help, "concat_gold": base - gold_help / 2,
            "placebo_gold": base,
        }
    return out


def test_aggregate_gold_block_and_relevance_slope():
    cross = _grid_gold(40, retrieved_help=0.2, gold_help=1.0)
    dc = _grid_gold(40, retrieved_help=0.2, gold_help=0.4)
    agg = aggregate_grid(cross, dc)
    assert abs(agg["aux_lift_grant_gold_cross"]["mean"] - 1.0) < 1e-9
    assert abs(agg["aux_lift_grant_gold_doc_causal"]["mean"] - 0.4) < 1e-9
    # slope = grant - grant_gold = extra lift from a better aux
    assert abs(agg["relevance_slope_cross"]["mean"] - 0.8) < 1e-9
    assert abs(agg["relevance_slope_doc_causal"]["mean"] - 0.2) < 1e-9
    assert abs(agg["relevance_slope_interaction"]["mean"] - 0.6) < 1e-9
    assert agg["relevance_slope_interaction"]["significant"]
    assert abs(agg["training_grant_gold_interaction"]["mean"] - 0.6) < 1e-9
    # retrieved-aux block unchanged by the gold cells
    assert abs(agg["training_grant_interaction"]["mean"]) < 1e-9
    assert abs(agg["placebo_sep_gold_cross"]["mean"] - 1.0) < 1e-9


def test_aggregate_without_gold_cells_emits_no_gold_keys():
    cross = {k: {c: 1.0 for c in ("baseline", "grant", "concat", "invisible", "placebo")} for k in range(3)}
    agg = aggregate_grid(cross, cross)
    assert not any(k.endswith("_gold_interaction") or k.startswith("relevance_slope") for k in agg)
