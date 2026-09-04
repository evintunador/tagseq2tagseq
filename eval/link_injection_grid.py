"""
eval/link_injection_grid.py — causal 2x2 (model x mask) link-injection eval.

Turns the existing "annotated" injection eval (eval/link_annotator.py +
run_benchmark_annotated) into a controlled causal test. The SAME injected link +
aux doc is scored across a 2x2 grid of

    {cross-doc-trained ckpt, doc-causal-trained ckpt}
      x {grant-on (cross_doc_link), raw-concat (doc_concatenated)}

plus a no-aux baseline and a derangement placebo. The headline quantity is the
INTERACTION — how much MORE the aux doc helps the cross-doc-trained model than the
doc-causal-trained model — not any absolute score. Absolute lift conflates "cross-doc
training taught the model to use links" with "relevant context helps any LM"; only the
interaction separates them.

Two phases:

  1. annotate_items(model, annotator, ...): run the annotator ONCE, with one designated
     model, to inject a link + acquire an aux doc per benchmark item, and cache the
     result (AnnotatedRecord). Holding the annotation fixed across cells is what makes
     the comparison content-matched. Serialize with save_records / load_records so the
     identical aux is replayed to every checkpoint (and later to placebo / gold / LLM
     variants).

  2. score_grid(model, records, ...): for one loaded checkpoint (loaded under a
     cross_doc_link mask via generate.load_inference_model), score every link-fired
     record under each cell + placebo. Run once per checkpoint.

  3. aggregate_grid(cross, doc_causal): combine the two checkpoints' per-record scores
     into the mechanism / training / placebo effects with bootstrap CIs.

The cells map onto tested scoring primitives:
  baseline  score_completion(context_with_link, comp)                 doc_causal, no aux
  grant     score_completion_with_context_docs(aux, ...)              cross_doc_link
  concat    score_completion_concat(aux, ..., 'doc_concatenated')     raw-concat
  invisible score_completion_concat(aux, ..., 'doc_causal')           sanity ~ baseline
  placebo   score_completion_with_context_docs(deranged_aux, ...)     grant-on, wrong aux

Gold-aux gradient (optional, sciq only): each record can also carry the benchmark's own
gold passage (sciq `support`) as `gold_aux_tokens`, scored under the same link through
  grant_gold / concat_gold / placebo_gold
so the relevance slope (gold vs retrieved aux) and the gold-aux training interaction can
be read off the same paired items. hotpotqa is excluded: its annotatable context already
contains the gold supporting sentences.

See docs/link_injection_causal_eval_DESIGN.md.
"""
from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

SEED = 42
BOOTSTRAP_RESAMPLES = 10_000

# Cells scored per checkpoint. "baseline" is the paired reference (link spliced, no
# aux) so every delta isolates the AUX-DOC effect, holding the injected link fixed.
CELLS = ("baseline", "grant", "concat", "invisible", "placebo")
# Extra cells scored only for records that carry a gold aux passage.
GOLD_CELLS = ("grant_gold", "concat_gold", "placebo_gold")
GOLD_AUX_BENCHMARKS = ("sciq",)


# ─── Records ────────────────────────────────────────────────────────────────────

@dataclass
class AnnotatedRecord:
    """One benchmark item after link injection, with everything needed to replay
    scoring across cells and checkpoints. All token fields are plain int lists so
    the record round-trips through JSON."""
    benchmark: str
    item_index: int
    is_mc: bool
    # Injected-prompt context (link syntax spliced in) — the reference context.
    context_tokens: List[int]
    # Completions: MC → one list per choice; fill-in → single list in completions[0].
    completions: List[List[int]]
    label: Optional[int]              # gold choice index (MC) or None (fill-in)
    # Aux doc(s) acquired for the injected link.
    aux_token_lists: List[List[int]]
    aux_raw_identifiers: List[str]
    target_str: str
    link_opener_prob: float
    link_fired: bool
    # Benchmark-native gold aux passage (sciq `support`), attached by attach_gold_aux.
    # None when absent / not requested; older record files load with None.
    gold_aux_tokens: Optional[List[int]] = None

    def gold_completion(self) -> List[int]:
        """The completion whose NLL is the paired continuous signal: the gold choice
        for MC, the single completion for fill-in."""
        if self.is_mc:
            idx = self.label if self.label is not None else 0
            return self.completions[idx]
        return self.completions[0]


def save_records(records: List[AnnotatedRecord], path: str) -> None:
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(asdict(r)) + "\n")


def load_records(path: str) -> List[AnnotatedRecord]:
    out: List[AnnotatedRecord] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(AnnotatedRecord(**json.loads(line)))
    return out


# ─── Placebo (derangement) ───────────────────────────────────────────────────────

def _derange(keyed: List[Tuple[int, Any]], seed: int) -> Dict[int, Any]:
    """{key: value of a DIFFERENT entry}. True derangement (no fixed points), rejection-
    sampled with a rotation fallback; a single entry maps to itself."""
    n = len(keyed)
    if n == 0:
        return {}
    if n == 1:
        # No other content to swap in; reuse own aux (placebo == real for this one).
        return {keyed[0][0]: keyed[0][1]}
    rng = random.Random(seed)
    order = list(range(n))
    for _ in range(1000):
        perm = order[:]
        rng.shuffle(perm)
        if all(perm[i] != i for i in range(n)):
            break
    else:
        perm = order[1:] + order[:1]
    return {keyed[i][0]: keyed[perm[i]][1] for i in range(n)}


def derange_aux(
    records: List[AnnotatedRecord], seed: int = SEED
) -> Dict[int, List[List[int]]]:
    """Map each fired record's index → a DIFFERENT fired record's aux token lists.

    Mirrors the tier2 placebo: swap aux CONTENT across fired examples while keeping
    each record's OWN injected link (so the grant still fires at the same link, but
    attends to wrong-but-plausible content). A true derangement (no fixed points) is
    used so no record keeps its own aux; falls back to a rotation when <2 fired.

    Returns {item_index: placebo_aux_token_lists}. Only fired records are keyed.
    """
    fired = [(r.item_index, r.aux_token_lists)
             for r in records if r.link_fired and any(r.aux_token_lists)]
    return _derange(fired, seed)


def derange_gold_aux(
    records: List[AnnotatedRecord], seed: int = SEED
) -> Dict[int, List[List[int]]]:
    """Placebo for the gold cells: each gold-carrying fired record's index → a
    DIFFERENT record's gold passage (as a one-element aux list)."""
    keyed = [(r.item_index, [list(r.gold_aux_tokens)])
             for r in records
             if r.link_fired and any(r.aux_token_lists) and r.gold_aux_tokens]
    return _derange(keyed, seed)


# ─── Gold aux (relevance-gradient ceiling) ──────────────────────────────────────

def attach_gold_aux(
    records: List[AnnotatedRecord],
    benchmark_name: str,
    enc,
    cache_dir: Optional[str] = None,
) -> int:
    """Attach each record's benchmark-native gold passage as `gold_aux_tokens`.

    sciq: the HF dataset's `support` field (the passage the question was written
    from). `item_index` is the raw validation-split index — SciQDataset keeps raw
    order and only truncates (`items[:limit]`), so indices line up. Empty supports
    leave gold_aux_tokens=None. Returns the number of records that received gold.
    """
    if benchmark_name not in GOLD_AUX_BENCHMARKS:
        raise ValueError(
            f"gold aux is only defined for {GOLD_AUX_BENCHMARKS}; hotpotqa's annotatable "
            "context already contains the gold supporting sentences, so an injected gold "
            "aux would be redundant there."
        )
    from datasets import load_dataset
    raw = load_dataset(
        "allenai/sciq", split="validation",
        cache_dir=cache_dir or os.path.join("data", ".cache", "sciq"),
    )
    supports = [(ex.get("support") or "").strip() for ex in raw]
    n = 0
    for r in records:
        text = supports[r.item_index] if r.item_index < len(supports) else ""
        r.gold_aux_tokens = list(enc(text)) if text else None
        n += bool(r.gold_aux_tokens)
    logger.info("attach_gold_aux(%s): %d/%d records carry a gold passage",
                benchmark_name, n, len(records))
    return n


# ─── Annotation (phase 1) ──────────────────────────────────────────────────────

def annotate_items(
    model,
    annotator,
    benchmark_name: str,
    items: List[Dict[str, Any]],
    device: str = "cuda",
) -> List[AnnotatedRecord]:
    """Inject a link + acquire an aux doc for each benchmark item, ONCE.

    `model` is the single designated annotator (use the cross-doc-link checkpoint —
    it is the model that "knows" links); the resulting records are replayed to every
    checkpoint so the injected link + aux is identical across the grid.

    `items` is the output of eval.nlp_benchmarks._load_benchmark_items: dicts with
    "type" ('mc'|'fill'), "context_tokens", "completion_token_lists" (mc) or
    "completion_tokens" (fill), and "label" (mc).
    """
    records: List[AnnotatedRecord] = []
    for i, item in enumerate(items):
        is_mc = item["type"] == "mc"
        completions = (
            item["completion_token_lists"] if is_mc else [item["completion_tokens"]]
        )
        try:
            ann = annotator.annotate(model, item["context_tokens"], device)
        except Exception as exc:
            logger.warning("annotate_items: item %d failed: %s", i, exc)
            ann = None

        if ann is None:
            records.append(AnnotatedRecord(
                benchmark=benchmark_name, item_index=i, is_mc=is_mc,
                context_tokens=list(item["context_tokens"]), completions=completions,
                label=item.get("label"), aux_token_lists=[], aux_raw_identifiers=[],
                target_str="", link_opener_prob=0.0, link_fired=False,
            ))
            continue

        records.append(AnnotatedRecord(
            benchmark=benchmark_name, item_index=i, is_mc=is_mc,
            context_tokens=list(ann.context_tokens), completions=completions,
            label=item.get("label"),
            aux_token_lists=[list(a) for a in ann.aux_token_lists],
            aux_raw_identifiers=list(ann.aux_raw_identifiers),
            target_str=ann.target_str, link_opener_prob=float(ann.link_opener_prob),
            link_fired=bool(ann.link_fired),
        ))
    n_fired = sum(1 for r in records if r.link_fired)
    logger.info(
        "annotate_items(%s): %d items, %d link-fired", benchmark_name, len(records), n_fired
    )
    return records


# ─── Scoring one checkpoint over cached records (phase 2) ─────────────────────────

def score_grid(
    model,
    records: List[AnnotatedRecord],
    device: str = "cuda",
    placebo_aux: Optional[Dict[int, List[List[int]]]] = None,
    placebo_gold_aux: Optional[Dict[int, List[List[int]]]] = None,
) -> Dict[int, Dict[str, Optional[float]]]:
    """Score each link-fired record under every cell for ONE loaded checkpoint.

    `model` must be loaded under a cross_doc_link mask with a MarkdownLinkDetector
    (for the doc-causal checkpoint, load with mask_type_override='cross_doc_link' and
    link_detector_override='markdown'). All cells share the same weights; only the
    mask / aux content differs.

    Returns {item_index: {cell: gold_completion_nll_or_None}}. Only fired records with
    non-empty aux are scored (unfired items carry no aux signal). Records carrying
    `gold_aux_tokens` additionally get the GOLD_CELLS (same link, gold passage as the
    single aux).
    """
    from eval.scoring import (
        score_completion, score_completion_concat, score_completion_with_context_docs,
    )

    if placebo_aux is None:
        placebo_aux = derange_aux(records)
    if placebo_gold_aux is None:
        placebo_gold_aux = derange_gold_aux(records)

    out: Dict[int, Dict[str, Optional[float]]] = {}
    for r in records:
        if not (r.link_fired and any(r.aux_token_lists)):
            continue
        comp = r.gold_completion()
        scores: Dict[str, Optional[float]] = {}

        # baseline: link spliced, no aux (doc_causal reference).
        scores["baseline"] = score_completion(model, r.context_tokens, comp, device=device)

        # grant-on: real aux via the cross_doc_link mask. Coarse grant mode
        # (aux_raw_identifiers=None): the re-detected link grants access to every
        # packed aux span. With one injected link and one aux doc per record this is
        # exactly the precise grant, but it does not depend on the detector's
        # re-extracted target_str matching the aux identifier — which fails for
        # titles containing ')' (detect_links truncates at the first ')') and
        # silently dropped ~25% of fired items from the grant cell in the smoke.
        scores["grant"] = score_completion_with_context_docs(
            model, aux_token_lists=r.aux_token_lists, context_tokens=r.context_tokens,
            completion_tokens=comp, link_detector=model.link_detector,
            aux_raw_identifiers=None, device=device,
        )

        # raw-concat: same aux as ordinary prior context (doc_concatenated).
        scores["concat"] = score_completion_concat(
            model, aux_token_lists=r.aux_token_lists, context_tokens=r.context_tokens,
            completion_tokens=comp, mask_type="doc_concatenated", device=device,
        )

        # invisible sanity: aux packed but masked out; should ~ baseline.
        scores["invisible"] = score_completion_concat(
            model, aux_token_lists=r.aux_token_lists, context_tokens=r.context_tokens,
            completion_tokens=comp, mask_type="doc_causal", device=device,
        )

        # placebo: grant-on with deranged (wrong-item) aux, original identifiers.
        p_aux = placebo_aux.get(r.item_index)
        if p_aux is not None:
            scores["placebo"] = score_completion_with_context_docs(
                model, aux_token_lists=p_aux, context_tokens=r.context_tokens,
                completion_tokens=comp, link_detector=model.link_detector,
                aux_raw_identifiers=None, device=device,
            )
        else:
            scores["placebo"] = None

        # gold-aux gradient: the benchmark's own gold passage through the same link.
        if r.gold_aux_tokens:
            gold = [list(r.gold_aux_tokens)]
            scores["grant_gold"] = score_completion_with_context_docs(
                model, aux_token_lists=gold, context_tokens=r.context_tokens,
                completion_tokens=comp, link_detector=model.link_detector,
                aux_raw_identifiers=None, device=device,
            )
            scores["concat_gold"] = score_completion_concat(
                model, aux_token_lists=gold, context_tokens=r.context_tokens,
                completion_tokens=comp, mask_type="doc_concatenated", device=device,
            )
            pg = placebo_gold_aux.get(r.item_index)
            scores["placebo_gold"] = (
                score_completion_with_context_docs(
                    model, aux_token_lists=pg, context_tokens=r.context_tokens,
                    completion_tokens=comp, link_detector=model.link_detector,
                    aux_raw_identifiers=None, device=device,
                ) if pg is not None else None
            )

        out[r.item_index] = scores
    return out


# ─── Aggregation & interaction (phase 3) ─────────────────────────────────────────

def _bootstrap_ci(
    deltas: List[float], seed: int = SEED, resamples: int = BOOTSTRAP_RESAMPLES
) -> Tuple[float, float]:
    """95% percentile bootstrap CI of the mean of `deltas`."""
    import numpy as np
    if not deltas:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    arr = np.asarray(deltas, dtype=float)
    means = rng.choice(arr, size=(resamples, len(arr)), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def _paired(a: Dict[int, Dict[str, Optional[float]]], keys: List[int],
            cell_pos: str, cell_neg: str) -> List[float]:
    """Per-item (cell_neg - cell_pos) over items where both cells scored.

    Convention: NLL is lower-is-better, so (baseline - cell) > 0 means the cell HELPED.
    """
    out = []
    for k in keys:
        s = a.get(k, {})
        p, n = s.get(cell_pos), s.get(cell_neg)
        if p is not None and n is not None:
            out.append(n - p)
    return out


def _mean_ci(deltas: List[float]) -> Dict[str, Any]:
    """Mean with bootstrap CI, plus heavy-tail-robust companions: the median and the
    fraction of items with a positive delta (aux helped). Per-item NLL deltas are
    heavy-tailed (a long irrelevant aux can add several nats on one item), so a mean
    alone can be driven by a handful of items."""
    import numpy as np
    lo, hi = _bootstrap_ci(deltas)
    arr = np.asarray(deltas, dtype=float)
    return {
        "mean": float(arr.mean()) if deltas else float("nan"),
        "ci95": [lo, hi],
        "median": float(np.median(arr)) if deltas else float("nan"),
        "frac_positive": float((arr > 0).mean()) if deltas else float("nan"),
        "n": len(deltas),
        "significant": (not (lo != lo)) and (lo > 0 or hi < 0),  # CI excludes 0
    }


def aggregate_grid(
    cross: Dict[int, Dict[str, Optional[float]]],
    doc_causal: Dict[int, Dict[str, Optional[float]]],
) -> Dict[str, Any]:
    """Combine the two checkpoints' per-record cell NLLs into the causal effects.

    All effects are on the GOLD-completion NLL (lower is better). Deltas are paired
    per item; each carries a bootstrap 95% CI and a `significant` flag (CI excludes 0).

      aux_lift_grant[W]   = baseline - grant   (does the real aux help ckpt W?)
      aux_lift_concat[W]  = baseline - concat  (does raw-concat aux help ckpt W?)
      mechanism[W]        = concat - grant     (grant beyond plain concatenation)
      placebo_sep[W]      = placebo - grant     paired vs real (right doc vs any doc)
      training_grant      = aux_lift_grant[cross] - aux_lift_grant[doc_causal]
                            (the HEADLINE interaction: cross-doc TRAINING advantage)
      invisible_check[W]  = |mean(invisible - baseline)|  (should be ~0)

    When gold cells are present (see GOLD_CELLS), the same block is emitted with a
    `_gold` suffix (aux_lift_grant_gold, mechanism_gold, placebo_sep_gold,
    training_grant_gold_interaction) plus
      relevance_slope[W]  = grant - grant_gold  (extra lift from gold over retrieved;
                            > 0 means the model exploits a BETTER aux more)
      relevance_slope_interaction = relevance_slope[cross] - relevance_slope[doc_causal]
    """
    result: Dict[str, Any] = {}
    per_ckpt = {"cross": cross, "doc_causal": doc_causal}

    def _block(grant: str, concat: str, placebo: str, suffix: str) -> None:
        lift_items: Dict[str, Dict[int, float]] = {}
        for name, a in per_ckpt.items():
            keys = list(a.keys())
            lift_items[name] = _per_item_map(a, grant, "baseline")      # baseline - grant
            result[f"aux_lift_grant{suffix}_{name}"] = _mean_ci(_paired(a, keys, grant, "baseline"))
            result[f"aux_lift_concat{suffix}_{name}"] = _mean_ci(_paired(a, keys, concat, "baseline"))
            result[f"mechanism{suffix}_{name}"] = _mean_ci(_paired(a, keys, grant, concat))
            result[f"placebo_sep{suffix}_{name}"] = _mean_ci(_paired(a, keys, grant, placebo))
        # Interaction: cross-doc training advantage in aux utilization, paired over
        # items scored by BOTH checkpoints.
        shared = sorted(set(lift_items["cross"]) & set(lift_items["doc_causal"]))
        result[f"training_grant{suffix}_interaction"] = _mean_ci(
            [lift_items["cross"][k] - lift_items["doc_causal"][k] for k in shared]
        )

    _block("grant", "concat", "placebo", "")
    for name, a in per_ckpt.items():
        inv = _paired(a, list(a.keys()), "invisible", "baseline")  # baseline - invisible ~ 0
        result[f"invisible_check_{name}"] = _mean_ci(inv)

    has_gold = any("grant_gold" in s for a in per_ckpt.values() for s in a.values())
    if has_gold:
        _block("grant_gold", "concat_gold", "placebo_gold", "_gold")
        slope_items: Dict[str, Dict[int, float]] = {}
        for name, a in per_ckpt.items():
            slope_items[name] = _per_item_map(a, "grant_gold", "grant")  # grant - grant_gold
            result[f"relevance_slope_{name}"] = _mean_ci(list(slope_items[name].values()))
        shared = sorted(set(slope_items["cross"]) & set(slope_items["doc_causal"]))
        result["relevance_slope_interaction"] = _mean_ci(
            [slope_items["cross"][k] - slope_items["doc_causal"][k] for k in shared]
        )
    return result


def _per_item_map(a: Dict[int, Dict[str, Optional[float]]],
                  cell_pos: str, cell_neg: str) -> Dict[int, float]:
    """{item_index: (cell_neg - cell_pos)} for items where both cells scored."""
    out: Dict[int, float] = {}
    for k, s in a.items():
        p, n = s.get(cell_pos), s.get(cell_neg)
        if p is not None and n is not None:
            out[k] = n - p
    return out


# ─── Orchestration (tie phases together) ─────────────────────────────────────────

def _build_markdown_annotator(
    model, annotator_corpus_dir: str, annotator_mode: str, use_trie: bool = True,
    beam_width: int = 1,
):
    """Construct a MarkdownPromptAnnotator over the wiki corpus, mirroring the
    annotated-condition setup in eval_checkpoints.py.

    use_trie=True builds a TrieTitleIndex, which constrains title generation to real
    corpus titles so every injected link resolves and fires (a corpus miss under
    HashNormTitleIndex leaves link_fired=False, which starves the causal grid of
    scored items). For a *forced-injection* eval we want reliable firing, so trie is
    the default; HashNormTitleIndex remains the post-hoc fallback.
    """
    from model.document_corpus import PretokCorpus
    from eval.link_annotator import MarkdownPromptAnnotator, TrieTitleIndex
    from eval.title_index import HashNormTitleIndex

    corpus = PretokCorpus(annotator_corpus_dir, link_detector=model.link_detector)
    raw_ids = [
        node["raw_identifier"]
        for node in corpus._graph.nodes.values()
        if "raw_identifier" in node
    ]
    fallback = HashNormTitleIndex(
        raw_ids,
        strategies=("exact", "norm", "word_overlap_ordered"),  # edit_distance dropped (slow)
    )
    if use_trie:
        title_index = TrieTitleIndex(
            raw_ids, model.tokenizer, beam_width=beam_width, fallback_index=fallback,
        )
    else:
        title_index = fallback
    annotator = MarkdownPromptAnnotator(
        corpus=corpus,
        title_index=title_index,
        link_retrieval_mode=annotator_mode,
        layout_policy=getattr(model, "inference_layout_policy", None),
    )
    return annotator, corpus


def _score_pair_and_report(
    cross_model,
    cross_ckpt: str,
    doc_causal_ckpt: str,
    records: List[AnnotatedRecord],
    out_dir: str,
    benchmark_name: str,
    device: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Phases 2-3 for an already-loaded cross-doc model: score both checkpoints over
    the cached records (fixed placebo mappings → paired), aggregate, write the report.
    Frees `cross_model` before loading the doc-causal checkpoint."""
    import torch
    from generate import load_inference_model

    placebo_aux = derange_aux(records)
    placebo_gold_aux = derange_gold_aux(records)

    # ── Phase 2a: score the cross-doc checkpoint ──────────────────────────────────
    cross_scores = score_grid(
        cross_model, records, device=device,
        placebo_aux=placebo_aux, placebo_gold_aux=placebo_gold_aux,
    )
    del cross_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Phase 2b: score the doc-causal checkpoint under a cross_doc_link mask ──────
    logger.info("Loading doc-causal checkpoint under cross_doc_link mask: %s", doc_causal_ckpt)
    dc_model, _ = load_inference_model(
        doc_causal_ckpt, device=device,
        mask_type_override="cross_doc_link", link_detector_override="markdown",
    )
    dc_scores = score_grid(
        dc_model, records, device=device,
        placebo_aux=placebo_aux, placebo_gold_aux=placebo_gold_aux,
    )
    del dc_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Phase 3: aggregate + report ───────────────────────────────────────────────
    agg = aggregate_grid(cross_scores, dc_scores)
    report = {
        "benchmark": benchmark_name,
        "cross_ckpt": cross_ckpt,
        "doc_causal_ckpt": doc_causal_ckpt,
        "n_items": len(records),
        "n_fired": sum(1 for r in records if r.link_fired),
        "n_gold": sum(1 for r in records if r.gold_aux_tokens),
        "n_scored_cross": len(cross_scores),
        "n_scored_doc_causal": len(dc_scores),
        **(extra or {}),
        "effects": agg,
    }
    report_path = os.path.join(out_dir, f"{benchmark_name}_grid_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    scores_path = os.path.join(out_dir, f"{benchmark_name}_cell_scores.json")
    with open(scores_path, "w") as f:
        json.dump({"cross": cross_scores, "doc_causal": dc_scores}, f)
    logger.info("Wrote report → %s (per-item cell scores → %s)", report_path, scores_path)

    for key in ("training_grant_interaction", "training_grant_gold_interaction",
                "relevance_slope_interaction"):
        if key in agg:
            e = agg[key]
            logger.info("HEADLINE %s: mean=%.4f ci95=%s n=%d significant=%s",
                        key, e["mean"], e["ci95"], e["n"], e["significant"])
    return report


def run_link_injection_grid(
    cross_ckpt: str,
    doc_causal_ckpt: str,
    benchmark_name: str,
    annotator_corpus_dir: str,
    out_dir: str,
    annotator_mode: str = "corpus_only",
    use_trie: bool = True,
    beam_width: int = 1,
    max_examples: Optional[int] = None,
    cache_dir: Optional[str] = None,
    device: str = "cuda",
    gold_aux: bool = False,
) -> Dict[str, Any]:
    """End-to-end causal 2x2 link-injection eval on a matched checkpoint pair.

    Annotates each benchmark item ONCE with the cross-doc checkpoint (the designated
    annotator), optionally attaches the benchmark's gold passage (`gold_aux`, sciq),
    scores the cached records under both checkpoints (the doc-causal one loaded under
    a cross_doc_link mask + markdown detector), and writes the aggregated interaction
    report. Returns the report dict.
    """
    from generate import load_inference_model
    from eval.nlp_benchmarks import _load_benchmark_items, _make_encoder

    os.makedirs(out_dir, exist_ok=True)

    # ── Phase 1: annotate once with the cross-doc checkpoint ──────────────────────
    logger.info("Loading cross-doc checkpoint: %s", cross_ckpt)
    cross_model, _ = load_inference_model(cross_ckpt, device=device)
    enc = _make_encoder(cross_model.tokenizer)
    items = _load_benchmark_items(benchmark_name, enc, max_examples, cache_dir)
    logger.info("Loaded %d items for %s", len(items), benchmark_name)

    annotator, corpus = _build_markdown_annotator(
        cross_model, annotator_corpus_dir, annotator_mode,
        use_trie=use_trie, beam_width=beam_width,
    )
    records = annotate_items(cross_model, annotator, benchmark_name, items, device=device)
    corpus.close()
    if gold_aux:
        attach_gold_aux(records, benchmark_name, enc, cache_dir)
    records_path = os.path.join(out_dir, f"{benchmark_name}_records.jsonl")
    save_records(records, records_path)
    logger.info("Saved %d records → %s", len(records), records_path)

    return _score_pair_and_report(
        cross_model, cross_ckpt, doc_causal_ckpt, records, out_dir, benchmark_name,
        device, extra={"annotator_mode": annotator_mode, "records": records_path},
    )


def replay_link_injection_grid(
    cross_ckpt: str,
    doc_causal_ckpt: str,
    records_path: str,
    out_dir: str,
    cache_dir: Optional[str] = None,
    device: str = "cuda",
    gold_aux: bool = False,
) -> Dict[str, Any]:
    """Re-score cached annotations (no re-annotation, no corpus load) — e.g. to add the
    gold-aux cells to a finished run, or to score a different checkpoint pair on the
    identical injected links + aux. Writes a fresh report into `out_dir`."""
    from generate import load_inference_model
    from eval.nlp_benchmarks import _make_encoder

    os.makedirs(out_dir, exist_ok=True)
    records = load_records(records_path)
    benchmark_name = records[0].benchmark if records else "unknown"
    logger.info("Replaying %d cached records (%s) from %s",
                len(records), benchmark_name, records_path)

    logger.info("Loading cross-doc checkpoint: %s", cross_ckpt)
    cross_model, _ = load_inference_model(cross_ckpt, device=device)
    if gold_aux and not any(r.gold_aux_tokens for r in records):
        attach_gold_aux(records, benchmark_name, _make_encoder(cross_model.tokenizer), cache_dir)
        new_path = os.path.join(out_dir, f"{benchmark_name}_records.jsonl")
        if os.path.abspath(new_path) != os.path.abspath(records_path):
            save_records(records, new_path)
            logger.info("Saved gold-augmented records → %s", new_path)

    return _score_pair_and_report(
        cross_model, cross_ckpt, doc_causal_ckpt, records, out_dir, benchmark_name,
        device, extra={"replayed_from": records_path},
    )


def _main():
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Causal 2x2 link-injection eval.")
    p.add_argument("--cross-ckpt", required=True, help="cross_doc_link best_model.pt")
    p.add_argument("--doc-causal-ckpt", required=True, help="doc_causal best_model.pt")
    p.add_argument("--benchmark", help="ANNOTATABLE benchmark name (annotate mode)")
    p.add_argument("--annotator-corpus", help="wiki_merged pretok dir (annotate mode)")
    p.add_argument("--replay-records", default=None,
                   help="Cached *_records.jsonl from a previous run: skip annotation and "
                        "re-score both checkpoints on the identical injected links + aux.")
    p.add_argument("--gold-aux", action="store_true",
                   help="Also score the benchmark's gold passage (sciq `support`) as an "
                        "aux through the same link (grant_gold/concat_gold/placebo_gold).")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--annotator-mode", default="corpus_only",
                   choices=["corpus_only", "generate_only", "corpus_then_generate"])
    p.add_argument("--no-trie", action="store_true",
                   help="Use HashNormTitleIndex (post-hoc match) instead of the "
                        "fire-guaranteeing TrieTitleIndex.")
    p.add_argument("--beam-width", type=int, default=1)
    p.add_argument("--max-examples", type=int, default=None)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    if args.replay_records:
        replay_link_injection_grid(
            cross_ckpt=args.cross_ckpt, doc_causal_ckpt=args.doc_causal_ckpt,
            records_path=args.replay_records, out_dir=args.out_dir,
            cache_dir=args.cache_dir, device=args.device, gold_aux=args.gold_aux,
        )
        return
    if not (args.benchmark and args.annotator_corpus):
        p.error("--benchmark and --annotator-corpus are required unless --replay-records")
    run_link_injection_grid(
        cross_ckpt=args.cross_ckpt, doc_causal_ckpt=args.doc_causal_ckpt,
        benchmark_name=args.benchmark, annotator_corpus_dir=args.annotator_corpus,
        out_dir=args.out_dir, annotator_mode=args.annotator_mode,
        use_trie=not args.no_trie, beam_width=args.beam_width,
        max_examples=args.max_examples, cache_dir=args.cache_dir, device=args.device,
        gold_aux=args.gold_aux,
    )


if __name__ == "__main__":
    _main()
