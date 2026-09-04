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

See docs/link_injection_causal_eval_DESIGN.md.
"""
from __future__ import annotations

import json
import logging
import random
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

SEED = 42
BOOTSTRAP_RESAMPLES = 10_000

# Cells scored per checkpoint. "baseline" is the paired reference (link spliced, no
# aux) so every delta isolates the AUX-DOC effect, holding the injected link fixed.
CELLS = ("baseline", "grant", "concat", "invisible", "placebo")


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

def derange_aux(
    records: List[AnnotatedRecord], seed: int = SEED
) -> Dict[int, List[List[int]]]:
    """Map each fired record's index → a DIFFERENT fired record's aux token lists.

    Mirrors the tier2 placebo: swap aux CONTENT across fired examples while keeping
    each record's OWN aux_raw_identifiers (so the grant still fires at the same link,
    but attends to wrong-but-plausible content). A true derangement (no fixed points)
    is used so no record keeps its own aux; falls back to a rotation when <2 fired.

    Returns {item_index: placebo_aux_token_lists}. Only fired records are keyed.
    """
    fired = [r for r in records if r.link_fired and any(r.aux_token_lists)]
    n = len(fired)
    if n == 0:
        return {}
    if n == 1:
        # No other content to swap in; reuse own aux (placebo == real for this one).
        return {fired[0].item_index: fired[0].aux_token_lists}

    rng = random.Random(seed)
    order = list(range(n))
    # Rejection-sample a permutation with no fixed points (derangement).
    for _ in range(1000):
        perm = order[:]
        rng.shuffle(perm)
        if all(perm[i] != i for i in range(n)):
            break
    else:
        # Deterministic fallback: single rotation is a derangement for n >= 2.
        perm = order[1:] + order[:1]

    return {fired[i].item_index: fired[perm[i]].aux_token_lists for i in range(n)}


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
) -> Dict[int, Dict[str, Optional[float]]]:
    """Score each link-fired record under every cell for ONE loaded checkpoint.

    `model` must be loaded under a cross_doc_link mask with a MarkdownLinkDetector
    (for the doc-causal checkpoint, load with mask_type_override='cross_doc_link' and
    link_detector_override='markdown'). All cells share the same weights; only the
    mask / aux content differs.

    Returns {item_index: {cell: gold_completion_nll_or_None}}. Only fired records with
    non-empty aux are scored (unfired items carry no aux signal).
    """
    from eval.scoring import (
        score_completion, score_completion_concat, score_completion_with_context_docs,
    )

    if placebo_aux is None:
        placebo_aux = derange_aux(records)

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
    import numpy as np
    lo, hi = _bootstrap_ci(deltas)
    return {
        "mean": float(np.mean(deltas)) if deltas else float("nan"),
        "ci95": [lo, hi],
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
    """
    result: Dict[str, Any] = {}

    per_ckpt = {"cross": cross, "doc_causal": doc_causal}
    aux_lift_grant_items: Dict[str, List[float]] = {}
    for name, a in per_ckpt.items():
        keys = list(a.keys())
        aux_lift_grant = _paired(a, keys, "grant", "baseline")     # baseline - grant
        aux_lift_grant_items[name] = _per_item_map(a, "grant", "baseline")
        result[f"aux_lift_grant_{name}"] = _mean_ci(aux_lift_grant)
        result[f"aux_lift_concat_{name}"] = _mean_ci(_paired(a, keys, "concat", "baseline"))
        result[f"mechanism_{name}"] = _mean_ci(_paired(a, keys, "grant", "concat"))
        result[f"placebo_sep_{name}"] = _mean_ci(_paired(a, keys, "grant", "placebo"))
        inv = _paired(a, keys, "invisible", "baseline")  # baseline - invisible ~ 0
        result[f"invisible_check_{name}"] = _mean_ci(inv)

    # Headline interaction: cross-doc training advantage in aux utilization, paired
    # over items scored by BOTH checkpoints.
    shared = sorted(set(aux_lift_grant_items["cross"]) & set(aux_lift_grant_items["doc_causal"]))
    interaction = [
        aux_lift_grant_items["cross"][k] - aux_lift_grant_items["doc_causal"][k]
        for k in shared
    ]
    result["training_grant_interaction"] = _mean_ci(interaction)
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
) -> Dict[str, Any]:
    """End-to-end causal 2x2 link-injection eval on a matched checkpoint pair.

    Annotates each benchmark item ONCE with the cross-doc checkpoint (the designated
    annotator), scores the cached records under both checkpoints (the doc-causal one
    loaded under a cross_doc_link mask + markdown detector), and writes the aggregated
    interaction report. Returns the aggregate dict.
    """
    import os
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
    records_path = os.path.join(out_dir, f"{benchmark_name}_records.jsonl")
    save_records(records, records_path)
    logger.info("Saved %d records → %s", len(records), records_path)

    # Fixed placebo mapping, shared across both checkpoints for a paired comparison.
    placebo_aux = derange_aux(records)

    # ── Phase 2a: score the cross-doc checkpoint ──────────────────────────────────
    cross_scores = score_grid(cross_model, records, device=device, placebo_aux=placebo_aux)
    corpus.close()
    del cross_model
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Phase 2b: score the doc-causal checkpoint under a cross_doc_link mask ──────
    logger.info("Loading doc-causal checkpoint under cross_doc_link mask: %s", doc_causal_ckpt)
    dc_model, _ = load_inference_model(
        doc_causal_ckpt, device=device,
        mask_type_override="cross_doc_link", link_detector_override="markdown",
    )
    dc_scores = score_grid(dc_model, records, device=device, placebo_aux=placebo_aux)
    del dc_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Phase 3: aggregate + report ───────────────────────────────────────────────
    agg = aggregate_grid(cross_scores, dc_scores)
    report = {
        "benchmark": benchmark_name,
        "annotator_mode": annotator_mode,
        "cross_ckpt": cross_ckpt,
        "doc_causal_ckpt": doc_causal_ckpt,
        "n_items": len(records),
        "n_fired": sum(1 for r in records if r.link_fired),
        "n_scored_cross": len(cross_scores),
        "n_scored_doc_causal": len(dc_scores),
        "effects": agg,
    }
    report_path = os.path.join(out_dir, f"{benchmark_name}_grid_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Wrote report → %s", report_path)

    inter = agg["training_grant_interaction"]
    logger.info(
        "HEADLINE training-grant interaction: mean=%.4f ci95=%s n=%d significant=%s",
        inter["mean"], inter["ci95"], inter["n"], inter["significant"],
    )
    return report


def _main():
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Causal 2x2 link-injection eval.")
    p.add_argument("--cross-ckpt", required=True, help="cross_doc_link best_model.pt")
    p.add_argument("--doc-causal-ckpt", required=True, help="doc_causal best_model.pt")
    p.add_argument("--benchmark", required=True, help="ANNOTATABLE benchmark name")
    p.add_argument("--annotator-corpus", required=True, help="wiki_merged pretok dir")
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
    run_link_injection_grid(
        cross_ckpt=args.cross_ckpt, doc_causal_ckpt=args.doc_causal_ckpt,
        benchmark_name=args.benchmark, annotator_corpus_dir=args.annotator_corpus,
        out_dir=args.out_dir, annotator_mode=args.annotator_mode,
        use_trie=not args.no_trie, beam_width=args.beam_width,
        max_examples=args.max_examples, cache_dir=args.cache_dir, device=args.device,
    )


if __name__ == "__main__":
    _main()
