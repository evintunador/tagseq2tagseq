"""
scripts/compare_title_strategies.py

Compare corpus hit rates across any two HashNormTitleIndex strategy configurations.

Loads the model once, builds two annotators differing only in TitleIndex strategies,
runs the same examples through both, and reports n_link_fired / n_annotated per
benchmark. When B fires but A didn't, prints the generated title and the matched
corpus entry so you can visually confirm the new matches are sensible.

Usage:
    # Compare default strategies vs default + edit_distance
    python scripts/compare_title_strategies.py \\
        --checkpoint /fss/evin_t/tagseq2tagseq/runs/20260308_012516/checkpoints/best_model.pt \\
        --dataset data/pretokenized_datasets/simplewiki \\
        --strategies-a exact norm word_overlap_ordered \\
        --strategies-b exact norm word_overlap_ordered edit_distance \\
        --max-examples 200

    # Compare ordered vs unordered overlap
    python scripts/compare_title_strategies.py \\
        --checkpoint ... --dataset ... \\
        --strategies-a exact norm word_overlap_ordered \\
        --strategies-b exact norm word_overlap_unordered
"""

import argparse
import logging
import sys
from dataclasses import dataclass, field
from typing import List, Optional

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("compare_strategies")
logger.setLevel(logging.INFO)

BENCHMARKS = ["wiki_qa", "hellaswag", "boolq", "openbookqa"]
DEFAULT_A = ["exact", "norm", "word_overlap_ordered"]
DEFAULT_B = ["exact", "norm", "word_overlap_ordered", "edit_distance"]


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--strategies-a", nargs="+", default=DEFAULT_A, metavar="S",
                   help="Strategy list A (baseline). Default: exact norm word_overlap_ordered")
    p.add_argument("--strategies-b", nargs="+", default=DEFAULT_B, metavar="S",
                   help="Strategy list B (comparison). Default: adds edit_distance")
    p.add_argument("--benchmarks", nargs="+", default=BENCHMARKS, metavar="B",
                   help=f"Benchmarks to run. Default: {' '.join(BENCHMARKS)}")
    p.add_argument("--max-examples", type=int, default=200)
    p.add_argument("--ed-threshold", type=float, default=0.2,
                   help="edit_distance_threshold for both indexes (default 0.2)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--cache-dir", default="data/.cache")
    return p.parse_args()


@dataclass
class _ItemResult:
    fired_a: bool
    fired_b: bool
    target_str_a: str = ""
    target_str_b: str = ""
    matched_id_b: str = ""  # aux_raw_identifiers[0] when B fires


def _run_pair(model, ann_a, ann_b, items, device) -> List[_ItemResult]:
    """Run both annotators over items; return per-item results."""
    results = []
    for item in items:
        ctx = item["context_tokens"]
        r = _ItemResult(fired_a=False, fired_b=False)
        try:
            ann = ann_a.annotate(model, ctx, device=device)
            if ann is not None:
                r.fired_a = ann.link_fired
                r.target_str_a = ann.target_str
        except Exception as exc:
            logger.warning("ann_a.annotate() raised %s: %s", type(exc).__name__, exc)
        try:
            ann = ann_b.annotate(model, ctx, device=device)
            if ann is not None:
                r.fired_b = ann.link_fired
                r.target_str_b = ann.target_str
                if ann.link_fired and ann.aux_raw_identifiers:
                    r.matched_id_b = ann.aux_raw_identifiers[0]
        except Exception as exc:
            logger.warning("ann_b.annotate() raised %s: %s", type(exc).__name__, exc)
        results.append(r)
    return results


def main():
    args = _parse_args()
    sys.path.insert(0, ".")

    logger.info("Loading model from %s ...", args.checkpoint)
    from generate import load_inference_model, PretokCorpus
    model = load_inference_model(args.checkpoint, device=args.device)
    logger.info("mask_type=%s", getattr(model, "mask_type", "?"))

    from model.graph_traversal.markdown_link_detector import MarkdownLinkDetector
    if getattr(model, "mask_type", None) != "cross_doc_link" or \
       not isinstance(getattr(model, "link_detector", None), MarkdownLinkDetector):
        logger.error(
            "Requires cross_doc_link model with MarkdownLinkDetector "
            "(mask_type=%r, link_detector=%r)",
            getattr(model, "mask_type", None),
            type(getattr(model, "link_detector", None)).__name__,
        )
        sys.exit(1)

    logger.info("Loading corpus from %s ...", args.dataset)
    corpus = PretokCorpus(args.dataset)
    raw_ids = [
        node["raw_identifier"]
        for node in corpus._graph.nodes.values()
        if "raw_identifier" in node
    ]
    logger.info("Corpus: %d titles", len(raw_ids))

    from eval.title_index import HashNormTitleIndex
    from eval.link_annotator import MarkdownPromptAnnotator

    layout = getattr(model, "inference_layout_policy", None)

    def _make_annotator(strategies):
        idx = HashNormTitleIndex(
            raw_ids,
            strategies=strategies,
            edit_distance_threshold=args.ed_threshold,
        )
        return MarkdownPromptAnnotator(
            corpus=corpus, title_index=idx,
            link_retrieval_mode="corpus_only", layout_policy=layout,
        )

    ann_a = _make_annotator(args.strategies_a)
    ann_b = _make_annotator(args.strategies_b)

    label_a = " ".join(args.strategies_a)
    label_b = " ".join(args.strategies_b)
    col = max(len(label_a), len(label_b), 18)

    print(f"\n{'Benchmark':<16}  {'n':>5}  {label_a:>{col}}  {label_b:>{col}}  {'delta':>8}")
    print("-" * (16 + 5 + col * 2 + 20))

    newly_matched: List[dict] = []  # accumulate B-only hits across all benchmarks

    for bname in args.benchmarks:
        logger.info("Loading %s (%d examples) ...", bname, args.max_examples)
        try:
            from eval.nlp_benchmarks import _load_benchmark_items
            items = _load_benchmark_items(
                bname, max_examples=args.max_examples,
                cache_dir=args.cache_dir, device=args.device,
            )
        except Exception as exc:
            logger.error("Failed to load %s: %s", bname, exc)
            continue
        if not items:
            logger.warning("No items for %s", bname)
            continue

        pair_results = _run_pair(model, ann_a, ann_b, items, device=args.device)

        n_annotated = len(pair_results)
        fired_a = sum(r.fired_a for r in pair_results)
        fired_b = sum(r.fired_b for r in pair_results)
        pct_a = 100.0 * fired_a / n_annotated if n_annotated else 0.0
        pct_b = 100.0 * fired_b / n_annotated if n_annotated else 0.0
        delta = pct_b - pct_a

        a_str = f"{fired_a} ({pct_a:.1f}%)"
        b_str = f"{fired_b} ({pct_b:.1f}%)"
        print(f"{bname:<16}  {n_annotated:>5}  {a_str:>{col}}  {b_str:>{col}}  {delta:>+7.1f}%")

        for r in pair_results:
            if r.fired_b and not r.fired_a:
                newly_matched.append({
                    "benchmark": bname,
                    "generated": r.target_str_b,
                    "matched":   r.matched_id_b,
                })

    print()

    if newly_matched:
        print(f"Newly matched by B ({len(newly_matched)} total):")
        print(f"  {'benchmark':<16}  {'generated title':<40}  matched corpus entry")
        print(f"  {'-'*16}  {'-'*40}  {'-'*40}")
        for m in newly_matched:
            print(f"  {m['benchmark']:<16}  {m['generated']!r:<40}  {m['matched']!r}")
        print()

    corpus.close()


if __name__ == "__main__":
    main()
