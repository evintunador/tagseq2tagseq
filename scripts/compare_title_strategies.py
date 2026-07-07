"""
scripts/compare_title_strategies.py

Compare corpus hit rates across HashNormTitleIndex strategy configurations and
optionally TrieTitleIndex (trie-constrained generation).

Loads the model once, builds two or three annotators differing only in TitleIndex,
runs the same examples through all, and reports n_link_fired / n_annotated per
benchmark. When B (or C) fires but A didn't, prints the generated title and the
matched corpus entry so you can visually confirm the new matches are sensible.

Usage:
    # Compare default strategies vs default + edit_distance
    python scripts/compare_title_strategies.py \\
        --checkpoint /fss/evin_t/tagseq2tagseq/runs/20260308_012516/checkpoints/best_model.pt \\
        --dataset data/pretokenized_datasets/simplewiki \\
        --strategies-a exact norm word_overlap_ordered \\
        --strategies-b exact norm word_overlap_ordered edit_distance \\
        --max-examples 200

    # Add trie as a third column (C)
    python scripts/compare_title_strategies.py \\
        --checkpoint ... --dataset ... \\
        --strategies-a exact norm word_overlap_ordered \\
        --strategies-b exact norm word_overlap_ordered edit_distance \\
        --use-trie \\
        --max-examples 200
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
    p.add_argument("--use-trie", action="store_true",
                   help="Add a third column C: TrieTitleIndex with B's strategies as fallback.")
    p.add_argument("--trie-min-logprob", type=float, default=None, metavar="LOGPROB",
                   help="min_joint_logprob for TrieTitleIndex. None = no threshold (default).")
    p.add_argument("--beam-width", type=int, default=1, metavar="W",
                   help="Beam width for TrieTitleIndex. 1 = greedy (default).")
    p.add_argument("--length-penalty", type=float, default=0.0, metavar="ALPHA",
                   help="Length penalty exponent for TrieTitleIndex candidate scoring. "
                        "0.0 = raw joint log-prob (default); 1.0 = per-token mean log-prob. "
                        "0.6 is a recommended middle ground.")
    p.add_argument("--benchmarks", nargs="+", default=BENCHMARKS, metavar="B",
                   help=f"Benchmarks to run. Default: {' '.join(BENCHMARKS)}")
    p.add_argument("--max-examples", type=int, default=200)
    p.add_argument("--ed-threshold", type=float, default=0.2,
                   help="edit_distance_threshold for both indexes (default 0.2)")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="Sampling temperature for all annotators. Use 0.0 for greedy (default 1.0).")
    p.add_argument("--top-k", type=int, default=None,
                   help="Top-k sampling cutoff (default: disabled).")
    p.add_argument("--top-p", type=float, default=None,
                   help="Nucleus sampling cutoff (default: disabled).")
    p.add_argument("--device", default="cuda")
    p.add_argument("--cache-dir", default="data/.cache")
    return p.parse_args()


@dataclass
class _ItemResult:
    fired_a: bool
    fired_b: bool
    fired_c: bool = False
    target_str_a: str = ""
    target_str_b: str = ""
    target_str_c: str = ""
    matched_id_b: str = ""  # aux_raw_identifiers[0] when B fires
    matched_id_c: str = ""  # aux_raw_identifiers[0] when C fires


def _run_pair(model, ann_a, ann_b, items, device, ann_c=None) -> List[_ItemResult]:
    """Run two (or three) annotators over items; return per-item results."""
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
        if ann_c is not None:
            try:
                ann = ann_c.annotate(model, ctx, device=device)
                if ann is not None:
                    r.fired_c = ann.link_fired
                    r.target_str_c = ann.target_str
                    if ann.link_fired and ann.aux_raw_identifiers:
                        r.matched_id_c = ann.aux_raw_identifiers[0]
            except Exception as exc:
                logger.warning("ann_c.annotate() raised %s: %s", type(exc).__name__, exc)
        results.append(r)
    return results


def main():
    args = _parse_args()
    sys.path.insert(0, ".")

    logger.info("Loading model from %s ...", args.checkpoint)
    from generate import load_inference_model
    from model.document_corpus import PretokCorpus
    model, _ = load_inference_model(args.checkpoint, device=args.device)
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
    corpus = PretokCorpus(args.dataset, link_detector=model.link_detector)
    raw_ids = [
        node["raw_identifier"]
        for node in corpus._graph.nodes.values()
        if "raw_identifier" in node
    ]
    logger.info("Corpus: %d titles", len(raw_ids))

    from eval.title_index import HashNormTitleIndex
    from eval.link_annotator import MarkdownPromptAnnotator, TrieTitleIndex
    from model.generation_config import GenerationConfig

    layout = getattr(model, "inference_layout_policy", None)
    gen_cfg = GenerationConfig(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )

    def _make_hashnorm(strategies):
        return HashNormTitleIndex(
            raw_ids,
            strategies=strategies,
            edit_distance_threshold=args.ed_threshold,
        )

    def _make_annotator(idx):
        return MarkdownPromptAnnotator(
            corpus=corpus, title_index=idx,
            link_retrieval_mode="corpus_only", layout_policy=layout,
            generation_config=gen_cfg,
        )

    ann_a = _make_annotator(_make_hashnorm(args.strategies_a))
    idx_b = _make_hashnorm(args.strategies_b)
    ann_b = _make_annotator(idx_b)

    ann_c = None
    label_c = ""
    if args.use_trie:
        trie_idx = TrieTitleIndex(
            raw_ids,
            model.tokenizer,
            beam_width=args.beam_width,
            length_penalty=args.length_penalty,
            min_joint_logprob=args.trie_min_logprob,
            fallback_index=idx_b,
        )
        ann_c = _make_annotator(trie_idx)
        label_c = f"trie beam={args.beam_width}(+B fallback)"

    label_a = " ".join(args.strategies_a)
    label_b = " ".join(args.strategies_b)
    col = max(len(label_a), len(label_b), len(label_c), 18)

    header = f"\n{'Benchmark':<16}  {'n':>5}  {label_a:>{col}}  {label_b:>{col}}"
    if ann_c is not None:
        header += f"  {label_c:>{col}}"
    header += f"  {'B-A':>8}"
    if ann_c is not None:
        header += f"  {'C-A':>8}"
    print(header)
    print("-" * (len(header) - 1))

    newly_matched: List[dict] = []  # accumulate B-only / C-only hits across all benchmarks

    for bname in args.benchmarks:
        logger.info("Loading %s (%d examples) ...", bname, args.max_examples)
        try:
            from eval.nlp_benchmarks import _load_benchmark_items
            items = _load_benchmark_items(
                bname,
                enc=model.tokenizer.encode,
                max_examples=args.max_examples,
                cache_dir=args.cache_dir,
            )
        except Exception as exc:
            logger.error("Failed to load %s: %s", bname, exc)
            continue
        if not items:
            logger.warning("No items for %s", bname)
            continue

        pair_results = _run_pair(model, ann_a, ann_b, items, device=args.device, ann_c=ann_c)

        n_annotated = len(pair_results)
        fired_a = sum(r.fired_a for r in pair_results)
        fired_b = sum(r.fired_b for r in pair_results)
        fired_c = sum(r.fired_c for r in pair_results)
        pct_a = 100.0 * fired_a / n_annotated if n_annotated else 0.0
        pct_b = 100.0 * fired_b / n_annotated if n_annotated else 0.0
        pct_c = 100.0 * fired_c / n_annotated if n_annotated else 0.0

        row = (
            f"{bname:<16}  {n_annotated:>5}"
            f"  {f'{fired_a} ({pct_a:.1f}%)':>{col}}"
            f"  {f'{fired_b} ({pct_b:.1f}%)':>{col}}"
        )
        if ann_c is not None:
            row += f"  {f'{fired_c} ({pct_c:.1f}%)':>{col}}"
        row += f"  {pct_b - pct_a:>+7.1f}%"
        if ann_c is not None:
            row += f"  {pct_c - pct_a:>+7.1f}%"
        print(row)

        for r in pair_results:
            if r.fired_b and not r.fired_a:
                newly_matched.append({
                    "benchmark": bname, "col": "B",
                    "generated": r.target_str_b, "matched": r.matched_id_b,
                })
            if ann_c is not None and r.fired_c and not r.fired_a:
                newly_matched.append({
                    "benchmark": bname, "col": "C",
                    "generated": r.target_str_c, "matched": r.matched_id_c,
                })

    print()

    if newly_matched:
        print(f"Newly matched vs A ({len(newly_matched)} total):")
        print(f"  {'col':<4}  {'benchmark':<16}  {'generated title':<40}  matched corpus entry")
        print(f"  {'-'*4}  {'-'*16}  {'-'*40}  {'-'*40}")
        for m in newly_matched:
            print(f"  {m['col']:<4}  {m['benchmark']:<16}  {m['generated']!r:<40}  {m['matched']!r}")
        print()

    corpus.close()


if __name__ == "__main__":
    main()
