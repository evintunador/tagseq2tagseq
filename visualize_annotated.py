"""
visualize_annotated.py — render annotated benchmark examples from a config.

Loads a checkpoint and eval config, finds all benchmarks with 'annotated' in
their conditions, annotates one example per benchmark, and prints the colored
render_annotated_example panels.

Usage:
    python visualize_annotated.py \
        --checkpoint runs/20260308_012516/checkpoints/best_model.pt \
        --config configs/simplewiki_cross_doc.yaml \
        [--n 1] [--no-color] [--device cuda]
"""

import argparse
import sys
from pathlib import Path

import torch

from eval.link_annotator import MarkdownPromptAnnotator, render_annotated_example
from eval.nlp_benchmarks import ANNOTATABLE_BENCHMARKS, _load_benchmark_items
from eval.title_index import HashNormTitleIndex


def _load_config(config_path: str) -> dict:
    import yaml
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Render annotated benchmark examples from a trained checkpoint + eval config.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", required=True, metavar="PATH",
        help="Path to best_model.pt checkpoint.",
    )
    parser.add_argument(
        "--config", required=True, metavar="PATH",
        help="Eval config YAML (e.g. configs/simplewiki_cross_doc.yaml). "
             "Reads eval.benchmarks, eval.annotator_corpus, eval.annotator_mode.",
    )
    parser.add_argument(
        "--n", type=int, default=1, metavar="N",
        help="Number of examples to render per benchmark.",
    )
    parser.add_argument(
        "--no-color", action="store_true",
        help="Disable ANSI color codes (for file output).",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    cfg = _load_config(args.config)
    eval_cfg = cfg.get("eval", {})

    # Collect benchmarks with 'annotated' condition
    benchmark_specs = eval_cfg.get("benchmarks", [])
    annotated_benchmarks = [
        spec["name"]
        for spec in benchmark_specs
        if "annotated" in spec.get("conditions", [])
        and spec["name"] in ANNOTATABLE_BENCHMARKS
    ]
    if not annotated_benchmarks:
        print("No annotatable benchmarks found in config (no benchmark has 'annotated' in conditions).")
        sys.exit(0)

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}", flush=True)
    from generate import load_inference_model, PretokCorpus
    model, _ = load_inference_model(args.checkpoint, device=args.device)
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        print("ERROR: model has no tokenizer.", file=sys.stderr)
        sys.exit(1)

    # Build corpus + title index
    corpus_dir = eval_cfg.get("annotator_corpus") or cfg.get("data", {}).get("dataset_dir")
    if corpus_dir is None:
        print("ERROR: no annotator_corpus in config.", file=sys.stderr)
        sys.exit(1)
    print(f"Loading corpus: {corpus_dir}", flush=True)
    corpus = PretokCorpus(corpus_dir)
    title_index = HashNormTitleIndex(
        corpus._graph.get_raw_identifier(corpus._graph.get_normed_identifier(i))
        for i in range(len(corpus._graph))
    )
    annotator_mode = eval_cfg.get("annotator_mode", "corpus_only")
    layout_policy = getattr(model, "inference_layout_policy", None)
    annotator = MarkdownPromptAnnotator(
        corpus=corpus,
        title_index=title_index,
        link_retrieval_mode=annotator_mode,
        layout_policy=layout_policy,
    )

    use_color = not args.no_color

    for bname in annotated_benchmarks:
        print(f"\n{'━'*72}", flush=True)
        print(f"  BENCHMARK: {bname}  (showing {args.n} example(s))", flush=True)
        print(f"{'━'*72}", flush=True)

        items = _load_benchmark_items(
            benchmark_name=bname,
            enc=tokenizer.encode,
            max_examples=args.n,
            cache_dir=None,
        )
        if not items:
            print(f"  (no items loaded for {bname})")
            continue

        for idx, item in enumerate(items[:args.n]):
            original_ctx = item["context_tokens"]
            annotated = annotator.annotate(model, original_ctx, device=args.device)

            choices = item.get("completion_token_lists")
            completion = item.get("completion_tokens")
            label = item.get("label")

            rendered = render_annotated_example(
                original_tokens=original_ctx,
                annotated=annotated,
                tokenizer=tokenizer,
                choices=choices,
                completion_tokens=completion,
                label=label,
                use_color=use_color,
            )
            print(f"\n  Example {idx + 1}")
            print(rendered)

    corpus.close()


if __name__ == "__main__":
    main()
