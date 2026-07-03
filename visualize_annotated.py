"""
visualize_annotated.py — render annotated benchmark examples from a config.

Loads a checkpoint and eval config, finds all benchmarks with 'annotated' in
their conditions, annotates one example per benchmark, and prints the colored
render_annotated_example panels.

Usage:
    python visualize_annotated.py \
        --checkpoint runs/20260308_012516/checkpoints/best_model.pt \
        --config configs/wiki_merged_cross_doc.yaml \
        [--n 1] [--no-color] [--device cuda]
"""

import argparse
import sys
from pathlib import Path

import torch

from eval.link_annotator import MarkdownPromptAnnotator, TrieTitleIndex, render_annotated_example
from eval.nlp_benchmarks import ANNOTATABLE_BENCHMARKS, _load_benchmark_items
from eval.scoring import score_completions_batched, score_completion, score_completion_with_context_docs
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
        help="Eval config YAML (e.g. configs/wiki_merged_cross_doc.yaml). "
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
    parser.add_argument(
        "--use-trie", action="store_true",
        help="Use TrieTitleIndex (trie-constrained generation) instead of "
             "HashNormTitleIndex. HashNormTitleIndex is still used as fallback.",
    )
    parser.add_argument(
        "--trie-min-logprob", type=float, default=None, metavar="LOGPROB",
        help="min_joint_logprob for TrieTitleIndex. None = no threshold (default).",
    )
    parser.add_argument(
        "--beam-width", type=int, default=1, metavar="W",
        help="Beam width for TrieTitleIndex. 1 = greedy (default).",
    )
    parser.add_argument(
        "--length-penalty", type=float, default=0.0, metavar="ALPHA",
        help="Length penalty exponent for TrieTitleIndex candidate scoring "
             "(score = joint_log_prob / n_tokens**alpha). 0.0 = no normalization "
             "(default); 0.6 = recommended; 1.0 = full per-token mean log-prob.",
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0,
        help="Sampling temperature for title generation. 0.0 = greedy (default 1.0).",
    )
    parser.add_argument(
        "--top-k", type=int, default=None,
        help="Top-k sampling cutoff (default: disabled).",
    )
    parser.add_argument(
        "--top-p", type=float, default=None,
        help="Nucleus sampling cutoff (default: disabled).",
    )
    parser.add_argument(
        "--show-beam-candidates", action="store_true",
        help="Show all beam candidates and their scores in the output (trie only).",
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
    raw_ids = [
        node["raw_identifier"]
        for node in corpus._graph.nodes.values()
        if "raw_identifier" in node
    ]
    hashnorm = HashNormTitleIndex(raw_ids)
    if args.use_trie:
        print("Building TrieTitleIndex ...", flush=True)
        title_index = TrieTitleIndex(
            raw_ids,
            model.tokenizer,
            beam_width=args.beam_width,
            length_penalty=args.length_penalty,
            min_joint_logprob=args.trie_min_logprob,
            fallback_index=hashnorm,
        )
    else:
        title_index = hashnorm
    annotator_mode = eval_cfg.get("annotator_mode", "corpus_only")
    layout_policy = getattr(model, "inference_layout_policy", None)
    from model.generation_config import GenerationConfig
    annotator = MarkdownPromptAnnotator(
        corpus=corpus,
        title_index=title_index,
        link_retrieval_mode=annotator_mode,
        layout_policy=layout_policy,
        generation_config=GenerationConfig(
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        ),
        show_beam_candidates=args.show_beam_candidates,
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

            # Score choices under both flat and annotated contexts to show
            # model predictions alongside the ground-truth label.
            pred_flat = pred_annotated = None
            if choices:
                nlls_flat = score_completions_batched(
                    model, original_ctx, choices, device=args.device
                )
                pred_flat = int(min(range(len(nlls_flat)), key=lambda i: nlls_flat[i]))

                ann_ctx = annotated.context_tokens
                if annotated.link_fired and annotated.aux_token_lists:
                    nlls_ann = []
                    for ch in choices:
                        nll = score_completion_with_context_docs(
                            model,
                            aux_token_lists=annotated.aux_token_lists,
                            context_tokens=ann_ctx,
                            completion_tokens=ch,
                            link_detector=model.link_detector,
                            aux_raw_identifiers=annotated.aux_raw_identifiers,
                            device=args.device,
                        )
                        if nll is None:
                            nll = score_completion(model, ann_ctx, ch, device=args.device)
                        nlls_ann.append(nll)
                    pred_annotated = int(min(range(len(nlls_ann)), key=lambda i: nlls_ann[i]))
                else:
                    nlls_ann = score_completions_batched(
                        model, ann_ctx, choices, device=args.device
                    )
                    pred_annotated = int(min(range(len(nlls_ann)), key=lambda i: nlls_ann[i]))

            rendered = render_annotated_example(
                original_tokens=original_ctx,
                annotated=annotated,
                tokenizer=tokenizer,
                choices=choices,
                completion_tokens=completion,
                label=label,
                pred_flat=pred_flat,
                pred_annotated=pred_annotated,
                use_color=use_color,
            )
            print(f"\n  Example {idx + 1}")
            print(rendered)

    corpus.close()


if __name__ == "__main__":
    main()
