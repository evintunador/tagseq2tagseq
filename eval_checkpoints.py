"""
eval_checkpoints.py — downstream benchmark evaluation for TS2TS checkpoints.

Loads a checkpoint, reconstructs the model in inference mode, and runs the
configured benchmark suite. Results are written as JSON.

Benchmarks can be run under multiple named conditions in a single pass.
Each condition specifies a mask_type and layout_policy override, allowing
direct comparison between e.g. the model's experimental cross_doc_link
behaviour and a doc_causal baseline.

Usage (CLI):
    python eval_checkpoints.py \\
        --checkpoint runs/YYYYMMDD/checkpoints/best_model.pt \\
        --dataset data/pretokenized_datasets/stack_10m \\
        [--benchmarks held_out_perplexity] \\
        [--split val_community] \\
        [--max-docs 500] \\
        [--output eval_results.json] \\
        [--device cuda]

Importable:
    from eval_checkpoints import run_eval
    results = run_eval(checkpoint_path, dataset_dir, eval_cfg, device)

Results dict structure (with conditions):
    {
        "held_out_perplexity/baseline":     {...},
        "held_out_perplexity/experimental": {...},
        "hellaswag/baseline":               {...},
    }
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch

from generate import load_inference_model

logger = logging.getLogger(__name__)

# ─── Benchmark registry ──────────────────────────────────────────────────────

_KNOWN_BENCHMARKS = ("held_out_perplexity", "hellaswag", "pack_contrastive_perplexity")

# ─── Condition registry ───────────────────────────────────────────────────────
# Condition dicts specify per-call overrides for forward_inference.
# Special layout_policy string values:
#   'eos'       → EOS-suffix-only layout (no identifier prefix, EOS suffix only).
#   'inference' → model.inference_layout_policy
#   'training'  → model.training_layout_policy
#   'null'      → NullLayoutPolicy()
# mask_type None means "use model default" (self.mask_type).
#
# requires_cross_doc_link: True → condition is silently skipped when the model's
#   mask_type is 'doc_causal'. Doc-causal models ARE the baseline; running the
#   baseline condition on them produces identical results to experimental and
#   adds no information.

_BUILTIN_CONDITIONS: Dict[str, Dict[str, Any]] = {
    "baseline": {
        "mask_type":              "doc_causal",
        "layout_policy":          "eos",
        "requires_cross_doc_link": True,   # meaningless on doc_causal models
    },
    "experimental": {
        "mask_type":    None,        # use model's trained mask_type
        "layout_policy": "inference",
    },
}

_DEFAULTS: Dict[str, Any] = {
    "benchmarks": [
        {"name": "held_out_perplexity", "conditions": ["experimental"]},
    ],
    "conditions": {},   # user-defined conditions merged with _BUILTIN_CONDITIONS
    "max_docs": 500,
    "split": "all",
    "epoch_dirs": [],   # pre-computed epoch dirs for pack_contrastive_perplexity
}


def _resolve_layout_policy(policy_str: Optional[str], model):
    """Resolve a layout_policy string to a DocLayoutPolicy instance."""
    from data.layout import EOSLayoutPolicy, NullLayoutPolicy
    if policy_str is None or policy_str == "inference":
        return model.inference_layout_policy
    elif policy_str == "training":
        return model.training_layout_policy
    elif policy_str == "null":
        return NullLayoutPolicy()
    elif policy_str == "eos":
        # EOS-suffix-only layout: no identifier prefix, just an EOS stop token.
        # Always in-distribution: StochasticIdentifierPrefixLayoutPolicy always
        # appends EOS during training, so the model always sees this suffix.
        eos_id = getattr(model.inference_layout_policy, 'eos_token_id', 50256)
        return EOSLayoutPolicy(eos_token_id=eos_id)
    else:
        raise ValueError(
            f"Unknown layout_policy condition value {policy_str!r}. "
            "Expected 'eos', 'inference', 'training', or 'null'."
        )


# ─── Core dispatch ───────────────────────────────────────────────────────────

def run_benchmarks_on_model(
    model,
    dataset_dir: Union[str, Path],
    eval_cfg: Optional[Dict[str, Any]] = None,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Run the configured benchmark suite on an already-built inference model.

    Used by main.py to evaluate the end-of-training model without loading from
    disk again (which would trigger a fresh torch.compile).

    Each benchmark is run once per listed condition. Results are keyed as
    ``'{benchmark}/{condition}'`` (e.g. ``'held_out_perplexity/experimental'``).
    When a benchmark lists only one condition, the condition suffix is still
    included for consistency.

    Args:
        model:       TS2TSModel in eval mode. Must have layout policies set.
        dataset_dir: Path to the pretokenized dataset directory.
        eval_cfg:    Optional config dict from the YAML ``eval:`` block.
        device:      Device string.

    Returns:
        Dict mapping ``'{benchmark}/{condition}'`` to its result dict.
    """
    cfg = {**_DEFAULTS, **(eval_cfg or {})}
    max_docs: int  = int(cfg.get("max_docs", _DEFAULTS["max_docs"]))
    split: str     = cfg.get("split", _DEFAULTS["split"])
    epoch_dirs     = cfg.get("epoch_dirs", [])

    # Merge built-in conditions with any user-defined overrides
    all_conditions = {**_BUILTIN_CONDITIONS, **cfg.get("conditions", {})}

    # Normalise benchmark list — supports both string shorthand and dict form:
    #   "held_out_perplexity"  →  {"name": "held_out_perplexity", "conditions": ["experimental"]}
    raw_benchmarks = cfg.get("benchmarks", _DEFAULTS["benchmarks"])
    benchmark_specs: List[Dict[str, Any]] = []
    for b in raw_benchmarks:
        if isinstance(b, str):
            benchmark_specs.append({"name": b, "conditions": ["experimental"]})
        else:
            benchmark_specs.append(b)

    unknown = [b["name"] for b in benchmark_specs if b["name"] not in _KNOWN_BENCHMARKS]
    if unknown:
        raise ValueError(
            f"Unknown benchmarks: {unknown}. "
            f"Valid options: {list(_KNOWN_BENCHMARKS)}"
        )

    results: Dict[str, Any] = {}

    for spec in benchmark_specs:
        bname      = spec["name"]
        conditions = spec.get("conditions", ["experimental"])

        for cname in conditions:
            if cname not in all_conditions:
                raise ValueError(
                    f"Unknown condition {cname!r} for benchmark {bname!r}. "
                    f"Available: {sorted(all_conditions)}."
                )
            cond = all_conditions[cname]

            # Skip conditions that only make sense for cross_doc_link models.
            # Doc-causal models ARE the baseline; running the baseline condition
            # on them gives identical mask_type as experimental — no information.
            if cond.get("requires_cross_doc_link") and model.mask_type != "cross_doc_link":
                logger.debug(
                    "Skipping condition %r for %s/%r: requires cross_doc_link model",
                    cname, bname, model.mask_type,
                )
                continue

            mask_type = cond.get("mask_type")    # None → model default
            layout    = _resolve_layout_policy(cond.get("layout_policy"), model)
            key       = f"{bname}/{cname}"

            logger.info("Running %s (condition=%s)", bname, cname)

            if bname == "held_out_perplexity":
                from eval.perplexity import run_held_out_perplexity
                results[key] = run_held_out_perplexity(
                    model=model,
                    dataset_dir=dataset_dir,
                    layout_policy=layout,
                    split=split,
                    max_docs=max_docs,
                    device=device,
                    mask_type_override=mask_type,
                )

            elif bname == "hellaswag":
                from eval.hellaswag import run_hellaswag
                results[key] = run_hellaswag(
                    model=model,
                    max_examples=max_docs,
                    device=device,
                )

            elif bname == "pack_contrastive_perplexity":
                if model.mask_type != "cross_doc_link":
                    logger.info(
                        "Skipping pack_contrastive_perplexity: "
                        "model.mask_type=%r (requires cross_doc_link)",
                        model.mask_type,
                    )
                    continue
                if not epoch_dirs:
                    logger.warning(
                        "Skipping pack_contrastive_perplexity: "
                        "no epoch_dirs configured (set eval.epoch_dirs in config)."
                    )
                    continue
                from eval.perplexity import run_pack_contrastive_perplexity
                results[key] = run_pack_contrastive_perplexity(
                    model=model,
                    epoch_dirs=epoch_dirs,
                    dataset_dir=dataset_dir,
                    layout_policy=layout,
                    max_packs=max_docs,
                    device=device,
                )

    _log_summary(results)
    return results


# ─── Main callable (CLI / standalone use) ────────────────────────────────────

def run_eval(
    checkpoint_path: Union[str, Path],
    dataset_dir: Union[str, Path],
    eval_cfg: Optional[Dict[str, Any]] = None,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Load a checkpoint and run the configured benchmark suite.

    For post-training evaluation inside main.py use run_benchmarks_on_model()
    instead to avoid a redundant torch.compile.

    Args:
        checkpoint_path: Path to ``best_model.pt``.
        dataset_dir:     Path to the pretokenized dataset directory.
        eval_cfg:        Optional config dict from the YAML ``eval:`` block.
        device:          Device to run inference on.

    Returns:
        Dict mapping ``'{benchmark}/{condition}'`` to its result dict.
    """
    logger.info("Loading checkpoint: %s", checkpoint_path)
    model, _hp = load_inference_model(str(checkpoint_path), device=device)
    model.eval()
    return run_benchmarks_on_model(model, dataset_dir, eval_cfg=eval_cfg, device=device)


# ─── Logging helper ───────────────────────────────────────────────────────────

def _log_summary(results: Dict[str, Any]) -> None:
    logger.info("─── Benchmark results ───────────────────────────────")
    for name, res in results.items():
        if isinstance(res, dict) and "perplexity" in res:
            logger.info(
                "  %-40s  ppl=%.3f  mean_nll=%.4f  n=%d",
                name,
                res.get("perplexity", float("nan")),
                res.get("mean_nll", float("nan")),
                res.get("num_docs", 0),
            )
        elif isinstance(res, dict) and "accuracy" in res:
            logger.info(
                "  %-40s  acc=%.4f  n=%d",
                name,
                res.get("accuracy", float("nan")),
                res.get("total_examples", 0),
            )
        elif isinstance(res, dict) and any(
            isinstance(v, dict) and "mean_delta" in v for v in res.values()
        ):
            # pack_contrastive_perplexity: result is a dict of strategy -> stats
            for strategy, stats in res.items():
                logger.info(
                    "  %-40s  delta=%.4f [%.4f, %.4f]  cross=%.4f  base=%.4f  n=%d",
                    f"{name}/{strategy}",
                    stats.get("mean_delta", float("nan")),
                    stats.get("delta_ci_low", float("nan")),
                    stats.get("delta_ci_high", float("nan")),
                    stats.get("mean_nll_cross_doc", float("nan")),
                    stats.get("mean_nll_baseline", float("nan")),
                    stats.get("n_packs", 0),
                )
        else:
            logger.info("  %-40s  %s", name, res)
    logger.info("─────────────────────────────────────────────────────")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        stream=sys.stderr,
    )

    parser = argparse.ArgumentParser(
        description="Evaluate a trained TS2TS checkpoint on downstream benchmarks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to best_model.pt checkpoint.",
    )
    parser.add_argument(
        "--dataset", required=True,
        help="Path to pretokenized dataset directory.",
    )
    parser.add_argument(
        "--benchmarks", nargs="+", default=["held_out_perplexity"],
        choices=list(_KNOWN_BENCHMARKS),
        help="Benchmarks to run.",
    )
    parser.add_argument(
        "--conditions", nargs="+", default=["experimental"],
        help="Named conditions to run each benchmark under. "
             "Built-in: 'baseline' (doc_causal + null layout), "
             "'experimental' (model default). "
             "Multiple conditions produce side-by-side results.",
    )
    parser.add_argument(
        "--split", default="all",
        help='Graph split to evaluate (held_out_perplexity only). '
             'Use "all" to sample randomly from the full graph.',
    )
    parser.add_argument(
        "--max-docs", type=int, default=500,
        help="Maximum number of documents / examples per benchmark+condition.",
    )
    parser.add_argument(
        "--output", default=None,
        help="Path to write JSON results (default: print to stdout).",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on.",
    )
    args = parser.parse_args()

    eval_cfg = {
        "benchmarks": [
            {"name": b, "conditions": args.conditions}
            for b in args.benchmarks
        ],
        "split": args.split,
        "max_docs": args.max_docs,
    }

    results = run_eval(
        checkpoint_path=args.checkpoint,
        dataset_dir=args.dataset,
        eval_cfg=eval_cfg,
        device=args.device,
    )

    output_str = json.dumps(results, ensure_ascii=False, indent=2)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output_str, encoding="utf-8")
        logger.info("Results written to %s", args.output)
    else:
        print(output_str)


if __name__ == "__main__":
    main()
