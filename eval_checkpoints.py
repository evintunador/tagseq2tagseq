"""
eval_checkpoints.py — downstream benchmark evaluation for TS2TS checkpoints.

Loads a checkpoint, reconstructs the model in inference mode, and runs the
configured benchmark suite. Results are written as JSON.

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

# ─── Registry ────────────────────────────────────────────────────────────────

_KNOWN_BENCHMARKS = ("held_out_perplexity", "hellaswag")

_DEFAULTS: Dict[str, Any] = {
    "benchmarks": ["held_out_perplexity"],
    "max_docs": 500,
    "split": "all",  # "all" = random sample; use "val_community" for datasets with split annotations
}


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

    Args:
        model: TS2TSModel in eval mode. Must have ``model.layout_policy`` set.
        dataset_dir: Path to the pretokenized dataset directory.
        eval_cfg: Optional config dict from the YAML ``eval:`` block.
        device: Device string.

    Returns:
        Dict mapping benchmark name to its result dict.
    """
    cfg = {**_DEFAULTS, **(eval_cfg or {})}
    benchmarks: List[str] = cfg.get("benchmarks", _DEFAULTS["benchmarks"])
    max_docs: int = int(cfg.get("max_docs", _DEFAULTS["max_docs"]))
    split: str = cfg.get("split", _DEFAULTS["split"])

    unknown = [b for b in benchmarks if b not in _KNOWN_BENCHMARKS]
    if unknown:
        raise ValueError(
            f"Unknown benchmarks: {unknown}. "
            f"Valid options: {list(_KNOWN_BENCHMARKS)}"
        )

    layout_policy = model.layout_policy
    results: Dict[str, Any] = {}

    for benchmark in benchmarks:
        logger.info("Running benchmark: %s", benchmark)

        if benchmark == "held_out_perplexity":
            from eval.perplexity import run_held_out_perplexity
            results["held_out_perplexity"] = run_held_out_perplexity(
                model=model,
                dataset_dir=dataset_dir,
                layout_policy=layout_policy,
                split=split,
                max_docs=max_docs,
                device=device,
            )

        elif benchmark == "hellaswag":
            from eval.hellaswag import run_hellaswag
            results["hellaswag"] = run_hellaswag(
                model=model,
                max_examples=max_docs,
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

    Loads the checkpoint from disk (calls load_inference_model). For
    post-training evaluation inside main.py use run_benchmarks_on_model()
    instead to avoid a redundant torch.compile.

    Args:
        checkpoint_path: Path to ``best_model.pt``.
        dataset_dir: Path to the pretokenized dataset directory.
        eval_cfg: Optional config dict from the YAML ``eval:`` block.
        device: Device to run inference on.

    Returns:
        Dict mapping benchmark name to its result dict.
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
                "  %-30s  ppl=%.3f  mean_nll=%.4f  n=%d",
                name,
                res.get("perplexity", float("nan")),
                res.get("mean_nll", float("nan")),
                res.get("num_docs", 0),
            )
        elif isinstance(res, dict) and "accuracy" in res:
            logger.info(
                "  %-30s  acc=%.4f  n=%d",
                name,
                res.get("accuracy", float("nan")),
                res.get("total_examples", 0),
            )
        else:
            logger.info("  %-30s  %s", name, res)
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
        "--split", default="all",
        help='Graph split to evaluate (held_out_perplexity only). '
             'Use "all" (default) to sample randomly from the full graph; '
             'use "val_community" or "val_random" for datasets with split annotations.',
    )
    parser.add_argument(
        "--max-docs", type=int, default=500,
        help="Maximum number of documents / examples per benchmark.",
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
        "benchmarks": args.benchmarks,
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
