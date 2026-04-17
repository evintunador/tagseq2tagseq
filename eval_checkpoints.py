"""
eval_checkpoints.py — downstream benchmark evaluation for TS2TS checkpoints.

Loads one or more checkpoints, reconstructs each model in inference mode, and
runs the configured benchmark suite. Results are written as JSON.

Benchmarks can be run under multiple named conditions in a single pass.
Each condition specifies a mask_type and layout_policy override, allowing
direct comparison between e.g. the model's experimental cross_doc_link
behaviour and a doc_causal baseline.

Usage (CLI — single checkpoint):
    python eval_checkpoints.py \\
        --checkpoints runs/YYYYMMDD/checkpoints/best_model.pt \\
        --dataset data/pretokenized_datasets/stack_10m \\
        [--benchmarks held_out_perplexity] \\
        [--split val_community] \\
        [--max-docs 500] \\
        [--output eval_results.json] \\
        [--device cuda]

    Results are auto-saved to {run_dir}/eval_results.json.
    Pass --output to override the save path.

Usage (CLI — multiple checkpoints):
    python eval_checkpoints.py \\
        --checkpoints runs/RUN_A/checkpoints/best_model.pt \\
                      runs/RUN_B/checkpoints/best_model.pt \\
        --dataset data/pretokenized_datasets/stack_10m

    Each checkpoint's results are saved to its own run dir.
    A combined comparison table is written to evals/YYYYMMDD_HHMMSS/.

Importable:
    from eval_checkpoints import run_eval, run_benchmarks_on_model
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
import datetime
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch

from generate import load_inference_model

logger = logging.getLogger(__name__)

# ─── Benchmark registry ───────────────────────────────────────────────────────

_KNOWN_BENCHMARKS = (
    "held_out_perplexity",
    # NLP — commonsense / general
    "hellaswag",
    "winogrande",
    "piqa",
    "boolq",
    "commonsense_qa",
    "copa",
    # NLP — science
    "arc_easy",
    "arc_challenge",
    "openbookqa",
    "sciq",
    # NLP — language modeling
    "wiki_qa",
    "lambada",
    # Code
    "codexglue_line_completion",
    # Graph-structured
    "pack_contrastive_perplexity",
)

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
#
# Single-doc benchmarks score each document in isolation — cross-doc grants
# can never fire regardless of mask_type, so 'experimental' on a cross_doc_link
# model is identical to 'baseline' but pays full BIM construction cost (~20s/doc).
# Benchmarks in _SINGLE_DOC_BENCHMARKS auto-skip conditions whose mask_type is
# None (i.e. would use the model's cross_doc_link default).

_SINGLE_DOC_BENCHMARKS = frozenset({
    "held_out_perplexity",
    "hellaswag",
    "winogrande",
    "piqa",
    "boolq",
    "commonsense_qa",
    "copa",
    "wiki_qa",
    "lambada",
    "arc_easy",
    "arc_challenge",
    "openbookqa",
    "sciq",
    "codexglue_line_completion",
})

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
        # held_out_perplexity uses doc_causal (baseline) only — single-doc scoring
        # means cross_doc_link grants can never fire, so experimental == baseline.
        {"name": "held_out_perplexity", "conditions": ["baseline", "experimental"]},
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


def _default_run_output_path(checkpoint_path: Path) -> Path:
    """Infer eval_results.json save path from checkpoint location.

    Convention: checkpoint lives at {run_dir}/checkpoints/best_model.pt
    → run_dir = checkpoint.parent.parent.
    Falls back to checkpoint.parent if the immediate parent dir is not
    named 'checkpoints'.
    """
    p = checkpoint_path.resolve()
    if p.parent.name == "checkpoints":
        return p.parent.parent / "eval_results.json"
    return p.parent / "eval_results.json"


# ─── Core dispatch ────────────────────────────────────────────────────────────

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

            # Skip the experimental condition on single-doc benchmarks when the
            # model is cross_doc_link. Each document is scored in isolation so
            # cross-doc grants can never fire — the result is identical to
            # baseline/doc_causal but pays full BIM construction cost per doc.
            if (
                bname in _SINGLE_DOC_BENCHMARKS
                and cond.get("mask_type") is None
                and model.mask_type == "cross_doc_link"
            ):
                logger.info(
                    "Skipping condition %r for %s: single-doc benchmark, "
                    "cross_doc_link mask has no effect (identical to baseline).",
                    cname, bname,
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
                from eval.nlp_benchmarks import run_hellaswag
                results[key] = run_hellaswag(
                    model=model,
                    max_examples=max_docs,
                    device=device,
                )

            elif bname == "wiki_qa":
                from eval.nlp_benchmarks import run_wiki_qa
                results[key] = run_wiki_qa(
                    model=model,
                    max_examples=max_docs,
                    device=device,
                )

            elif bname == "lambada":
                from eval.nlp_benchmarks import run_lambada
                results[key] = run_lambada(
                    model=model,
                    max_examples=max_docs,
                    device=device,
                )

            elif bname in ("arc_easy", "arc_challenge"):
                from eval.nlp_benchmarks import run_arc
                results[key] = run_arc(
                    model=model,
                    config="easy" if bname == "arc_easy" else "challenge",
                    max_examples=max_docs,
                    device=device,
                )

            elif bname == "winogrande":
                from eval.nlp_benchmarks import run_winogrande
                results[key] = run_winogrande(
                    model=model, max_examples=max_docs, device=device,
                )

            elif bname == "piqa":
                from eval.nlp_benchmarks import run_piqa
                results[key] = run_piqa(
                    model=model, max_examples=max_docs, device=device,
                )

            elif bname == "boolq":
                from eval.nlp_benchmarks import run_boolq
                results[key] = run_boolq(
                    model=model, max_examples=max_docs, device=device,
                )

            elif bname == "commonsense_qa":
                from eval.nlp_benchmarks import run_commonsense_qa
                results[key] = run_commonsense_qa(
                    model=model, max_examples=max_docs, device=device,
                )

            elif bname == "copa":
                from eval.nlp_benchmarks import run_copa
                results[key] = run_copa(
                    model=model, max_examples=max_docs, device=device,
                )

            elif bname == "openbookqa":
                from eval.nlp_benchmarks import run_openbookqa
                results[key] = run_openbookqa(
                    model=model, max_examples=max_docs, device=device,
                )

            elif bname == "sciq":
                from eval.nlp_benchmarks import run_sciq
                results[key] = run_sciq(
                    model=model, max_examples=max_docs, device=device,
                )

            elif bname == "codexglue_line_completion":
                from eval.nlp_benchmarks import run_codexglue_line_completion
                results[key] = run_codexglue_line_completion(
                    model=model, max_examples=max_docs, device=device,
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
            # held_out_perplexity uses "mean_nll"/"num_docs";
            # lambada (FillInTheBlank) uses "average_nll"/"total_examples".
            nll = res.get("mean_nll") or res.get("average_nll", float("nan"))
            n   = res.get("num_docs") or res.get("total_examples", 0)
            logger.info(
                "  %-40s  ppl=%.3f  nll=%.4f  n=%d",
                name,
                res.get("perplexity", float("nan")),
                nll,
                n,
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


# ─── Comparison table helpers ─────────────────────────────────────────────────

def _build_comparison_entry(
    checkpoint_path: Path,
    dataset_dir: str,
    results: Dict[str, Any],
    hp: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build one row for the comparison table from a single checkpoint's results.

    The entry includes checkpoint path, run_dir basename, model metadata read
    from hyperparameters.json (mask_type, num_layers, model_dim, dataset), and
    the full results dict for every benchmark/condition that was run.

    Args:
        checkpoint_path: Resolved path to best_model.pt.
        dataset_dir:     Dataset path string (for the table).
        results:         Output of run_benchmarks_on_model for this checkpoint.
        hp:              Already-loaded hyperparameters dict, or None to load
                         from {run_dir}/hyperparameters.json if it exists.
    """
    p = checkpoint_path.resolve()
    run_dir = p.parent.parent if p.parent.name == "checkpoints" else p.parent

    # Load hyperparameters if not provided
    if hp is None:
        hp_path = run_dir / "hyperparameters.json"
        if hp_path.exists():
            try:
                with open(hp_path) as f:
                    hp = json.load(f)
            except Exception:
                hp = {}
        else:
            hp = {}

    model_cfg = hp.get("model", {})
    data_cfg  = hp.get("data", {})

    entry: Dict[str, Any] = {
        "checkpoint":  str(checkpoint_path),
        "run_dir":     str(run_dir),
        "run_dir_name": run_dir.name,
        "mask_type":   model_cfg.get("mask_type", "unknown"),
        "num_layers":  model_cfg.get("num_layers"),
        "model_dim":   model_cfg.get("model_dim"),
        "dataset":     Path(data_cfg.get("dataset_dir", dataset_dir)).name,
        "strategy":    data_cfg.get("strategy"),
    }
    entry.update(results)
    return entry


def _headline_metric(key: str, result: Any) -> str:
    """Extract a single printable number from a benchmark result for the .txt table."""
    nan = float("nan")
    if not isinstance(result, dict):
        return "—"

    # held_out_perplexity → show perplexity
    if "perplexity" in result:
        v = result.get("perplexity", nan)
        return "—" if (v is None or (isinstance(v, float) and math.isnan(v))) else f"{v:.2f}"

    # hellaswag / MC → show accuracy
    if "accuracy" in result:
        v = result.get("accuracy", nan)
        return "—" if (v is None or (isinstance(v, float) and math.isnan(v))) else f"{v:.4f}"

    # pack_contrastive_perplexity → dict of strategy → stats; show first strategy's delta
    for _strategy, stats in result.items():
        if isinstance(stats, dict) and "mean_delta" in stats:
            v = stats.get("mean_delta", nan)
            return "—" if (v is None or (isinstance(v, float) and math.isnan(v))) else f"{v:+.4f}"

    return "—"


def _write_comparison_table(eval_dir: Path, entries: List[Dict[str, Any]]) -> None:
    """Write comparison_table.json and comparison_table.txt to eval_dir.

    comparison_table.json — list of full entry dicts (one per checkpoint).
    comparison_table.txt  — human-readable fixed-width ASCII table with one
        column per benchmark/condition showing its headline metric.
    """
    # JSON
    json_path = eval_dir / "comparison_table.json"
    json_path.write_text(
        json.dumps(entries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Collect all benchmark/condition keys present across any entry.
    meta_keys = {"checkpoint", "run_dir", "run_dir_name", "mask_type",
                 "num_layers", "model_dim", "dataset", "strategy"}
    bench_keys: List[str] = []
    seen: set = set()
    for entry in entries:
        for k in entry:
            if k not in meta_keys and k not in seen:
                bench_keys.append(k)
                seen.add(k)

    # Build rows: [run_dir_name, mask_type, dataset, strategy, ...headline metrics...]
    header = ["run_dir", "mask_type", "dataset", "strategy"] + bench_keys
    rows: List[List[str]] = [header]
    for entry in entries:
        row = [
            entry.get("run_dir_name", "—"),
            entry.get("mask_type", "—"),
            entry.get("dataset", "—"),
            entry.get("strategy") or "—",
        ]
        for k in bench_keys:
            row.append(_headline_metric(k, entry.get(k)))
        rows.append(row)

    # Compute column widths
    col_widths = [max(len(r[i]) for r in rows) for i in range(len(header))]

    lines = []
    for i, row in enumerate(rows):
        lines.append("  ".join(cell.ljust(col_widths[j]) for j, cell in enumerate(row)))
        if i == 0:
            lines.append("  ".join("-" * w for w in col_widths))

    txt_path = eval_dir / "comparison_table.txt"
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    logger.info("Comparison table written to %s", eval_dir)
    logger.info("\n%s", "\n".join(lines))


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        stream=sys.stderr,
    )

    parser = argparse.ArgumentParser(
        description="Evaluate one or more trained TS2TS checkpoints on downstream benchmarks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoints", nargs="+", required=True, metavar="PATH",
        help="One or more paths to best_model.pt checkpoints. "
             "Each checkpoint's results are saved to its own run dir. "
             "With multiple checkpoints a comparison table is also written "
             "to evals/YYYYMMDD_HHMMSS/.",
    )
    parser.add_argument(
        "--dataset", required=True,
        help="Path to pretokenized dataset directory.",
    )
    parser.add_argument(
        "--benchmarks", nargs="+", default=["held_out_perplexity"],
        choices=list(_KNOWN_BENCHMARKS),
        help=(
            "Benchmarks to run. NLP commonsense: hellaswag, winogrande, piqa, boolq, "
            "commonsense_qa, copa. NLP science: arc_easy, arc_challenge, openbookqa, sciq. "
            "NLP language: wiki_qa, lambada. Code: codexglue_line_completion. "
            "Graph: pack_contrastive_perplexity."
        ),
    )
    parser.add_argument(
        "--conditions", nargs="+", default=["experimental"],
        help="Named conditions to run each benchmark under. "
             "Built-in: 'baseline' (doc_causal + eos layout), "
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
        help="Override the save path for results JSON. "
             "Only valid when a single checkpoint is given; "
             "ignored when evaluating multiple checkpoints.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on.",
    )
    args = parser.parse_args()

    checkpoints = [Path(c) for c in args.checkpoints]
    multi = len(checkpoints) > 1

    if multi and args.output:
        logger.warning(
            "--output is ignored when evaluating multiple checkpoints; "
            "results are written to each checkpoint's run dir."
        )

    eval_cfg = {
        "benchmarks": [
            {"name": b, "conditions": args.conditions}
            for b in args.benchmarks
        ],
        "split": args.split,
        "max_docs": args.max_docs,
    }

    from tunalab.reproducibility import ReproducibilityManager

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create eval dir for multi-checkpoint runs.
    eval_dir: Optional[Path] = None
    if multi:
        eval_dir = Path(__file__).parent / "evals" / ts
        eval_dir.mkdir(parents=True, exist_ok=True)

    # ReproducibilityManager captures git state, pip freeze, and env for the
    # entire eval session.  For single runs, use a timestamped subdir inside
    # the run dir so training-time and eval-time snapshots don't collide and
    # successive re-runs each get their own clean directory.
    if eval_dir is not None:
        rm_output_dir = eval_dir
    else:
        run_dir = _default_run_output_path(checkpoints[0]).parent
        rm_output_dir = run_dir / "eval" / ts

    with ReproducibilityManager(output_dir=str(rm_output_dir), is_main_process=True):
        all_entries: List[Dict[str, Any]] = []

        for ckpt_path in checkpoints:
            results = run_eval(
                checkpoint_path=ckpt_path,
                dataset_dir=args.dataset,
                eval_cfg=eval_cfg,
                device=args.device,
            )

            # Per-checkpoint save path
            if not multi and args.output:
                out_path = Path(args.output)
            else:
                out_path = _default_run_output_path(ckpt_path)

            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(
                json.dumps(results, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.info("Results written to %s", out_path)

            if multi:
                all_entries.append(
                    _build_comparison_entry(ckpt_path, args.dataset, results)
                )

        if multi and all_entries:
            _write_comparison_table(eval_dir, all_entries)


if __name__ == "__main__":
    main()
