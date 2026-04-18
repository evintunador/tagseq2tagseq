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
from eval.perplexity import run_held_out_perplexity, run_pack_contrastive_perplexity
from eval.nlp_benchmarks import (
    run_hellaswag, run_wiki_qa, run_arc, run_lambada,
    run_winogrande, run_piqa, run_boolq, run_commonsense_qa, run_copa,
    run_openbookqa, run_sciq, run_codexglue_line_completion,
    run_mmlu, run_mathqa, run_math,
    run_codexglue_code_to_text, run_repobench, run_humaneval_buggy,
)

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
    # STEM / math (HF-direct)
    "mmlu",               # requires --mmlu-subject
    "mathqa",
    "math",               # requires --math-subject
    # Code (tunalab-backed)
    "codexglue_line_completion",
    # Code (HF-direct)
    "codexglue_code_to_text",
    "repobench",          # requires --repobench-split
    "humaneval_buggy",    # requires --humaneval-language
    # Graph-structured
    "pack_contrastive_perplexity",
)

# ─── Condition registry ───────────────────────────────────────────────────────
# Conditions specify forward_inference overrides applied per benchmark run.
#
# layout_policy string values:
#   'eos'       → EOSLayoutPolicy (no identifier prefix, EOS suffix only).
#                 Always in-distribution: training always appends EOS.
#   'inference' → model.inference_layout_policy (set at checkpoint load time).
#   'training'  → model.training_layout_policy.
#   'null'      → NullLayoutPolicy() — no decoration.
# mask_type None → use model's trained mask_type unchanged.
#
# Built-in conditions and their intended use:
#
#   'doceval'     — doc_causal + eos layout, runs on ALL models.
#                   Use for head-to-head comparison tables where every model
#                   fills one column per benchmark. This is the standard
#                   condition for cross-model comparisons (e.g. the 2×4 grid).
#
#   'baseline'    — doc_causal + eos layout, cross_doc_link models ONLY
#                   (requires_cross_doc_link=True skips doc_causal models).
#                   Use to measure the doc_causal floor for a cross_doc_link
#                   model, isolating the benefit of cross-doc attention.
#
#   'experimental'— model's own trained mask_type + inference layout.
#                   Skipped automatically on single-doc benchmarks for
#                   cross_doc_link models (grants can never fire on isolated
#                   docs, result is identical to baseline but wastes BIM cost).
#                   Use for graph-structured benchmarks (pack_contrastive_perplexity)
#                   or to measure a cross_doc_link model's native behaviour.
#
# Validated checkpoint grid (as of 2026-04-17):
#   20260308_012514  doc_causal      simplewiki  36L/1280D  dfs  → use doceval
#   20260308_012516  cross_doc_link  simplewiki  36L/1280D  dfs  → use doceval / baseline+experimental
#   20260308_012518  doc_causal      stack_10m   36L/1280D  dfs  → use doceval
#   20260308_012521  cross_doc_link  stack_10m   36L/1280D  dfs  → use doceval / baseline+experimental
#   run_20260311_184203_685319  cross_doc_link  stack_100m  24L/1024D  bfs  step=3000  → early ckpt
#   run_20260313_183004_686307  cross_doc_link  stack_100m  24L/1024D  bfs  step=900   → early ckpt
#
# CHECKLIST FOR NEW BENCHMARKS:
#   1. Add name to _KNOWN_BENCHMARKS (with category comment).
#   2. Add to _SINGLE_DOC_BENCHMARKS if it scores documents in isolation.
#   3. Add a dispatch case in run_benchmarks_on_model.
#   4. Add to _SINGLE_DOC_BENCHMARKS tests in tests/test_eval_checkpoints.py.
#   5. Consider which conditions make sense: doceval for standard comparison;
#      baseline+experimental if the benchmark can benefit from cross-doc attention
#      (i.e. it is NOT in _SINGLE_DOC_BENCHMARKS).
#
# Single-doc benchmarks auto-skip 'experimental' on cross_doc_link models:
# each item is scored in isolation so cross-doc grants can never fire —
# the result is identical to 'baseline' but pays full BIM construction cost.

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
    # STEM / math
    "mmlu",
    "mathqa",
    "math",
    # Code
    "codexglue_line_completion",
    "codexglue_code_to_text",
    "repobench",
    "humaneval_buggy",
})

_BUILTIN_CONDITIONS: Dict[str, Dict[str, Any]] = {
    "baseline": {
        "mask_type":               "doc_causal",
        "layout_policy":           "eos",
        "requires_cross_doc_link": True,
    },
    "experimental": {
        "mask_type":    None,
        "layout_policy": "inference",
    },
    "doceval": {
        "mask_type":    "doc_causal",
        "layout_policy": "eos",
    },
}

_DEFAULTS: Dict[str, Any] = {
    "benchmarks": [
        {"name": "held_out_perplexity", "conditions": ["baseline", "experimental"]},
    ],
    "conditions": {},
    "max_docs": 500,
    "split": "all",
    "epoch_dirs": [],
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

            try:
                if bname == "held_out_perplexity":
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
                    results[key] = run_hellaswag(
                        model=model, max_examples=max_docs, device=device,
                    )

                elif bname == "wiki_qa":
                    results[key] = run_wiki_qa(
                        model=model, max_examples=max_docs, device=device,
                    )

                elif bname == "lambada":
                    results[key] = run_lambada(
                        model=model, max_examples=max_docs, device=device,
                    )

                elif bname in ("arc_easy", "arc_challenge"):
                    results[key] = run_arc(
                        model=model,
                        config="easy" if bname == "arc_easy" else "challenge",
                        max_examples=max_docs,
                        device=device,
                    )

                elif bname == "winogrande":
                    results[key] = run_winogrande(
                        model=model, max_examples=max_docs, device=device,
                    )

                elif bname == "piqa":
                    results[key] = run_piqa(
                        model=model, max_examples=max_docs, device=device,
                    )

                elif bname == "boolq":
                    results[key] = run_boolq(
                        model=model, max_examples=max_docs, device=device,
                    )

                elif bname == "commonsense_qa":
                    results[key] = run_commonsense_qa(
                        model=model, max_examples=max_docs, device=device,
                    )

                elif bname == "copa":
                    results[key] = run_copa(
                        model=model, max_examples=max_docs, device=device,
                    )

                elif bname == "openbookqa":
                    results[key] = run_openbookqa(
                        model=model, max_examples=max_docs, device=device,
                    )

                elif bname == "sciq":
                    results[key] = run_sciq(
                        model=model, max_examples=max_docs, device=device,
                    )

                elif bname == "codexglue_line_completion":
                    results[key] = run_codexglue_line_completion(
                        model=model, max_examples=max_docs, device=device,
                    )

                elif bname == "mmlu":
                    results[key] = run_mmlu(
                        model=model,
                        subject=spec.get("subject", "college_mathematics"),
                        max_examples=max_docs,
                        device=device,
                    )

                elif bname == "mathqa":
                    results[key] = run_mathqa(
                        model=model, max_examples=max_docs, device=device,
                    )

                elif bname == "math":
                    results[key] = run_math(
                        model=model,
                        subject=spec.get("subject", "algebra"),
                        max_examples=max_docs,
                        device=device,
                    )

                elif bname == "codexglue_code_to_text":
                    results[key] = run_codexglue_code_to_text(
                        model=model, max_examples=max_docs, device=device,
                    )

                elif bname == "repobench":
                    results[key] = run_repobench(
                        model=model,
                        split=spec.get("split", "cross_file_first"),
                        max_examples=max_docs,
                        device=device,
                    )

                elif bname == "humaneval_buggy":
                    results[key] = run_humaneval_buggy(
                        model=model,
                        language=spec.get("language", "python"),
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
                    results[key] = run_pack_contrastive_perplexity(
                        model=model,
                        epoch_dirs=epoch_dirs,
                        dataset_dir=dataset_dir,
                        layout_policy=layout,
                        max_packs=max_docs,
                        device=device,
                    )

            except Exception as _exc:
                logger.error(
                    "Benchmark %s (condition=%s) failed and will be skipped: %s: %s",
                    bname, cname, type(_exc).__name__, _exc,
                )
                results[key] = {"error": str(_exc)}

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
            "Benchmarks to run. "
            "NLP commonsense: hellaswag, winogrande, piqa, boolq, commonsense_qa, copa. "
            "NLP science: arc_easy, arc_challenge, openbookqa, sciq. "
            "NLP language: wiki_qa, lambada. "
            "STEM/math: mmlu (see --mmlu-subject), mathqa, math (see --math-subject). "
            "Code: codexglue_line_completion, codexglue_code_to_text, "
            "repobench (see --repobench-split), humaneval_buggy (see --humaneval-language). "
            "Graph: pack_contrastive_perplexity."
        ),
    )
    parser.add_argument(
        "--mmlu-subject", default="college_mathematics",
        help="MMLU subject to evaluate (used when 'mmlu' is in --benchmarks). "
             "Examples: college_mathematics, high_school_physics, machine_learning, "
             "college_computer_science. See eval.nlp_benchmarks.MMLU_STEM_SUBJECTS.",
    )
    parser.add_argument(
        "--math-subject", default="algebra",
        help="MATH dataset subject (used when 'math' is in --benchmarks). "
             "One of: algebra, counting_and_probability, geometry, "
             "intermediate_algebra, number_theory, prealgebra, precalculus.",
    )
    parser.add_argument(
        "--repobench-split", default="cross_file_first",
        choices=["cross_file_first", "cross_file_random", "in_file"],
        help="RepoBench-C split (used when 'repobench' is in --benchmarks). "
             "cross_file_first is most relevant for cross_doc_link models.",
    )
    parser.add_argument(
        "--humaneval-language", default="python",
        choices=["python", "cpp", "go", "java", "js", "rust"],
        help="HumanEvalPack language (used when 'humaneval_buggy' is in --benchmarks).",
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

    # Per-benchmark extra params from CLI — folded into each spec dict.
    _bench_extras: Dict[str, Dict[str, Any]] = {
        "mmlu":            {"subject": args.mmlu_subject},
        "math":            {"subject": args.math_subject},
        "repobench":       {"split": args.repobench_split},
        "humaneval_buggy": {"language": args.humaneval_language},
    }

    eval_cfg = {
        "benchmarks": [
            {"name": b, "conditions": args.conditions, **_bench_extras.get(b, {})}
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
