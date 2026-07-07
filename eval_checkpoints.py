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
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

import torch

from generate import load_inference_model
from eval.perplexity import run_held_out_perplexity, run_pack_contrastive_perplexity, run_community_pack_perplexity
from eval.nlp_benchmarks import (
    run_hellaswag, run_wiki_qa, run_arc, run_lambada,
    run_winogrande, run_piqa, run_boolq, run_commonsense_qa, run_copa,
    run_openbookqa, run_sciq, run_codexglue_line_completion,
    run_mmlu, run_mathqa, run_math,
    run_codexglue_code_to_text, run_repobench, run_repobench_cross_doc,
    run_humaneval_buggy,
    run_hotpotqa, run_hotpotqa_cross_doc,
    run_benchmark_annotated, ANNOTATABLE_BENCHMARKS,
    MMLU_STEM_SUBJECTS, MATH_SUBJECTS,
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
    "repobench_cross_doc",
    "humaneval_buggy",    # requires --humaneval-language
    # Wikipedia multi-hop QA
    "hotpotqa",           # bridge-type, flat concat baseline
    "hotpotqa_cross_doc", # bridge-type, cross-doc-link structured variant
    # Graph-structured (multi-doc; experimental condition auto-applies)
    "pack_contrastive_perplexity",  # cross_doc_link models only (requires precomputed epoch dirs)
    "community_pack_perplexity",    # all models; cross_doc_link gets contrastive delta, doc_causal gets baseline
    # Annotated benchmarks — NLP benchmarks run under injected cross-doc links
    # Dispatched specially (not via condition loop); uses MarkdownPromptAnnotator.
    # Any benchmark in ANNOTATABLE_BENCHMARKS can carry "annotated" in its conditions list.
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
#   20260308_012518  doc_causal      stack_10m   36L/1280D  dfs  → use doceval  # TODO: stack_10m deleted, replace with thestack version when available
#   20260308_012521  cross_doc_link  stack_10m   36L/1280D  dfs  → use doceval / baseline+experimental  # TODO: stack_10m deleted, replace with thestack version when available
#   run_20260311_184203_685319  cross_doc_link  stack_100m  24L/1024D  bfs  step=3000  → early ckpt  # TODO: stack_100m deleted, replace with thestack version when available
#   run_20260313_183004_686307  cross_doc_link  stack_100m  24L/1024D  bfs  step=900   → early ckpt  # TODO: stack_100m deleted, replace with thestack version when available
#
# CHECKLIST FOR NEW BENCHMARKS:
#   1. Add name to _KNOWN_BENCHMARKS (with category comment).
#   2. Add to _SINGLE_DOC_BENCHMARKS if it scores documents in isolation.
#      (hotpotqa YES — flat concat; hotpotqa_cross_doc NO — multi-doc.)
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
    # hotpotqa is single-doc (flat concat); hotpotqa_cross_doc is NOT in this set
    "hotpotqa",
})

# Mask types whose attention is multi-document (so the 'experimental' condition
# differs from the doc_causal 'baseline'). Single-doc benchmarks score each doc
# in isolation, so for these models 'experimental' == 'baseline' and is skipped.
_MULTI_DOC_MASK_TYPES = frozenset({
    "cross_doc_link",
    "doc_concat_link",
    "doc_concatenated",
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
    # Sentinel for the annotated condition — handled separately after the main
    # condition loop by run_benchmarks_on_model. The main loop skips it.
    "annotated": {
        "_is_annotated": True,
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

            # The "annotated" condition is handled separately after this loop.
            if cond.get("_is_annotated"):
                continue

            # Skip the doc_causal 'baseline' condition on models whose own mask
            # is already doc_causal — it would be identical to experimental, no
            # information. Multi-doc models (cross_doc_link, doc_concat*) all get
            # a meaningful doc_causal baseline floor.
            if cond.get("requires_cross_doc_link") and model.mask_type not in _MULTI_DOC_MASK_TYPES:
                logger.debug(
                    "Skipping condition %r for %s/%r: requires multi-doc model",
                    cname, bname, model.mask_type,
                )
                continue

            # Skip the experimental condition on single-doc benchmarks when the
            # model uses a multi-doc mask. Each document is scored in isolation so
            # cross-doc attention can never fire — the result is identical to
            # baseline/doc_causal but pays full mask construction cost per doc.
            if (
                bname in _SINGLE_DOC_BENCHMARKS
                and cond.get("mask_type") is None
                and model.mask_type in _MULTI_DOC_MASK_TYPES
            ):
                logger.info(
                    "Skipping condition %r for %s: single-doc benchmark, "
                    "%s mask has no effect (identical to baseline).",
                    cname, bname, model.mask_type,
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
                    for subject in MMLU_STEM_SUBJECTS:
                        subj_key = f"mmlu/{subject}/{cname}"
                        try:
                            results[subj_key] = run_mmlu(
                                model=model,
                                subject=subject,
                                max_examples=max_docs,
                                device=device,
                            )
                        except Exception as _exc:
                            logger.error(
                                "Benchmark mmlu/%s (condition=%s) failed: %s: %s",
                                subject, cname, type(_exc).__name__, _exc,
                            )
                            results[subj_key] = {"error": str(_exc)}
                    continue  # skip the generic results[key] assignment below

                elif bname == "mathqa":
                    results[key] = run_mathqa(
                        model=model, max_examples=max_docs, device=device,
                    )

                elif bname == "math":
                    for subject in MATH_SUBJECTS:
                        subj_key = f"math/{subject}/{cname}"
                        try:
                            results[subj_key] = run_math(
                                model=model,
                                subject=subject,
                                max_examples=max_docs,
                                device=device,
                            )
                        except Exception as _exc:
                            logger.error(
                                "Benchmark math/%s (condition=%s) failed: %s: %s",
                                subject, cname, type(_exc).__name__, _exc,
                            )
                            results[subj_key] = {"error": str(_exc)}
                    continue  # skip the generic results[key] assignment below

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

                elif bname == "repobench_cross_doc":
                    if model.mask_type != "cross_doc_link":
                        logger.info(
                            "Skipping repobench_cross_doc: requires cross_doc_link model "
                            "(model.mask_type=%r).",
                            model.mask_type,
                        )
                        continue
                    results[key] = run_repobench_cross_doc(
                        model=model,
                        max_examples=max_docs,
                        device=device,
                    )

                elif bname == "hotpotqa":
                    results[key] = run_hotpotqa(
                        model=model,
                        max_examples=max_docs,
                        device=device,
                    )

                elif bname == "hotpotqa_cross_doc":
                    if model.mask_type != "cross_doc_link":
                        logger.info(
                            "Skipping hotpotqa_cross_doc: requires cross_doc_link model "
                            "(model.mask_type=%r).",
                            model.mask_type,
                        )
                        continue
                    from model.graph_traversal.markdown_link_detector import MarkdownLinkDetector as _MLD
                    if not isinstance(getattr(model, "link_detector", None), _MLD):
                        logger.info(
                            "Skipping hotpotqa_cross_doc: requires MarkdownLinkDetector "
                            "(model.link_detector=%r). Only Wikipedia models apply.",
                            type(getattr(model, "link_detector", None)).__name__,
                        )
                        continue
                    results[key] = run_hotpotqa_cross_doc(
                        model=model,
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

                elif bname == "community_pack_perplexity":
                    results[key] = run_community_pack_perplexity(
                        model=model,
                        dataset_dir=dataset_dir,
                        layout_policy=layout,
                        split=spec.get("split", "val_community"),
                        max_packs=max_docs,
                        device=device,
                    )

            except Exception as _exc:
                logger.error(
                    "Benchmark %s (condition=%s) failed and will be skipped: %s: %s",
                    bname, cname, type(_exc).__name__, _exc,
                )
                results[key] = {"error": str(_exc)}

    # ── Annotated condition pass ──────────────────────────────────────────────
    # Benchmarks that include "annotated" in their conditions list are dispatched
    # here rather than in the main loop, because they require building a
    # Prompt annotators inject dataset-specific link syntax into benchmark prompts.
    # MarkdownPromptAnnotator: [display](Title) for wiki/markdown models.
    # ArxivPromptAnnotator:   \cite{Title}      for arxiv models.
    # Guard: cross_doc_link models with a supported link detector.
    annotated_specs = [
        spec for spec in benchmark_specs
        if "annotated" in spec.get("conditions", [])
        and spec["name"] in ANNOTATABLE_BENCHMARKS
    ]
    if annotated_specs:
        from model.graph_traversal.markdown_link_detector import MarkdownLinkDetector as _MLD
        from model.graph_traversal.arxiv_cite_detector import ArxivCiteDetector as _ACD
        _detector = getattr(model, "link_detector", None)
        _mask_type = getattr(model, "mask_type", None)
        _is_markdown = _mask_type == "cross_doc_link" and isinstance(_detector, _MLD)
        _is_arxiv    = _mask_type == "cross_doc_link" and isinstance(_detector, _ACD)
        _is_annotatable = _is_markdown or _is_arxiv
        if not _is_annotatable:
            logger.info(
                "Skipping annotated condition: requires cross_doc_link model with "
                "MarkdownLinkDetector or ArxivCiteDetector (mask_type=%r, link_detector=%r).",
                _mask_type,
                type(_detector).__name__,
            )
        else:
            annotator_corpus_dir = cfg.get("annotator_corpus") or str(dataset_dir)
            annotator_mode = cfg.get("annotator_mode", "full_skip")
            try:
                from model.document_corpus import PretokCorpus
                from eval.link_annotator import (
                    MarkdownPromptAnnotator, ArxivPromptAnnotator, TrieTitleIndex,
                )
                from eval.title_index import HashNormTitleIndex
                _corpus = PretokCorpus(
                    annotator_corpus_dir, link_detector=model.link_detector
                )
                _raw_ids = [
                    node["raw_identifier"]
                    for node in _corpus._graph.nodes.values()
                    if "raw_identifier" in node
                ]
                _fallback = HashNormTitleIndex(
                    _raw_ids,
                    strategies=cfg.get(
                        "annotator_strategies",
                        ("exact", "norm", "word_overlap_ordered", "edit_distance"),
                    ),
                    edit_distance_threshold=cfg.get("annotator_ed_threshold", 0.2),
                )
                if _is_markdown and cfg.get("annotator_use_trie", False):
                    _title_index = TrieTitleIndex(
                        _raw_ids,
                        model.tokenizer,
                        beam_width=cfg.get("annotator_beam_width", 1),
                        length_penalty=cfg.get("annotator_length_penalty", 0.0),
                        min_joint_logprob=cfg.get("annotator_trie_min_logprob"),
                        fallback_index=_fallback,
                    )
                else:
                    _title_index = _fallback
                if _is_arxiv:
                    _annotator = ArxivPromptAnnotator(
                        corpus=_corpus,
                        title_index=_title_index,
                        link_retrieval_mode=annotator_mode,
                        layout_policy=getattr(model, "inference_layout_policy", None),
                    )
                else:
                    _annotator = MarkdownPromptAnnotator(
                        corpus=_corpus,
                        title_index=_title_index,
                        link_retrieval_mode=annotator_mode,
                        layout_policy=getattr(model, "inference_layout_policy", None),
                    )
                logger.info(
                    "Annotated eval: annotator=%s corpus=%s mode=%s",
                    type(_annotator).__name__, annotator_corpus_dir, annotator_mode,
                )
                for spec in annotated_specs:
                    bname = spec["name"]
                    key = f"{bname}/annotated"
                    logger.info("Running %s (condition=annotated)", bname)
                    try:
                        results[key] = run_benchmark_annotated(
                            model=model,
                            benchmark_name=bname,
                            annotator=_annotator,
                            max_examples=max_docs,
                            device=device,
                        )
                    except Exception as _exc:
                        logger.error(
                            "Benchmark %s (condition=annotated) failed: %s: %s",
                            bname, type(_exc).__name__, _exc,
                        )
                        results[key] = {"error": str(_exc)}
                _corpus.close()
            except Exception as _exc:
                logger.error(
                    "Annotated eval setup failed: %s: %s", type(_exc).__name__, _exc
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
        elif isinstance(res, dict) and "perplexity_cross_doc_only" in res:
            # repobench_cross_doc / hotpotqa_cross_doc: cross-doc + paired flat + delta
            flat_nll  = res.get("average_nll_flat_linked_only", float("nan"))
            cross_nll = res.get("average_nll_cross_doc_only", float("nan"))
            delta = flat_nll - cross_nll  # positive = cross-doc helps
            logger.info(
                "  %-40s  ppl_cross=%.3f  ppl_flat=%.3f  Δnll=%+.4f  "
                "(n=%d)  ppl_all=%.3f (n=%d)  link_found=%d/%d",
                name,
                res.get("perplexity_cross_doc_only", float("nan")),
                res.get("perplexity_flat_linked_only", float("nan")),
                delta,
                res.get("n_cross_doc", 0),
                res.get("perplexity_with_fallback", float("nan")),
                res.get("total_examples", 0),
                res.get("n_link_found", 0),
                res.get("total_examples", 0),
            )
        elif isinstance(res, dict) and "mean_delta" in res:
            # community_pack_perplexity: flat dict with mean_delta
            logger.info(
                "  %-40s  delta=%.4f [%.4f, %.4f]  cross=%.4f  base=%.4f  n=%d",
                name,
                res.get("mean_delta", float("nan")),
                res.get("delta_ci_low", float("nan")),
                res.get("delta_ci_high", float("nan")),
                res.get("mean_nll_cross_doc", float("nan")),
                res.get("mean_nll_baseline", float("nan")),
                res.get("n_packs", 0),
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
        elif isinstance(res, dict) and "baseline_flat" in res and "t=0.0" in res:
            # run_benchmark_annotated: multi-threshold result
            for t_label in ("baseline_flat", "t=0.0", "t=p25", "t=p50", "t=p75"):
                stats = res.get(t_label)
                if not isinstance(stats, dict):
                    continue
                n_ann  = stats.get("n_annotated", "—")
                n_fire = stats.get("n_link_fired", "—")
                if "accuracy" in stats:
                    logger.info(
                        "  %-40s  acc=%.4f  n=%d  annotated=%s  link_fired=%s",
                        f"{name}[{t_label}]",
                        stats.get("accuracy", float("nan")),
                        stats.get("total_examples", 0),
                        n_ann, n_fire,
                    )
                else:
                    logger.info(
                        "  %-40s  ppl=%.3f  nll=%.4f  n=%d  annotated=%s  link_fired=%s",
                        f"{name}[{t_label}]",
                        stats.get("perplexity", float("nan")),
                        stats.get("average_nll", float("nan")),
                        stats.get("total_examples", 0),
                        n_ann, n_fire,
                    )
            tv = res.get("threshold_values", {})
            logger.info(
                "  %-40s  p25=%.4f  p50=%.4f  p75=%.4f",
                f"{name}[thresholds]",
                tv.get("p25", float("nan")),
                tv.get("p50", float("nan")),
                tv.get("p75", float("nan")),
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

    # repobench_cross_doc / hotpotqa_cross_doc → cross-doc ppl + paired NLL delta
    if "perplexity_cross_doc_only" in result:
        v = result.get("perplexity_cross_doc_only", nan)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "—"
        flat_nll  = result.get("average_nll_flat_linked_only", nan)
        cross_nll = result.get("average_nll_cross_doc_only", nan)
        if not (math.isnan(flat_nll) or math.isnan(cross_nll)):
            delta = flat_nll - cross_nll  # positive = cross-doc improves NLL
            return f"{v:.2f} (Δnll={delta:+.3f})"
        return f"{v:.2f}"

    # held_out_perplexity / fill-in-the-blank → show perplexity
    if "perplexity" in result:
        v = result.get("perplexity", nan)
        return "—" if (v is None or (isinstance(v, float) and math.isnan(v))) else f"{v:.2f}"

    # hellaswag / MC → show accuracy
    if "accuracy" in result:
        v = result.get("accuracy", nan)
        return "—" if (v is None or (isinstance(v, float) and math.isnan(v))) else f"{v:.4f}"

    # community_pack_perplexity → flat dict with mean_delta
    if "mean_delta" in result:
        v = result.get("mean_delta", nan)
        return "—" if (v is None or (isinstance(v, float) and math.isnan(v))) else f"{v:+.4f}"

    # pack_contrastive_perplexity → dict of strategy → stats; show first strategy's delta
    for _strategy, stats in result.items():
        if isinstance(stats, dict) and "mean_delta" in stats:
            v = stats.get("mean_delta", nan)
            return "—" if (v is None or (isinstance(v, float) and math.isnan(v))) else f"{v:+.4f}"

    # run_benchmark_annotated → show t=p50 accuracy or perplexity as headline
    if "baseline_flat" in result and "t=0.0" in result:
        p50 = result.get("t=p50", {})
        if isinstance(p50, dict):
            if "accuracy" in p50:
                v = p50.get("accuracy", nan)
                base = result.get("baseline_flat", {}).get("accuracy", nan)
                if not (math.isnan(v) or math.isnan(base)):
                    return f"{v:.4f} (Δ={v - base:+.4f} vs flat)"
                return "—" if (v is None or (isinstance(v, float) and math.isnan(v))) else f"{v:.4f}"
            elif "perplexity" in p50:
                v = p50.get("perplexity", nan)
                return "—" if (v is None or (isinstance(v, float) and math.isnan(v))) else f"{v:.2f}"

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
            "STEM/math: mmlu (all %d STEM subjects run automatically), mathqa, "
            "math (all %d subjects run automatically). "
            "Code: codexglue_line_completion, codexglue_code_to_text, "
            "repobench (see --repobench-split), humaneval_buggy (see --humaneval-language). "
            "Wikipedia QA: hotpotqa (flat), hotpotqa_cross_doc (cross-doc-link, bridge only). "
            "Graph: pack_contrastive_perplexity." % (len(MMLU_STEM_SUBJECTS), len(MATH_SUBJECTS))
        ),
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
        help='Graph split to evaluate. For held_out_perplexity: use "all", '
             '"val_random", "val_community", etc. For community_pack_perplexity: '
             'use "val_community" (default) or "test_community".',
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
    parser.add_argument(
        "--annotator-corpus", default=None, metavar="PATH",
        help="Path to pretokenized dataset directory used for corpus lookup by the "
             "link annotator (used when 'annotated' is in --conditions). "
             "Defaults to --dataset if not specified. Use the full unsplit dataset "
             "dir (not splits/train) so all articles are searchable.",
    )
    parser.add_argument(
        "--annotator-mode", default="full_skip",
        choices=["full_skip", "link_but_skip", "corpus_only", "generate_only",
                 "corpus_then_generate"],
        help="Link retrieval mode for the prompt annotator (used when 'annotated' "
             "is in --conditions). Shared with GenerationConfig; default full_skip "
             "(no link injected — the no-link baseline).",
    )
    parser.add_argument(
        "--annotator-strategies", nargs="+",
        default=["exact", "norm", "word_overlap_ordered", "edit_distance"],
        metavar="STRATEGY",
        help="Ordered list of title-matching strategies for HashNormTitleIndex "
             "(used when 'annotated' is in --conditions). "
             "Valid: exact, norm, word_overlap_ordered, word_overlap_unordered, "
             "edit_distance. Default includes edit_distance as final fallback.",
    )
    parser.add_argument(
        "--annotator-ed-threshold", type=float, default=0.2,
        metavar="THRESH",
        help="Normalized edit distance threshold for the edit_distance strategy "
             "[0, 1]. Lower = stricter. Default 0.2 (≥80%% similarity required).",
    )
    parser.add_argument(
        "--annotator-use-trie", action="store_true",
        help="Use TrieTitleIndex (trie-constrained generation) instead of "
             "HashNormTitleIndex. HashNormTitleIndex is still used as the fallback "
             "lookup when the trie returns None.",
    )
    parser.add_argument(
        "--annotator-trie-min-logprob", type=float, default=None,
        metavar="LOGPROB",
        help="Minimum joint log-prob for trie-constrained generation. If the "
             "running sum of log P(chosen token) drops below this value the trie "
             "aborts and falls back to free generation. None (default) = no threshold.",
    )
    parser.add_argument(
        "--annotator-beam-width", type=int, default=1,
        metavar="W",
        help="Beam width for TrieTitleIndex. 1 = greedy (default). Higher values "
             "keep more active paths and select the completed title with highest "
             "total log-prob, helping longer titles beat short high-first-token ones.",
    )
    parser.add_argument(
        "--annotator-length-penalty", type=float, default=0.0,
        metavar="ALPHA",
        help="Length penalty exponent for TrieTitleIndex candidate scoring "
             "(Wu et al. formula: score = joint_log_prob / n_tokens**alpha). "
             "0.0 = no normalization (default); 1.0 = per-token mean log-prob; "
             "0.6 = recommended middle ground.",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Global RNG seed (torch, numpy, cuda, Python random). "
             "Set this to make annotated title generation reproducible across runs.",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity. Use DEBUG to see per-example corpus hit/miss messages.",
    )
    args = parser.parse_args()

    logging.getLogger().setLevel(getattr(logging, args.log_level))

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        logger.info("Global seed set to %d", args.seed)

    checkpoints = [Path(c) for c in args.checkpoints]
    multi = len(checkpoints) > 1

    if multi and args.output:
        logger.warning(
            "--output is ignored when evaluating multiple checkpoints; "
            "results are written to each checkpoint's run dir."
        )

    # Per-benchmark extra params from CLI — folded into each spec dict.
    _bench_extras: Dict[str, Dict[str, Any]] = {
        "repobench":                {"split": args.repobench_split},
        "humaneval_buggy":          {"language": args.humaneval_language},
        "community_pack_perplexity": {"split": args.split},
    }

    eval_cfg = {
        "benchmarks": [
            {"name": b, "conditions": args.conditions, **_bench_extras.get(b, {})}
            for b in args.benchmarks
        ],
        "split": args.split,
        "max_docs": args.max_docs,
        "annotator_corpus": args.annotator_corpus or args.dataset,
        "annotator_mode": args.annotator_mode,
        "annotator_strategies": args.annotator_strategies,
        "annotator_ed_threshold": args.annotator_ed_threshold,
        "annotator_use_trie": args.annotator_use_trie,
        "annotator_trie_min_logprob": args.annotator_trie_min_logprob,
        "annotator_beam_width": args.annotator_beam_width,
        "annotator_length_penalty": args.annotator_length_penalty,
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
            # Merge into any existing results rather than clobbering them, so a
            # later run of a *different* benchmark/condition subset doesn't wipe
            # earlier keys. Same "{benchmark}/{condition}" keys are overwritten
            # with the fresh values (intended: re-running updates in place).
            merged = results
            if out_path.exists():
                try:
                    prior = json.loads(out_path.read_text(encoding="utf-8"))
                    if isinstance(prior, dict):
                        merged = {**prior, **results}
                except (json.JSONDecodeError, OSError) as _exc:
                    logger.warning(
                        "Could not read existing %s to merge (%s); overwriting.",
                        out_path, _exc,
                    )
            out_path.write_text(
                json.dumps(merged, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.info(
                "Results written to %s (%d total benchmark/condition keys)",
                out_path, len(merged),
            )

            if multi:
                all_entries.append(
                    _build_comparison_entry(ckpt_path, args.dataset, results)
                )

        if multi and all_entries:
            _write_comparison_table(eval_dir, all_entries)


if __name__ == "__main__":
    main()
