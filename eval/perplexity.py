"""
eval/perplexity.py — corpus perplexity benchmarks.

Provides two entry points:

  run_held_out_perplexity(model, dataset_dir, layout_policy,
                          split, max_docs, device)
      -> Dict[str, float]

      Measures mean NLL and perplexity on individual documents from the
      corpus. If tokenized_graph.jsonl contains 'split' annotations pass
      that split name; otherwise use split="all" to sample uniformly at
      random (seeded by seed=42).

  run_pack_contrastive_perplexity(model, epoch_dirs, dataset_dir,
                                   layout_policy, max_packs, device)
      -> Dict[str, Dict[str, float]]

      Scores body tokens of docs with incoming cross-doc edges within each
      pack under two conditions — cross_doc_link mask vs doc_causal mask —
      and reports the mean NLL delta per traversal strategy. epoch_dirs is
      a list of pre-computed epoch directories (one per strategy); the
      strategy name is read from each directory's metadata.json.

  run_community_pack_perplexity(model, dataset_dir, layout_policy,
                                 split, max_packs, device)
      -> Dict[str, float]

      Like run_pack_contrastive_perplexity but builds packs live from a
      held-out community split rather than pre-computed epoch directories.
      Uses PackBatchSampler(allowed_ids=community_ids) with BFS so the
      packs respect the community's internal link structure.
      Only meaningful for cross_doc_link models — caller should guard.
"""

import itertools
import json
import logging
import math
import random
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import numpy as np

from tunalab.stats_funcs import calculate_bootstrap_ci

from data.bucketed_pack_dataset import BucketedPackDataset
from data.dataset import GraphIndex, PretokShardedBackend
from data.layout import DocLayoutPolicy
from data.pack_sampler import PackBatchSampler
from data.packed_dataset import PackedSequenceDataset
from data.traversal import BFSStrategy
from eval.scoring import score_docs_batched, score_doc_with_context

logger = logging.getLogger(__name__)


@contextmanager
def _open_dataset(
    dataset_dir: Path,
) -> Generator[Tuple[GraphIndex, PretokShardedBackend], None, None]:
    """Open a pretokenized dataset and guarantee backend cleanup."""
    graph = GraphIndex(dataset_dir)
    backend = PretokShardedBackend(graph)
    try:
        yield graph, backend
    finally:
        backend.close()


def run_held_out_perplexity(
    model,
    dataset_dir: Union[str, Path],
    layout_policy: Optional[DocLayoutPolicy] = None,
    split: str = "all",
    max_docs: int = 500,
    device: str = "cuda",
    mask_type_override: Optional[str] = None,
) -> Dict[str, float]:
    """Evaluate model perplexity on held-out corpus documents.

    Documents are selected by the ``split`` annotation in
    ``tokenized_graph.jsonl``. The first ``max_docs`` matching documents are
    used (deterministic order — consistent across checkpoints for trend
    tracking).

    Args:
        model: TS2TSModel in eval mode.
        dataset_dir: Path to a pretokenized dataset directory containing
            ``tokenized_graph.jsonl``, ``metadata.json``, and shard files.
        layout_policy: Layout policy for prefix/suffix decoration. Defaults
            to model.active_layout_policy if not provided.
        split: Graph split name to evaluate on. Common values:
            ``"val_community"``, ``"val_random"``. Pass ``"all"`` to sample
            ``max_docs`` documents uniformly at random from the full graph
            (deterministic, seed=42). Use ``"all"`` for datasets that have
            no split annotations in tokenized_graph.jsonl.
        max_docs: Maximum number of documents to evaluate. Capped to the
            number of documents available in the split.
        device: Device for token tensors.
        mask_type_override: Passed to forward_inference. Has no practical
            effect here — each document is scored in isolation (single
            DocSpan, no other documents in context), so cross-doc grants
            can never fire regardless of mask type.

    Returns:
        Dictionary with keys:
            split, num_docs, mean_nll, perplexity,
            nll_ci_low, nll_ci_high, perplexity_ci_low, perplexity_ci_high
    """
    dataset_dir = Path(dataset_dir)

    with _open_dataset(dataset_dir) as (graph, backend):
        if split == "all":
            all_ids = list(range(len(graph)))
            rng = random.Random(42)
            rng.shuffle(all_ids)
            doc_ids = all_ids[:max_docs]
        else:
            doc_ids = graph.get_split_ids(split)
            if not doc_ids:
                logger.warning(
                    "No documents found for split=%r in %s. "
                    "Check that tokenized_graph.jsonl has 'split' annotations, "
                    "or use split='all' to sample from the full graph.",
                    split, dataset_dir,
                )
                return _empty_result(split)

        doc_ids = doc_ids[:max_docs]
        logger.info(
            "Evaluating held-out perplexity on %d documents (split=%r) from %s",
            len(doc_ids), split, dataset_dir,
        )

        nll_list = []
        skipped = 0

        # Gather scoreable docs, then batch-score them: score_docs_batched packs
        # multiple docs into one forward_inference (doc_causal isolation → per-doc
        # results identical to score_doc), eliminating the per-doc batch-1 passes
        # that dominate this benchmark. mask_type_override is intentionally NOT
        # forwarded — each doc is scored in isolation (the override has no effect
        # on an isolated doc, per this function's contract), and doc_causal is
        # required so packed docs never grant cross-doc attention to each other.
        docs_to_score = []   # (body_tokens, raw_id, normed_id)
        for doc_id in doc_ids:
            tokens_arr = backend.get_tokens_by_id(doc_id)
            if tokens_arr is None or len(tokens_arr) < 2:
                skipped += 1
                continue
            normed_id = graph.get_normed_identifier(doc_id)
            raw_id = graph.get_raw_identifier(normed_id) or normed_id
            docs_to_score.append((tokens_arr.tolist(), raw_id, normed_id))

        results = score_docs_batched(
            model=model,
            docs=docs_to_score,
            layout_policy=layout_policy,
            device=device,
        )

        for result in results:
            if result["num_tokens"] > 0:
                nll_list.append(result["mean_nll"])
            else:
                skipped += 1

        if skipped:
            logger.info("Skipped %d documents (empty body or too short).", skipped)

        if not nll_list:
            logger.warning("No documents could be scored in split=%r.", split)
            return _empty_result(split)

        mean_nll = float(np.mean(nll_list))
        try:
            perplexity = math.exp(mean_nll)
        except OverflowError:
            perplexity = float('inf')
        nll_ci = calculate_bootstrap_ci(nll_list)
        try:
            perplexity_ci = (math.exp(nll_ci[0]), math.exp(nll_ci[1]))
        except OverflowError:
            perplexity_ci = (float('inf'), float('inf'))

        logger.info(
            "Held-out perplexity (%s, n=%d): ppl=%.3f  mean_nll=%.4f  "
            "95%% CI nll=[%.4f, %.4f]",
            split, len(nll_list), perplexity, mean_nll, nll_ci[0], nll_ci[1],
        )

        return {
            "split": split,
            "num_docs": len(nll_list),
            "mean_nll": mean_nll,
            "perplexity": perplexity,
            "nll_ci_low": float(nll_ci[0]),
            "nll_ci_high": float(nll_ci[1]),
            "perplexity_ci_low": float(perplexity_ci[0]),
            "perplexity_ci_high": float(perplexity_ci[1]),
        }


def _empty_result(split: str) -> Dict[str, float]:
    return {
        "split": split,
        "num_docs": 0,
        "mean_nll": float("nan"),
        "perplexity": float("nan"),
        "nll_ci_low": float("nan"),
        "nll_ci_high": float("nan"),
        "perplexity_ci_low": float("nan"),
        "perplexity_ci_high": float("nan"),
    }


def run_pack_contrastive_perplexity(
    model,
    epoch_dirs: List[Union[str, Path]],
    dataset_dir: Union[str, Path],
    layout_policy: Optional[DocLayoutPolicy] = None,
    max_packs: int = 50,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Score pre-computed training packs under cross_doc_link vs doc_causal masks.

    For each epoch directory, reads the traversal strategy from metadata.json,
    iterates up to max_packs packs via BucketedPackDataset, and scores all body
    tokens in each pack under both mask conditions. Reports the mean NLL delta
    (baseline - cross_doc_link; positive means cross_doc_link helps) per strategy.

    Args:
        model: TS2TSModel in eval mode. Must have mask_type='cross_doc_link'.
        epoch_dirs: List of pre-computed epoch directories, each containing
            ``packs.parquet`` and ``metadata.json``. The traversal strategy
            is read from ``metadata.json["strategy"]``.
        dataset_dir: Path to the pretokenized dataset directory (GraphIndex +
            PretokShardedBackend), shared across all epoch dirs.
        layout_policy: Layout policy for prefix/suffix decoration. Defaults to
            model.active_layout_policy if not provided.
        max_packs: Maximum number of packs to score per epoch dir. Packs with
            no cross-doc edges (no target docs) are skipped and do not count
            toward this limit.
        device: Device string.

    Returns:
        Dict keyed by strategy name. Each value is a dict with keys:
            strategy, n_packs,
            mean_nll_cross_doc, mean_nll_baseline, mean_delta,
            delta_ci_low, delta_ci_high,
            cross_doc_ci_low, cross_doc_ci_high,
            baseline_ci_low, baseline_ci_high.
    """
    dataset_dir = Path(dataset_dir)

    with _open_dataset(dataset_dir) as (graph, backend):
        if layout_policy is None:
            layout_policy = model.active_layout_policy

        results: Dict[str, Any] = {}

        for epoch_dir in epoch_dirs:
            epoch_dir = Path(epoch_dir)

            meta_path = epoch_dir / "metadata.json"
            if not meta_path.exists():
                logger.warning("metadata.json not found in %s; skipping.", epoch_dir)
                continue
            with open(meta_path) as f:
                meta = json.load(f)
            strategy_name = meta.get("strategy", str(epoch_dir.name))

            logger.info(
                "Scoring packs for strategy=%r from %s (max_packs=%d)",
                strategy_name, epoch_dir, max_packs,
            )

            dataset = BucketedPackDataset(
                epoch_dirs=[str(epoch_dir)],
                graph=graph,
                backend=backend,
                layout=layout_policy,
                rank=0,
                world_size=1,
            )

            cross_nlls: List[float] = []
            base_nlls: List[float] = []
            skipped = 0

            for batch in itertools.islice(dataset, max_packs):
                result_cross = score_doc_with_context(
                    model, batch, layout_policy, device, mask_type=None,
                )
                result_base = score_doc_with_context(
                    model, batch, layout_policy, device, mask_type="doc_causal",
                )

                if result_cross["num_tokens"] == 0 or result_base["num_tokens"] == 0:
                    skipped += 1
                    continue

                cross_nlls.append(result_cross["mean_nll"])
                base_nlls.append(result_base["mean_nll"])

            if skipped:
                logger.info("  Skipped %d packs with no scoreable tokens.", skipped)

            n = len(cross_nlls)
            if n == 0:
                logger.warning("No packs could be scored for strategy=%r.", strategy_name)
                results[strategy_name] = _empty_contrastive_result(strategy_name)
                continue

            delta_list = [b - c for c, b in zip(cross_nlls, base_nlls)]
            mean_cross = float(np.mean(cross_nlls))
            mean_base = float(np.mean(base_nlls))
            mean_delta = float(np.mean(delta_list))

            delta_ci = calculate_bootstrap_ci(delta_list)
            cross_ci = calculate_bootstrap_ci(cross_nlls)
            base_ci = calculate_bootstrap_ci(base_nlls)

            logger.info(
                "strategy=%r  n=%d  delta=%.4f [%.4f, %.4f]  "
                "cross=%.4f  base=%.4f",
                strategy_name, n, mean_delta, delta_ci[0], delta_ci[1],
                mean_cross, mean_base,
            )

            results[strategy_name] = {
                "strategy": strategy_name,
                "n_packs": n,
                "mean_nll_cross_doc": mean_cross,
                "mean_nll_baseline": mean_base,
                "mean_delta": mean_delta,
                "delta_ci_low": float(delta_ci[0]),
                "delta_ci_high": float(delta_ci[1]),
                "cross_doc_ci_low": float(cross_ci[0]),
                "cross_doc_ci_high": float(cross_ci[1]),
                "baseline_ci_low": float(base_ci[0]),
                "baseline_ci_high": float(base_ci[1]),
            }

    return results


def _empty_contrastive_result(strategy: str) -> Dict[str, Any]:
    nan = float("nan")
    return {
        "strategy": strategy,
        "n_packs": 0,
        "mean_nll_cross_doc": nan,
        "mean_nll_baseline": nan,
        "mean_delta": nan,
        "delta_ci_low": nan,
        "delta_ci_high": nan,
        "cross_doc_ci_low": nan,
        "cross_doc_ci_high": nan,
        "baseline_ci_low": nan,
        "baseline_ci_high": nan,
    }


def run_community_pack_perplexity(
    model,
    dataset_dir: Union[str, Path],
    layout_policy: Optional[DocLayoutPolicy] = None,
    split: str = "val_community",
    max_packs: int = 500,
    device: str = "cuda",
    keep_frac: float = 1.0,
    keep_seed: int = 0,
    keep_mode: str = "edge",
) -> Dict[str, Any]:
    """Score cross-doc attention on live-packed held-out community nodes.

    Builds packs on-the-fly from the community split using BFS with
    ``PackBatchSampler(allowed_ids=community_ids)``, then scores each pack
    under cross_doc_link and doc_causal masks — identical reporting to
    ``run_pack_contrastive_perplexity`` but without pre-computed epoch dirs.

    Only meaningful for cross_doc_link models. The caller (eval_checkpoints)
    is responsible for skipping this benchmark on doc_causal models.

    Args:
        model: TS2TSModel in eval mode (must be cross_doc_link).
        dataset_dir: Path to pretokenized dataset directory.
        layout_policy: Layout policy for prefix/suffix decoration.
        split: Split name to use. Should be ``"val_community"`` or
            ``"test_community"`` — the point is that it's a BFS-identified
            subgraph whose internal link structure is intact.
        max_packs: Maximum number of packs to score.
        device: Device for token tensors.
        keep_frac: graph-sparsity ablation — fraction of resolved cross-doc
            grants to keep on the cross_doc arm (seeded, per-pack deterministic).
            1.0 = full density (default); 0.0 makes the cross arm ≡ doc_causal.
            The doc_causal baseline is unaffected. See
            eval.scoring.subsample_link_to_target and memory
            [[graph-sparsity-scaling-law]].
        keep_seed: global seed for the keep_frac subsample.
        keep_mode: 'edge' (per-edge density line) or 'node' (per-target-doc
            robustness check).

    Returns:
        Dict with the same keys as ``run_pack_contrastive_perplexity``:
        strategy, n_packs, mean_nll_cross_doc, mean_nll_baseline,
        mean_delta, delta_ci_low, delta_ci_high, cross_doc_ci_low,
        cross_doc_ci_high, baseline_ci_low, baseline_ci_high.
        Also includes ``split``, ``keep_frac``, ``keep_mode`` keys.
    """
    dataset_dir = Path(dataset_dir)
    split_dir = dataset_dir / "splits" / split

    if not split_dir.is_dir():
        logger.warning(
            "Split directory not found: %s — run data/split_graph.py first. "
            "Returning empty result.",
            split_dir,
        )
        result = _empty_contrastive_result(split)
        result["split"] = split
        return result

    with _open_dataset(split_dir) as (graph, backend):
        if layout_policy is None:
            layout_policy = model.active_layout_policy

        if len(graph) == 0:
            logger.warning("Split %r is empty; returning empty result.", split)
            result = _empty_contrastive_result(split)
            result["split"] = split
            return result

        logger.info(
            "Building community packs: split=%r, n_nodes=%d, max_packs=%d",
            split, len(graph), max_packs,
        )

        # Resolve the pack token budget from the backbone (TS2TSModel has no
        # top-level .max_seq_len, and the backbone exposes .max_seq_len directly,
        # NOT a HF-style .config.max_position_embeddings). The old chain fell all
        # the way through to the 2048 default, silently packing at 2048 tokens
        # regardless of the trained 32768 — which collapsed long-doc sources
        # (arxiv: median 14.7k tok/doc) to a handful of scoreable packs (n=5) and
        # under-packed every other source too. Mirror score_doc's resolution.
        token_budget = getattr(getattr(model, "backbone", None), "max_seq_len", None)
        if token_budget is None:
            token_budget = getattr(model, "max_seq_len", None)
        if token_budget is None:
            try:
                token_budget = model.backbone.config.max_position_embeddings
            except AttributeError:
                token_budget = 2048

        sampler = PackBatchSampler(
            graph=graph,
            strategy_factory=lambda: BFSStrategy(edge_mode="outgoing"),
            token_budget=token_budget,
            overflow_policy="truncate",
            doc_level_trim_side="tail",
            pack_level_trim_side="head",
            max_candidates_per_component=1000,
            seed=42,
            order_mode="prefer_targets_first",
            layout_policy=layout_policy,
        )
        dataset = PackedSequenceDataset(
            graph=graph,
            backend=backend,
            pack_sampler=sampler,
            layout_policy=layout_policy,
            as_2d=True,
        )

        cross_nlls: List[float] = []
        base_nlls: List[float] = []
        skipped = 0

        for batch in itertools.islice(dataset, max_packs):
            # Option B: resolve cross-doc grants from the pack's KNOWN graph edges,
            # not by re-detecting text (a merged model's single detector fires on
            # only one source → Δ=0 on all others). doc_causal baseline ignores it.
            result_cross = score_doc_with_context(
                model, batch, layout_policy, device, mask_type=None,
                grants_from_graph_edges=True,
                keep_frac=keep_frac, keep_seed=keep_seed, keep_mode=keep_mode,
            )
            result_base = score_doc_with_context(
                model, batch, layout_policy, device, mask_type="doc_causal",
            )

            if result_cross["num_tokens"] == 0 or result_base["num_tokens"] == 0:
                skipped += 1
                continue

            cross_nlls.append(result_cross["mean_nll"])
            base_nlls.append(result_base["mean_nll"])

        if skipped:
            logger.info("  Skipped %d packs with no scoreable tokens.", skipped)

        n = len(cross_nlls)
        if n == 0:
            logger.warning("No packs could be scored for split=%r.", split)
            result = _empty_contrastive_result(split)
            result["split"] = split
            return result

        delta_list = [b - c for c, b in zip(cross_nlls, base_nlls)]
        mean_cross = float(np.mean(cross_nlls))
        mean_base  = float(np.mean(base_nlls))
        mean_delta = float(np.mean(delta_list))

        delta_ci = calculate_bootstrap_ci(delta_list)
        cross_ci = calculate_bootstrap_ci(cross_nlls)
        base_ci  = calculate_bootstrap_ci(base_nlls)

        logger.info(
            "split=%r  keep=%.2f(%s)  n=%d  delta=%.4f [%.4f, %.4f]  cross=%.4f  base=%.4f",
            split, keep_frac, keep_mode, n, mean_delta, delta_ci[0], delta_ci[1],
            mean_cross, mean_base,
        )

        return {
            "split":               split,
            "keep_frac":           float(keep_frac),
            "keep_mode":           keep_mode,
            "n_packs":             n,
            "mean_nll_cross_doc":  mean_cross,
            "mean_nll_baseline":   mean_base,
            "mean_delta":          mean_delta,
            "delta_ci_low":        float(delta_ci[0]),
            "delta_ci_high":       float(delta_ci[1]),
            "cross_doc_ci_low":    float(cross_ci[0]),
            "cross_doc_ci_high":   float(cross_ci[1]),
            "baseline_ci_low":     float(base_ci[0]),
            "baseline_ci_high":    float(base_ci[1]),
        }
