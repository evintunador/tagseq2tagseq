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
from eval.scoring import score_doc, score_doc_with_context

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

        for i, doc_id in enumerate(doc_ids):
            if i > 0 and i % 50 == 0:
                logger.info(
                    "  %d / %d docs scored (%.1f%%)  running mean_nll=%.4f",
                    i, len(doc_ids), 100.0 * i / len(doc_ids),
                    float(np.mean(nll_list)) if nll_list else float("nan"),
                )

            tokens_arr = backend.get_tokens_by_id(doc_id)
            if tokens_arr is None or len(tokens_arr) < 2:
                skipped += 1
                continue

            normed_id = graph.get_normed_identifier(doc_id)
            raw_id = graph.get_raw_identifier(normed_id) or normed_id

            result = score_doc(
                model=model,
                tokens=tokens_arr.tolist(),
                layout_policy=layout_policy,
                raw_identifier=raw_id,
                normed_identifier=normed_id,
                device=device,
                mask_type=mask_type_override,
            )

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
