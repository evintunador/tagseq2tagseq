"""
eval/perplexity.py — held-out corpus perplexity benchmark.

Measures mean NLL and perplexity of a TS2TS model on documents from the
corpus. If tokenized_graph.jsonl contains 'split' annotations (e.g.
"val_community", "val_random"), pass that split name to evaluate only on
held-out documents. If the dataset has no split annotations, pass
split="all" to sample max_docs documents uniformly at random from the
full graph (deterministic: seeded by seed=42).

Entry point:

  run_held_out_perplexity(model, dataset_dir, layout_policy,
                          split, max_docs, device)
      -> Dict[str, float]

Uses the same layout_policy the model was trained with, so the evaluation
distribution matches training as closely as possible.
"""

import logging
import math
import random
from pathlib import Path
from typing import Dict, Union

import numpy as np

from data.dataset import GraphIndex, PretokShardedBackend
from data.layout import DocLayoutPolicy
from eval.scoring import score_doc

logger = logging.getLogger(__name__)


def run_held_out_perplexity(
    model,
    dataset_dir: Union[str, Path],
    layout_policy: DocLayoutPolicy,
    split: str = "all",
    max_docs: int = 500,
    device: str = "cuda",
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
        layout_policy: The layout policy used during training. Applied when
            building each document's token sequence so the model sees the
            same prefix/suffix tokens it saw during training.
        split: Graph split name to evaluate on. Common values:
            ``"val_community"``, ``"val_random"``. Pass ``"all"`` to sample
            ``max_docs`` documents uniformly at random from the full graph
            (deterministic, seed=42). Use ``"all"`` for datasets that have
            no split annotations in tokenized_graph.jsonl.
        max_docs: Maximum number of documents to evaluate. Capped to the
            number of documents available in the split.
        device: Device for token tensors.

    Returns:
        Dictionary with keys:
            split, num_docs, mean_nll, perplexity,
            nll_ci_low, nll_ci_high, perplexity_ci_low, perplexity_ci_high
    """
    try:
        from tunalab.stats_funcs import calculate_bootstrap_ci
    except ImportError as e:
        raise ImportError(
            "tunalab must be installed for bootstrap CI computation. "
            "Install via: pip install -e /fss/evin_t/tunalab"
        ) from e

    dataset_dir = Path(dataset_dir)
    graph = GraphIndex(dataset_dir)
    backend = PretokShardedBackend(graph)

    try:
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
        perplexity = math.exp(mean_nll)
        nll_ci = calculate_bootstrap_ci(nll_list)
        perplexity_ci = (math.exp(nll_ci[0]), math.exp(nll_ci[1]))

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

    finally:
        backend.close()


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
