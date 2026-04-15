"""
tests/eval/test_perplexity.py — unit tests for eval.perplexity.

Uses a synthetic dataset with split annotations. All tests run on CPU.
No CUDA required.
"""
import json
import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

from data.collate import DocSpan
from data.layout import NullLayoutPolicy
from eval.perplexity import run_held_out_perplexity, run_pack_contrastive_perplexity


# ─── Dataset fixture ──────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def dummy_split_dataset(tmp_path_factory):
    """Tiny dataset with 3 nodes and explicit split annotations.

    Node A: split=train          (tokens: [0, 1, 2, 3, 4])
    Node B: split=val_community  (tokens: [10, 11, 12, 13])
    Node C: split=val_community  (tokens: [20, 21, 22, 23, 24, 25])
    """
    run_dir = tmp_path_factory.mktemp("eval_perplexity_dataset")

    tokens_a = np.array([0, 1, 2, 3, 4],          dtype=np.uint16)
    tokens_b = np.array([10, 11, 12, 13],          dtype=np.uint16)
    tokens_c = np.array([20, 21, 22, 23, 24, 25],  dtype=np.uint16)
    all_tokens = np.concatenate([tokens_a, tokens_b, tokens_c])

    metadata = {
        "tokenizer": "dummy",
        "dtype_str": "uint16",
        "shard_filenames": ["shard_000000.bin"],
    }
    with open(run_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)

    header = np.zeros(256, dtype=np.int32)
    header[0] = 11041999  # magic number (matches BinaryShardIO)
    header[1] = 1         # version
    header[2] = len(all_tokens)
    header[3] = np.dtype("uint16").itemsize

    shard_path = run_dir / "shard_000000.bin"
    with open(shard_path, "wb") as f:
        f.write(header.tobytes())
        f.write(all_tokens.tobytes())

    header_bytes = 256 * 4
    item_bytes = 2  # uint16

    graph_data = [
        {
            "normed_identifier": "node_a", "raw_identifier": "Node A",
            "outgoing": [], "incoming": [],
            "tok_shard_idx": 0,
            "tok_offset_bytes": header_bytes + 0 * item_bytes,
            "tok_len": len(tokens_a),
            "split": "train",
        },
        {
            "normed_identifier": "node_b", "raw_identifier": "Node B",
            "outgoing": [], "incoming": [],
            "tok_shard_idx": 0,
            "tok_offset_bytes": header_bytes + len(tokens_a) * item_bytes,
            "tok_len": len(tokens_b),
            "split": "val_community",
        },
        {
            "normed_identifier": "node_c", "raw_identifier": "Node C",
            "outgoing": [], "incoming": [],
            "tok_shard_idx": 0,
            "tok_offset_bytes": header_bytes + (len(tokens_a) + len(tokens_b)) * item_bytes,
            "tok_len": len(tokens_c),
            "split": "val_community",
        },
    ]
    with open(run_dir / "tokenized_graph.jsonl", "w") as f:
        for entry in graph_data:
            f.write(json.dumps(entry) + "\n")

    return run_dir


@pytest.fixture
def mock_model():
    """Mock TS2TSModel with uniform logits (NLL = log(256))."""
    model = MagicMock()

    def _forward(tokens, doc_spans, **kwargs):
        T = tokens.shape[1]
        return torch.zeros(1, T, 256)

    model.forward_inference.side_effect = _forward
    model.active_layout_policy = MagicMock()
    model.active_layout_policy.prefix_tokens.return_value = []
    model.active_layout_policy.suffix_tokens.return_value = []
    dummy_param = nn.Parameter(torch.zeros(1))
    model.backbone.parameters.return_value = iter([dummy_param])
    return model


@pytest.fixture
def null_policy():
    return NullLayoutPolicy()


# ─── Tests ────────────────────────────────────────────────────────────────────

def test_returns_all_expected_keys(mock_model, dummy_split_dataset, null_policy):
    result = run_held_out_perplexity(
        mock_model, dummy_split_dataset, null_policy,
        split="val_community", max_docs=10, device="cpu",
    )
    expected_keys = {
        "split", "num_docs",
        "mean_nll", "perplexity",
        "nll_ci_low", "nll_ci_high",
        "perplexity_ci_low", "perplexity_ci_high",
    }
    assert set(result.keys()) == expected_keys


def test_filters_by_split(mock_model, dummy_split_dataset, null_policy):
    result = run_held_out_perplexity(
        mock_model, dummy_split_dataset, null_policy,
        split="val_community", max_docs=100, device="cpu",
    )
    # Only nodes B and C have split=val_community
    assert result["num_docs"] == 2


def test_train_split_returns_one_doc(mock_model, dummy_split_dataset, null_policy):
    result = run_held_out_perplexity(
        mock_model, dummy_split_dataset, null_policy,
        split="train", max_docs=100, device="cpu",
    )
    assert result["num_docs"] == 1


def test_respects_max_docs(mock_model, dummy_split_dataset, null_policy):
    result = run_held_out_perplexity(
        mock_model, dummy_split_dataset, null_policy,
        split="val_community", max_docs=1, device="cpu",
    )
    assert result["num_docs"] == 1


def test_perplexity_equals_exp_of_mean_nll(mock_model, dummy_split_dataset, null_policy):
    result = run_held_out_perplexity(
        mock_model, dummy_split_dataset, null_policy,
        split="val_community", max_docs=10, device="cpu",
    )
    assert math.isfinite(result["perplexity"])
    assert math.isclose(result["perplexity"], math.exp(result["mean_nll"]), rel_tol=1e-5)


def test_ci_straddles_mean(mock_model, dummy_split_dataset, null_policy):
    result = run_held_out_perplexity(
        mock_model, dummy_split_dataset, null_policy,
        split="val_community", max_docs=10, device="cpu",
    )
    assert result["nll_ci_low"] <= result["mean_nll"] <= result["nll_ci_high"]


def test_split_field_in_result(mock_model, dummy_split_dataset, null_policy):
    result = run_held_out_perplexity(
        mock_model, dummy_split_dataset, null_policy,
        split="val_community", max_docs=10, device="cpu",
    )
    assert result["split"] == "val_community"


def test_uniform_logits_nll_approx_log_V(mock_model, dummy_split_dataset, null_policy):
    result = run_held_out_perplexity(
        mock_model, dummy_split_dataset, null_policy,
        split="val_community", max_docs=10, device="cpu",
    )
    # With zero logits, NLL ≈ log(256) ≈ 5.545
    assert math.isfinite(result["mean_nll"])
    assert abs(result["mean_nll"] - math.log(256)) < 0.5


def test_nonexistent_split_returns_empty(mock_model, dummy_split_dataset, null_policy):
    result = run_held_out_perplexity(
        mock_model, dummy_split_dataset, null_policy,
        split="val_random", max_docs=10, device="cpu",
    )
    assert result["num_docs"] == 0
    assert math.isnan(result["perplexity"])


# ─── run_pack_contrastive_perplexity tests ────────────────────────────────────

VOCAB_SIZE = 256


def _make_contrastive_mock_model():
    """Mock TS2TSModel for contrastive tests. Returns uniform logits."""
    model = MagicMock()

    def _forward(tokens, doc_spans, **kwargs):
        T = tokens.shape[1]
        return torch.zeros(1, T, VOCAB_SIZE)

    model.forward_inference.side_effect = _forward
    model.active_layout_policy = NullLayoutPolicy()
    model.mask_type = "cross_doc_link"
    return model


def _make_contrastive_batch(T: int = 20, n_docs: int = 2):
    """Synthetic batch dict as yielded by BucketedPackDataset.

    doc_0 links to doc_1 so that score_doc_with_context has a target to score.
    All subsequent docs (if n_docs > 2) have no outgoing links and are
    context-only.
    """
    tokens = torch.zeros(1, T, dtype=torch.long)
    doc_len = T // n_docs
    spans = []
    for i in range(n_docs):
        # doc_0 links to doc_1 — provides a cross-doc edge for scoring.
        outgoing = [f"doc_{i + 1}"] if i == 0 and n_docs > 1 else []
        spans.append(DocSpan(
            doc_id=i,
            normed_identifier=f"doc_{i}",
            raw_identifier=f"Doc {i}",
            start=i * doc_len,
            end=(i + 1) * doc_len,
            truncated=False,
            outgoing_identifiers=outgoing,
        ))
    return {"tokens": tokens, "doc_spans": spans, "link_to_target": {}}


@pytest.fixture
def epoch_dir_bfs(tmp_path):
    """Fake epoch dir with metadata.json strategy=bfs and 5 synthetic packs."""
    epoch_dir = tmp_path / "epoch_bfs"
    epoch_dir.mkdir()
    meta = {"strategy": "bfs", "n_packs": 5, "token_budget": 20}
    with open(epoch_dir / "metadata.json", "w") as f:
        json.dump(meta, f)
    return epoch_dir


def _mock_bucketed_dataset(batches):
    """Return a MagicMock that iterates over the given batches."""
    mock_ds = MagicMock()
    mock_ds.__iter__ = MagicMock(return_value=iter(batches))
    return mock_ds


def test_pack_contrastive_returns_strategy_key(tmp_path, epoch_dir_bfs, dummy_split_dataset):
    model = _make_contrastive_mock_model()
    batches = [_make_contrastive_batch() for _ in range(3)]

    with patch("eval.perplexity.BucketedPackDataset", return_value=_mock_bucketed_dataset(batches)), \
         patch("eval.perplexity.GraphIndex"), \
         patch("eval.perplexity.PretokShardedBackend") as mock_backend_cls:
        mock_backend_cls.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_backend_cls.return_value.close = MagicMock()
        result = run_pack_contrastive_perplexity(
            model=model,
            epoch_dirs=[epoch_dir_bfs],
            dataset_dir=dummy_split_dataset,
            layout_policy=NullLayoutPolicy(),
            max_packs=10,
            device="cpu",
        )

    assert "bfs" in result


def test_pack_contrastive_result_structure(tmp_path, epoch_dir_bfs, dummy_split_dataset):
    model = _make_contrastive_mock_model()
    batches = [_make_contrastive_batch() for _ in range(3)]

    with patch("eval.perplexity.BucketedPackDataset", return_value=_mock_bucketed_dataset(batches)), \
         patch("eval.perplexity.GraphIndex"), \
         patch("eval.perplexity.PretokShardedBackend") as mock_backend_cls:
        mock_backend_cls.return_value.close = MagicMock()
        result = run_pack_contrastive_perplexity(
            model=model,
            epoch_dirs=[epoch_dir_bfs],
            dataset_dir=dummy_split_dataset,
            layout_policy=NullLayoutPolicy(),
            max_packs=10,
            device="cpu",
        )

    expected_keys = {
        "strategy", "n_packs",
        "mean_nll_cross_doc", "mean_nll_baseline", "mean_delta",
        "delta_ci_low", "delta_ci_high",
        "cross_doc_ci_low", "cross_doc_ci_high",
        "baseline_ci_low", "baseline_ci_high",
    }
    assert set(result["bfs"].keys()) == expected_keys


def test_pack_contrastive_n_packs(tmp_path, epoch_dir_bfs, dummy_split_dataset):
    model = _make_contrastive_mock_model()
    batches = [_make_contrastive_batch() for _ in range(5)]

    with patch("eval.perplexity.BucketedPackDataset", return_value=_mock_bucketed_dataset(batches)), \
         patch("eval.perplexity.GraphIndex"), \
         patch("eval.perplexity.PretokShardedBackend") as mock_backend_cls:
        mock_backend_cls.return_value.close = MagicMock()
        result = run_pack_contrastive_perplexity(
            model=model,
            epoch_dirs=[epoch_dir_bfs],
            dataset_dir=dummy_split_dataset,
            layout_policy=NullLayoutPolicy(),
            max_packs=3,
            device="cpu",
        )

    assert result["bfs"]["n_packs"] == 3


def test_pack_contrastive_respects_max_packs(tmp_path, epoch_dir_bfs, dummy_split_dataset):
    """max_packs=3 with 5 available batches → only 3 scored."""
    model = _make_contrastive_mock_model()
    batches = [_make_contrastive_batch() for _ in range(5)]

    with patch("eval.perplexity.BucketedPackDataset", return_value=_mock_bucketed_dataset(batches)), \
         patch("eval.perplexity.GraphIndex"), \
         patch("eval.perplexity.PretokShardedBackend") as mock_backend_cls:
        mock_backend_cls.return_value.close = MagicMock()
        result = run_pack_contrastive_perplexity(
            model=model,
            epoch_dirs=[epoch_dir_bfs],
            dataset_dir=dummy_split_dataset,
            layout_policy=NullLayoutPolicy(),
            max_packs=3,
            device="cpu",
        )

    # 3 packs × 2 calls (cross + base) = 6 total forward_inference calls
    assert model.forward_inference.call_count == 6


def test_pack_contrastive_delta_equals_base_minus_cross(tmp_path, epoch_dir_bfs, dummy_split_dataset):
    """With uniform logits both conditions return the same NLL, so delta == 0."""
    model = _make_contrastive_mock_model()
    batches = [_make_contrastive_batch() for _ in range(3)]

    with patch("eval.perplexity.BucketedPackDataset", return_value=_mock_bucketed_dataset(batches)), \
         patch("eval.perplexity.GraphIndex"), \
         patch("eval.perplexity.PretokShardedBackend") as mock_backend_cls:
        mock_backend_cls.return_value.close = MagicMock()
        result = run_pack_contrastive_perplexity(
            model=model,
            epoch_dirs=[epoch_dir_bfs],
            dataset_dir=dummy_split_dataset,
            layout_policy=NullLayoutPolicy(),
            max_packs=10,
            device="cpu",
        )

    stats = result["bfs"]
    assert abs(stats["mean_delta"]) < 1e-6
    assert math.isclose(
        stats["mean_delta"],
        stats["mean_nll_baseline"] - stats["mean_nll_cross_doc"],
        abs_tol=1e-6,
    )


def test_pack_contrastive_empty_epoch_dirs(dummy_split_dataset):
    """Empty epoch_dirs list returns empty dict."""
    model = _make_contrastive_mock_model()
    with patch("eval.perplexity.GraphIndex"), \
         patch("eval.perplexity.PretokShardedBackend") as mock_backend_cls:
        mock_backend_cls.return_value.close = MagicMock()
        result = run_pack_contrastive_perplexity(
            model=model,
            epoch_dirs=[],
            dataset_dir=dummy_split_dataset,
            layout_policy=NullLayoutPolicy(),
            device="cpu",
        )
    assert result == {}
