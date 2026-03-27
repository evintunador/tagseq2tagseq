"""
tests/eval/test_perplexity.py — unit tests for eval.perplexity.

Uses a synthetic dataset with split annotations. All tests run on CPU.
No CUDA required.
"""
import json
import math
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torch.nn as nn
from pathlib import Path

from data.layout import NullLayoutPolicy
from eval.perplexity import run_held_out_perplexity


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

    def _forward(tokens, doc_spans):
        T = tokens.shape[1]
        return torch.zeros(1, T, 256)

    model.forward_inference.side_effect = _forward
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
