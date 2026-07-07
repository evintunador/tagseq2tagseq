"""
tests/data/test_make_repo_corpus.py — unit tests for the single-repo corpus
carver. Writes a tiny fake parent dataset (metadata.json + tokenized_graph.jsonl,
no real shards) to a tmp dir and checks the filtered output.
"""
import json
from pathlib import Path

import pytest

from data.make_repo_corpus import make_repo_corpus, _repo_of, _safe_repo_name


def _write_parent(tmp_path: Path):
    """Write a minimal multi-repo parent dataset; return its dir."""
    d = tmp_path / "thestack"
    d.mkdir()
    (d / "shard_000000.bin").write_bytes(b"\x00\x00")  # placeholder shard
    meta = {"tokenizer": "gpt2", "dtype_str": "uint16",
            "shard_filenames": ["shard_000000.bin"]}
    (d / "metadata.json").write_text(json.dumps(meta))
    nodes = [
        {"raw_identifier": "repoA/x:a/main.py", "normed_identifier": "a1",
         "outgoing": ["a2"], "incoming": [], "tok_shard_idx": 0,
         "tok_offset_bytes": 0, "tok_len": 1},
        {"raw_identifier": "repoA/x:a/util.py", "normed_identifier": "a2",
         "outgoing": [], "incoming": ["a1", "b1"], "tok_shard_idx": 0,
         "tok_offset_bytes": 2, "tok_len": 1},
        {"raw_identifier": "repoB/y:b/main.py", "normed_identifier": "b1",
         "outgoing": ["a2"], "incoming": [], "tok_shard_idx": 0,
         "tok_offset_bytes": 4, "tok_len": 1},
    ]
    with open(d / "tokenized_graph.jsonl", "w") as f:
        for n in nodes:
            f.write(json.dumps(n) + "\n")
    return d


def test_repo_of_and_safe_name():
    assert _repo_of("owner/repo:path/to/f.py") == "owner/repo"
    assert _repo_of("no_colon") == "no_colon"
    assert _safe_repo_name("owner/repo") == "owner_repo"


def test_filters_to_single_repo(tmp_path):
    parent = _write_parent(tmp_path)
    out = tmp_path / "out_repoA"
    n = make_repo_corpus(parent, "repoA/x", out)
    assert n == 2

    written = [json.loads(l) for l in (out / "tokenized_graph.jsonl").read_text().splitlines()]
    raws = {node["raw_identifier"] for node in written}
    assert raws == {"repoA/x:a/main.py", "repoA/x:a/util.py"}


def test_cross_repo_edges_filtered_out(tmp_path):
    """Edges pointing outside the repo are dropped; in-repo edges kept."""
    parent = _write_parent(tmp_path)
    out = tmp_path / "out_repoB"
    make_repo_corpus(parent, "repoB/y", out)
    written = {n["normed_identifier"]: n for n in
               (json.loads(l) for l in (out / "tokenized_graph.jsonl").read_text().splitlines())}
    # b1's outgoing edge to a2 (repoA) must be dropped — a2 is not in this corpus.
    assert written["b1"]["outgoing"] == []


def test_metadata_shard_paths_absolute(tmp_path):
    parent = _write_parent(tmp_path)
    out = tmp_path / "out"
    make_repo_corpus(parent, "repoA/x", out)
    meta = json.loads((out / "metadata.json").read_text())
    assert len(meta["shard_filenames"]) == 1
    shard = meta["shard_filenames"][0]
    assert Path(shard).is_absolute()
    assert Path(shard).name == "shard_000000.bin"
    # tokenizer/dtype carried over from parent.
    assert meta["tokenizer"] == "gpt2" and meta["dtype_str"] == "uint16"


def test_unknown_repo_raises(tmp_path):
    parent = _write_parent(tmp_path)
    with pytest.raises(ValueError, match="No nodes found for repo"):
        make_repo_corpus(parent, "does/not-exist", tmp_path / "out")
