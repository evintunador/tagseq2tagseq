"""Tests for data/merge_datasets.py.

Builds small real pretokenized datasets on disk (real shard headers + token
bytes), merges them, and verifies the merged artifact both structurally and via
a round-trip token read through GraphIndex/PretokShardedBackend — the true test
that shard remapping preserved every offset.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from tunalab.pretokenized_data.shard_io import BinaryShardIO

from data.dataset import GraphIndex, PretokShardedBackend
from data.merge_datasets import (
    _parse_inputs,
    _resolve_priority,
    merge_datasets,
    merge_nodes,
)


# ---------------------------------------------------------------------------
# Fixtures: write a real pretokenized dataset with correct headers + offsets
# ---------------------------------------------------------------------------

HEADER_BYTES = 256 * 4


def _write_dataset(
    dataset_dir: Path,
    docs: dict,
    outgoing: dict | None = None,
    tokenizer: str = "gpt2",
    dtype_str: str = "uint16",
) -> None:
    """Write a one-shard pretokenized dataset.

    docs: normed_id -> list[int] tokens. Nodes are laid out contiguously in a
    single shard in dict order; offsets/lengths computed so a round-trip read
    returns exactly the given tokens.
    outgoing: normed_id -> list[normed_id] (defaults to empty).
    """
    outgoing = outgoing or {}
    dtype = np.dtype(dtype_str)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Concatenate all token arrays into one payload; record per-doc offsets.
    payload = []
    node_meta = {}
    offset = HEADER_BYTES
    for nid, toks in docs.items():
        arr = np.asarray(toks, dtype=dtype)
        node_meta[nid] = {"tok_shard_idx": 0, "tok_offset_bytes": offset, "tok_len": len(arr)}
        payload.append(arr)
        offset += arr.nbytes

    all_toks = (
        np.concatenate(payload) if payload else np.array([], dtype=dtype)
    )
    BinaryShardIO.write_datafile(str(dataset_dir / "shard_000000.bin"), all_toks)

    with open(dataset_dir / "tokenized_graph.jsonl", "w", encoding="utf-8") as f:
        for nid, toks in docs.items():
            node = {
                "normed_identifier": nid,
                "raw_identifier": nid,
                "char_count": len(nid),
                "outgoing": sorted(outgoing.get(nid, [])),
                "incoming": [],
                **node_meta[nid],
            }
            f.write(json.dumps(node) + "\n")

    with open(dataset_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {"tokenizer": tokenizer, "dtype_str": dtype_str,
             "shard_filenames": ["shard_000000.bin"]},
            f,
        )


def _read_tokens(dataset_dir: Path, nid: str) -> list[int]:
    idx = GraphIndex(dataset_dir)
    backend = PretokShardedBackend(idx)
    return list(backend.get_tokens(nid))


# ---------------------------------------------------------------------------
# CLI arg parsing
# ---------------------------------------------------------------------------

def test_parse_inputs_ok():
    inputs = _parse_inputs(["a=/x/a", "b=/y/b"])
    assert [t for t, _ in inputs] == ["a", "b"]
    assert str(inputs[0][1]) == "/x/a"


def test_parse_inputs_rejects_bad_and_dupes():
    with pytest.raises(ValueError):
        _parse_inputs(["noequals"])
    with pytest.raises(ValueError):
        _parse_inputs(["a=/x", "a=/y"])
    with pytest.raises(ValueError):
        _parse_inputs([])


def test_resolve_priority():
    assert _resolve_priority(None, ["a", "b"]) == ["a", "b"]
    assert _resolve_priority("b,a", ["a", "b"]) == ["b", "a"]
    with pytest.raises(ValueError):
        _resolve_priority("a", ["a", "b"])       # missing b
    with pytest.raises(ValueError):
        _resolve_priority("a,b,c", ["a", "b"])   # extra c


# ---------------------------------------------------------------------------
# End-to-end merge
# ---------------------------------------------------------------------------

def test_merge_roundtrip_tokens_and_shard_remap(tmp_path):
    """Tokens read back correctly after shard renumbering across two inputs."""
    a = tmp_path / "a"
    b = tmp_path / "b"
    _write_dataset(a, {"a1": [1, 2, 3], "a2": [4, 5]})
    _write_dataset(b, {"b1": [10, 11, 12, 13], "b2": [20]})

    out = tmp_path / "merged"
    merge_datasets(
        inputs=[("dsa", a), ("dsb", b)],
        output_dir=out,
        priority=["dsa", "dsb"],
        shard_mode="hardlink",
    )

    meta = json.loads((out / "metadata.json").read_text())
    # Two inputs × one shard each → two globally-numbered shards.
    assert meta["shard_filenames"] == ["shard_000000.bin", "shard_000001.bin"]

    # dsb's nodes must now point at global shard 1, and read back correctly.
    assert _read_tokens(out, "a1") == [1, 2, 3]
    assert _read_tokens(out, "a2") == [4, 5]
    assert _read_tokens(out, "b1") == [10, 11, 12, 13]
    assert _read_tokens(out, "b2") == [20]

    idx = GraphIndex(out)
    assert idx.get_node("b1")["tok_shard_idx"] == 1
    assert idx.get_node("b1")["source"] == "dsb"
    assert idx.get_node("a1")["source"] == "dsa"


def test_collision_priority_and_provenance(tmp_path):
    """Higher-priority source wins a shared id; loser is dropped from the graph."""
    a = tmp_path / "a"
    b = tmp_path / "b"
    # "shared" exists in both, with different tokens.
    _write_dataset(a, {"shared": [1, 1, 1], "a_only": [2]})
    _write_dataset(b, {"shared": [9, 9], "b_only": [3]})

    out = tmp_path / "merged"
    meta = merge_datasets(
        inputs=[("hi", a), ("lo", b)],
        output_dir=out,
        priority=["hi", "lo"],
        shard_mode="copy",
    )

    idx = GraphIndex(out)
    # hi wins: shared is hi's, with hi's tokens.
    assert idx.get_node("shared")["source"] == "hi"
    assert _read_tokens(out, "shared") == [1, 1, 1]
    assert len(idx) == 3  # shared + a_only + b_only

    srcs = {s["tag"]: s for s in meta["merged_from"]["sources"]}
    assert srcs["lo"]["nodes_dropped_to_collision"] == 1
    assert srcs["hi"]["nodes_dropped_to_collision"] == 0
    assert srcs["lo"]["nodes_kept"] == 1


def test_reverse_priority_flips_winner(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    _write_dataset(a, {"shared": [1, 1, 1]})
    _write_dataset(b, {"shared": [9, 9]})

    out = tmp_path / "merged"
    merge_datasets(
        inputs=[("hi", a), ("lo", b)],
        output_dir=out,
        priority=["lo", "hi"],   # lo now wins
        shard_mode="hardlink",
    )
    idx = GraphIndex(out)
    assert idx.get_node("shared")["source"] == "lo"
    assert _read_tokens(out, "shared") == [9, 9]


def test_cross_source_edges_light_up(tmp_path):
    """An outgoing link that dangled within one source resolves after merge."""
    a = tmp_path / "a"
    b = tmp_path / "b"
    # a1 links to b1 (dangling within a); b1 links back to a1 (dangling within b).
    _write_dataset(a, {"a1": [1]}, outgoing={"a1": ["b1"]})
    _write_dataset(b, {"b1": [2]}, outgoing={"b1": ["a1"]})

    out = tmp_path / "merged"
    merge_datasets(
        inputs=[("dsa", a), ("dsb", b)],
        output_dir=out,
        priority=["dsa", "dsb"],
        shard_mode="hardlink",
    )
    idx = GraphIndex(out)
    # Reverse edges now exist across sources.
    assert idx.get_node("b1")["incoming"] == ["a1"]
    assert idx.get_node("a1")["incoming"] == ["b1"]
    # neighbors_out resolves across sources.
    a1_id = idx.get_id("a1")
    b1_id = idx.get_id("b1")
    assert b1_id in idx.neighbors_out(a1_id)


def test_incoming_recomputed_not_double_counted(tmp_path):
    """Pre-existing incoming lists are cleared and rebuilt, not appended to."""
    a = tmp_path / "a"
    _write_dataset(a, {"x": [1], "y": [2]}, outgoing={"x": ["y"]})
    # Manually inject a stale/duplicate incoming on y to prove it gets cleared.
    graph_path = a / "tokenized_graph.jsonl"
    lines = [json.loads(l) for l in graph_path.read_text().splitlines()]
    for n in lines:
        if n["normed_identifier"] == "y":
            n["incoming"] = ["x", "x", "bogus"]
    graph_path.write_text("\n".join(json.dumps(n) for n in lines))

    out = tmp_path / "merged"
    merge_datasets(inputs=[("dsa", a)], output_dir=out,
                   priority=["dsa"], shard_mode="copy")
    idx = GraphIndex(out)
    assert idx.get_node("y")["incoming"] == ["x"]  # deduped-by-recompute, no bogus


def test_tokenizer_mismatch_aborts(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    _write_dataset(a, {"a1": [1]}, tokenizer="gpt2")
    _write_dataset(b, {"b1": [2]}, tokenizer="cl100k_base")
    with pytest.raises(ValueError, match="tokenizer"):
        merge_datasets(inputs=[("dsa", a), ("dsb", b)],
                       output_dir=tmp_path / "m", priority=["dsa", "dsb"],
                       shard_mode="copy")


def test_dtype_mismatch_aborts(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    _write_dataset(a, {"a1": [1]}, dtype_str="uint16")
    _write_dataset(b, {"b1": [2]}, dtype_str="uint32")
    with pytest.raises(ValueError, match="dtype"):
        merge_datasets(inputs=[("dsa", a), ("dsb", b)],
                       output_dir=tmp_path / "m", priority=["dsa", "dsb"],
                       shard_mode="copy")


def test_hardlink_shares_inode(tmp_path):
    """hardlink mode must not copy bytes: merged shard shares the source inode."""
    a = tmp_path / "a"
    _write_dataset(a, {"a1": [1, 2, 3]})
    out = tmp_path / "merged"
    merge_datasets(inputs=[("dsa", a)], output_dir=out,
                   priority=["dsa"], shard_mode="hardlink")
    src_stat = (a / "shard_000000.bin").stat()
    dst_stat = (out / "shard_000000.bin").stat()
    assert src_stat.st_ino == dst_stat.st_ino
