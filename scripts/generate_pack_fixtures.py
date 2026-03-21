"""
Generate real-data pack fixtures for kernel tests and benchmarks.

Samples packs from simplewiki using BFS, ranks by cross-doc grant density,
and saves three representative fixtures (sparse / medium / dense) as .pt files
in tests/fixtures/real_packs/.

Each fixture is a dict:
    tokens          [T] int64 token sequence (1-D, input tokens only)
    doc_spans       list of dicts: {doc_id, start, end, raw_identifier,
                                    normed_identifier, outgoing_identifiers, truncated}
    link_to_target  {link_pos: [target_doc_id, ...]} — pre-resolved grants
    n_grants        int  (total resolved grants in this pack)
    kv_block_count  int  (analytical block count including cross-doc grants)
    density_label   str  ("sparse" | "medium" | "dense")
    dataset         str  ("simplewiki")
    seq_len         int  (len(tokens))

Usage (run from repo root):
    python scripts/generate_pack_fixtures.py
    python scripts/generate_pack_fixtures.py --n-samples 1000 --token-budget 2048
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

# Ensure kernels-repo root is on sys.path so data.* / model.* are importable.
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

DATASET_DIR = "/fss/evin_t/tagseq2tagseq/data/pretokenized_datasets/simplewiki"
OUT_DIR = REPO_ROOT / "tests" / "fixtures" / "real_packs"


def _collect_packs(
    n_samples: int,
    token_budget: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """Sample n_samples packs from simplewiki and return metadata for each."""
    import tiktoken

    from data.collate import build_packed_batch
    from data.dataset import GraphIndex, PretokShardedBackend
    from data.layout import NullLayoutPolicy
    from data.pack_sampler import PackBatchSampler
    from data.traversal import BFSStrategy
    from model.graph_traversal.cross_doc_mask import (
        CrossDocLinkMaskCreator,
        _kv_block_count_analytical,
    )
    from model.graph_traversal.markdown_link_detector import MarkdownLinkDetector

    graph = GraphIndex(DATASET_DIR)
    backend = PretokShardedBackend(graph)
    enc = tiktoken.get_encoding("gpt2")
    detector = MarkdownLinkDetector(decode_fn=enc.decode)
    creator = CrossDocLinkMaskCreator(link_detector=detector)
    layout = NullLayoutPolicy()

    sampler = PackBatchSampler(
        graph=graph,
        strategy_factory=lambda: BFSStrategy(edge_mode="outgoing"),
        token_budget=token_budget + 1,
        overflow_policy="truncate",
        seed=seed,
        order_mode="prefer_targets_first",
    )

    records = []
    for i, placements in enumerate(sampler):
        if i >= n_samples:
            break
        if i % 100 == 0:
            print(f"  sampled {i}/{n_samples}...", flush=True)

        batch = build_packed_batch(graph, backend, layout, placements, as_2d=True)
        raw_tokens = batch["tokens"]  # [1, T]
        doc_spans_raw = batch["doc_spans"]

        # Input tokens (drop last — next-token target)
        tokens = raw_tokens[0, :-1] if raw_tokens.shape[1] > 1 else raw_tokens[0]
        actual_T = int(tokens.shape[0])

        links = detector.detect_links(tokens)
        link_to_target = creator._match_links_to_docs(links, doc_spans_raw)
        n_grants = sum(len(v) for v in link_to_target.values())

        kv_bc = _kv_block_count_analytical(doc_spans_raw, link_to_target, actual_T)

        doc_spans_dicts = [
            {
                "doc_id":              s.doc_id,
                "start":               s.start,
                "end":                 s.end,
                "raw_identifier":      s.raw_identifier,
                "normed_identifier":   s.normed_identifier,
                "outgoing_identifiers": list(s.outgoing_identifiers),
                "truncated":           s.truncated,
            }
            for s in doc_spans_raw
        ]

        # Serialise link_to_target with int keys (JSON / torch.save compat)
        ltt_serial = {int(k): [int(v) for v in vs] for k, vs in link_to_target.items()}

        records.append({
            "tokens":          tokens.clone(),
            "doc_spans":       doc_spans_dicts,
            "link_to_target":  ltt_serial,
            "n_grants":        n_grants,
            "kv_block_count":  kv_bc,
            "seq_len":         actual_T,
        })

    backend.close()
    print(f"  collected {len(records)} packs total.")
    return records


def _pick_three(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Pick sparse / medium / dense packs by n_grants."""
    # Sort by n_grants ascending
    ranked = sorted(records, key=lambda r: r["n_grants"])

    n = len(ranked)
    assert n >= 3, f"Need at least 3 packs; got {n}"

    # Sparse: first record with n_grants == 0 (pure doc_causal, no cross-doc)
    sparse_candidates = [r for r in ranked if r["n_grants"] == 0]
    if not sparse_candidates:
        sparse_candidates = ranked[:1]
        print("  WARNING: no zero-grant pack found; using lowest-grant pack as sparse.")
    sparse = sparse_candidates[0]

    # Dense: highest n_grants
    dense = ranked[-1]

    # Medium: record closest to median n_grants
    med_grants = ranked[n // 2]["n_grants"]
    # Among records with > 0 grants if possible
    medium_pool = [r for r in ranked if r["n_grants"] > 0]
    if not medium_pool:
        medium_pool = ranked
    medium = min(medium_pool, key=lambda r: abs(r["n_grants"] - med_grants))

    print(f"\n  Selected fixtures:")
    print(f"    sparse  — n_grants={sparse['n_grants']:3d}  "
          f"kv_blocks={sparse['kv_block_count']:4d}  seq_len={sparse['seq_len']}")
    print(f"    medium  — n_grants={medium['n_grants']:3d}  "
          f"kv_blocks={medium['kv_block_count']:4d}  seq_len={medium['seq_len']}")
    print(f"    dense   — n_grants={dense['n_grants']:3d}  "
          f"kv_blocks={dense['kv_block_count']:4d}  seq_len={dense['seq_len']}")

    return {"sparse": sparse, "medium": medium, "dense": dense}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=500,
                        help="Number of packs to sample (default: 500)")
    parser.add_argument("--token-budget", type=int, default=1024,
                        help="Token budget per pack (default: 1024)")
    parser.add_argument("--seed", type=int, default=7,
                        help="RNG seed for pack sampler (default: 7)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Sampling {args.n_samples} packs from simplewiki "
          f"(budget={args.token_budget}, seed={args.seed})...")
    records = _collect_packs(args.n_samples, args.token_budget, args.seed)

    picks = _pick_three(records)

    for label, rec in picks.items():
        rec["density_label"] = label
        rec["dataset"] = "simplewiki"
        rec["token_budget"] = args.token_budget
        rec["generation_seed"] = args.seed

        out_path = OUT_DIR / f"{label}.pt"
        torch.save(rec, str(out_path))
        print(f"  Saved → {out_path.relative_to(REPO_ROOT)}")

    # Print a summary for the README
    print(f"\n{'='*60}")
    print("Fixture summary (update tests/fixtures/real_packs/README.md):")
    for label, rec in picks.items():
        print(f"  {label}.pt: n_grants={rec['n_grants']}, "
              f"kv_block_count={rec['kv_block_count']}, "
              f"seq_len={rec['seq_len']}, "
              f"n_docs={len(rec['doc_spans'])}")
    dense_grants = picks["dense"]["n_grants"]
    # Suggest max_grants for truncation testing: half of dense grants, min 4
    suggested_mg = max(4, dense_grants // 2)
    print(f"\n  Suggested max_grants for dense-truncation tests: {suggested_mg}")
    print(f"  (dense has {dense_grants} grants; {suggested_mg} will truncate "
          f"{dense_grants - suggested_mg} of them)")
    print('='*60)


if __name__ == "__main__":
    main()
