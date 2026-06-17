"""
Phase 0 — ArXiv citation-graph density validation (parallelized).

Streams all unarXive 2024 shards and measures the in-corpus citation graph density
using ONLY directly-resolvable arXiv-id citations (no OpenAlex enrichment yet). This
is the decision gate: if direct-id density already supports BFS packing, enrichment is
a bonus; if sparse (expected), Phase 1 OpenAlex enrichment is required.

A citation contributes an edge iff the cited arXiv id (from `bib_entries[*].contained_arXiv_ids`
or `bib_entries[*].ids.arxiv_id`) canonicalizes to a paper that exists in the corpus.

Parallelized across CPUs with multiprocessing.Pool (JSON parsing of 371 GB is the
bottleneck). The per-shard parse pattern here is reused by the Phase 2 extractor.

  Pass 1: workers return each shard's set of canonical paper ids → parent unions them
          into a global id→index map, written to a temp file.
  Pass 2: a Pool initializer loads that id map once per worker; workers return each
          shard's edge list (src_idx, [tgt_idx...]) → parent accumulates degree and
          a union-find for connected components.

Run via SLURM (CPU), output to /fss-data. See measure_density.sh.

Usage:
    python data/arxiv_graph_extractor/measure_density.py \
        --corpus-dir /fss-data/.../extracted/processed_unarxive_extended_data \
        --out /fss-data/.../graphs/arxiv_density_report.json \
        --workers 8
"""
import argparse
import glob
import json
import logging
import os
import pickle
import re
import time
from multiprocessing import Pool

logger = logging.getLogger(__name__)

# arXiv id with optional version suffix, e.g. "2401.12345", "2401.12345v2".
_VERSION_RE = re.compile(r"v\d+$")


def canonical_arxiv_id(raw: str) -> str:
    """Strip a trailing version suffix and whitespace so ids match across cite/paper forms."""
    return _VERSION_RE.sub("", raw.strip())


def iter_shards(corpus_dir: str):
    """Return shard file paths in sorted order."""
    return sorted(glob.glob(os.path.join(corpus_dir, "**", "*.jsonl"), recursive=True))


def _cited_ids(record: dict):
    """Yield canonical arXiv ids referenced by this paper's bibliography (direct-id only)."""
    for be in record.get("bib_entries", {}).values():
        for cax in be.get("contained_arXiv_ids") or []:
            cid = cax.get("id") if isinstance(cax, dict) else None
            if cid:
                yield canonical_arxiv_id(cid)
        ax = (be.get("ids") or {}).get("arxiv_id")
        if ax:
            yield canonical_arxiv_id(ax)


# ---------------------------------------------------------------------------
# Pass 1 worker: collect canonical paper ids present in a shard.
# ---------------------------------------------------------------------------
def _pass1_worker(shard: str) -> set:
    ids = set()
    with open(shard, "r", encoding="utf-8") as f:
        for line in f:
            try:
                ids.add(canonical_arxiv_id(json.loads(line)["paper_id"]))
            except Exception:
                continue
    return ids


# ---------------------------------------------------------------------------
# Pass 2 worker: build this shard's edges against the global id→idx map.
# The map is loaded once per worker process via the Pool initializer.
# ---------------------------------------------------------------------------
_ID_TO_IDX: dict = {}


def _pass2_init(id_map_path: str) -> None:
    global _ID_TO_IDX
    with open(id_map_path, "rb") as f:
        _ID_TO_IDX = pickle.load(f)


def _pass2_worker(shard: str):
    """Return (edges, n_cite_total, papers_with_cite) for one shard.

    edges: list of (src_idx, tgt_idx) for in-corpus, non-self citations (deduped per paper).
    """
    edges = []
    n_cite_total = 0
    papers_with_cite = 0
    idx = _ID_TO_IDX
    with open(shard, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                src = idx[canonical_arxiv_id(rec["paper_id"])]
            except Exception:
                continue
            targets = set()
            has_cite = False
            for cid in _cited_ids(rec):
                has_cite = True
                n_cite_total += 1
                tgt = idx.get(cid)
                if tgt is not None and tgt != src:
                    targets.add(tgt)
            if has_cite:
                papers_with_cite += 1
            for tgt in targets:
                edges.append((src, tgt))
    return edges, n_cite_total, papers_with_cite


class UnionFind:
    """Minimal union-find over integer node indices for connected-component sizing."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--limit-shards", type=int, default=0, help="0 = all (debug only)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    shards = iter_shards(args.corpus_dir)
    if args.limit_shards:
        shards = shards[: args.limit_shards]
    logger.info("Found %d shards; using %d workers", len(shards), args.workers)

    # ---- Pass 1: union per-shard id sets into a global id→idx map. ----
    t0 = time.time()
    id_to_idx: dict[str, int] = {}
    with Pool(args.workers) as pool:
        for i, shard_ids in enumerate(pool.imap_unordered(_pass1_worker, shards, chunksize=8)):
            for cid in shard_ids:
                if cid not in id_to_idx:
                    id_to_idx[cid] = len(id_to_idx)
            if (i + 1) % 2000 == 0:
                logger.info("pass1 %d/%d shards, %d nodes", i + 1, len(shards), len(id_to_idx))
    n_nodes = len(id_to_idx)
    logger.info("pass1 done: %d nodes in %.1fs", n_nodes, time.time() - t0)

    # Persist the id map so each pass-2 worker loads it once (not per task).
    id_map_path = args.out + ".idmap.pkl"
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(id_map_path, "wb") as f:
        pickle.dump(id_to_idx, f, protocol=pickle.HIGHEST_PROTOCOL)

    # ---- Pass 2: accumulate edges, degree, union-find across shards. ----
    t0 = time.time()
    uf = UnionFind(n_nodes)
    out_degree = [0] * n_nodes
    in_degree = [0] * n_nodes
    n_edges = 0
    n_cite_total = 0
    papers_with_cite = 0
    with Pool(args.workers, initializer=_pass2_init, initargs=(id_map_path,)) as pool:
        for i, (edges, n_cite, n_paper) in enumerate(
            pool.imap_unordered(_pass2_worker, shards, chunksize=8)
        ):
            n_cite_total += n_cite
            papers_with_cite += n_paper
            for src, tgt in edges:
                n_edges += 1
                out_degree[src] += 1
                in_degree[tgt] += 1
                uf.union(src, tgt)
            if (i + 1) % 2000 == 0:
                logger.info("pass2 %d/%d shards, %d edges", i + 1, len(shards), n_edges)
    logger.info("pass2 done: %d edges in %.1fs", n_edges, time.time() - t0)

    # ---- Component sizing + summary stats. ----
    comp_size: dict[int, int] = {}
    for i in range(n_nodes):
        r = uf.find(i)
        comp_size[r] = comp_size.get(r, 0) + 1
    sizes = sorted(comp_size.values(), reverse=True)
    largest_cc = sizes[0] if sizes else 0
    singletons = sum(1 for s in sizes if s == 1)
    nodes_with_out = sum(1 for d in out_degree if d > 0)
    nodes_with_any = sum(1 for i in range(n_nodes) if out_degree[i] or in_degree[i])

    report = {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "direct_cite_total": n_cite_total,
        "direct_cite_incorpus": n_edges,
        "incorpus_cite_fraction": round(n_edges / n_cite_total, 4) if n_cite_total else 0,
        "papers_with_any_direct_cite": papers_with_cite,
        "nodes_with_outgoing_edge": nodes_with_out,
        "pct_nodes_with_outgoing": round(100 * nodes_with_out / n_nodes, 2) if n_nodes else 0,
        "nodes_with_any_edge": nodes_with_any,
        "pct_nodes_with_any_edge": round(100 * nodes_with_any / n_nodes, 2) if n_nodes else 0,
        "mean_out_degree": round(n_edges / n_nodes, 3) if n_nodes else 0,
        "n_connected_components": len(sizes),
        "largest_cc_size": largest_cc,
        "largest_cc_fraction": round(largest_cc / n_nodes, 4) if n_nodes else 0,
        "singleton_components": singletons,
        "top10_cc_sizes": sizes[:10],
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    os.remove(id_map_path)  # cleanup temp map
    logger.info("wrote report to %s", args.out)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
