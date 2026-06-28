r"""
Phase 2 — ArXiv (unarXive 2024) graph + content extraction.

Produces the two artifacts the pretokenizer consumes, mirroring the wiki/stack
extractors but for arXiv papers (nodes) linked by citations (edges):

  graph.jsonl   — one node per paper:
      {normed_identifier, raw_identifier (title), categories, char_count,
       outgoing[], incoming[]}
  content.jsonl — one body per paper:
      {normed_identifier, content (rehydrated LaTeX)}

Link-target scheme (see memory `arxiv-dataset`): the model emits and we match on
the paper TITLE (raw_identifier); normed_identifier = normalize_arxiv(paper_id) is
the internal, globally-unique key. In-text citations are rewritten to LaTeX-native
``\\cite{<cited title>}`` so the existing ArxivCiteDetector recovers them.

Citation resolution is dual-path, direct first then OpenAlex-enriched:
  1. bib_entries[*].contained_arXiv_ids / .ids.arxiv_id  -> arXiv id  (direct, ~14.5%)
  2. bib_entries[*].ids.open_alex_id -> arxiv_openalex_map -> arXiv id  (enriched, +~52%)
An edge (and a ``\\cite{Title}``) is created only when the resolved arXiv id is
in-corpus; out-of-corpus citations are dropped to a bare textual marker.

Body rehydration keeps everything as faithful LaTeX:
  {{formula:uuid}} -> $<ref_entries[uuid].latex>$
  {{figure|table:uuid}} -> \ref-like marker carrying the caption (when present)
  (REF ) cross-refs -> \ref{}
  {{cite:hexkey}} -> \cite{<title>} (in-corpus) or dropped

Two passes (mirrors measure_density.py's Pool-over-shards):
  Pass 1 — per shard, collect {canonical_arxiv_id -> (normed_id, title, categories)}
           and {openalex_id -> canonical_arxiv_id} restricted to corpus papers.
           Parent unions into the global corpus index; this is what makes a
           cross-shard citation resolvable to its target's title.
  Pass 2 — per shard, rehydrate bodies + resolve citations against the global
           index; workers write their own graph/content shard files; parent
           concatenates and computes incoming edges.

Run via SLURM (CPU), output to /fss-data. See extract.sh.
"""
import argparse
import glob
import json
import logging
import os
import re
import time
from multiprocessing import Pool

from tunalab.reproducibility import ReproducibilityManager

from data.normalization import normalize_arxiv, canonical_arxiv_id

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Placeholder / cross-ref rehydration patterns
# ---------------------------------------------------------------------------
_FORMULA_RE = re.compile(r"\{\{formula:([0-9a-f\-]+)\}\}")
_FIGURE_RE = re.compile(r"\{\{(figure|table):([0-9a-f\-]+)\}\}")
_CITE_RE = re.compile(r"\{\{cite:([0-9a-f]+)\}\}")
# unarXive renders resolved internal cross-references (\ref, \eqref) as the literal
# token "(REF )"; turn it back into a neutral LaTeX \ref{} so the body stays valid-ish.
_XREF_RE = re.compile(r"\(REF \)")


# ---------------------------------------------------------------------------
# Globals populated in pass-2 workers via the Pool initializer (avoids pickling
# the ~2M-entry corpus index to every task).
# ---------------------------------------------------------------------------
_ARXIV_TO_TITLE: dict = {}     # canonical arXiv id -> title (in-corpus targets)
_ARXIV_TO_NORMED: dict = {}    # canonical arXiv id -> normed_identifier
_OA_TO_ARXIV: dict = {}        # openalex id 'W...' -> canonical arXiv id (full OA map)


def _norm_text(s: str) -> str:
    return s if s is not None else ""


# ---------------------------------------------------------------------------
# Pass 1 worker: index corpus papers in one shard.
# ---------------------------------------------------------------------------
def _pass1_worker(shard: str):
    """Return (corpus_entries, ) where corpus_entries maps canonical arXiv id ->
    (normed_id, title, categories) for every paper in this shard."""
    entries = {}
    with open(shard, "r", encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                pid = r["paper_id"]
            except Exception:
                continue
            cid = canonical_arxiv_id(pid)
            meta = r.get("metadata", {}) or {}
            title = _norm_text(meta.get("title")).strip().replace("\n", " ")
            # collapse internal whitespace runs in titles (latex wrapping artifacts)
            title = re.sub(r"\s+", " ", title)
            categories = _norm_text(meta.get("categories")).strip()
            if title:  # a paper with no title can't be a learnable link target
                entries[cid] = (normalize_arxiv(pid), title, categories)
    return entries


# ---------------------------------------------------------------------------
# Pass 2: rehydration + citation resolution.
# ---------------------------------------------------------------------------
def _pass2_init(index_path: str, oa_map_path: str | None) -> None:
    global _ARXIV_TO_TITLE, _ARXIV_TO_NORMED, _OA_TO_ARXIV
    with open(index_path, "r", encoding="utf-8") as f:
        for line in f:
            cid, normed, title, cats = json.loads(line)
            _ARXIV_TO_TITLE[cid] = title
            _ARXIV_TO_NORMED[cid] = normed
    if oa_map_path and os.path.exists(oa_map_path):
        with open(oa_map_path, "r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                _OA_TO_ARXIV[r["openalex_id"]] = r["arxiv_id"]


def _resolve_bib_to_arxiv(bib_entry: dict) -> str | None:
    """Resolve one bib entry to a canonical in-corpus arXiv id, or None.

    Direct arXiv id first (most reliable), then OpenAlex-id -> arXiv via the map.
    Returns the id only if it is in-corpus (a known target title exists)."""
    ids = bib_entry.get("ids") or {}
    # 1) direct arXiv id
    cand = None
    cax = bib_entry.get("contained_arXiv_ids") or []
    if cax and isinstance(cax[0], dict) and cax[0].get("id"):
        cand = canonical_arxiv_id(cax[0]["id"])
    elif ids.get("arxiv_id"):
        cand = canonical_arxiv_id(ids["arxiv_id"])
    if cand and cand in _ARXIV_TO_TITLE:
        return cand
    # 2) OpenAlex-id -> arXiv id (enrichment)
    oa = ids.get("open_alex_id")
    if oa:
        ax = _OA_TO_ARXIV.get(oa.rsplit("/", 1)[-1])
        if ax and ax in _ARXIV_TO_TITLE:
            return ax
    return None


def _rehydrate_body(record: dict) -> tuple[str, list[str]]:
    """Rehydrate a paper's sections into faithful LaTeX and rewrite in-corpus
    citations to ``\\cite{Title}``.

    Returns (body_text, outgoing_normed_ids).
    """
    ref_entries = record.get("ref_entries", {}) or {}
    bib_entries = record.get("bib_entries", {}) or {}

    # Pre-resolve every bib key -> (title, normed_id) for in-corpus targets.
    bibkey_to_target: dict[str, tuple[str, str]] = {}
    for bibkey, be in bib_entries.items():
        ax = _resolve_bib_to_arxiv(be)
        if ax is not None:
            bibkey_to_target[bibkey] = (_ARXIV_TO_TITLE[ax], _ARXIV_TO_NORMED[ax])

    outgoing: set[str] = set()

    def sub_formula(m):
        latex = (ref_entries.get(m.group(1), {}) or {}).get("latex", "")
        return f"${latex}$" if latex else ""

    def sub_figure(m):
        kind, uid = m.group(1), m.group(2)
        caption = (ref_entries.get(uid, {}) or {}).get("caption", "NO_CAPTION")
        if caption and caption != "NO_CAPTION":
            return f"\\ref{{{kind}}} ({caption})"
        return f"\\ref{{{kind}}}"

    def sub_cite(m):
        target = bibkey_to_target.get(m.group(1))
        if target is None:
            return ""  # out-of-corpus citation: drop the marker (no edge, no text noise)
        title, normed = target
        outgoing.add(normed)
        return f"\\cite{{{title}}}"

    parts: list[str] = []
    for section in record.get("sections", {}).values():
        if not isinstance(section, dict):
            continue
        text = _norm_text(section.get("text"))
        if not text:
            continue
        text = _FORMULA_RE.sub(sub_formula, text)
        text = _FIGURE_RE.sub(sub_figure, text)
        text = _CITE_RE.sub(sub_cite, text)
        text = _XREF_RE.sub(r"\\ref{}", text)
        parts.append(text)

    body = "\n\n".join(parts)
    return body, sorted(outgoing)


def _pass2_worker(args):
    """Rehydrate one shard; write its graph + content shard files; return counts."""
    shard, out_dir, shard_idx = args
    graph_path = os.path.join(out_dir, f"_graph_shard_{shard_idx:05d}.jsonl")
    content_path = os.path.join(out_dir, f"_content_shard_{shard_idx:05d}.jsonl")
    n_nodes = 0
    n_edges = 0
    with open(shard, "r", encoding="utf-8") as f, \
         open(graph_path, "w", encoding="utf-8") as gout, \
         open(content_path, "w", encoding="utf-8") as cout:
        for line in f:
            try:
                r = json.loads(line)
                pid = r["paper_id"]
            except Exception:
                continue
            cid = canonical_arxiv_id(pid)
            normed = _ARXIV_TO_NORMED.get(cid)
            title = _ARXIV_TO_TITLE.get(cid)
            if not normed or not title:
                continue  # untitled paper, skipped in pass 1
            categories = (r.get("metadata", {}) or {}).get("categories", "") or ""
            body, outgoing = _rehydrate_body(r)
            if not body:
                continue  # empty body contributes nothing trainable
            gout.write(json.dumps({
                "normed_identifier": normed,
                "raw_identifier": title,
                "categories": categories.strip(),
                "char_count": len(body),
                "outgoing": outgoing,
                "incoming": [],  # filled in the parent merge pass
            }) + "\n")
            cout.write(json.dumps({"normed_identifier": normed, "content": body}) + "\n")
            n_nodes += 1
            n_edges += len(outgoing)
    return graph_path, content_path, n_nodes, n_edges


def run_extraction(args, rep: ReproducibilityManager):
    """Extract the ArXiv graph + content into the ReproducibilityManager's run dir.

    Artifacts (graph.jsonl, content.jsonl, extract_summary.json, and the
    transient per-shard / index files) are written under ``rep.output_dir`` so
    every run is captured alongside its git state, environment, and invocation
    metadata — matching the convention used by data/pretokenize*.py.
    """
    out_dir = rep.output_dir

    shards = sorted(glob.glob(os.path.join(args.corpus_dir, "**", "*.jsonl"), recursive=True))
    if args.limit_shards:
        shards = shards[: args.limit_shards]
    logger.info("Found %d shards; %d workers; oa_map=%s",
                len(shards), args.workers, args.oa_map or "(none)")

    # ---- Pass 1: build the global corpus index (arxiv id -> normed, title, cats). ----
    t0 = time.time()
    index_path = os.path.join(out_dir, "_corpus_index.jsonl")
    arxiv_to_normed: dict[str, str] = {}
    with Pool(args.workers) as pool, open(index_path, "w", encoding="utf-8") as idxf:
        for i, entries in enumerate(pool.imap_unordered(_pass1_worker, shards, chunksize=8)):
            for cid, (normed, title, cats) in entries.items():
                if cid in arxiv_to_normed:
                    continue  # first occurrence wins (canonical dedup)
                arxiv_to_normed[cid] = normed
                idxf.write(json.dumps([cid, normed, title, cats]) + "\n")
            if (i + 1) % 2000 == 0:
                logger.info("pass1 %d/%d shards, %d corpus papers", i + 1, len(shards), len(arxiv_to_normed))
    logger.info("pass1 done: %d corpus papers in %.0fs", len(arxiv_to_normed), time.time() - t0)

    # ---- Pass 2: rehydrate + resolve citations, writing per-shard graph/content. ----
    t0 = time.time()
    tasks = [(shard, out_dir, i) for i, shard in enumerate(shards)]
    graph_shards: list[str] = []
    content_shards: list[str] = []
    total_nodes = total_edges = 0
    with Pool(args.workers, initializer=_pass2_init, initargs=(index_path, args.oa_map)) as pool:
        for i, (gpath, cpath, n_nodes, n_edges) in enumerate(
            pool.imap_unordered(_pass2_worker, tasks, chunksize=8)
        ):
            graph_shards.append(gpath)
            content_shards.append(cpath)
            total_nodes += n_nodes
            total_edges += n_edges
            if (i + 1) % 2000 == 0:
                logger.info("pass2 %d/%d shards, %d nodes, %d edges",
                            i + 1, len(shards), total_nodes, total_edges)
    logger.info("pass2 done: %d nodes, %d edges in %.0fs", total_nodes, total_edges, time.time() - t0)

    # ---- Merge: concatenate content; compute incoming edges; write final graph.jsonl. ----
    t0 = time.time()
    content_out = os.path.join(out_dir, "content.jsonl")
    with open(content_out, "w", encoding="utf-8") as cout:
        for cpath in sorted(content_shards):
            with open(cpath, "r", encoding="utf-8") as cin:
                for line in cin:
                    cout.write(line)
            os.remove(cpath)

    # First pass over graph shards: load nodes, detect title collisions, compute incoming.
    nodes: dict[str, dict] = {}
    incoming: dict[str, list[str]] = {}
    seen_titles: dict[str, str] = {}   # title -> first normed_id that claimed it
    n_title_collisions = 0
    for gpath in sorted(graph_shards):
        with open(gpath, "r", encoding="utf-8") as gin:
            for line in gin:
                node = json.loads(line)
                normed = node["normed_identifier"]
                title = node["raw_identifier"]
                # Disambiguate colliding titles: the second paper to claim a title
                # gets "(arxiv:YYYY.NNNNN)" appended so matching remains exact and unique.
                # The arxiv id is recoverable from the normed_identifier suffix-stripped body.
                if title in seen_titles:
                    n_title_collisions += 1
                    # Recover the canonical arxiv id from the normed_identifier:
                    # normed form is e.g. "2401_12345_abc123" -> "2401.12345".
                    arxiv_hint = re.sub(r"_[0-9a-f]{6}$", "", normed).replace("_", ".", 1)
                    node["raw_identifier"] = f"{title} (arxiv:{arxiv_hint})"
                else:
                    seen_titles[title] = normed
                nodes[normed] = node
        os.remove(gpath)
    if n_title_collisions:
        logger.info("disambiguated %d title collisions (0.13%% expected)", n_title_collisions)

    for normed, node in nodes.items():
        for tgt in node["outgoing"]:
            incoming.setdefault(tgt, []).append(normed)

    graph_out = os.path.join(out_dir, "graph.jsonl")
    with open(graph_out, "w", encoding="utf-8") as gout:
        for normed, node in nodes.items():
            node["incoming"] = incoming.get(normed, [])
            gout.write(json.dumps(node) + "\n")
    os.remove(index_path)
    logger.info("merge done in %.0fs", time.time() - t0)

    summary = {
        "n_nodes": len(nodes),
        "n_edges": total_edges,
        "mean_out_degree": round(total_edges / len(nodes), 3) if nodes else 0,
        "title_collisions_disambiguated": n_title_collisions,
        "graph": graph_out,
        "content": content_out,
        "oa_map_used": bool(args.oa_map),
    }
    with open(os.path.join(out_dir, "extract_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-dir", required=True)
    ap.add_argument(
        "-o", "--runs-dir", "--out-dir",
        dest="runs_dir",
        required=True,
        help="Root directory to store experiment runs. The ReproducibilityManager "
             "writes graph.jsonl + content.jsonl + extract_summary.json here, "
             "alongside a reproducibility/ folder capturing git state and the run "
             "invocation.",
    )
    ap.add_argument("--oa-map", default=None, help="arxiv_openalex_map.jsonl (enrichment); omit to skip")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 8)
    ap.add_argument("--limit-shards", type=int, default=0)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    # The ReproducibilityManager creates/owns the output directory, captures git
    # state + environment + invocation, and refuses to clobber a run that already
    # has reproducibility artifacts — matching data/pretokenize*.py.
    with ReproducibilityManager(output_dir=str(args.runs_dir), is_main_process=True) as rep:
        run_extraction(args, rep)


if __name__ == "__main__":
    main()
