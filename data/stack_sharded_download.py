"""
Parquet-sharded full-subset downloader for bigcode/the-stack-dedup.

The per-language `download_<lang>.py` scripts stream the subset sequentially from a
single reader — fine for a 2M sample, too slow for a FULL subset (TypeScript is
~10.6M files across 71 parquet shards). This downloader instead reads the language's
parquet files DIRECTLY and splits them across N workers, so a SLURM array job can
download the whole subset in parallel. Each worker writes one JSONL per parquet
shard it owns; the shards are concatenated (or read as a glob) by the graph builder.

Keeps the same fields + per-language extension filter as the sequential downloaders
(so the graph builders are unchanged). Resumable: a shard whose `.done` marker
exists is skipped.

Usage (single shard-range, e.g. one SLURM array task):
    python -m data.stack_sharded_download \
        --lang rust \
        --out-dir /fss-data/evin_t/tagseq2tagseq_artifacts/raw/rust/shards \
        --num-workers 21 --worker-id $SLURM_ARRAY_TASK_ID

Usage (list parquet files only, for planning / array sizing):
    python -m data.stack_sharded_download --lang rust --list-only
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from typing import Callable, Dict, List

logger = logging.getLogger(__name__)

_KEEP = ("max_stars_repo_name", "max_stars_repo_path", "content", "ext", "lang")

# Per-language "keep this record?" predicate on the repo-relative path. Mirrors the
# endswith filters in the sequential download_<lang>.py scripts EXACTLY.
_KEEP_PATH: Dict[str, Callable[[str], bool]] = {
    "rust":       lambda p: p.endswith(".rs"),
    "typescript": lambda p: p.endswith(".ts") or p.endswith(".tsx"),
    # .kt but NOT .kts (kotlin scripts excluded — match download_kotlin.py)
    "kotlin":     lambda p: p.endswith(".kt") and not p.endswith(".kts"),
    "go":         lambda p: p.endswith(".go") or p == "go.mod" or p.endswith("/go.mod"),
    "java":       lambda p: p.endswith(".java"),
    "python":     lambda p: p.endswith(".py"),
    # JS: .js/.jsx/.mjs/.cjs but NOT minified bundles (stem ending in .min —
    # matches build_javascript_graph's _is_js_node exclusion exactly).
    "javascript": lambda p: (p.endswith((".js", ".jsx", ".mjs", ".cjs"))
                             and not p.rsplit("/", 1)[-1].rsplit(".", 1)[0].endswith(".min")),
    "zig":        lambda p: p.endswith(".zig"),
    "dart":       lambda p: p.endswith(".dart"),
}


def _token() -> str:
    return open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()


def _list_parquet(lang: str, token: str) -> List[str]:
    from huggingface_hub import HfApi
    files = HfApi().list_repo_files("bigcode/the-stack-dedup",
                                    repo_type="dataset", token=token)
    pfs = sorted(p for p in files
                 if p.startswith(f"data/{lang}/") and p.endswith(".parquet"))
    if not pfs:
        raise SystemExit(f"No parquet files for data/{lang}")
    return pfs


def _download_via_cache(parquet_path: str, token: str) -> str:
    """Download the parquet to the local HF cache and return the local path.

    Downloading the whole file once (with hf_hub_download's built-in retry) then
    reading it locally is far more rate-limit-friendly than fsspec range-reads,
    which issue many concurrent HTTP requests per file (the 429 source).
    """
    from huggingface_hub import hf_hub_download
    return hf_hub_download("bigcode/the-stack-dedup", parquet_path,
                           repo_type="dataset", token=token)


def _download_shard(parquet_path: str, out_jsonl: str, keep_path, token: str,
                    max_retries: int = 8, base_backoff: float = 5.0,
                    jitter: float = 0.0) -> dict:
    """Stream one parquet file → JSONL, applying the extension filter.

    Retries with exponential backoff on transient HTTP errors (429/5xx). The
    parquet is fetched to the local HF cache first (single request with the hub's
    own retry) rather than range-read over HTTP, which avoids the 429 storm.
    """
    import pyarrow.parquet as pq

    done_marker = out_jsonl + ".done"
    if os.path.exists(done_marker):
        logger.info("skip (done): %s", out_jsonl)
        return {"skipped": True, "parquet": parquet_path}

    kept = read = 0
    t0 = time.time()
    tmp = out_jsonl + ".tmp"
    attempt = 0
    while True:
        try:
            local = _download_via_cache(parquet_path, token)
            pf = pq.ParquetFile(local)
            cols = [c for c in _KEEP if c in pf.schema_arrow.names]
            kept = read = 0
            with open(tmp, "w", encoding="utf-8", buffering=8 * 1024 * 1024) as out:
                for batch in pf.iter_batches(batch_size=4096, columns=cols):
                    d = batch.to_pydict()
                    n = len(next(iter(d.values())))
                    for i in range(n):
                        read += 1
                        path = d.get("max_stars_repo_path", [None] * n)[i] or ""
                        content = d.get("content", [None] * n)[i]
                        repo = d.get("max_stars_repo_name", [None] * n)[i]
                        if not content or not repo or not keep_path(path):
                            continue
                        rec = {k: d.get(k, [None] * n)[i] for k in _KEEP}
                        out.write(json.dumps(rec) + "\n")
                        kept += 1
            break
        except Exception as e:  # noqa: BLE001 — retry any transient fetch/parse error
            attempt += 1
            if attempt > max_retries:
                logger.error("shard %s FAILED after %d retries: %s",
                             parquet_path, max_retries, e)
                raise
            wait = base_backoff * (2 ** (attempt - 1)) + jitter
            logger.warning("shard %s attempt %d hit %s: %s — backoff %.0fs",
                           parquet_path, attempt, type(e).__name__, e, wait)
            time.sleep(wait)

    os.replace(tmp, out_jsonl)
    open(done_marker, "w").write(f"{kept}\n")
    dt = time.time() - t0
    logger.info("shard %s: read %d, kept %d in %.0fs", parquet_path, read, kept, dt)
    return {"parquet": parquet_path, "read": read, "kept": kept, "seconds": round(dt)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", required=True, choices=sorted(_KEEP_PATH))
    ap.add_argument("--out-dir", help="dir for per-shard JSONL (required unless --list-only)")
    ap.add_argument("--num-workers", type=int, default=1,
                    help="total workers splitting the parquet list")
    ap.add_argument("--worker-id", type=int, default=0,
                    help="this worker's index in [0, num-workers)")
    ap.add_argument("--stagger", type=float, default=0.0,
                    help="per-worker startup delay (seconds) = worker_id * stagger, "
                         "to avoid a synchronized HF-endpoint stampede (429)")
    ap.add_argument("--list-only", action="store_true",
                    help="print the parquet file count + names and exit")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    token = _token()
    pfs = _list_parquet(args.lang, token)

    if args.list_only:
        print(json.dumps({"lang": args.lang, "num_parquet": len(pfs),
                          "files": pfs}, indent=2))
        return 0

    if not args.out_dir:
        ap.error("--out-dir required unless --list-only")
    os.makedirs(args.out_dir, exist_ok=True)

    # Round-robin assign parquet files to this worker.
    mine = [p for i, p in enumerate(pfs) if i % args.num_workers == args.worker_id]
    keep_path = _KEEP_PATH[args.lang]
    logger.info("worker %d/%d handling %d/%d parquet files for %s",
                args.worker_id, args.num_workers, len(mine), len(pfs), args.lang)

    # Stagger workers' first request to avoid a synchronized HF-endpoint stampede
    # (429). Deterministic per worker, so resumes don't all wake at once.
    startup_delay = (args.worker_id % max(args.num_workers, 1)) * args.stagger
    if startup_delay:
        logger.info("worker %d staggered start: sleeping %.0fs",
                    args.worker_id, startup_delay)
        time.sleep(startup_delay)

    results = []
    for p in mine:
        # data/rust/train-00007-of-00021.parquet -> rust_00007.jsonl
        stem = p.rsplit("/", 1)[-1].replace(".parquet", "")
        out_jsonl = os.path.join(args.out_dir, f"{stem}.jsonl")
        results.append(_download_shard(p, out_jsonl, keep_path, token,
                                       jitter=(args.worker_id % 8)))

    total_kept = sum(r.get("kept", 0) for r in results)
    logger.info("worker %d done: %d shards, %d records kept",
                args.worker_id, len(mine), total_kept)
    print(json.dumps({"worker": args.worker_id, "shards": len(mine),
                      "kept": total_kept}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
