"""
Phase 1 — Build the arXiv <-> OpenAlex id map for the corpus.

Streams the OpenAlex `works` snapshot (s3://openalex/data/works/, ~639 GB gzip,
~2128 parts) directly from S3 — no full download to disk — and extracts, for every
work that has an arXiv source location, the pair (canonical_arxiv_id, openalex_id).

Why we need it: unarXive resolves only ~14.5% of citations to a direct arXiv id but
~46% to an OpenAlex id. The direct-id citation graph is hub-dominated (a few
universally-cited papers create a 58.7% giant component that fragments under the
training no-repeats rule). Mapping cited OpenAlex ids -> arXiv ids recovers non-hub
lateral edges. With this map, a citation's `ids.open_alex_id` can be resolved to an
in-corpus arXiv paper at extraction time (Phase 2).

The output is a single JSONL of {"arxiv_id", "openalex_id"} (canonical, version-less
arxiv ids; bare 'W...' openalex ids). The reverse direction is recoverable by swapping.
~3M arXiv works -> a few hundred MB; written to /fss-data, never /fss.

Parallelized with multiprocessing.Pool over snapshot parts (mirrors
measure_density.py). Each worker streams its assigned parts over the public S3
HTTPS endpoint (the `aws` CLI is not installed on compute nodes; HTTPS + urllib
needs no external binary and no credentials — OpenAlex's bucket is public).

Usage (via SLURM, CPU-only):
    python data/arxiv_graph_extractor/build_openalex_map.py \
        --out /fss-data/.../graphs/arxiv_openalex_map.jsonl \
        --workers 32
"""
import argparse
import gzip
import json
import logging
import os
import re
import time
import urllib.request
from multiprocessing import Pool

logger = logging.getLogger(__name__)

# Public, unauthenticated S3 HTTPS endpoint for the OpenAlex bucket.
OPENALEX_HTTPS_BASE = "https://openalex.s3.amazonaws.com/"
OPENALEX_MANIFEST_KEY = "data/works/manifest"
ARXIV_SOURCE_ID = "https://openalex.org/S4306400194"  # the arXiv source in OpenAlex


def _s3_url_to_https(s3_url: str) -> str:
    """Convert 's3://openalex/data/works/.../part.gz' to its public HTTPS URL."""
    assert s3_url.startswith("s3://openalex/"), s3_url
    return OPENALEX_HTTPS_BASE + s3_url[len("s3://openalex/"):]

# Extract the arXiv id from a landing/pdf URL. Handles:
#   http://arxiv.org/abs/2403.11716
#   https://arxiv.org/pdf/1706.03762v5
#   https://doi.org/10.48550/arxiv.1706.03762
# New-style (YYMM.NNNNN) and old-style (archive/NNNNNNN) ids; version suffix stripped.
_ARXIV_URL_RE = re.compile(
    r"arxiv(?:\.org/(?:abs|pdf)/|[./])"
    r"([a-z\-]+(?:\.[A-Z]{2})?/\d{7}|\d{4}\.\d{4,5})",
    re.IGNORECASE,
)
_VERSION_RE = re.compile(r"v\d+$")


def extract_arxiv_id(url: str) -> str | None:
    """Return the canonical (version-less) arXiv id embedded in a URL, or None."""
    if not url:
        return None
    m = _ARXIV_URL_RE.search(url)
    if not m:
        return None
    return _VERSION_RE.sub("", m.group(1))


def list_part_keys() -> list[str]:
    """Read the snapshot manifest and return the full s3:// URLs of all works parts."""
    with urllib.request.urlopen(OPENALEX_HTTPS_BASE + OPENALEX_MANIFEST_KEY, timeout=60) as r:
        manifest = json.loads(r.read())
    # manifest entries look like {"url": "s3://openalex/data/works/updated_date=.../part_000.gz", ...}
    keys = [e["url"] for e in manifest.get("entries", [])]
    if not keys:
        raise RuntimeError("No entries in OpenAlex works manifest")
    return keys


def _worker(part_url: str, _retries: int = 3):
    """Stream one snapshot part over HTTPS and return its (arxiv_id, openalex_id) pairs."""
    https_url = _s3_url_to_https(part_url)
    last_err = None
    for attempt in range(_retries):
        pairs = []
        try:
            with urllib.request.urlopen(https_url, timeout=300) as resp, \
                 gzip.GzipFile(fileobj=resp) as gz:
                for line in gz:
                    try:
                        w = json.loads(line)
                    except Exception:
                        continue
                    # Find an arXiv-source location and pull its URL.
                    arxiv_url = None
                    for loc in (w.get("locations") or []):
                        src = loc.get("source") or {}
                        if src.get("id") == ARXIV_SOURCE_ID:
                            arxiv_url = loc.get("landing_page_url") or loc.get("pdf_url")
                            if arxiv_url:
                                break
                    if not arxiv_url:
                        continue
                    arxiv_id = extract_arxiv_id(arxiv_url)
                    if not arxiv_id:
                        continue
                    oa = w.get("id")  # full https://openalex.org/W... URL
                    if not oa:
                        continue
                    pairs.append((arxiv_id, oa.rsplit("/", 1)[-1]))  # bare 'W...'
            return pairs
        except Exception as e:  # transient network/gzip error: retry the whole part
            last_err = e
    # Exhausted retries: return a sentinel so one flaky part doesn't kill the whole
    # run; the parent logs it as a gap (no silent loss — failures are counted).
    return ("__FAILED__", part_url, str(last_err))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output JSONL of {arxiv_id, openalex_id}")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 8)
    ap.add_argument("--limit-parts", type=int, default=0, help="0 = all parts (debug only)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    parts = list_part_keys()
    if args.limit_parts:
        parts = parts[: args.limit_parts]
    logger.info("OpenAlex works snapshot: %d parts; using %d workers", len(parts), args.workers)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    t0 = time.time()
    n_pairs = 0
    n_dup_arxiv = 0
    failed_parts: list[str] = []
    seen_arxiv: dict[str, str] = {}
    # Write streaming so we never hold all pairs in RAM beyond the dedup dict.
    with open(args.out, "w", encoding="utf-8") as fout, \
         Pool(args.workers) as pool:
        for i, result in enumerate(pool.imap_unordered(_worker, parts, chunksize=2)):
            # A failed part comes back as the 3-tuple ("__FAILED__", url, err).
            if result and result[0] == "__FAILED__":
                failed_parts.append(result[1])
                logger.warning("PART FAILED (skipped): %s — %s", result[1], result[2])
                continue
            for arxiv_id, oa_id in result:
                # Keep the first OpenAlex id seen for an arXiv id; OpenAlex
                # occasionally has duplicate works for the same preprint.
                if arxiv_id in seen_arxiv:
                    n_dup_arxiv += 1
                    continue
                seen_arxiv[arxiv_id] = oa_id
                fout.write(json.dumps({"arxiv_id": arxiv_id, "openalex_id": oa_id}) + "\n")
                n_pairs += 1
            if (i + 1) % 100 == 0:
                logger.info(
                    "%d/%d parts, %d unique arxiv<->OA pairs (%d dup, %d failed parts), %.0fs",
                    i + 1, len(parts), n_pairs, n_dup_arxiv, len(failed_parts), time.time() - t0,
                )

    logger.info(
        "DONE: %d unique pairs (%d dup arxiv skipped, %d FAILED parts) in %.0fs -> %s",
        n_pairs, n_dup_arxiv, len(failed_parts), time.time() - t0, args.out,
    )
    if failed_parts:
        # Persist the gap list so a targeted re-run can fill it (no silent truncation).
        fail_path = args.out + ".failed_parts.txt"
        with open(fail_path, "w") as f:
            f.write("\n".join(failed_parts) + "\n")
        logger.warning("%d parts failed; listed in %s", len(failed_parts), fail_path)
    print(json.dumps({
        "unique_pairs": n_pairs,
        "duplicate_arxiv_skipped": n_dup_arxiv,
        "parts_processed": len(parts),
        "failed_parts": len(failed_parts),
        "elapsed_sec": round(time.time() - t0, 1),
    }, indent=2))


if __name__ == "__main__":
    main()
