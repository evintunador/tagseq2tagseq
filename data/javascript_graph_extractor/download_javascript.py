"""
Stream the JavaScript subset of bigcode/the-stack-dedup to a JSONL file.

JavaScript analogue of typescript_graph_extractor/download_typescript.py. Keeps
only the fields the builder needs, retaining ``.js``/``.jsx``/``.mjs``/``.cjs``
source files. Minified ``.min.js`` bundles are kept in the stream but excluded from
nodes at build time (see build_javascript_graph._is_node_path).

Writes to /fss-data (never /fss for bulk I/O — see project memory).

Usage:
    python -m data.javascript_graph_extractor.download_javascript \\
        -o /fss-data/evin_t/tagseq2tagseq_artifacts/raw/javascript/sample_javascript.jsonl \\
        --limit 2000000
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from itertools import islice

from datasets import load_dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)

# fields we keep (everything the builder reads + a little provenance)
_KEEP = ("max_stars_repo_name", "max_stars_repo_path", "content", "ext", "lang")

_EXTS = (".js", ".jsx", ".mjs", ".cjs")


def _js_stream(token_enabled: bool = True):
    try:
        return load_dataset(
            "bigcode/the-stack-dedup", data_dir="data/javascript",
            split="train", streaming=True, token=token_enabled,
        )
    except TypeError:
        return load_dataset(
            "bigcode/the-stack-dedup", data_dir="data/javascript",
            split="train", streaming=True, use_auth_token=token_enabled,
        )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--output", required=True, help="output JSONL path")
    ap.add_argument("--limit", type=int, default=2_000_000)
    ap.add_argument("--flush-every", type=int, default=50_000)
    ap.add_argument("--max-content-chars", type=int, default=0,
                    help="truncate content to N chars (0 = no truncation)")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    logger.info("Streaming bigcode/the-stack-dedup data/javascript -> %s (limit %d)",
                args.output, args.limit)
    ds = _js_stream(True)

    written = kept_js = 0
    start = time.time()
    with open(args.output, "w", encoding="utf-8", buffering=8 * 1024 * 1024) as f:
        buf = []
        with tqdm(total=args.limit, unit="item", desc="download") as pbar:
            for item in islice(ds, args.limit):
                pbar.update(1)
                if not isinstance(item, dict):
                    continue
                rec = {k: item.get(k) for k in _KEEP}
                content = rec.get("content")
                path = rec.get("max_stars_repo_path") or ""
                if not content or not rec.get("max_stars_repo_name"):
                    continue
                # keep .js / .jsx / .mjs / .cjs source files only
                if not path.endswith(_EXTS):
                    continue
                if args.max_content_chars > 0 and len(content) > args.max_content_chars:
                    rec["content"] = content[: args.max_content_chars]
                kept_js += 1
                buf.append(json.dumps(rec))
                written += 1
                if len(buf) >= args.flush_every:
                    f.write("\n".join(buf) + "\n")
                    buf.clear()
        if buf:
            f.write("\n".join(buf) + "\n")

    dt = time.time() - start
    logger.info("Done: wrote %d records (%d .js/.jsx/.mjs/.cjs files) in %.0fs",
                written, kept_js, dt)
    print(json.dumps({"written": written, "js_files": kept_js, "seconds": round(dt)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
