"""
FineWeb (edgeless) dataset builder.

Streams high-quality web text from HuggingFaceFW/fineweb(-edu) and emits the
same two-file layout the ArXiv extractor produces, so the shared
pretokenization pipeline (data/pretokenize.py via data/pretokenize_fineweb.py)
can consume it unchanged:

  graph.jsonl    one node per doc: {normed_identifier, raw_identifier,
                 char_count, outgoing: [], incoming: []}. FineWeb has NO link
                 structure, so every edge list is empty — the corpus is a flat
                 control/baseline. Only `doc_causal` is meaningful downstream
                 (cross_doc_link would find nothing to grant).
  content.jsonl  one {normed_identifier, content} per line, aligned to graph.

Docs are accumulated until a target token budget is reached (FineWeb records
carry a precomputed gpt2 ``token_count``, so we can hit the budget precisely
without tokenizing here — pretokenize.py re-tokenizes with the real tokenizer).

Usage:
    python -m data.fineweb_graph_extractor.build_fineweb \\
        -o /fss-data/evin_t/tagseq2tagseq_artifacts/graphs/fineweb_run \\
        --target-tokens 1_300_000_000 \\
        --edu --size 10BT --seed 42
"""
import argparse
import json
import logging
from pathlib import Path

from datasets import load_dataset

from data.normalization import identifier_hash, _norm_body

logger = logging.getLogger(__name__)


# FineWeb ``token_count`` is a gpt2 count; our pretokenizer also defaults to
# gpt2, so the budget estimate is essentially exact. If a different tokenizer
# is used downstream the realized token total will differ slightly.
def _normed_id(raw_id: str, seen: set) -> str:
    """Derive a filesystem/graph-safe normed id from a FineWeb record id.

    FineWeb ids look like ``<urn:uuid:d853d453-...-efc2b26c40d2>``. We normalize
    the body and append a short hash of the raw id to guarantee uniqueness even
    if two normalized bodies collide.
    """
    body = _norm_body(raw_id)
    normed = f"{body}_{identifier_hash(raw_id)}" if body else identifier_hash(raw_id)
    # Defensive: guarantee global uniqueness (hash collisions are astronomically
    # unlikely but a duplicate would silently drop a doc in the writer merge).
    if normed in seen:
        suffix = 1
        while f"{normed}_{suffix}" in seen:
            suffix += 1
        normed = f"{normed}_{suffix}"
    return normed


def build(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    graph_path = out_dir / "graph.jsonl"
    content_path = out_dir / "content.jsonl"

    name = "HuggingFaceFW/fineweb" + ("-edu" if args.edu else "")
    config = "sample-" + args.size
    logger.info("Streaming %s (config=%s) ...", name, config)
    ds = load_dataset(
        name,
        name=config,
        split="train",
        streaming=True,
        cache_dir=args.cache_dir,
    )
    # Shuffle the stream so a token-capped prefix is a representative sample
    # rather than the first N crawl-ordered docs.
    ds = ds.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)

    seen: set = set()
    n_docs = 0
    total_tokens = 0
    total_chars = 0

    with open(graph_path, "w", encoding="utf-8") as gf, \
         open(content_path, "w", encoding="utf-8") as cf:
        for rec in ds:
            content = rec.get("text", "")
            if not content:
                continue
            raw_id = rec.get("id") or f"doc_{n_docs}"
            normed = _normed_id(raw_id, seen)
            seen.add(normed)

            gf.write(json.dumps({
                "normed_identifier": normed,
                "raw_identifier": raw_id,
                "char_count": len(content),
                "outgoing": [],
                "incoming": [],
            }) + "\n")
            cf.write(json.dumps({
                "normed_identifier": normed,
                "content": content,
            }) + "\n")

            n_docs += 1
            total_chars += len(content)
            # FineWeb provides a precomputed gpt2 token_count; fall back to a
            # ~4 chars/token estimate if a record ever lacks it.
            total_tokens += int(rec.get("token_count") or (len(content) // 4))

            if n_docs % 50_000 == 0:
                logger.info("  %d docs, ~%.1fM tokens", n_docs, total_tokens / 1e6)
            if total_tokens >= args.target_tokens:
                break

    summary = {
        "source": name,
        "config": config,
        "seed": args.seed,
        "n_docs": n_docs,
        "approx_tokens_gpt2": total_tokens,
        "total_chars": total_chars,
        "target_tokens": args.target_tokens,
    }
    with open(out_dir / "extract_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(
        "Done: %d docs, ~%.1fM gpt2 tokens (target %.1fM) → %s",
        n_docs, total_tokens / 1e6, args.target_tokens / 1e6, out_dir,
    )


def main():
    parser = argparse.ArgumentParser(
        prog="FineWeb builder",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-o", "--output-dir", required=True,
                        help="Run dir for graph.jsonl + content.jsonl.")
    parser.add_argument("--target-tokens", type=lambda s: int(s.replace("_", "")),
                        default=1_300_000_000,
                        help="Stop after ~this many (gpt2) tokens (default 1.3B).")
    parser.add_argument("--edu", action="store_true",
                        help="Use fineweb-edu (educational-filtered) instead of base fineweb.")
    parser.add_argument("--size", type=str, default="10BT",
                        choices=["10BT", "100BT", "350BT"],
                        help="FineWeb sample config (default 10BT; plenty for a 1.3B slice).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle-buffer", type=int, default=10_000,
                        help="Streaming shuffle buffer size.")
    parser.add_argument("--cache-dir", type=str,
                        default="/fss-data/evin_t/tagseq2tagseq_artifacts/raw/fineweb_hf_cache",
                        help="HF datasets cache dir (on /fss-data, not /fss).")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(levelname)s: %(message)s")
    build(args)


if __name__ == "__main__":
    main()
