"""
CLI: dump random (source doc, detected link, resolved target, target snippet)
tuples from a built dataset — the artifact a HUMAN reads to confirm links point
where they claim (design doc §3c, feeds the mandatory visual gate).

Usage:
    python -m data.graph_harness.run_sample_dump <dataset_dir> --detector <name> \
        [--n 20] [--seed 0] [--snippet-chars 300]

For each of N randomly sampled source nodes with at least one detected link:
  * prints the source identifier + a snippet,
  * runs the language's LinkDetector on the source tokens,
  * for each detected link: prints the emitted target_str, whether it RESOLVES
    (via the same PretokCorpus resolution training uses), the resolved target
    identifier, and a snippet of the target document.

No model / checkpoint needed. This is deliberately human-facing prose, not a
machine gate — the numbers come from run_audit / the fixtures runner; this shows a
human the actual content so "looks fine but is subtly wrong" data gets caught.
"""
from __future__ import annotations

import argparse
import random
import sys

import torch


def _snippet(text: str, n: int) -> str:
    text = text.replace("\n", "\\n")
    return text[:n] + ("…" if len(text) > n else "")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset_dir")
    ap.add_argument("--detector", required=True,
                    help="link_detector name (python, go, markdown, arxiv, ...)")
    ap.add_argument("--n", type=int, default=20, help="source docs to sample")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--snippet-chars", type=int, default=300)
    ap.add_argument("--tokenizer", default="gpt2")
    ap.add_argument("--max-scan", type=int, default=2000,
                    help="max nodes to scan looking for N with links")
    args = ap.parse_args(argv)

    import tiktoken
    from data.dataset import GraphIndex, PretokShardedBackend
    from model.document_corpus import PretokCorpus
    from model.graph_traversal.link_detector import make_link_detector

    enc = tiktoken.get_encoding(args.tokenizer)
    detector = make_link_detector(args.detector, enc.decode)
    graph = GraphIndex(args.dataset_dir)
    backend = PretokShardedBackend(graph)
    corpus = PretokCorpus(args.dataset_dir, link_detector=detector)

    all_normed = graph.get_all_normed_identifiers()
    rng = random.Random(args.seed)
    rng.shuffle(all_normed)

    shown = 0
    scanned = 0
    for normed in all_normed:
        if shown >= args.n or scanned >= args.max_scan:
            break
        scanned += 1
        tokens = backend.get_tokens(normed)
        if tokens is None or len(tokens) == 0:
            continue
        ids = torch.tensor(tokens.tolist(), dtype=torch.long)
        raw = graph.get_raw_identifier(normed) or normed
        # prefer per-doc detection when available (relative imports)
        if hasattr(detector, "detect_links_for_doc"):
            links = detector.detect_links_for_doc(ids, raw)
        else:
            links = detector.detect_links(ids)
        if not links:
            continue

        shown += 1
        src_text = enc.decode(tokens.tolist())
        print("=" * 78)
        print(f"SOURCE [{shown}/{args.n}]  {raw}")
        print(f"  {_snippet(src_text, args.snippet_chars)}")
        # dedupe targets, keep resolution status
        seen = set()
        for li in links:
            if li.target_str in seen:
                continue
            seen.add(li.target_str)
            resolved = corpus.has_document(li.target_str)
            status = "RESOLVES" if resolved else "unresolved"
            line = f"  link@{li.link_end_pos} -> {li.target_str!r}  [{status}]"
            print(line)
            if resolved:
                tgt_tokens = list(corpus.get_document(li.target_str))
                if tgt_tokens:
                    tgt_text = enc.decode(tgt_tokens)
                    print(f"      target: {_snippet(tgt_text, args.snippet_chars)}")

    print("=" * 78)
    print(f"Shown {shown} source docs with links (scanned {scanned}).")
    backend.close()
    corpus.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
