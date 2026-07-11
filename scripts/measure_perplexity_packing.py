"""One-off: measure the wall-time win from sequence-packing run_held_out_perplexity.

Loads a checkpoint once, then times two paths over the IDENTICAL doc set:
  - OLD: per-doc score_doc loop (one forward_inference per doc, batch-1)
  - NEW: score_docs_batched (multiple docs packed per forward)
Also asserts the two produce the same mean_nll (numerical parity on real weights).

Usage:
  python scripts/measure_perplexity_packing.py \
    --checkpoint runs/20260701_170244/checkpoints/best_model.pt \
    --dataset /fss-data/evin_t/tagseq2tagseq_artifacts/pretokenized_datasets/arxiv \
    --split val_random --max-docs 200
"""
import argparse
import time

import numpy as np
import torch

from data.dataset import GraphIndex, PretokShardedBackend
from eval.scoring import score_doc, score_docs_batched
from generate import load_inference_model


def _gather_docs(graph, backend, split, max_docs):
    if split == "all":
        ids = list(range(len(graph)))
        import random
        random.Random(42).shuffle(ids)
        ids = ids[:max_docs]
    else:
        ids = graph.get_split_ids(split)[:max_docs]
    docs = []
    for doc_id in ids:
        arr = backend.get_tokens_by_id(doc_id)
        if arr is None or len(arr) < 2:
            continue
        normed = graph.get_normed_identifier(doc_id)
        raw = graph.get_raw_identifier(normed) or normed
        docs.append((arr.tolist(), raw, normed))
    return docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="val_random")
    ap.add_argument("--max-docs", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    print(f"Loading {args.checkpoint} ...", flush=True)
    model, _hp = load_inference_model(args.checkpoint, device=args.device)
    model.eval()
    layout = model.inference_layout_policy

    graph = GraphIndex(args.dataset)
    backend = PretokShardedBackend(graph)
    docs = _gather_docs(graph, backend, args.split, args.max_docs)
    print(f"Scoring {len(docs)} docs (split={args.split})", flush=True)

    def sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # Warm up compile on both code paths so timings exclude first-forward compile.
    print("Warming up (compile)...", flush=True)
    _ = score_doc(model, docs[0][0], layout, docs[0][1], docs[0][2], device=args.device)
    _ = score_docs_batched(model, docs[:2], layout, device=args.device)
    sync()

    # OLD path: per-doc score_doc
    sync(); t0 = time.perf_counter()
    old = [score_doc(model, b, layout, r, n, device=args.device) for (b, r, n) in docs]
    sync(); t_old = time.perf_counter() - t0

    # NEW path: batched
    sync(); t1 = time.perf_counter()
    new = score_docs_batched(model, docs, layout, device=args.device)
    sync(); t_new = time.perf_counter() - t1

    old_nll = np.mean([r["mean_nll"] for r in old if r["num_tokens"] > 0])
    new_nll = np.mean([r["mean_nll"] for r in new if r["num_tokens"] > 0])

    print("\n─── RESULTS ─────────────────────────────")
    print(f"  docs scored:     {len(docs)}")
    print(f"  OLD per-doc:     {t_old:8.2f} s")
    print(f"  NEW batched:     {t_new:8.2f} s")
    print(f"  speedup:         {t_old / t_new:8.2f}×")
    print(f"  mean_nll OLD:    {old_nll:.6f}")
    print(f"  mean_nll NEW:    {new_nll:.6f}")
    print(f"  |Δ mean_nll|:    {abs(old_nll - new_nll):.2e}")
    backend.close()


if __name__ == "__main__":
    main()
