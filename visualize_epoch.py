#!/usr/bin/env python
"""
Visualize a pre-computed epoch's density distribution and attention masks.

Generates up to three figures:

  {output_dir}/density_overview.png  -- kv_block_count histogram + per-bucket violin
  {output_dir}/masks.png             -- block-level attention masks (sparsest / densest bucket)
  {output_dir}/step_timing.png       -- per-step wall-clock timing coloured by bucket
                                        (requires --live-run and/or --precomputed-run)

Usage examples
--------------
# Density + masks only (no timing data):
python visualize_epoch.py \\
    --epoch-dir  schedules/smoke_stack10m_bfs/epoch_0 \\
    --dataset-dir data/pretokenized_datasets/stack_10m

# Full report including timing comparison:
python visualize_epoch.py \\
    --epoch-dir         schedules/smoke_stack10m_bfs/epoch_0 \\
    --dataset-dir       data/pretokenized_datasets/stack_10m \\
    --live-run          runs/run_20260319_220022_460348 \\
    --precomputed-run   runs/run_20260319_220230_319138 \\
    --output-dir        artifacts/smoke_report
"""

import argparse
import collections
import copy
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import pyarrow.parquet as pq


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_epoch(epoch_dir: str):
    """Return (records, metadata) from packs.parquet + metadata.json."""
    from data.epoch_precompute import _table_to_records
    path = Path(epoch_dir)
    table = pq.read_table(str(path / "packs.parquet"))
    records = _table_to_records(table)
    meta = json.load(open(path / "metadata.json"))
    return records, meta


def load_timing_csv(run_dir: str, rank: int = 0) -> List[dict]:
    """Load step_timing_rank{rank}.csv, skip compile step (total_s > 30s)."""
    path = Path(run_dir) / f"step_timing_rank{rank}.csv"
    if not path.exists():
        return []
    with open(path) as f:
        rows = list(csv.DictReader(f))
    # Skip the torch.compile step (first step, usually >> 30s)
    return [r for r in rows if float(r["total_s"]) < 30]


def all_ranks_timing(run_dir: str) -> List[List[dict]]:
    """Load timing CSVs for all available ranks in a run dir."""
    result = []
    for rank in range(16):
        rows = load_timing_csv(run_dir, rank)
        if not rows:
            break
        result.append(rows)
    return result


def identify_run_mode(run_dir: str) -> str:
    """Return 'PRECOMPUTED' or 'LIVE' by inspecting hyperparameters.json."""
    hp_path = Path(run_dir) / "hyperparameters.json"
    if hp_path.exists():
        hp = json.load(open(hp_path))
        if hp.get("data", {}).get("epoch_dirs"):
            return "PRECOMPUTED"
    return "LIVE"


# ---------------------------------------------------------------------------
# Block-mask reconstruction (CPU, no CUDA needed)
# ---------------------------------------------------------------------------

def compute_block_mask_grid(
    record,
    graph,
    backend,
    layout,
    block_size: int = 128,
    seq_limit: int = None,
) -> Tuple[np.ndarray, List[int], int, int, int, bool]:
    """
    Reconstruct the block-level attention mask for one PackRecord, classifying
    each ``block_size`` × ``block_size`` tile as empty / partial / full.

    The classification is derived from the EXACT token-level mask
    (causal & (same_doc | cross_doc_grant), the same composition the kernel
    uses), then reduced per tile, so it faithfully reflects what the kernel
    computes:

      * 0 = empty  — no token pair in the tile may attend (kernel skips it);
      * 1 = partial — some but not all token pairs attend (e.g. the causal
            triangle edge, or a grant region that starts mid-block); and
      * 2 = full   — every token pair in the tile attends (kernel can skip the
            per-element mask_mod entirely — this is FlexAttention's
            ``full_kv_num_blocks``).

    Partial tiles are exactly where the token-level structure (causal diagonal,
    grant rectangle edges) lives, so colouring them distinctly recovers the
    detail a pure non-empty/empty block view hides.

    ``seq_limit`` optionally crops the pack to its first N tokens (and the docs
    that fall within) so a real 32k pack can be visualised at a legible
    resolution without recomputing a smaller-seq-len epoch.

    Returns
    -------
    grid        : (n_blocks, n_blocks) int8 ndarray with values {0, 1, 2}
    boundaries  : list of block indices where doc boundaries occur
    seq_len     : token count actually rendered (after any seq_limit crop)
    n_docs      : number of documents within the rendered span
    n_links     : number of cross-doc links within the rendered span
    stale       : True if the pack materialized shorter than its recorded body
                  budget (schedule is stale vs the on-disk dataset)
    """
    import torch

    from data.epoch_precompute import _record_to_placements
    from data.collate import build_packed_batch
    from model.graph_traversal.cross_doc_mask import CrossDocLinkMaskCreator

    placements = _record_to_placements(record)
    batch = build_packed_batch(graph, backend, layout, placements)
    tokens = batch["tokens"]            # [1, T]
    doc_spans = batch["doc_spans"]
    full_seq_len = tokens.shape[-1]

    # Staleness check: a record's effective_lens are body-token budgets decided at
    # precompute time. If the on-disk dataset was re-pretokenized since (shorter
    # bodies, or doc_ids now out of range), build_packed_batch silently clips and
    # the pack materializes short. Detect that here so the figure can flag it
    # rather than rendering a misleadingly tiny pack. (effective_lens exclude
    # layout decoration, so the materialized length is normally >= their sum.)
    stale = full_seq_len < sum(record.effective_lens)

    # Cross-doc grants come from the STORED record links (link_end_positions →
    # link_target_doc_ids), exactly as BucketedPackDataset feeds the kernel at
    # training time. We deliberately do NOT re-run the link detector here: the
    # precompute worker already resolved links to in-pack doc ids, and
    # re-detecting + re-matching does not reproduce that (it can silently yield
    # zero matches, collapsing the mask to doc_causal).
    link_to_target = dict(zip(record.link_end_positions, record.link_target_doc_ids))

    # Optional crop: render only the first seq_limit tokens. Keep doc_spans that
    # begin within the crop, clipping their end so per-doc triangles stay exact,
    # and drop grants whose source link position falls past the crop.
    seq_len = full_seq_len if seq_limit is None else min(seq_limit, full_seq_len)
    if seq_limit is not None and seq_len < full_seq_len:
        tokens = tokens[:, :seq_len]
        cropped = []
        for s in doc_spans:
            if s.start >= seq_len:
                continue
            if s.end <= seq_len:
                cropped.append(s)
            else:
                # Shallow copy with a clipped end so the dense mask matches.
                cs = copy.copy(s)
                cs.end = seq_len
                cropped.append(cs)
        doc_spans = cropped
        link_to_target = {p: t for p, t in link_to_target.items() if p < seq_len}

    n_links = len(link_to_target)

    # Build the EXACT token-level mask = causal & (same_doc | cross_doc_grant),
    # the same composition CrossDocLinkMaskCreator.__call__ uses for the kernel.
    creator = CrossDocLinkMaskCreator(link_detector=None)
    device = torch.device("cpu")
    cross = creator._build_cross_doc_mask(seq_len, doc_spans, link_to_target, device)

    document_ids = torch.full((seq_len,), -1, dtype=torch.int32, device=device)
    for s in doc_spans:
        a, b = max(0, s.start), min(seq_len, s.end)
        if a < b:
            document_ids[a:b] = s.doc_id
    q_idx = torch.arange(seq_len, device=device).unsqueeze(1)
    k_idx = torch.arange(seq_len, device=device).unsqueeze(0)
    causal = q_idx >= k_idx
    same_doc = document_ids.unsqueeze(1) == document_ids.unsqueeze(0)
    dense = causal & (same_doc | cross)  # [T, T] bool

    # Reduce the dense token mask to a per-tile {empty, partial, full} grid.
    n_blocks = math.ceil(seq_len / block_size)
    grid = np.zeros((n_blocks, n_blocks), dtype=np.int8)
    dense_np = dense.numpy()
    for qb in range(n_blocks):
        q0, q1 = qb * block_size, min((qb + 1) * block_size, seq_len)
        for kb in range(qb + 1):  # lower triangle only (causal)
            k0, k1 = kb * block_size, min((kb + 1) * block_size, seq_len)
            tile = dense_np[q0:q1, k0:k1]
            n_attend = int(tile.sum())
            if n_attend == 0:
                continue
            grid[qb, kb] = 2 if n_attend == tile.size else 1

    boundaries = sorted({
        span.start // block_size
        for span in doc_spans
        if span.start > 0
    })

    return grid, boundaries, seq_len, len(doc_spans), n_links, stale


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

def bucket_colors(n_buckets: int):
    return [plt.cm.plasma(i / max(n_buckets - 1, 1)) for i in range(n_buckets)]


# ---------------------------------------------------------------------------
# Figure 1 — density overview
# ---------------------------------------------------------------------------

def fig_density_overview(records, meta, output_path: str):
    n_buckets = meta["n_buckets"]
    colors = bucket_colors(n_buckets)

    bucket_kv: Dict[int, List[int]] = collections.defaultdict(list)
    for r in records:
        bucket_kv[r.bucket_id].append(r.kv_block_count)

    all_kv = [r.kv_block_count for r in records]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"kv_block_count density distribution  "
        f"({meta.get('n_packs', len(records)):,} packs, "
        f"{meta.get('strategy','?').upper()}, "
        f"{meta.get('token_budget', '?')} tokens)",
        fontsize=13, fontweight="bold",
    )

    # Stacked histogram
    ax = axes[0]
    bins = np.linspace(min(all_kv), max(all_kv), 60)
    bottoms = np.zeros(len(bins) - 1)
    for b in range(n_buckets):
        counts, _ = np.histogram(bucket_kv[b], bins=bins)
        ax.bar(bins[:-1], counts, width=np.diff(bins), bottom=bottoms,
               color=colors[b], alpha=0.85, label=f"bucket {b}")
        bottoms += counts
    ax.set_xlabel("kv_block_count (non-empty 128×128 attention block pairs)")
    ax.set_ylabel("# packs")
    ax.set_title("kv_block_count histogram (stacked by bucket)")
    ax.legend(fontsize=7, ncol=2)

    # Per-bucket violin
    ax2 = axes[1]
    data = [bucket_kv[b] for b in range(n_buckets)]
    means = [np.mean(d) for d in data]
    stds  = [np.std(d)  for d in data]
    covs  = [s / max(m, 1) * 100 for s, m in zip(stds, means)]
    parts = ax2.violinplot(data, positions=range(n_buckets), showmedians=True)
    for pc, c in zip(parts["bodies"], colors):
        pc.set_facecolor(c)
        pc.set_alpha(0.8)
    parts["cmedians"].set_color("white")
    for key in ("cbars", "cmins", "cmaxes"):
        parts[key].set_color("gray")
    ax2.scatter(range(n_buckets), means, color="white", s=30, zorder=5, label="mean")
    # Annotate per-bucket CoV above each violin
    y_top = ax2.get_ylim()[1] if ax2.get_ylim()[1] > 0 else max(max(d) for d in data)
    for b in range(n_buckets):
        top = max(data[b])
        ax2.text(b, top * 1.01, f"CoV\n{covs[b]:.0f}%",
                 ha="center", va="bottom", fontsize=6, color="dimgray")
    ax2.set_xticks(range(n_buckets))
    ax2.set_xticklabels(
        [f"b{b}\n{means[b]:.0f}" for b in range(n_buckets)], fontsize=8
    )
    ratio = means[-1] / max(means[0], 1)
    overall_cov = np.std(all_kv) / max(np.mean(all_kv), 1) * 100
    ax2.set_xlabel("bucket  (mean kv_block_count below)")
    ax2.set_ylabel("kv_block_count")
    ax2.set_title(
        f"Per-bucket spread  ({ratio:.1f}× ratio b0→b{n_buckets-1})\n"
        f"overall CoV={overall_cov:.0f}%  |  "
        f"within-bucket CoV: min={min(covs):.0f}%  max={max(covs):.0f}%  "
        f"mean={np.mean(covs):.0f}%"
    )
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {output_path}")


# ---------------------------------------------------------------------------
# Figure 2 — block-level attention masks
# ---------------------------------------------------------------------------

def fig_masks(records, meta, dataset_dir: str, output_path: str,
              block_size: int = 128, seq_limit: int = None, pack_ids=None):
    import tiktoken
    from data.dataset import GraphIndex, PretokShardedBackend
    from data.layout import make_layout_policy

    graph = GraphIndex(dataset_dir)
    backend = PretokShardedBackend(graph)
    enc = tiktoken.get_encoding(graph.metadata.get("tokenizer", "gpt2"))
    layout = make_layout_policy(
        meta.get("layout_policy", "null"), encode_fn=enc.encode_ordinary
    )

    n_buckets = meta["n_buckets"]

    bucket_lists: Dict[int, list] = collections.defaultdict(list)
    for r in records:
        bucket_lists[r.bucket_id].append(r)

    def pick_pack(b, seq_lim=None):
        """Pick the most link-rich pack in bucket b (in-crop links first, median fallback)."""
        lst = sorted(bucket_lists[b], key=lambda r: r.kv_block_count)
        limit = seq_lim or (1 << 30)
        # Count links whose source position falls within the crop window.
        def in_crop_links(r):
            return sum(1 for p in r.link_end_positions if p < limit)
        best = max(lst, key=in_crop_links)
        if in_crop_links(best) > 0:
            return best
        return lst[len(lst) // 2]  # fallback: median by density

    if pack_ids:
        # Explicit pack selection (e.g. to showcase packs with cross-doc grants).
        by_id = {r.pack_id: r for r in records}
        selections = []
        for pid in pack_ids:
            if pid not in by_id:
                print(f"  (pack_id {pid} not found; skipping)")
                continue
            r = by_id[pid]
            selections.append((r.bucket_id, f"bucket {r.bucket_id}", r))
    else:
        # Five percentile buckets: 0 / 25 / 50 / 75 / 100.
        # Within each bucket, prefer the pack with the most in-crop cross-doc
        # links so the off-diagonal grant rectangles are visible where they exist.
        pcts = [0, 25, 50, 75, 100]
        labels = ["p0 (lowest)", "p25", "p50 (median)", "p75", "p100 (highest)"]
        def pct_bucket(p):
            return int(round(p / 100 * (n_buckets - 1)))
        selections = [
            (pct_bucket(p), f"{lbl} density", pick_pack(pct_bucket(p), seq_lim=seq_limit))
            for p, lbl in zip(pcts, labels)
        ]

    # 3-state colormap: 0=empty (white), 1=partial (light), 2=full (dark).
    # Partial tiles are where the token-level structure (causal edge / grant
    # boundary) lives — distinguishing them recovers the detail a binary
    # non-empty/empty view hides at block resolution.
    cmap = ListedColormap(["#f5f8ff", "#7fb3e6", "#0b2a5b"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

    fig, axes = plt.subplots(1, len(selections), figsize=(6.5 * len(selections), 7.9))
    if len(selections) == 1:
        axes = [axes]
    crop_note = f", first {seq_limit} tok shown" if seq_limit else ""
    mode_note = "5 percentile buckets; link-richest pack per bucket" if not pack_ids else "explicit pack selection"
    fig.suptitle(
        f"Block-level attention masks  ({mode_note}{crop_note})\n"
        f"each cell = one {block_size}-token block  ·  partial vs full distinguished",
        fontsize=12, fontweight="bold", y=1.01,
    )

    any_stale = False
    for ax, (b_idx, label, rec) in zip(axes, selections):
        print(f"  building mask for bucket {b_idx} ({label}) pack {rec.pack_id}...")
        grid, boundaries, seq_len, n_docs, n_links, stale = compute_block_mask_grid(
            rec, graph, backend, layout,
            block_size=block_size, seq_limit=seq_limit,
        )
        any_stale = any_stale or stale
        n = grid.shape[0]
        ax.imshow(grid, origin="upper", aspect="equal",
                  cmap=cmap, norm=norm, interpolation="nearest")
        for bnd in boundaries:
            ax.axhline(bnd - 0.5, color="tomato", linewidth=0.6, alpha=0.9)
            ax.axvline(bnd - 0.5, color="tomato", linewidth=0.6, alpha=0.9)
        ax.set_xlabel(f"KV block  (0–{n-1}, {seq_len} tokens)")
        ax.set_ylabel(f"Q block   (0–{n-1})")
        stale_tag = "  ⚠ STALE (clipped)" if stale else ""
        title_color = "crimson" if stale else "black"
        ax.set_title(
            f"Bucket {b_idx} ({label})  —  kv_block_count={rec.kv_block_count:,}{stale_tag}\n"
            f"{n_docs} docs, {n_links} cross-doc links, {seq_len} tokens",
            fontsize=10, color=title_color,
        )
        total_causal = n * (n + 1) // 2
        n_full = int((grid == 2).sum())
        n_partial = int((grid == 1).sum())
        nnz = n_full + n_partial
        sparsity = 1.0 - nnz / max(total_causal, 1)
        ax.text(
            0.97, 0.03,
            f"full: {n_full:,}   partial: {n_partial:,}\n"
            f"non-empty: {nnz:,}   sparsity: {sparsity:.1%}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
            color="black",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
        )

    # Shared legend for the 3 tile states.
    handles = [
        mpatches.Patch(color="#0b2a5b", label="full block (all token pairs attend)"),
        mpatches.Patch(color="#7fb3e6", label="partial block (some token pairs attend)"),
        mpatches.Patch(facecolor="#f5f8ff", edgecolor="gray",
                       label="empty block (skipped by kernel)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, -0.02))

    if any_stale:
        warning = (
            "⚠ At least one pack materialized SHORTER than its recorded body budget — "
            "the schedule is STALE vs the on-disk dataset (re-pretokenized since precompute). "
            "Regenerate with precompute_epochs.py."
        )
        fig.text(0.5, 0.965, warning, ha="center", va="top", fontsize=9.5,
                 color="white", fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="crimson", alpha=0.92))
        print(f"  WARNING: {warning}")

    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {output_path}")


# ---------------------------------------------------------------------------
# Figure 3 — step timing
# ---------------------------------------------------------------------------

def _bucket_timing_stats(
    rows: List[dict],
    n_buckets: int,
    bucket_seq: List[int],
) -> Dict[int, List[float]]:
    """Group step wall-times by bucket index using the bucket sequence."""
    by_bucket: Dict[int, List[float]] = collections.defaultdict(list)
    for r in rows:
        step = int(r["step"])
        t = float(r["total_s"])
        b = bucket_seq[step % len(bucket_seq)]
        by_bucket[b].append(t)
    return by_bucket


def fig_step_timing(meta, run_configs: List[Tuple[str, str]], output_path: str):
    """Per-step bar charts (one panel per run), coloured by bucket for precomputed runs."""
    from data.bucketed_pack_dataset import _make_bucket_sequence

    n_buckets = meta["n_buckets"]
    colors = bucket_colors(n_buckets)
    bucket_seq = _make_bucket_sequence(n_buckets, seed=0)

    n_runs = len(run_configs)
    fig, axes = plt.subplots(n_runs, 1, figsize=(14, 4.5 * n_runs), sharex=True)
    if n_runs == 1:
        axes = [axes]
    fig.suptitle(
        f"Step timing  (rank 0, {meta.get('token_budget', '?')} tokens)",
        fontsize=13, fontweight="bold",
    )

    for ax, (run_dir, label) in zip(axes, run_configs):
        rows = load_timing_csv(run_dir, rank=0)
        if not rows:
            ax.set_title(f"{label} — no timing data found in {run_dir}")
            continue

        xs = [int(r["step"]) for r in rows]
        ys = [float(r["total_s"]) for r in rows]
        mu, sigma = np.mean(ys), np.std(ys)

        is_pre = "PRECOMPUTED" in label.upper()
        if is_pre:
            for x, y in zip(xs, ys):
                b = bucket_seq[x % len(bucket_seq)]
                ax.bar(x, y, color=colors[b], alpha=0.85, width=0.8)
            legend_patches = [
                mpatches.Patch(color=colors[b], label=f"b{b}")
                for b in range(n_buckets)
            ]
            ax.legend(handles=legend_patches, ncol=min(n_buckets, 16),
                      fontsize=7, loc="upper right")
        else:
            ax.bar(xs, ys, color="steelblue", alpha=0.75, width=0.8)

        ax.axhline(mu, color="red", linewidth=1.5, linestyle="--")
        ax.set_ylabel("wall time (s)")
        ax.set_title(
            f"{label}   —   mean={mu:.2f}s  std={sigma:.2f}s  "
            f"CoV={sigma/mu*100:.0f}%  (n={len(ys)} steps)",
            fontsize=10,
        )

    axes[-1].set_xlabel("optimizer step")
    fig.tight_layout()
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {output_path}")


def fig_step_timing_by_bucket(
    meta, run_configs: List[Tuple[str, str]], output_path: str
):
    """Per-bucket timing breakdown.

    Precomputed runs: mean ± std per bucket with within-bucket CoV annotated.
    Live runs: histogram of all step times with percentile markers
               (live has no buckets so per-bucket stats would be meaningless).
    """
    from data.bucketed_pack_dataset import _make_bucket_sequence

    n_buckets = meta["n_buckets"]
    colors = bucket_colors(n_buckets)
    bucket_seq = _make_bucket_sequence(n_buckets, seed=0)

    n_runs = len(run_configs)
    fig, axes = plt.subplots(1, n_runs, figsize=(7 * n_runs, 5))
    if n_runs == 1:
        axes = [axes]
    fig.suptitle(
        f"Step-time breakdown  (rank 0, {meta.get('token_budget', '?')} tokens)",
        fontsize=13, fontweight="bold",
    )

    for ax, (run_dir, label) in zip(axes, run_configs):
        rows = load_timing_csv(run_dir, rank=0)
        if not rows:
            ax.set_title(f"{label} — no data")
            continue

        ys = [float(r["total_s"]) for r in rows]
        mu = np.mean(ys)
        overall_cov = np.std(ys) / mu * 100 if mu > 0 else 0
        is_pre = "PRECOMPUTED" in label.upper()

        if is_pre:
            by_bucket = _bucket_timing_stats(rows, n_buckets, bucket_seq)
            bkt_indices = sorted(by_bucket)
            bkt_means = [np.mean(by_bucket[b]) for b in bkt_indices]
            bkt_stds  = [np.std(by_bucket[b])  for b in bkt_indices]
            bkt_covs  = [s / max(m, 1) * 100
                         for s, m in zip(bkt_stds, bkt_means)]
            bkt_ns    = [len(by_bucket[b]) for b in bkt_indices]

            ax.bar(bkt_indices, bkt_means, color=[colors[b] for b in bkt_indices],
                   alpha=0.85, yerr=bkt_stds, capsize=4,
                   error_kw={"elinewidth": 1.5, "ecolor": "dimgray"})
            ax.axhline(mu, color="red", linewidth=1.2, linestyle="--",
                       label=f"overall mean={mu:.2f}s")
            ax.legend(fontsize=8)

            for b, m, cov, n in zip(bkt_indices, bkt_means, bkt_covs, bkt_ns):
                ax.text(b, m + bkt_stds[bkt_indices.index(b)] + 0.08,
                        f"CoV={cov:.0f}%\n(n={n})",
                        ha="center", va="bottom", fontsize=7, color="dimgray")

            within_cov = [c for c in bkt_covs if not np.isnan(c)]
            ax.set_xticks(bkt_indices)
            ax.set_xticklabels([f"b{b}" for b in bkt_indices], fontsize=8)
            ax.set_xlabel("bucket index")
            ax.set_ylabel("mean ± 1 std  (s)")
            ax.set_title(
                f"{label.split(' — ')[0]}\n"
                f"within-bucket CoV: min={min(within_cov):.0f}%  "
                f"max={max(within_cov):.0f}%  mean={np.mean(within_cov):.0f}%\n"
                f"(overall CoV={overall_cov:.0f}%)",
                fontsize=9,
            )

        else:
            pcts = [10, 25, 50, 75, 90]
            pct_vals = np.percentile(ys, pcts)
            ax.hist(ys, bins=20, color="steelblue", alpha=0.75, edgecolor="white")
            for pct, val in zip(pcts, pct_vals):
                ax.axvline(val, linewidth=1.3, linestyle="--",
                           label=f"p{pct}={val:.1f}s")
            ax.set_xlabel("wall time (s)")
            ax.set_ylabel("# steps")
            ax.legend(fontsize=7.5)
            ax.set_title(
                f"{label.split(' — ')[0]}\n"
                f"p10={pct_vals[0]:.2f}s  p50={pct_vals[2]:.2f}s  "
                f"p90={pct_vals[4]:.2f}s\n"
                f"IQR={pct_vals[3]-pct_vals[1]:.2f}s  "
                f"overall CoV={overall_cov:.0f}%",
                fontsize=9,
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Visualize a pre-computed epoch's density and attention masks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--epoch-dir", required=True,
                        help="Path to epoch dir containing packs.parquet + metadata.json.")
    parser.add_argument("--dataset-dir", default=None,
                        help="Path to pretokenized dataset dir (needed for mask figures).")
    parser.add_argument("--live-run", default=None,
                        help="Run dir with step_timing_rank*.csv from live training.")
    parser.add_argument("--precomputed-run", default=None,
                        help="Run dir with step_timing_rank*.csv from precomputed training.")
    parser.add_argument("--output-dir", default="artifacts",
                        help="Directory to write PNG figures.")
    parser.add_argument("--block-size", type=int, default=128,
                        help="FlexAttention block size (default 128).")
    parser.add_argument("--seq-limit", type=int, default=None,
                        help="Crop each pack to its first N tokens for the mask "
                             "figure (e.g. 8192) so a real 32k pack renders at a "
                             "legible resolution without recomputing a smaller epoch.")
    parser.add_argument("--pack-ids", type=str, default=None,
                        help="Comma-separated pack_ids to render in the mask figure "
                             "instead of the default low/medium/high bucket medians "
                             "(e.g. to showcase packs with cross-doc grants).")
    parser.add_argument("--timing-rank", type=int, default=0,
                        help="Which rank's CSV to use for timing plots.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading epoch from {args.epoch_dir} ...")
    records, meta = load_epoch(args.epoch_dir)
    print(f"  {len(records):,} packs, {meta['n_buckets']} buckets, "
          f"kv_method={meta.get('kv_method','?')}")

    # ---- Figure 1: density overview (always) --------------------------------
    fig_density_overview(
        records, meta,
        output_path=os.path.join(args.output_dir, "density_overview.png"),
    )

    # ---- Figure 2: masks (requires dataset_dir) -----------------------------
    if args.dataset_dir:
        print("Building block-level attention masks ...")
        pack_ids = (
            [int(x) for x in args.pack_ids.split(",")] if args.pack_ids else None
        )
        fig_masks(
            records, meta,
            dataset_dir=args.dataset_dir,
            output_path=os.path.join(args.output_dir, "masks.png"),
            block_size=args.block_size,
            seq_limit=args.seq_limit,
            pack_ids=pack_ids,
        )
    else:
        print("Skipping masks (pass --dataset-dir to enable).")

    # ---- Figure 3: step timing (requires at least one run dir) --------------
    run_configs = []
    for run_dir, default_label in [
        (args.live_run,        "LIVE (online link detection)"),
        (args.precomputed_run, "PRECOMPUTED (density-bucketed)"),
    ]:
        if run_dir is None:
            continue
        label = identify_run_mode(run_dir)
        label = f"{label} — {Path(run_dir).name}"
        run_configs.append((run_dir, label))

    if run_configs:
        fig_step_timing(
            meta, run_configs,
            output_path=os.path.join(args.output_dir, "step_timing.png"),
        )
        fig_step_timing_by_bucket(
            meta, run_configs,
            output_path=os.path.join(args.output_dir, "step_timing_by_bucket.png"),
        )
    else:
        print("Skipping step timing (pass --live-run / --precomputed-run to enable).")

    print("Done.")


if __name__ == "__main__":
    main()
