#!/usr/bin/env python
"""Reusable figure tool for the graph-sparsity scaling law.

Reads Phase-1 sweep output (one {dataset}.json per dataset, as written by
eval/sparsity_sweep.py) and renders two panels:

  A. Density lines — Δnll vs kept link fraction, small-multiples (one cell per
     dataset), coloured by CODE (blue) vs TEXT (orange). Shows the within-dataset
     dose-response (near-linear, pinned to 0 at keep=0).
  B. Cross-dataset law — Δ@1.0 vs effective grants/pack (log x), with an
     OLS fit over the code datasets. Shows whether density predicts benefit
     across datasets, and extrapolates past today's densest corpus.

Re-runnable: point --sweep-dir at any updated sweep output and it re-renders.
Pass --merged-dir to overlay the merged model's per-source Δ@1.0 in panel B, and
draw solo-vs-merged in panel A.

Colour: blue #2a78d6 (code) / orange #eb6834 (text) — validated (validate_palette.py,
CVD ΔE 24.7, both modes pass). Marks/labels follow the dataviz skill: thin 2px
lines, ≥8px markers, direct labels, recessive grid, legend for the 2 groups.

Usage:
    python -m eval.viz.plot_sparsity \
        --sweep-dir  /fss-data/.../sparsity_scaling/phase1_eval \
        --merged-dir /fss-data/.../sparsity_scaling/phase1_eval_merged3p9b \
        --effective  /fss-data/.../sparsity_scaling/effective_density.json \
        --out        /fss-data/.../sparsity_scaling/fig_sparsity
"""
import argparse, glob, json, math, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# thestack == Python (first code dataset); all 9 code datasets are from The Stack.
# Only wiki + arxiv are text.
CODE = {"thestack", "python", "go", "java", "typescript", "kotlin",
        "rust", "javascript", "zig", "dart"}
DISPLAY = {"thestack": "python (thestack)", "wiki_merged": "wiki"}
C_CODE, C_TEXT = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e6e3"


def _disp(ds):
    return DISPLAY.get(ds, ds)


def _group_color(ds):
    return C_CODE if ds in CODE else C_TEXT


def load_sweep(sweep_dir):
    """dataset -> {keep_frac: {mean_delta, ci_low, ci_high, n_packs}} (edge mode)."""
    out = {}
    for f in sorted(glob.glob(os.path.join(sweep_dir, "*.json"))):
        d = json.load(open(f))
        ds = d["dataset"]
        edge = sorted([r for r in d["rows"] if r.get("keep_mode") == "edge"],
                      key=lambda r: r["keep_frac"])
        if not edge:
            continue
        out[ds] = {
            "keeps": [r["keep_frac"] for r in edge],
            "delta": [r["mean_delta"] for r in edge],
            "lo": [r.get("delta_ci_low", r["mean_delta"]) for r in edge],
            "hi": [r.get("delta_ci_high", r["mean_delta"]) for r in edge],
            "delta1": next((r["mean_delta"] for r in edge if r["keep_frac"] == 1.0), None),
            "n": edge[-1].get("n_packs"),
        }
    return out


def panel_density_lines(ax_list, solo, merged, order):
    """Small-multiples: one cell per dataset, Δ vs keep_frac."""
    for ax, ds in zip(ax_list, order):
        s = solo[ds]
        col = _group_color(ds)
        ax.axhline(0, color=GRID, lw=1, zorder=0)
        # solo line + CI band
        ax.fill_between(s["keeps"], s["lo"], s["hi"], color=col, alpha=0.12, lw=0)
        ax.plot(s["keeps"], s["delta"], color=col, lw=2, marker="o", ms=4,
                zorder=3, label="solo")
        # merged overlay (dashed) if present
        if merged and ds in merged:
            m = merged[ds]
            ax.plot(m["keeps"], m["delta"], color=col, lw=2, ls="--", marker="s",
                    ms=3.5, alpha=0.85, zorder=2, label="merged")
        ax.set_title(_disp(ds), fontsize=9, color=INK, pad=3)
        ax.tick_params(labelsize=7, colors=MUTED)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(GRID)
        ax.margins(x=0.05)


def panel_cross_dataset(ax, solo, merged, eff):
    """Δ@1.0 vs effective grants/pack, log-x, OLS fit over code datasets."""
    xs_c, ys_c, xs_t, ys_t = [], [], [], []
    for ds, s in solo.items():
        if ds not in eff or s["delta1"] is None:
            continue
        x = eff[ds]["mean_grants_per_pack"]
        y = s["delta1"]
        (xs_c if ds in CODE else xs_t).append((x, ds))
        (ys_c if ds in CODE else ys_t).append(y)
    # scatter
    for (pts, ys, col, lab) in [(xs_c, ys_c, C_CODE, "code (The Stack)"),
                                 (xs_t, ys_t, C_TEXT, "text (wiki/arxiv)")]:
        if not pts:
            continue
        xv = [p[0] for p in pts]
        ax.scatter(xv, ys, s=48, color=col, zorder=3, label=lab,
                   edgecolor="white", linewidth=1)
        for (x, ds), y in zip(pts, ys):
            ax.annotate(_disp(ds), (x, y), fontsize=7, color=MUTED,
                        xytext=(4, 4), textcoords="offset points")
    # OLS fit over code (the r=0.97 line); fit in log-x
    if len(xs_c) >= 3:
        lx = np.log10([p[0] for p in xs_c])
        b, a = np.polyfit(lx, ys_c, 1)
        r = np.corrcoef(lx, ys_c)[0, 1]
        xr = np.linspace(min(lx), max(lx), 50)
        ax.plot(10**xr, a + b*xr, color=C_CODE, lw=1.5, ls=":", zorder=2,
                label=f"code fit (r={r:.2f})")
    ax.set_xscale("log")
    ax.axhline(0, color=GRID, lw=1, zorder=0)
    ax.set_xlabel("effective grants / pack  (density the mask saw, log scale)",
                  fontsize=9, color=MUTED)
    ax.set_ylabel("Δnll @ full density (cross − doc_causal)", fontsize=9, color=MUTED)
    ax.tick_params(labelsize=8, colors=MUTED)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.legend(fontsize=8, frameon=False, loc="upper left")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep-dir", required=True, help="Solo sweep dir ({ds}.json).")
    ap.add_argument("--merged-dir", default=None, help="Optional merged-model sweep dir.")
    ap.add_argument("--effective", required=True, help="effective_density.json.")
    ap.add_argument("--out", required=True, help="Output path stem (writes .png and .svg).")
    ap.add_argument("--title", default="Graph-sparsity scaling law (Phase 1, eval-time)")
    a = ap.parse_args()

    solo = load_sweep(a.sweep_dir)
    merged = load_sweep(a.merged_dir) if a.merged_dir else None
    eff = json.load(open(a.effective))
    if not solo:
        raise SystemExit(f"No sweep JSONs in {a.sweep_dir}")

    # order: code first (by Δ@1.0 desc), then text
    order = sorted(solo, key=lambda d: (d not in CODE, -(solo[d]["delta1"] or 0)))

    ncol = 4
    nrow = math.ceil(len(order) / ncol)
    fig = plt.figure(figsize=(13, 3.4 + 2.2*nrow), facecolor="#fcfcfb")
    gs = fig.add_gridspec(nrow + 2, ncol, height_ratios=[*([1]*nrow), 0.25, 2.4],
                          hspace=0.62, wspace=0.30, top=0.93)
    # Panel A: small multiples
    axes = [fig.add_subplot(gs[i // ncol, i % ncol]) for i in range(len(order))]
    panel_density_lines(axes, solo, merged, order)
    fig.text(0.5, 0.975, a.title, ha="center", fontsize=13, color=INK, weight="bold")
    fig.text(0.5, 0.952,
             "A · Δnll vs kept link fraction (solo = solid, merged = dashed) — "
             "blue = code, orange = text",
             ha="center", fontsize=9.5, color=MUTED)
    # shared axis labels for panel A
    axes[0].set_ylabel("Δnll", fontsize=8, color=MUTED)
    for ax in axes[-ncol:]:
        ax.set_xlabel("keep frac", fontsize=8, color=MUTED)
    # legend for A (solo/merged) if merged present
    if merged:
        from matplotlib.lines import Line2D
        h = [Line2D([0], [0], color=MUTED, lw=2, marker="o", ms=4, label="solo"),
             Line2D([0], [0], color=MUTED, lw=2, ls="--", marker="s", ms=3.5, label="merged")]
        axes[ncol-1].legend(handles=h, fontsize=7, frameon=False, loc="upper left")

    # Panel B: cross-dataset law (spans full width, bottom row)
    axB = fig.add_subplot(gs[nrow+1, :])
    panel_cross_dataset(axB, solo, merged, eff)
    axB.set_title("B · Cross-dataset law: does density predict cross-doc benefit?  "
                  f"(effective-grant-density x-axis; solo model)",
                  fontsize=10, color=INK, pad=6, loc="left")

    for ext in ("png", "svg"):
        p = f"{a.out}.{ext}"
        fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="#fcfcfb")
        print("wrote", p)


if __name__ == "__main__":
    main()
