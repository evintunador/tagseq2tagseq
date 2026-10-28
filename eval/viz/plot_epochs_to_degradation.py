#!/usr/bin/env python
"""Figure for the epochs-to-degradation experiment: held-out NLL vs. repetition
count for doc_causal vs. cross_doc_link on java, fresh-mode schedules, corrected
(fraction-of-run) warmup.

Reads eval/viz/epochs_to_degradation_java_runs.json, a mask -> epoch -> run_dir
manifest, and for each run loads memorization_gap_eos_best.json (written by
`python -m eval.memorization --checkpoint .../checkpoints/best_model.pt --mode gap
--max-docs 9000 --layout-policy eos`). Uses best_model.pt, NOT latest.pt: training
val loss is noisy checkpoint-to-checkpoint (~0.02-0.04 swings throughout a run, not
just near the end), so latest.pt can land on an arbitrary local uptick and make a
fine otherwise-monotonic arm look like it degraded. best_model.pt tracks the
historical minimum and is the noise-robust choice for a cross-epoch-count
comparison. Plots val_nll (solid, primary) and train_nll (dashed, muted) per mask
on one axis. Re-runnable: add a run dir to the manifest (e.g. cross_doc_link's "16"
once that arm finishes and its gap probe has been run) and re-invoke.

Colour: amber #c7752f (doc_causal) / teal #0e9488 (cross_doc_link) — validated,
CVD dE 11.2 (protan), normal-vision dE 21.5, both >= floor (validate_palette.py).
Same pair as the tracking dashboard, for continuity across artifacts.

Usage:
    python -m eval.viz.plot_epochs_to_degradation \
        --manifest eval/viz/epochs_to_degradation_java_runs.json \
        --runs-root /fss-data/evin_t/tagseq2tagseq_artifacts/runs \
        --out paper/figures/epochs_to_degradation_java
"""
import argparse, json, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C_DC, C_CDL = "#c7752f", "#0e9488"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e6e3"
DISPLAY = {"doc_causal": "doc_causal", "cross_doc_link": "cross_doc_link"}
COLOR = {"doc_causal": C_DC, "cross_doc_link": C_CDL}


def load_series(manifest_path, runs_root, lr_key="muon_lr_0.003"):
    """mask -> sorted list of (epoch, train_nll, val_nll), skipping missing runs."""
    manifest = json.load(open(manifest_path))[lr_key]
    out = {}
    for mask in ("doc_causal", "cross_doc_link"):
        pts = []
        for epoch_str, run_dir in manifest.get(mask, {}).items():
            if not run_dir:
                continue
            gap_path = os.path.join(runs_root, run_dir, "memorization_gap_eos_best.json")
            if not os.path.exists(gap_path):
                print(f"  skip {mask} e{epoch_str}: no memorization_gap_eos.json at {gap_path}")
                continue
            g = json.load(open(gap_path))["perplexity_gap"]
            pts.append((int(epoch_str), g["train"]["mean_nll"], g["val"]["mean_nll"]))
        pts.sort()
        out[mask] = pts
    return out


def plot(series, out_stem, series_alt=None):
    fig, ax = plt.subplots(figsize=(5.6, 4.2), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    all_epochs = sorted({e for ser in (series, series_alt or {}) for pts in ser.values() for e, _, _ in pts})

    for mask in ("doc_causal", "cross_doc_link"):
        pts = series[mask]
        if not pts:
            continue
        color = COLOR[mask]
        epochs = [e for e, _, _ in pts]
        train_y = [t for _, t, _ in pts]
        val_y = [v for _, _, v in pts]

        ax.plot(epochs, val_y, color=color, linewidth=2, marker="o",
                markersize=8, markeredgewidth=0, zorder=3)
        ax.plot(epochs, train_y, color=color, linewidth=1.5, linestyle=(0, (3, 2)),
                marker="o", markersize=5, markeredgewidth=0, alpha=0.55, zorder=2)

        # Direct label at the last available point of the val (primary) line.
        lx, ly = epochs[-1], val_y[-1]
        dy = -0.03 if mask == "doc_causal" else 0.03
        ax.annotate(DISPLAY[mask], (lx, ly), xytext=(6, dy * 300),
                    textcoords="offset points", color=color, fontsize=10,
                    fontweight="bold", va="center")

    # Second series (muon_lr 0.002): val only, thin line, hollow markers.
    for mask, pts in (series_alt or {}).items():
        if not pts:
            continue
        ax.plot([e for e, _, _ in pts], [v for _, _, v in pts], color=COLOR[mask], linewidth=1.1,
                marker="o", markersize=6, markerfacecolor="#fcfcfb", markeredgewidth=1.4,
                alpha=0.9, zorder=2)

    # Line style (solid vs dashed) is a separate channel from color (mask identity),
    # so it gets its own neutral-ink legend rather than relying on the axis label.
    style_handles = [
        plt.Line2D([], [], color=INK, linewidth=2, marker="o", markersize=7,
                   markeredgewidth=0, label="held-out (val)"),
        plt.Line2D([], [], color=INK, linewidth=1.5, linestyle=(0, (3, 2)),
                   marker="o", markersize=4.5, markeredgewidth=0, alpha=0.55, label="train"),
        plt.Line2D([], [], color=INK, linewidth=1.1, marker="o", markersize=6,
                   markerfacecolor="#fcfcfb", markeredgewidth=1.4, label="val, muon_lr 0.002"),
    ]
    ax.legend(handles=style_handles, loc="upper right", frameon=False,
              fontsize=9, labelcolor=MUTED, handlelength=2.4, borderaxespad=0.2)

    ax.set_xlabel("training epochs (data repetitions)", fontsize=10, color=MUTED)
    ax.set_ylabel("held-out NLL", fontsize=10, color=MUTED)
    ax.set_xticks(all_epochs)
    ax.tick_params(colors=MUTED, labelsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    fig.suptitle("java: held-out NLL vs. data repetitions (muon_lr 0.003 unless noted)",
                 fontsize=10.5, color=INK, y=0.99)

    fig.tight_layout()
    p = f"{out_stem}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="#fcfcfb")
    print(f"wrote {p}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=os.path.join(os.path.dirname(__file__),
                     "epochs_to_degradation_java_runs.json"))
    ap.add_argument("--runs-root", default="/fss-data/evin_t/tagseq2tagseq_artifacts/runs")
    ap.add_argument("--out", default="paper/figures/epochs_to_degradation_java",
                     help="Output path stem (writes .png, matching the other paper figures).")
    a = ap.parse_args()

    series = load_series(a.manifest, a.runs_root)
    series_alt = load_series(a.manifest, a.runs_root, "muon_lr_0.002")
    for tag, ser in (("0.003", series), ("0.002", series_alt)):
        for mask, pts in ser.items():
            print(f"lr {tag} {mask}: {len(pts)} point(s) -> epochs {[e for e, _, _ in pts]}")
    plot(series, a.out, series_alt)


if __name__ == "__main__":
    main()
