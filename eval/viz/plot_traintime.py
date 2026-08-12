#!/usr/bin/env python
"""Train-time vs eval-time graph-sparsity density figure.

For the 5 Phase-2 code datasets, plots each dataset's cross-doc benefit (Δnll,
community_pack, evaluated at full density) as a function of the TRAINING keep
fraction (solid) — the "does density help LEARNING" line — overlaid with the
Phase-1 EVAL-time line (dashed) for the same dataset: same trained-at-100% model,
Δ measured while withholding grants at eval ("does the model USE density").

The gap between them is the headline: eval-time only interpolates a fixed model
downward; train-time re-trains at each density. If train-time > eval-time at low
keep, training on a denser graph helps beyond what a fixed model's usage predicts.

Re-runnable: reads traintime_lines.json (assemble_traintime.py) + phase1_eval/.
Colour: blue #2a78d6 (validated). Small-multiples, one cell per dataset.

Usage:
  python -m eval.viz.plot_traintime \
    --traintime /fss-data/.../sparsity_scaling/traintime_lines.json \
    --evaltime  /fss-data/.../sparsity_scaling/phase1_eval \
    --out       /fss-data/.../sparsity_scaling/fig_traintime
"""
import argparse, glob, json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DISP={"thestack":"python (thestack)"}
C_TRAIN="#2a78d6"; C_EVAL="#eb6834"; INK="#0b0b0b"; MUTED="#52514e"; GRID="#e6e6e3"

def load_evaltime(d, ds):
    """eval-time edge line for one dataset from phase1_eval/{ds}.json."""
    f=os.path.join(d, f"{ds}.json")
    if not os.path.exists(f): return None
    j=json.load(open(f))
    edge=sorted([r for r in j["rows"] if r.get("keep_mode")=="edge"], key=lambda r:r["keep_frac"])
    if not edge: return None
    return [r["keep_frac"] for r in edge], [r["mean_delta"] for r in edge]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--traintime", required=True)
    ap.add_argument("--evaltime", required=True)
    ap.add_argument("--out", required=True)
    a=ap.parse_args()
    tt=json.load(open(a.traintime))
    order=sorted(tt, key=lambda d:-(tt[d].get("delta1") or 0))
    n=len(order); ncol=min(3,n); nrow=int(np.ceil(n/ncol))
    fig,axes=plt.subplots(nrow,ncol,figsize=(4.2*ncol,3.4*nrow),facecolor="#fcfcfb",squeeze=False)
    for idx,ds in enumerate(order):
        ax=axes[idx//ncol][idx%ncol]
        ax.axhline(0,color=GRID,lw=1,zorder=0)
        # train-time (solid blue)
        tk=tt[ds]["keeps"]; td=tt[ds]["delta"]
        ax.plot(tk,td,color=C_TRAIN,lw=2,marker="o",ms=5,zorder=3,label="train-time")
        # eval-time (dashed orange)
        ev=load_evaltime(a.evaltime, ds)
        if ev:
            ax.plot(ev[0],ev[1],color=C_EVAL,lw=2,ls="--",marker="s",ms=4,alpha=0.9,zorder=2,label="eval-time")
        ax.set_title(DISP.get(ds,ds),fontsize=10,color=INK,pad=3)
        ax.tick_params(labelsize=8,colors=MUTED)
        for s in ("top","right"): ax.spines[s].set_visible(False)
        for s in ("left","bottom"): ax.spines[s].set_color(GRID)
        if idx%ncol==0: ax.set_ylabel("Δnll (cross − doc_causal)",fontsize=8,color=MUTED)
        if idx//ncol==nrow-1: ax.set_xlabel("keep fraction",fontsize=8,color=MUTED)
    # hide unused axes
    for j in range(n,nrow*ncol): axes[j//ncol][j%ncol].set_visible(False)
    axes[0][0].legend(fontsize=8,frameon=False,loc="upper left")
    fig.suptitle("Graph-sparsity: train-time (solid) vs eval-time (dashed) density lines",
                 fontsize=13,color=INK,y=0.99)
    fig.text(0.5,0.955,"train-time = model RE-TRAINED at each keep-frac; eval-time = fixed 100%-model with grants withheld. "
             "Δ = community_pack cross−doc_causal.",ha="center",fontsize=8.5,color=MUTED)
    fig.tight_layout(rect=[0,0,1,0.945])
    for ext in ("png","svg"):
        fig.savefig(f"{a.out}.{ext}",dpi=150,bbox_inches="tight",facecolor="#fcfcfb")
        print("wrote",f"{a.out}.{ext}")

if __name__=="__main__":
    main()
