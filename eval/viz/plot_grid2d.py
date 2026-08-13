#!/usr/bin/env python
"""2D train-keep × eval-keep density heatmaps (the "elevation map").

Reads grid2d/{ds}_train{K}.json (each = one trained-at-K checkpoint evaluated across
the eval-keep axis) and renders, per dataset, a heatmap of Δnll (community_pack
cross−doc_causal) over (y=train keep, x=eval keep). Diagonal-ish reading:
  - a COLUMN (fixed eval keep) = how training density matters at that inference density
  - a ROW (fixed train keep)   = how inference density matters for that trained model
  - the earlier train-time line = the eval_keep=1.0 column; the earlier eval-time
    line = the train_keep=1.0 row.

Re-runnable: point --grid at the grid2d dir; re-render after any eval refresh.

Usage:
  python -m eval.viz.plot_grid2d --grid /fss-data/.../sparsity_scaling/grid2d \
      --out /fss-data/.../sparsity_scaling/fig_grid2d
"""
import argparse, glob, json, os, re
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

DISP={"thestack":"python (thestack)"}
CODE={"thestack","javascript","typescript","rust","dart"}

def load(gridpath):
    # dataset -> {train_keep: {eval_keep: delta}}
    data={}
    for f in glob.glob(os.path.join(gridpath,"*_train*.json")):
        j=json.load(open(f)); ds=j["dataset"]
        m=re.search(r'_train([0-9p]+)\.json$', os.path.basename(f))
        tk=float(m.group(1).replace("p","."))
        for r in j["rows"]:
            if r.get("keep_mode")!="edge": continue
            data.setdefault(ds,{}).setdefault(tk,{})[float(r["keep_frac"])]=r["mean_delta"]
    return data

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--grid", required=True)
    ap.add_argument("--out", required=True)
    a=ap.parse_args()
    data=load(a.grid)
    order=[d for d in ["javascript","typescript","rust","dart","thestack","wiki_merged"] if d in data]
    order+=[d for d in data if d not in order]
    n=len(order); ncol=min(3,n); nrow=int(np.ceil(n/ncol))
    fig,axes=plt.subplots(nrow,ncol,figsize=(4.6*ncol,3.9*nrow),facecolor="#fcfcfb",squeeze=False)
    # shared symmetric color scale across code (diverging), so panels compare
    allv=[v for ds in data for tk in data[ds] for v in data[ds][tk].values()]
    vmax=max(abs(min(allv)),abs(max(allv))) if allv else 0.05
    for idx,ds in enumerate(order):
        ax=axes[idx//ncol][idx%ncol]
        tks=sorted(data[ds]); eks=sorted({e for tk in data[ds] for e in data[ds][tk]})
        M=np.full((len(tks),len(eks)),np.nan)
        for i,tk in enumerate(tks):
            for j,ek in enumerate(eks):
                if ek in data[ds][tk]: M[i,j]=data[ds][tk][ek]
        im=ax.imshow(M,origin="lower",cmap="RdBu_r",vmin=-vmax,vmax=vmax,aspect="auto")
        ax.set_xticks(range(len(eks))); ax.set_xticklabels([f"{e:g}" for e in eks],fontsize=8)
        ax.set_yticks(range(len(tks))); ax.set_yticklabels([f"{t:g}" for t in tks],fontsize=8)
        ax.set_title(DISP.get(ds,ds),fontsize=10)
        if idx//ncol==nrow-1: ax.set_xlabel("eval keep",fontsize=9)
        if idx%ncol==0: ax.set_ylabel("train keep",fontsize=9)
        # annotate cells
        for i in range(len(tks)):
            for j in range(len(eks)):
                if not np.isnan(M[i,j]):
                    ax.text(j,i,f"{M[i,j]:+.3f}",ha="center",va="center",fontsize=6.5,
                            color="black" if abs(M[i,j])<vmax*0.55 else "white")
    for k in range(n,nrow*ncol): axes[k//ncol][k%ncol].set_visible(False)
    fig.colorbar(im, ax=axes, shrink=0.6, label="Δnll (cross − doc_causal)")
    fig.suptitle("Graph-sparsity 2D grid: Δnll over (train keep × eval keep)",fontsize=13,y=0.99)
    for ext in ("png","svg"):
        fig.savefig(f"{a.out}.{ext}",dpi=150,bbox_inches="tight",facecolor="#fcfcfb")
        print("wrote",f"{a.out}.{ext}")

if __name__=="__main__":
    main()
