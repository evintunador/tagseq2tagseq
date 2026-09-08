#!/usr/bin/env python
"""Traversal-time vs mask-time validation overlay (RE-RUNNABLE).

Does the cheap MASK-time edge-dropout (thin recorded grants, packing fixed) predict
REAL sparser-corpus training (TRAVERSAL-time: thin graph adjacency before packing, so
co-packing itself changes)? Plots train-time Δnll@eval=1.0 vs EFFECTIVE grant density
for both, on typescript. If the traversal points fall on the mask-time curve, mask-time
is a faithful proxy.

Reads:
  --traintime  traintime_lines.json          (mask-time line; key 'typescript')
  --traversal  phase2_traversal/             ({ds}_travkeep*.json, keep=1.0 row)
  --manifest   traversal_runs_manifest.json  (actual grants/pack per traversal arm + full)
Writes {out}.png/.svg.

  python -m eval.viz.plot_traversal_check \
    --traintime /fss-data/.../sparsity_scaling/traintime_lines.json \
    --traversal /fss-data/.../sparsity_scaling/phase2_traversal \
    --manifest  /fss-data/.../sparsity_scaling/traversal_runs_manifest.json \
    --out       /fss-data/.../sparsity_scaling/fig_traversal_check
"""
import argparse, glob, json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

C_MASK="#2a78d6"; C_TRAV="#eb6834"; INK="#0b0b0b"; MUTED="#52514e"; GRID="#e6e6e3"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--traintime", required=True)
    ap.add_argument("--traversal", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset", default="typescript")
    a=ap.parse_args()

    man=json.load(open(a.manifest))
    full_gpp=man["full_grants_per_pack"]
    # mask-time: keep_frac * full_gpp = effective density; delta from traintime_lines
    tl=json.load(open(a.traintime))[a.dataset]
    mask_x=[k*full_gpp for k in tl["keeps"]]; mask_y=tl["delta"]

    # traversal-time: actual grants/pack (manifest) vs keep=1.0-eval delta (json)
    trav=[]
    for tag,info in man["arms"].items():
        keeptag=tag.replace("travkeep","").replace(".","p")
        f=os.path.join(a.traversal, f'{a.dataset}_travkeep{keeptag}.json')
        if not os.path.exists(f): continue
        rows=[r for r in json.load(open(f))["rows"]
              if r.get("keep_mode")=="edge" and float(r["keep_frac"])>=1.0]
        if rows:
            trav.append((info["train_grants_per_pack"], rows[0]["mean_delta"],
                         rows[0].get("delta_ci_low"), rows[0].get("delta_ci_high")))
    # shared endpoints anchor both curves: keep=1.0 (full_gpp) and keep=0 (Δ=0)
    trav.append((full_gpp, mask_y[-1], None, None))
    trav.sort()

    fig,ax=plt.subplots(figsize=(7.5,5),facecolor="#fcfcfb")
    ax.axhline(0,color=GRID,lw=1,zorder=0)
    ax.plot(mask_x,mask_y,"-o",color=C_MASK,zorder=3,label="mask-time (grants thinned, packing fixed)")
    tx=[t[0] for t in trav]; ty=[t[1] for t in trav]
    yerr_lo=[t[1]-t[2] if t[2] is not None else 0 for t in trav]
    yerr_hi=[t[3]-t[1] if t[3] is not None else 0 for t in trav]
    ax.errorbar(tx,ty,yerr=[yerr_lo,yerr_hi],fmt="s",color=C_TRAV,ms=8,capsize=3,
                zorder=4,label="traversal-time (real sparser corpus)")
    ax.set_xlabel("effective training grants / pack",fontsize=10,color=MUTED)
    ax.set_ylabel("train-time Δnll @ eval keep=1.0",fontsize=10,color=MUTED)
    ax.set_title(f"Mask-time vs traversal-time dose-response ({a.dataset})\n"
                 "traversal points on the mask-time curve ⇒ mask-time is a faithful proxy",
                 fontsize=11,color=INK)
    for s in ("top","right"): ax.spines[s].set_visible(False)
    for s in ("left","bottom"): ax.spines[s].set_color(GRID)
    ax.tick_params(labelsize=9,colors=MUTED); ax.legend(fontsize=8.5,frameon=False,loc="lower right")
    for ext in ("png","svg"):
        fig.savefig(f"{a.out}.{ext}",dpi=150,bbox_inches="tight",facecolor="#fcfcfb")
        print("wrote",f"{a.out}.{ext}")

if __name__=="__main__":
    main()
