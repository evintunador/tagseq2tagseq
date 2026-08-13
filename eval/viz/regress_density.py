#!/usr/bin/env python
"""Cross-dataset graph-sparsity regression (RE-RUNNABLE).

Fits the "scaling law" — cross-doc benefit (Δnll) vs graph density — pooling all
(dataset, keep_fraction) points, on the honest x-axis: EFFECTIVE grants/pack the
mask saw, scaled by keep_frac (a keep-K arm was trained on ~K× the full grant
density). Reports both the TRAIN-TIME fit and, for comparison, the EVAL-TIME fit,
code-only and all-datasets, with Pearson r + OLS slope/intercept.

Everything is read from files, so re-run after any eval refresh:
  python -m eval.viz.regress_density \
    --traintime  /fss-data/.../sparsity_scaling/traintime_lines.json \
    --evaltime   /fss-data/.../sparsity_scaling/phase1_eval \
    --effdensity /fss-data/.../sparsity_scaling/effective_density.json \
    --out        /fss-data/.../sparsity_scaling/regression

Writes {out}.json (all fit stats + per-point table) and {out}.png (scatter+fits).
"""
import argparse, glob, json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

CODE={"thestack","python","go","java","typescript","kotlin","rust","javascript","zig","dart"}
C_CODE="#2a78d6"; C_TEXT="#eb6834"; INK="#0b0b0b"; MUTED="#52514e"; GRID="#e6e6e3"

def ols(x, y):
    x=np.asarray(x,float); y=np.asarray(y,float)
    if len(x)<2: return dict(n=len(x), r=float("nan"), slope=float("nan"), intercept=float("nan"))
    b,a=np.polyfit(x,y,1)
    r=float(np.corrcoef(x,y)[0,1]) if len(x)>1 else float("nan")
    return dict(n=int(len(x)), r=r, slope=float(b), intercept=float(a))

def load_eval_lines(d):
    """dataset -> {keep_frac: delta} from phase1_eval/{ds}.json (edge mode)."""
    out={}
    for f in glob.glob(os.path.join(d,"*.json")):
        j=json.load(open(f)); ds=j["dataset"]
        for r in j["rows"]:
            if r.get("keep_mode")=="edge":
                out.setdefault(ds,{})[float(r["keep_frac"])]=r["mean_delta"]
    return out

def build_points(lines, eff):
    """(ds, keep, x=eff_grants/pack*keep, y=delta) for keep>0 (keep=0 has x=0, Δ=0 trivially)."""
    pts=[]
    for ds, L in lines.items():
        if ds not in eff: continue
        gpp=eff[ds]["mean_grants_per_pack"]
        # lines may be {keeps:[],delta:[]} (traintime) or {keep:delta} (eval)
        if isinstance(L, dict) and "keeps" in L:
            items=list(zip(L["keeps"], L["delta"]))
        else:
            items=list(L.items())
        for keep, dl in items:
            keep=float(keep)
            if keep<=0: continue
            pts.append((ds, keep, gpp*keep, dl))
    return pts

def fit_report(pts, label):
    code=[(p[2],p[3]) for p in pts if p[0] in CODE]
    allp=[(p[2],p[3]) for p in pts]
    fc=ols([c[0] for c in code],[c[1] for c in code])
    fa=ols([c[0] for c in allp],[c[1] for c in allp])
    print(f"\n[{label}]  x = effective grants/pack × keep")
    print(f"  code-only (n={fc['n']}):  r={fc['r']:+.3f}  slope={fc['slope']:+.5f}  intercept={fc['intercept']:+.5f}")
    print(f"  all-data  (n={fa['n']}):  r={fa['r']:+.3f}  slope={fa['slope']:+.5f}  intercept={fa['intercept']:+.5f}")
    return {"code":fc, "all":fa}

def main():
    ap=argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traintime", required=True)
    ap.add_argument("--evaltime", required=True)
    ap.add_argument("--effdensity", required=True)
    ap.add_argument("--out", required=True)
    a=ap.parse_args()
    eff=json.load(open(a.effdensity))
    tt=json.load(open(a.traintime))
    ev=load_eval_lines(a.evaltime)
    tt_pts=build_points(tt, eff)
    ev_pts=build_points(ev, eff)

    res={"traintime":fit_report(tt_pts,"TRAIN-TIME"),
         "evaltime":fit_report(ev_pts,"EVAL-TIME"),
         "traintime_points":[dict(ds=p[0],keep=p[1],x_eff_density=p[2],delta=p[3]) for p in tt_pts],
         "evaltime_points":[dict(ds=p[0],keep=p[1],x_eff_density=p[2],delta=p[3]) for p in ev_pts]}
    json.dump(res, open(f"{a.out}.json","w"), indent=1)

    # scatter + code fit line, train (blue) vs eval (orange)
    fig,ax=plt.subplots(figsize=(8,5.5),facecolor="#fcfcfb")
    ax.axhline(0,color=GRID,lw=1,zorder=0)
    for pts,col,lab,fit in [(tt_pts,C_CODE,"train-time",res["traintime"]["code"]),
                            (ev_pts,C_TEXT,"eval-time",res["evaltime"]["code"])]:
        cx=[p[2] for p in pts if p[0] in CODE]; cy=[p[3] for p in pts if p[0] in CODE]
        tx=[p[2] for p in pts if p[0] not in CODE]; ty=[p[3] for p in pts if p[0] not in CODE]
        ax.scatter(cx,cy,s=42,color=col,zorder=3,edgecolor="white",linewidth=0.8,label=f"{lab} (code)")
        ax.scatter(tx,ty,s=42,color=col,marker="x",zorder=3,label=f"{lab} (text)")
        if len(cx)>=2 and not np.isnan(fit["slope"]):
            xr=np.linspace(min(cx),max(cx),50)
            ax.plot(xr,fit["intercept"]+fit["slope"]*xr,color=col,ls=":",lw=1.6,
                    label=f"{lab} code fit r={fit['r']:.2f}")
    ax.set_xlabel("effective training density  (grants/pack × keep_frac)",fontsize=10,color=MUTED)
    ax.set_ylabel("Δnll (cross − doc_causal)",fontsize=10,color=MUTED)
    ax.set_title("Cross-dataset graph-sparsity scaling law",fontsize=13,color=INK)
    for s in ("top","right"): ax.spines[s].set_visible(False)
    for s in ("left","bottom"): ax.spines[s].set_color(GRID)
    ax.tick_params(labelsize=9,colors=MUTED); ax.legend(fontsize=8,frameon=False)
    for ext in ("png","svg"):
        fig.savefig(f"{a.out}.{ext}",dpi=150,bbox_inches="tight",facecolor="#fcfcfb")
    print(f"\nwrote {a.out}.json / .png / .svg")

if __name__=="__main__":
    main()
