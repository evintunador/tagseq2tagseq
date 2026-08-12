#!/usr/bin/env python
"""Assemble the TRAIN-TIME density lines from Phase-2 eval JSONs.

Each phase2_traintime/{ds}_keep{K}.json is the community_pack eval (at FULL
density, keep=1.0 eval) of the model TRAINED at keep-fraction K. So the point
(train_keep=K, Δnll) is one point on that dataset's train-time density line.

Endpoints:
  train_keep=0.0  ==  the doc_causal arm (Δ≡0 by construction: no grants trained
                      or evaluated) — we anchor the line at (0,0).
  train_keep=1.0  ==  the solo cross_doc ENDPOINT (its recorded community_pack Δ).

Emits a JSON keyed by dataset: {keeps:[...], delta:[...]} matching what
plot_sparsity.load_sweep produces, so the same figure code can overlay it.

Usage:
  python -m eval.viz.assemble_traintime \
      --phase2 /fss-data/.../sparsity_scaling/phase2_traintime \
      --endpoints /fss-data/.../sparsity_scaling/phase1_solo_manifest.json \
      --out /fss-data/.../sparsity_scaling/traintime_lines.json
"""
import argparse, glob, json, os, re

CODE={"thestack","python","go","java","typescript","kotlin","rust","javascript","zig","dart"}

def endpoint_delta(run_dir):
    """community_pack experimental (cross_doc) mean_delta from an endpoint run's eval_results.json."""
    er=os.path.join(run_dir,"eval_results.json")
    if not os.path.exists(er): return None
    e=json.load(open(er))
    # prefer val_community experimental; fall back to plain experimental
    for k in ("community_pack_perplexity/experimental__val_community",
              "community_pack_perplexity/experimental",
              "community_pack_perplexity/baseline__val_community",
              "community_pack_perplexity/baseline"):
        if k in e and isinstance(e[k],dict) and e[k].get("mean_delta") is not None:
            return e[k]["mean_delta"]
    return None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--phase2", required=True)
    ap.add_argument("--endpoints", default=None, help="solo manifest json (run dir per dataset) for keep=1.0 endpoint")
    ap.add_argument("--out", required=True)
    a=ap.parse_args()

    # interior + any keep points from phase2 evals
    pts={}   # ds -> {keep_frac: delta}
    for f in sorted(glob.glob(os.path.join(a.phase2,"*.json"))):
        d=json.load(open(f))
        ds=d.get("dataset") or os.path.basename(f).split("_keep")[0]
        rows=d.get("rows",[])
        # the keep=1.0 edge row is this arm's full-density Δ
        r=next((x for x in rows if x.get("keep_mode")=="edge" and abs(x.get("keep_frac",-1)-1.0)<1e-9), None)
        if r is None and rows: r=rows[-1]
        if r is None: continue
        # train keep from filename
        km=re.search(r'_keep([0-9p]+)\.json$', os.path.basename(f))
        tk=float(km.group(1).replace("p",".")) if km else None
        if tk is None: continue
        pts.setdefault(ds,{})[tk]=r["mean_delta"]

    # keep=1.0 endpoint from the solo cross_doc run (if manifest given)
    if a.endpoints and os.path.exists(a.endpoints):
        for r in json.load(open(a.endpoints)):
            ds=r["dataset"]; run=r["run"]
            dl=endpoint_delta(run)
            if dl is not None:
                pts.setdefault(ds,{}).setdefault(1.0, dl)

    # anchor keep=0.0 at Δ=0 (doc_causal identity)
    out={}
    for ds,km in pts.items():
        km.setdefault(0.0, 0.0)
        keeps=sorted(km)
        out[ds]={"keeps":keeps,"delta":[km[k] for k in keeps],
                 "lo":[km[k] for k in keeps],"hi":[km[k] for k in keeps],
                 "delta1":km.get(1.0)}
    json.dump(out, open(a.out,"w"), indent=1)
    for ds in sorted(out):
        line=" ".join(f"{k:.2f}:{d:+.4f}" for k,d in zip(out[ds]["keeps"],out[ds]["delta"]))
        print(f"{ds:12s} {line}")
    print(f"\nwrote {a.out}  ({len(out)} datasets)")

if __name__=="__main__":
    main()
