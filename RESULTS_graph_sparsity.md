# Graph-Sparsity Scaling Law — Phase 1 (eval-time) results

**Question:** how much does the cross-doc attention benefit scale with graph link
density, and can we extrapolate the payoff of a *denser* corpus than we can build?

**Instrument:** seeded per-edge subsample of the resolved `link_to_target` grant
map, applied at mask-build time (packing held fixed). keep=0 ≡ doc_causal exactly;
keep=1 = full density. Metric = within-model Δnll (doc_causal − cross_doc) on each
dataset's `val_community` split (Option B graph-edge grants). PRELIMINARY —
eval-time only (fixed trained model); train-time is Phase 2.

## Provenance / how to reproduce the evidence

All results + manifests + analysis under
`/fss-data/evin_t/tagseq2tagseq_artifacts/sparsity_scaling/`:
- `phase1_eval/{ds}.json` — solo per-dataset density lines (11 datasets).
- `phase1_eval_merged3p9b/{ds}.json` — merged-3.9B model per-source density lines.
- `phase1_solo_manifest.json` / `phase1_merged_manifest.json` — exact checkpoints used.
- `effective_density.json` + `effective_density.py` — the effective-grant-density
  x-axis analysis (CPU; rebuilds the same BFS community packs).
- `inherent_density.json` — raw graph out-degree per split (the misleading x-axis).

Code (branch `sparsity-scaling-law`, commit "Graph-sparsity scaling law: Phase-1…"):
- `eval/scoring.py::subsample_link_to_target` — the instrument (edge/node modes).
- `eval/perplexity.py::run_community_pack_perplexity` — keep_frac/keep_seed/keep_mode.
- `eval/sparsity_sweep.py` — single-load per-checkpoint sweep driver.
- `scripts/sparsity_phase1_sweep.sh` — staggered per-GPU sweep launcher.
- `tests/eval/test_link_subsample.py` — 17 unit tests.
- `data/epoch_precompute.py` + `precompute_epochs.py --keep-frac` — TRAIN-TIME knob.

### Checkpoints (all cross_doc_link, ~3.9B tok = ~15k steps @ 262144, BFS)

Solo (config `configs/{ds}_cross_doc.yaml`; run dir under `/fss/evin_t/tagseq2tagseq/runs/`):

| dataset | run dir | step | dataset (val_community) |
|---------|---------|-----:|----|
| wiki_merged | run_20260717_173951_071952 | 14250 | wiki_merged |
| thestack | run_20260720_063128_690228 | 14750 | thestack |
| arxiv | run_20260718_003640_349805 | 12750 | arxiv |
| go | run_20260720_081428_847881 | 13000 | go |
| java | run_20260720_084228_887159 | 15000 | java |
| typescript | run_20260721_225606_146404 | 14000 | typescript |
| kotlin | run_20260722_181228_995658 | 14750 | kotlin |
| rust | run_20260722_172826_905950 | 14000 | rust |
| javascript | run_20260723_044101_595472 | 14000 | javascript |
| zig* | run_20260722_181852_210934 | 1500 | zig (*undertrained/capped) |
| dart | run_20260722_172933_276517 | 14750 | dart |

Merged 3.9B (config `configs/merged_v2_3p9b_cross_doc.yaml`, epoch_3p9b):
`run_20260730_183342_811412` — evaluated per-source against each source's val_community.

## Solo density lines (edge-mode, Δnll vs kept fraction)

| dataset | eff grants/pack | Δ@.25 | Δ@.50 | Δ@.75 | Δ@1.0 |
|---------|----------------:|------:|------:|------:|------:|
| javascript | 67.1 | +0.018 | +0.033 | +0.042 | **+0.048** |
| typescript | 48.8 | +0.013 | +0.024 | +0.032 | **+0.040** |
| rust | 21.1 | +0.008 | +0.014 | +0.020 | **+0.025** |
| dart | 26.8 | +0.008 | +0.014 | +0.020 | **+0.025** |
| thestack | 18.7 | +0.006 | +0.011 | +0.016 | **+0.020** |
| kotlin | 14.2 | +0.005 | +0.008 | +0.010 | **+0.011** |
| go | 5.4 | +0.001 | +0.001 | +0.002 | +0.003 |
| arxiv | 1.5 | +0.000 | +0.001 | +0.001 | +0.001 (flat) |
| java | 3.1 | +0.000 | +0.000 | −0.000 | −0.000 (flat) |
| zig* | 5.1 | −0.001 | −0.002 | −0.006 | −0.007 |
| wiki_merged | 31.8 | −0.006 | −0.011 | −0.016 | −0.020 |

Every line is near-perfectly LINEAR in kept fraction (no saturation to 100%),
pinned to Δ=0 at keep=0 (doc_causal identity — instrument verified on real ckpts).
Node-mode ≈ edge-mode everywhere (structure-insensitive: edge COUNT matters, not
which hubs).

## The cross-dataset law — x-axis matters

| density x-axis | Pearson r(solo Δ@1.0 vs x) |
|----------------|:--------------------------:|
| raw graph out-degree (all 11) | **−0.27** (misleading null) |
| effective grants/pack (all 11) | **+0.71** |
| effective grants/pack (drop zig) | +0.68 |
| **effective grants/pack (code only, n=8)** | **+0.97** |
| val_loss (convergence confound) | −0.63 |

Raw out-degree counts edges the mask never used (targets not co-packed). Measuring
the effective grant density the mask actually saw reveals a strong near-linear
cross-dataset law FOR CODE (r=0.97). It extrapolates above today's densest corpus
(javascript at 67 grants/pack).

## NEW (2026-08-04): the merged 3.9B model benefits on EVERY source

The merged model (trained jointly on all 11 sources) shows a POSITIVE cross-doc Δ
on all 11 — including the datasets that were flat/negative solo:

| dataset | SOLO Δ@1.0 | MERGED Δ@1.0 |
|---------|-----------:|-------------:|
| wiki_merged | −0.020 | **+0.159** |
| thestack | +0.020 | **+0.076** |
| javascript | +0.048 | +0.059 |
| typescript | +0.040 | +0.057 |
| dart | +0.025 | +0.031 |
| rust | +0.025 | +0.030 |
| kotlin | +0.011 | +0.021 |
| go | +0.003 | +0.010 |
| zig | −0.007 | +0.005 |
| java | −0.000 | +0.005 |
| arxiv | +0.001 | +0.004 |

wiki flips hardest (−0.020 → +0.159) and thestack ~4×. r(solo vs merged Δ@1.0) =
−0.10 — the merged benefit does NOT track the solo ranking. Cross-dataset density
law is weaker on the merged model (r=+0.52 all / +0.66 code) because the two
biggest movers (wiki, thestack) are exactly the ones diversity training rescued.

**Interpretation (preliminary):** joint multi-source training makes the model
*use* cross-doc grants far more — most dramatically on the text/hyperlink sources
that a solo model ignored. This is the diversity thesis and the density thesis
reinforcing each other, and it partially DE-CONFOUNDS the earlier "text doesn't
benefit" story: the merged text arms benefit strongly, so the solo text null was
substantially a solo-training/convergence artifact, not an intrinsic property of
hyperlinks. (Caveat: merged per-source Δ is scored on the same val_community splits
with Option-B grants; the merged model also simply trained longer/differently.)

## Caveats
1. Eval-time only — measures how a FIXED model USES density; Phase 2 (train-time)
   tests whether a model LEARNS to exploit density it's trained on.
2. Solo code-vs-text split is partly a convergence confound (r(Δ,val_loss)=−0.63;
   text models least converged) — and the merged result supports that reading.
3. Within-dataset slope is the most trustworthy signal; cross-dataset absolute-Δ
   law lives on the effective-grants-per-pack axis, not raw out-degree.

## Status / next
- Phase-1 solo + merged sweeps DONE. Train-time instrument built + validated
  (zig: keep=0.5→49% grants, keep=0.0→0, kv_block_count monotone, packing fixed).
- Phase 2 (train-time) NOT launched — needs free nodes; candidates = strong code
  datasets + wiki/thestack (biggest merged movers). Precompute subsampled epochs
  (25/50/75) via SLURM CPU jobs, then train the interior arms (~2h/run @ world=8).
