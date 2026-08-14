# Handoff: does cross-doc attention help Wikipedia-trained models? (3.9B solo wiki)

**Status:** open question. Solo-wiki cross-doc benefit is small but consistently
**negative**. A neutral from-scratch audit says it's real (not a harness/grant bug)
and **fit-dependent**. Another model should verify and decide whether to re-run.

**Metric throughout:** `community_pack_perplexity` on the dataset's `val_community`
split — score packed docs under the `cross_doc_link` mask vs the `doc_causal` mask,
report **Δnll = mean_nll(doc_causal) − mean_nll(cross_doc)**. Positive ⇒ cross-doc
attention helps; negative ⇒ it hurts. (`eval/scoring.py::link_to_target_from_graph_edges`
= "Option B": grants formed from the graph's `outgoing_identifiers`, not text detection.)

---

## 1. The finding

Solo Wikipedia models (trained ONLY on wiki, ~3.9B tokens ≈ 14k steps @ 262144
tok/step, `cross_doc_link` mask, VE-off recipe `configs/wiki_crossdoc_best.yaml`
adapted to `configs/sparsity/wiki_merged_keep*_cdl.yaml`): cross-doc **hurts** at
every training-graph density, monotone toward zero as density rises, never crosses
positive. Δ measured at full eval density (keep=1.0), 500 packs, val_community:

| train keep | Δnll | val_loss | run dir (runs/) |
|---|---|---|---|
| 0.25 | −0.0329 | 2.4402 | run_20260811_202735_269652 |
| 0.50 | −0.0284 | 2.4315 | run_20260812_092936_075333 |
| 0.75 | −0.0237 | 2.4308 | run_20260812_092534_042990 |
| 1.00 | −0.0209 | 2.4287 | run_20260812_133752_064352 |

(keep=0 ≡ doc_causal ≡ Δ=0 by construction — the doc_causal endpoint was NOT
separately evaluated; keep=0 is an anchor, not a measurement.)

An independent neutral agent (zero priors) separately measured a different solo
wiki_merged cross_doc run (`run_20260718_221959_635150`, step 14000) at Δ=**−0.0223**
[CI −0.0241,−0.0205] on wiki_merged (n=198) and −0.0351 on simplewiki — matching.

## 2. Why it's (probably) NOT a bug — verified by the neutral audit

- **Harness sound:** keep=0 → Δ=0.0000 exactly in every sweep (cross arm ≡ doc_causal).
- **Grants fire abundantly on wiki:** replaying the eval packer over 200 val_community
  packs: wiki_merged = 31.7 grant-edges/pack, ~198/200 packs have ≥1 grant — *denser*
  than healthy code refs (rust 22, typescript 47). So the negative is NOT grants
  failing to fire or a train-vs-eval grant mismatch.
- **CIs tight, sign stable** across models, steps, and both wiki eval sets.

## 3. Mechanism hypothesis (fit-dependent) — the thing to verify

Across all wiki evals the sign tracks **model fit (base nll)**, not solo-vs-merged:
the better a model already predicts wiki text, the more cross-doc attention hurts;
only an *under-fit* model nets benefit from linked context. Evidence the agent found
(NOTE: the merged points below are from KNOWN-BUGGY models — see §4 — treat as
suggestive only): an under-fit merged 3.9B model (wiki base nll 4.02) had Δ=+0.164,
while a better-fit merged 16B (base nll 2.73) flipped to −0.036. Consistent with the
solo runs (well-fit, val 2.43, all negative) and with the Phase-1 cross-dataset
r(Δ, val_loss) = −0.63. Reading: **cross-doc/linked context is useful scaffolding for
a weak LM and becomes noise for a strong one** — plausibly specific to "soft" link
types (topical hyperlinks) vs "hard" ones (code imports, which stay net-positive even
when well-fit).

## 4. CRITICAL caveats (read before trusting anything)

- **ALL `merged_all_v2` models are BUGGY and USELESS — do not cite them.** The merged
  dataloader served datasets sequentially instead of interleaving them across
  training. Every merged/per-source/diversity number (incl. the +0.164 above and any
  "diversity rescues wiki" claim) is invalid. They are being re-run. Only SOLO
  wiki models are trustworthy here.
- **keep=0 is an anchor (Δ=0), not measured.** A true keep=0 point = eval the
  `doc_causal` wiki endpoint under the cross_doc mask (may need the doc_causal ckpt
  to have a cross_doc creator; verify it doesn't KeyError).
- **Only `simplewiki` and `wiki_merged` have splits/val_community** and are evaluable.
  There is **no long-trained solo `simplewiki`** checkpoint (only 311-step smokes,
  val~13.6 — garbage). Genuine solo-wiki = the `wiki_merged` runs above.
- Sample size: 500 packs (agent used 200), single seed. Signs are robust; widen
  seeds/packs to tighten.

## 5. Datasets (edge density, from the audit)

| dataset (pretokenized_datasets/) | nodes | splits? | val_community resolved-edges/node |
|---|---|---|---|
| `wiki_merged` (8 dumps) | 9.65M | yes | 3.11 |
| `simplewiki` | 282k | yes | 7.28 |
| `wiki_enwiki{books,news,quote,source,versity,voyage}`, `wiki_enwiktionary`, `wiki_simplewiki` | various | **no splits** | not evaluable as-is |

All in `/fss-data/evin_t/tagseq2tagseq_artifacts/pretokenized_datasets/`. Training
epoch schedules under `.../schedules/wiki_merged_bfs/epoch_{0..3}` (full density) and
`.../sparsity_scaling/schedules/wiki_merged_bfs_keep{0p25,0p5,0p75,1p0}/` (subsampled).

## 6. How to reproduce / re-run

Env: `source /fss/evin_t/tagseq2tagseq/.venv/bin/activate`. Worktree with the eval
tooling: `/fss/evin_t/tagseq2tagseq-sparsity` (branch `sparsity-scaling-law`).

**Measure cross-doc Δ for a wiki checkpoint** (single-GPU; use a free local GPU or a
SLURM job — bad nodes GPU-495/954/943/749 have been flaky, GPU-943/749 drained):
```
CUDA_VISIBLE_DEVICES=<n> python -m eval.sparsity_sweep \
  --checkpoint runs/<run>/checkpoints/best_model.pt \
  --dataset /fss-data/evin_t/tagseq2tagseq_artifacts/pretokenized_datasets/wiki_merged \
  --split val_community --max-packs 500 --keep-fracs 0,1.0 --modes edge --seeds 0 \
  --dataset-tag wiki --output /tmp/wiki_probe.json
```
Output rows carry `mean_delta`, `mean_nll_cross_doc`, `mean_nll_baseline`, `n_packs`.
keep=0 must give delta≈0 (sanity). `keep-fracs 0,0.25,0.5,0.75,1.0` gives the full
eval-density sweep.

**Re-train a solo wiki arm** (if desired): configs `configs/sparsity/wiki_merged_keep{K}_cdl.yaml`
(recipe = `wiki_crossdoc_best.yaml`: muon_lr=0.003, wd=0.3, VE-off, 4 epochs). Launch:
`python launch_slurm.py --nodes 1 --gpus-per-node 8 --time 96:00:00 --exclude GPU-749,GPU-954,GPU-495,GPU-943 --config <cfg> [--resume-from <latest.pt>]`.
NOTE: pinned `max_optimizer_steps=14507` slightly exceeds 4-epoch data capacity
(~14436) — runs error "epoch dirs exhausted" trying a 5th epoch; the ckpts reach
~14250 which is the natural data-exhaustion point (== the reference endpoint). Use
`--train_loop.max_optimizer_steps 14400` (or leave null) to avoid the error.

## 7. Suggested next steps for the investigating model

1. **Confirm fit-dependence cleanly:** train (or find) solo-wiki checkpoints at
   several DIFFERENT fit levels (e.g. checkpoints at step 2k/6k/14k of the SAME run,
   or different LR) and plot Δ vs base_nll. If Δ→positive as base_nll rises, the
   fit-dependence hypothesis holds; find the crossover.
2. **True keep=0 point:** eval the doc_causal wiki endpoint under cross_doc mask.
3. **Re-test on FIXED merged models** once the interleaved-dataloader re-runs land —
   does a correctly-trained merged model change the wiki story? (Do NOT use current
   merged ckpts.)
4. **Widen packs/seeds** (max-packs 500 + seeds 0,1,2) to tighten CIs.
5. Consider whether the wiki graph edges (redirect-resolved hyperlinks) are the right
   "link" signal, or whether a stronger wiki link type would flip the sign.

## 8. Artifacts
- Train-time solo-wiki Δ (500-pack): `/fss-data/.../sparsity_scaling/phase2_traintime/wiki_merged_keep*.json`
- Assembled line: `/fss-data/.../sparsity_scaling/traintime_lines.json` (key `wiki_merged`)
- Neutral audit temp files: `/fss/evin_t/.claude/jobs/465d9eaf/tmp/wiki_probe/`
- Eval tool: `eval/sparsity_sweep.py`; grant primitive: `eval/scoring.py`.
- Broader context (the graph-sparsity scaling-law experiment this came from): memory
  note `[[graph-sparsity-scaling-law]]` and `RESULTS_graph_sparsity.md`.
