# RESULTS

Benchmark & held-out performance for the current model ablation: **4 datasets ×
4 mask conditions**. Cells are `-` where the run isn't finished or the eval
hasn't been run yet, `n/a` where the benchmark/condition doesn't apply to that
model.

## LR RETUNE MILESTONE (2026-07-16) — first above-chance wiki model

The 2026-07-01..07 runs were undertrained + LR-mis-tuned (all at chance; see
below). An 8-point muon_lr sweep on **wiki_merged / doc_causal** (VE banks OFF,
4-epoch data-repeat = ~14.5k steps / ~3.8B tok, cooldown + untie now actually
firing) found the inherited `muon_lr=0.001` was ~3× too low. **Final val_loss
U-curve bottoms at muon_lr≈0.003** (basin 0.002–0.004 within 0.03, noise-tight);
0.001 was 2nd-worst. adamw_lr scaled at muon_lr×0.01.

Winner (muon_lr=0.003, fully trained to step 14438, `doceval`, max_docs=1000):

| metric | value | chance | verdict |
|--------|:-----:|:------:|---------|
| held_out perplexity | **ppl 4.00 / nll 1.387** | — | vs old undertrained ppl ~124–137 |
| hellaswag | **0.333** [.304,.361] | 0.25 | ✅ **above chance** (CI clears) |
| arc_easy | **0.333** [.305,.363] | 0.25 | ✅ **above chance** (CI clears) |
| piqa | 0.524 [.492,.554] | 0.50 | ≈ chance |
| winogrande | 0.509 [.479,.539] | 0.50 | ≈ chance |
| openbookqa | 0.216 [.182,.252] | 0.25 | ≈ chance |
| boolq | 0.387 [.355,.417] | 0.50 | below (yes/no likelihood-scoring artifact) |

First wiki model **statistically above chance** (hellaswag + arc_easy) — the old
ablation was indistinguishable from chance on everything. Config: 1024d/24L/8H,
`ve_layers: []`, `weight_tying:true` + `untie_at_frac:0.667`, cooldown_frac 0.4,
min_lr_ratio 0.1. Run dir `runs/20260715_232111...` (job 45065).

**Follow-on tuning (2026-07-18):** weight-decay sweep found `muon_wd=1.2` (inherited)
was ~12× too high — clean U-curve minimum at **muon_wd=0.1** (val 2.424 vs 2.548 @1.2).
LR optimum stayed 0.003 across WD. Final tuned config: **muon_lr=0.003, muon_wd=0.1,
VE-off**. wd=0.1 winner re-eval: ppl **3.75**, arc_easy 0.368, hellaswag 0.322.

## CROSS-DOC THESIS CONFIRMED (2026-07-18) — the core result

`hotpotqa_cross_doc` on the tuned **cross_doc_link** wiki model (bfs, run 45314) scores
the same 2-hop questions two ways:

| condition | mean NLL | perplexity |
|-----------|:--------:|:----------:|
| **cross-doc** (linked doc via cross-doc attention) | **5.63** | **278** |
| flat-linked (same content concatenated, no cross-doc structure) | 6.92 | 1012 |

**Δnll = +1.29 (~19% lower NLL, 3.6× lower perplexity) favoring cross-doc attention**
(n=738). This is the project's central mechanism, confirmed at a properly-tuned
operating point (the old undertrained run showed only +13.4%). On single-doc
benchmarks all masks are ~identical (doc_causal / cross_doc_link / doc_concat_link
all ppl≈3.84, hellaswag≈0.33) — the cross-doc advantage appears **only** on the
multi-hop cross-doc benchmark, exactly as the hypothesis predicts.

## TRAVERSAL ABLATION (2026-07-19) — thesis robust; graph≫random for base LM

Full {bfs, dfs, random_walk, random} × {doc_causal, cross_doc_link} matrix at the tuned
config. Final val_loss (wd0.1 doc_causal / wd0.3 cross_doc_link, VE-off):

| strategy | cross_doc_link | doc_causal |
|----------|:--------------:|:----------:|
| dfs | 2.421 | 2.426 |
| bfs | 2.431 | 2.424 |
| random_walk | 2.492 | 2.495 |
| random | 3.001 | 3.176 |

Cross-doc Δnll (`hotpotqa_cross_doc`, n=738): dfs +1.329, bfs +1.291, rw +1.270, random +1.241.

- **Graph-structured packing matters hugely for the base LM**: bfs/dfs (~2.42) ≫ random (3.00),
  a 0.58 val gap. dfs ≈ bfs > random_walk > random.
- **Cross-doc benefit is robust across ALL traversals** (+1.24…+1.33) — the thesis is traversal-
  independent. bfs/dfs/rw are indistinguishable on Δnll (0.06 spread at n=738 = noise level; no
  winner claimed among them). `random` is a mild low-Δnll outlier.
- **Decoupling**: traversal strongly affects base-LM quality but only weakly affects the
  *incremental* cross-doc benefit — two separate axes. (Caveat: absolute NLL isn't cleanly
  cross-model comparable; the within-model Δnll cross-doc−flat is the robust contrast.)

## ARXIV SWEEP (2026-07-19) — 38B-token corpus, 15k-step budget (~3.9B tok), LR0.003

Final val: doc_causal veoff **2.156**, VE-ON 2.204, wd0.6 2.173, cross_doc_link 2.471,
doc_concatenated 2.140, doc_concat_link 2.133.
- **VE-off still beats VE-on (2.156 vs 2.204)** even on 38B tokens — the predicted VE "flip"
  did NOT fully materialize within this budget (VE closed the wiki gap but didn't overtake).
- Low-WD preference transfers from wiki (wd0.6 worse). ArXiv trains to lower val (2.14) than
  wiki (2.42) — 28× more data. concat variants edge out doc_causal on raw val (denser packing).

Full run-by-run log + the yield-watcher/auto-kill system are documented in `TUNING_LESSONS.md`
and `/fss-data/.../pipeline_logs/lrsweep_runmap.txt`.

## CODE CROSS-DOC SWEEP (2026-07-21) — thesis generalizes to code, ~10× weaker than wiki

Does the wiki cross-doc gain carry to code? 3 languages × 4 masks + a Python traversal
ablation (18 runs), all 15k steps (~3.9B tok), tuned recipe (muon_lr=0.003/wd=0.1, VE-off,
8×A100). thestack (Python, 9.29B tok) subset via 15k-step budget like arxiv; go (1.22B) /
java (0.78B) below chinchilla so multi-epoch (6/7 epochs) like wiki. Full detail +
per-run numbers in `RESULTS_code_crossdoc.md`; run map `runs/CODE_SWEEP_RUNMAP.txt`.

**Headline code cross-doc metric = `repobench_cross_doc`** (Python-only): same tokens
scored with cross-doc attention on vs flat-concat off.

| condition | held_ppl | repobench_ppl | humaneval | repobench_cross_doc Δnll |
|-----------|:--------:|:-------------:|:---------:|:------------------------:|
| PY doc_causal        | 4.230 |  8.941 | 0.659 |   —    |
| PY cross_doc_link    | 4.233 |  **7.248** | 0.640 | **+0.135** |
| PY doc_concat        | 4.279 | 10.417 | 0.640 |   —    |
| PY doc_concat_link   | 4.264 |  8.763 | 0.628 |   —    |
| GO doc_causal        | 3.773 |   —    | 0.665 |   —    |
| GO cross_doc_link    | 3.758 |   —    | 0.665 |   —    |
| GO doc_concat        | 3.789 |   —    | 0.671 |   —    |
| GO doc_concat_link   | 3.786 |   —    | 0.665 |   —    |
| JAVA doc_causal      | 3.206 |   —    | 0.622 |   —    |
| JAVA cross_doc_link  | 3.169 |   —    | 0.628 |   —    |
| JAVA doc_concat      | 3.237 |   —    | 0.628 |   —    |
| JAVA doc_concat_link | 3.181 |   —    | 0.659 |   —    |

Python traversal ablation (mirrors the wiki bfs/dfs/rw/random × dc/cdl matrix):

| traversal × mask | held_ppl | repobench_ppl | humaneval | repobench_cross_doc Δnll |
|------------------|:--------:|:-------------:|:---------:|:------------------------:|
| bfs    cross_doc_link | 4.233 | 7.248 | 0.640 | +0.135 |
| dfs    cross_doc_link | 4.425 | 7.346 | 0.665 | +0.100 |
| rw     cross_doc_link | 4.324 | 8.858 | 0.604 | +0.069 |
| random cross_doc_link | 4.161 | 7.797 | 0.622 | +0.151 |

- **Thesis generalizes — direction confirmed, ~10× smaller than wiki.** repobench_cross_doc
  Δnll +0.07…+0.15 across all traversals (wiki hotpotqa was +1.29). Cross-file code
  dependency is more local/predictable than Wikipedia bridge reasoning.
- **Clearest signal: cross_doc_link beats doc_causal on repobench_ppl** (7.25 vs 8.94, holds
  across every traversal) — a materially better cross-file next-line predictor.
- **FLOP-control concat conditions do NOT beat cross_doc_link.** doc_concat is *worse* than
  doc_causal on repobench (10.4 vs 8.9) — naive whole-file concat hurts; gated link attention wins.
- **`community_pack_perplexity` is near-noise for code** (deltas 0.0002–0.03; Java sparsest ≈0),
  unlike wiki — dense import graphs. Use repobench_cross_doc, not community_pack, for code.
- **Traversal ablation mirrors wiki's decoupling**: cross-doc Δnll robust across all traversals,
  no clean ordering (spread within noise at n≈430); traversal affects base-LM quality more than
  the incremental cross-doc benefit — same two-axes story.
- **Go/Java cross-doc claim INCONCLUSIVE**: no cross-doc code benchmark exists for them
  (repobench_cross_doc asserts a PythonImportDetector), so their signal rests only on the weak
  community_pack proxy. See TODO below.

---

> Last updated: 2026-07-13. All numbers below regenerated in a full 12-model
> re-sweep on the verified-sound eval path (max_docs=2000, seed=0, `doceval`
> condition for single-doc benchmarks, `experimental` for cross-doc/community).
> Numbers are transcribed from each run's `eval_results.json`.

## Caveats (read before trusting a number)

1. **Eval infrastructure VERIFIED SOUND 2026-07-13.** A long-suspected
   "forward_inference emits garbage / perplexity untrustworthy" bug was chased
   down and **disproven for the eval path**: on a genuinely-trained checkpoint
   the inference load + `forward_inference` + `score_doc` path matches the
   training-module forward to ~0.005 NLL (same argmax), loads `strict=True`
   cleanly on all 12 roster checkpoints, and the flex eval backend matches the
   training triton kernel. The old "garbage logits" observation came from a
   diverged step-150 smoke checkpoint whose weights genuinely score nll≈30 in
   *both* the training and inference forward — not an eval bug.
2. **Recorded `val_loss` numbers were DEFLATED (pre-2026-07-09 scale bug).** The
   roster's checkpoint-metadata val_losses (e.g. thestack 0.038, arxiv 0.11–0.15)
   are artifacts of the val-loss-scale bug and made the models look far better
   than they are. Real per-token NLL is ~2.8–4.9 (see held-out ppl below). The
   benchmark numbers here are the honest picture; the impressive val_losses are
   not. Models are genuinely **undertrained + LR-mis-tuned** (see TODOS.md →
   "Retune LR"), not broken.
3. **Untie-load bug FIXED 2026-07-13** (`generate.py`). Checkpoints trained with
   `weight_tying:true` + `untie_at_frac` store an `lm_head` that differs from the
   embedding on disk, but reconstruction tied them → `load_state_dict` last-key-
   wins collapsed the two. Fixed by breaking the tie before load + `strict=True`.
   Impact on *these* numbers is negligible (roster checkpoints diverged ≤0.3%
   post-split, changing NLL at the 4th decimal), but the fix matters for future
   well-trained untie runs. `main.py:839` resume path likely has the same bug —
   unaudited.
4. **Perplexity softcap FIXED 2026-07-03**: `forward_inference` replays training
   `logit_softcap=30` (`cap*tanh(logits/cap)`). All perplexity numbers here are
   on the corrected path.
5. **Conditions:** `doceval` = doc_causal + eos layout (the standard cross-model
   comparison column, runs on ALL models). `baseline` = doc_causal floor, only
   emitted for multi-doc models. `experimental` = model's own mask + inference
   layout (auto-skipped on single-doc benchmarks for multi-doc models — grants
   can't fire on isolated docs). `annotated` = links injected into prompts
   (requires cross_doc_link + Markdown/ArxivCite detector).
6. All runs are **1 epoch** at 262144 tok/step unless noted, and most were
   retired early (nodes reclaimed 2026-07-07) at partial training. **Undertrained.**
7. `simplewiki` models are **excluded** — superseded by `wiki_merged`.

## Model roster & training status

Recorded val_loss is the DEFLATED metadata value (caveat 2) — kept only to
identify relative training progress across the ablation, NOT as a real loss.

| Dataset | Condition | Run dir | Step | (deflated) val_loss |
|---------|-----------|---------|-----:|--------------------:|
| wiki_merged | doc_causal        | `runs/20260703_051129` | 3500  | 0.353 |
| wiki_merged | cross_doc_link    | `runs/20260703_050528` | 3500  | 0.351 |
| wiki_merged | doc_concatenated  | `runs/20260703_052240` | 3614  | 7.04  |
| wiki_merged | doc_concat_link   | `runs/20260703_053518` | 3614  | 6.973 |
| thestack    | doc_causal        | `runs/20260630_193825` | 27000 | 0.039 |
| thestack    | cross_doc_link    | `runs/20260630_195754` | 27300 | 0.038 |
| thestack    | doc_concatenated  | `runs/20260630_194919` | 27300 | 0.038 |
| thestack    | doc_concat_link   | `runs/20260702_010558` | 6300  | 1.555 |
| arxiv       | doc_causal        | `runs/20260701_170244` | 16200 | 0.116 |
| arxiv       | cross_doc_link    | `runs/20260701_170424` | 5800  | 0.154 |
| arxiv       | doc_concatenated  | `runs/20260701_170434` | 8800  | 0.128 |
| arxiv       | doc_concat_link   | `runs/20260703_000924` | 18900 | 0.109 |

---

## WIKI (wiki_merged) — commonsense MC + multi-hop QA

### Single-doc MC accuracy (condition = `doceval`; chance in header)

| Condition | hellaswag (0.25) | boolq (0.50) | piqa (0.50) | arc_easy (0.25) | openbookqa (0.25) | winogrande (0.50) | wiki_qa |
|-----------|:----------------:|:------------:|:-----------:|:---------------:|:-----------------:|:-----------------:|:-------:|
| doc_causal        | 0.283 [.263,.302] n=2000 | 0.433 [.410,.455] n=2000 | 0.477 [.453,.498] n=1838 | 0.287 [.268,.304] n=2376 | 0.196 [.164,.234] n=500 | 0.487 [.462,.514] n=1267 | 0.238 [.167,.310] n=126 |
| cross_doc_link    | 0.290 [.271,.309] n=2000 | 0.434 [.412,.456] n=2000 | 0.509 [.486,.532] n=1838 | 0.285 [.267,.302] n=2376 | 0.202 [.170,.238] n=500 | 0.498 [.471,.524] n=1267 | 0.238 [.167,.310] n=126 |
| doc_concatenated  | 0.284 [.265,.304] n=2000 | 0.429 [.406,.451] n=2000 | 0.501 [.479,.522] n=1838 | 0.275 [.258,.293] n=2376 | 0.202 [.172,.240] n=500 | 0.498 [.470,.526] n=1267 | 0.230 [.159,.302] n=126 |
| doc_concat_link   | 0.286 [.267,.305] n=2000 | 0.447 [.424,.467] n=2000 | 0.510 [.486,.533] n=1838 | 0.301 [.282,.319] n=2376 | 0.188 [.156,.222] n=500 | 0.498 [.470,.524] n=1267 | 0.246 [.167,.317] n=126 |

*Verdict @1ep: all four conditions statistically indistinguishable (CIs overlap)
and essentially at chance except arc_easy (~.29, marginally above). Expected —
single-doc MC gives cross-doc attention nothing to attend to.*

### Multi-hop QA (NLL)

| Condition | hotpotqa (doceval) NLL | hotpotqa_cross_doc: NLL cross-doc | NLL flat-linked | Δ (n_cd) |
|-----------|:---------------------:|:---------------------------------:|:---------------:|:--------:|
| doc_causal        | 14.13 (n=2000) | n/a | n/a | - |
| **cross_doc_link**| 14.23 (n=2000) | **13.43** | 15.51 | **+13.4%** (n=1466) |
| doc_concatenated  | 13.98 (n=2000) | n/a | n/a | n/a |
| doc_concat_link   | 13.83 (n=2000) | n/a | n/a | n/a |

*🟢 cross_doc attention gives 13.4% lower NLL than flat concat of the same linked
content (n=1466) — the core cross-doc thesis, confirmed on the verified-sound
path. Absolute NLL still high (~14) because 1-epoch QA is genuinely hard.*

### Held-out perplexity (split=all, val_random dir, n≈1931) + community-pack contrast

| Condition | held_out ppl | mean_nll | community-pack Δnll (cross−base, n) |
|-----------|:------------:|:--------:|:-----------------------------------:|
| doc_causal        | 137.5 | 4.923 | n/a |
| cross_doc_link    | 124.1 | 4.821 | −0.0037 (n=471) |
| doc_concatenated  | 132.2 | 4.884 | n/a |
| doc_concat_link   | 115.1 | 4.745 | −0.0034 (n=471) |

*Link-aware masks edge out plain concat on held-out ppl (doc_concat_link 115 <
doc_concatenated 132; cross_doc_link 124 < doc_causal 137) but the gaps are small
at 1 epoch. Community-pack cross-doc contrast is ~0 (undertrained; grants add
almost nothing yet).*

---

## THESTACK — code benchmarks

### Accuracy / perplexity (condition = `doceval` unless noted)

| Condition | humaneval_buggy acc (0.50) | repobench ppl | codexglue ppl | held_out ppl | repobench_cross_doc: NLL cross / flat |
|-----------|:--------------------------:|:-------------:|:-------------:|:------------:|:-------------------------------------:|
| **doc_causal**    | **0.640** [.567,.713] n=164 | 5.78 (n=2000) | 4.52 (n=2000) | 4.28 (n=2000) | n/a |
| **cross_doc_link**| **0.640** [.567,.707] n=164 | 5.92 (n=2000) | 4.06 (n=2000) | 4.22 (n=2000) | **2.321 / 2.418 = +4.0%** (n=1750) |
| doc_concatenated  | 0.591 [.512,.665] n=164 | 5.55 (n=2000) | 4.48 (n=2000) | 4.28 (n=2000) | n/a |
| doc_concat_link   | 0.561 [.488,.640] n=164 | 9.46 (n=2000) | 6.20 (n=2000) | 8.13 (n=2000) | n/a |

*🟢 thestack humaneval_buggy 0.64 (CI floor .567 > .50) for doc_causal &
cross_doc_link — genuinely distinguishes correct vs buggy code. Code-benchmark
perplexity is sane. doc_concat_link lags (least trained, step 6.3k vs 27k).*
*🟢 repobench_cross_doc: cross_doc_link scores linked source files +4.0% lower NLL
vs flat concat of the same content (n=1750) — the code analog of the wiki hotpot
cross-doc signal.*

### Community-pack perplexity (multi-doc masks; cross vs doc_causal baseline)

| Condition | community val: NLL cross / base | Δnll | n_packs |
|-----------|:-------------------------------:|:----:|:-------:|
| cross_doc_link   | 1.603 / 1.606 | +0.0028 | 276 |
| doc_concatenated | 1.728 / 1.761 | +0.0332 | 280 |
| doc_concat_link  | 2.452 / 2.476 | +0.0242 | 280 |

*Small positive cross-doc contrast (base − cross > 0 means the multi-doc mask
helps). doc_concatenated shows the largest Δ (+0.033). (thestack doc_concatenated
`held_out_perplexity/baseline*` keys OOM'd during the sweep — the `doceval`
number above is the valid one; the OOM'd baseline variants are ignored.)*

---

## ARXIV — commonsense MC (no code/hotpot; domain is science text)

Runs were RETIRED early at partial training with mis-scaled hyperparameters
(LR/schedule from moddedNanoGPT). Evaluated for above-chance signal.
n=2000/split=all, doceval.

### Single-doc MC accuracy (condition = `doceval`; chance in header)

| Condition | hellaswag (0.25) | boolq (0.50) | openbookqa (0.25) |
|-----------|:----------------:|:------------:|:-----------------:|
| doc_causal        | 0.288 [.268,.307] | 0.374 [.353,.395] | 0.262 [.224,.300] |
| cross_doc_link    | 0.280 [.262,.301] | 0.378 [.357,.400] | 0.266 [.228,.306] |
| doc_concatenated  | 0.284 [.265,.304] | 0.374 [.354,.396] | 0.264 [.226,.304] |
| doc_concat_link   | 0.285 [.264,.304] | 0.376 [.354,.398] | 0.248 [.210,.288] |

*All four barely-above-chance on hellaswag (~.28 vs .25), at/below on openbookqa,
BELOW chance on boolq (~.375 vs .50 — common for tiny/undertrained models on
yes-no). Conditions statistically indistinguishable. Undertrained + mis-tuned;
not a real capability signal.*

### Held-out perplexity (split=all, n=2000)

| Condition | held_out ppl | mean_nll | community-pack Δnll (n) |
|-----------|:------------:|:--------:|:-----------------------:|
| doc_causal        | 17.9 | 2.886 | n/a |
| cross_doc_link    | 44.2 | 3.787 | +0.0000 (n=5) |
| doc_concatenated  | 24.1 | 3.182 | n/a |
| doc_concat_link   | 16.1 | 2.776 | +0.0014 (n=5) |

*doc_concat_link (16.1) and doc_causal (17.9) best; cross_doc_link worst (44.2)
— but cross_doc resumed from step 5,800 (~45k live steps lost when nodes died),
so it's the least-trained here. Perplexity gaps mostly reflect training progress,
not architecture. arxiv community_pack has only n=5 packs (tiny val_community
subgraph) → no usable cross-doc contrast signal.*

---

## Headline findings so far

- **Eval infra verified sound; recorded val_losses were deflated.** The models
  are genuinely undertrained/mis-tuned (real held-out NLL 2.8–4.9), not broken by
  an eval bug. Retune LR/schedule before the next matrix (TODOS.md).
- **Cross-doc attention shows real signal** on both benchmarks built to test it:
  wiki `hotpotqa_cross_doc` **+13.4%** lower NLL vs flat concat (n=1466), thestack
  `repobench_cross_doc` **+4.0%** (n=1750). Two datasets, two domains, same
  direction — the core thesis holds even undertrained.
- **All 4 conditions × 3 datasets evaluated** at full n=2000 on the corrected
  path. Single-doc MC is at chance everywhere (expected @1ep, single-doc). Held-out
  ppl: link-aware masks slightly edge out plain concat on wiki; thestack code ppl
  sane (4–9); arxiv 16–44 (dominated by training progress).
- **thestack** doc_causal & cross_doc_link clearly above chance on humaneval_buggy
  (0.64, CI floor .567 > .50).
- **Deferred:** `annotated` (link-injection) eval remains blocked on annotator
  speed (see TODOS.md → Eval); a stray n=5 boolq/annotated probe fired on wiki
  cross_doc (link never resolved) — not meaningful, omitted.
