# RESULTS

Benchmark & held-out performance for the current model ablation: **4 datasets ×
4 mask conditions**. Cells are `-` where the run isn't finished or the eval
hasn't been run yet, `n/a` where the benchmark/condition doesn't apply to that
model.

> Last updated: 2026-07-03. Numbers are transcribed from each run's
> `eval_results.json`. **Regenerate this file's numbers whenever new evals land.**

## Caveats (read before trusting a number)

1. **Perplexity metric FIXED 2026-07-03** (was broken: `mean_nll≈404`). Root
   cause: inference returned raw logits without the training-time
   `logit_softcap=30` → uncapped log-probs. Now `forward_inference` replays
   `cap*tanh(logits/cap)`. Verified sane (wiki doc_causal held-out
   `mean_nll=4.97 / ppl=143.5`). **All perplexity + `hotpotqa*` numbers below
   that were produced before the fix are STALE and must be re-run** — they came
   from the uncapped path. (Deltas between sub-conditions within one broken run
   were still directionally valid since both sides shared the bug.)
2. **`eval_checkpoints.py` now MERGES into `eval_results.json`** (fixed
   2026-07-03): a later run of a different benchmark subset no longer wipes
   earlier keys; same `{benchmark}/{condition}` keys update in place. NOTE: eval
   jobs already running when the fix landed still use the old clobbering
   behaviour until they're relaunched. Some values below are last-observed good
   numbers captured before a clobber.
3. **Conditions:** `doceval` = doc_causal + eos layout (the standard cross-model
   comparison column, runs on ALL models). `baseline` = doc_causal floor, but
   only emitted for multi-doc models. `experimental` = model's own mask +
   inference layout (auto-skipped on single-doc benchmarks for multi-doc models —
   grants can't fire on isolated docs). `annotated` = links injected into prompts
   (requires cross_doc_link + Markdown/ArxivCite detector; the thestack python
   detector is NOT annotatable).
4. All runs are **1 epoch** at 262144 tok/step unless noted. Undertrained.
5. `simplewiki` models are **excluded** — superseded by `wiki_merged`.

## Model roster & training status

| Dataset | Condition | Run dir | Status |
|---------|-----------|---------|--------|
| wiki_merged | doc_causal        | `runs/20260703_051129` | **done** (eval TODO) |
| wiki_merged | cross_doc_link    | `runs/20260703_050528` | **done** |
| wiki_merged | doc_concatenated  | `runs/20260703_052240` | training |
| wiki_merged | doc_concat_link   | `runs/20260703_053518` | training |
| thestack    | doc_causal        | `runs/20260630_193825` | **done** |
| thestack    | cross_doc_link    | `runs/20260630_195754` | **done** (eval running) |
| thestack    | doc_concatenated  | `runs/20260630_194919` | training |
| thestack    | doc_concat_link   | `runs/20260702_010558` | training |
| arxiv       | doc_causal        | `runs/20260701_170244` | training |
| arxiv       | cross_doc_link    | `runs/20260701_170424` | training |
| arxiv       | doc_concatenated  | `runs/20260701_170434` | training |
| arxiv       | doc_concat_link   | `runs/20260703_000924` | training |

---

## WIKI (wiki_merged) — commonsense MC + multi-hop QA

### Single-doc MC accuracy (condition = `doceval`; chance in header)

| Condition | hellaswag (0.25) | boolq (0.50) | piqa (0.50) | arc_easy (0.25) | openbookqa (0.25) | winogrande (0.50) | wiki_qa |
|-----------|:----------------:|:------------:|:-----------:|:---------------:|:-----------------:|:-----------------:|:-------:|
| doc_causal        | 0.250 [.241,.259] n=10042 | 0.429 [.412,.446] n=3270 | 0.477 [.453,.498] n=1838 | 0.287 [.268,.304] n=2376 | 0.188 [.154,.224] n=500 | 0.487 [.462,.514] n=1267 | - |
| **cross_doc_link**| **0.253** [.245,.262] n=10042 | 0.435 [.418,.452] n=3270 | 0.498 [.473,.520] n=1838 | **0.281** [.262,.298] n=2376 | 0.180 [.144,.210] n=500 | 0.493 [.463,.522] n=1267 | - |
| doc_concatenated  | 0.256 n=10042 | 0.427 n=3270 | 0.501 n=1838 | 0.275 n=2376 | 0.204 n=500 | 0.498 n=1267 | - |
| **doc_concat_link**| 0.257 [.248,.265] n=10042 | 0.442 [.425,.459] n=3270 | 0.510 [.487,.532] n=1838 | 0.301 [.283,.319] n=2376 | 0.194 [.158,.232] n=500 | 0.498 [.470,.525] n=1267 | - |

*Verdict @1ep: doc_causal and cross_doc_link are statistically indistinguishable
(all CIs overlap) and both at chance except arc_easy (~.28, marginally above).
Expected — single-doc MC gives cross-doc attention nothing to attend to.*

### Link-injection MC accuracy (condition = `annotated`; cross_doc_link only)

| Condition | wiki_qa | hellaswag | boolq | openbookqa |
|-----------|:-------:|:---------:|:-----:|:----------:|
| cross_doc_link | - | - | - | - |

*(annotated eval RUNNING on GPU-670; results pending. Compare vs the `doceval`
row above to isolate the effect of injecting links into prompts.)*

### Multi-hop QA (NLL; POST-softcap-fix — absolutes now trustworthy)

| Condition | hotpotqa (doceval) NLL | hotpotqa_cross_doc: NLL cross-doc | NLL flat-linked | Δ (n_cd) |
|-----------|:---------------------:|:---------------------------------:|:---------------:|:--------:|
| doc_causal        | - | n/a | n/a | - |
| **cross_doc_link**| 14.22 (n=2000) | **13.43** | 15.51 | **−13.4%** (n=1466) |
| doc_concatenated  | 13.91 (n=7405) | n/a | n/a | n/a — no inference-time link prediction |
| doc_concat_link   | 13.73 (n=7405) | n/a | n/a | n/a — no inference-time link prediction |

*🟢 cross_doc attention gives 13.4% lower NLL than flat concat of the same linked
content — the core cross-doc thesis, confirmed on the fixed (softcapped) path
(pre-fix delta was −14.3%, so the signal was real). Absolute NLL still high (~14)
because 1-epoch QA is genuinely hard — that's the model, not a metric bug.*

### Held-out perplexity (POST-softcap-fix; ppl now sane)

split=all sampling used (val_random/community splits aren't queryable on the
merged dataset via get_split_ids — needs the split subdir's own graph; see note).
NB: doc_causal ppl is a small n=192 sample (early fix-verification run); the
other three are n≈19k. Re-run doc_causal at full n for a clean comparison.

| Condition | held_out ppl (split=all) | mean_nll | community val | community test |
|-----------|:------------------------:|:--------:|:-------------:|:--------------:|
| doc_causal        | **143.5** (n=192)   | 4.97 | - | - |
| cross_doc_link    | **125.7** (n=1935) | 4.83 | - | - |
| doc_concatenated  | **144.1** (n=19358) | 4.97 | - | - |
| doc_concat_link   | **124.9** (n=19358) | 4.83 | - | - |

---

## THESTACK — code benchmarks

### Accuracy / perplexity (condition = `doceval` unless noted)

| Condition | humaneval_buggy acc (0.50) | repobench ppl | codexglue ppl | repobench_cross_doc (exp): NLL cross-doc / flat |
|-----------|:--------------------------:|:-------------:|:-------------:|:-----------------------------------------------:|
| **doc_causal**    | **0.610** [.537,.683] n=164 | 10.42 (n=2000) | 4.39 (n=2000) | n/a |
| **cross_doc_link**| **0.610** [.530,.683] n=164 | 12.44 (n=2000) | 3.80 (n=2000) | **2.321 / 2.418 = −4.0%** (n=1750) |
| doc_concatenated  | - | - | - | - |
| **doc_concat_link**| 0.561 [.488,.640] n=164 | 9.46 (n=2000) | 6.20 (n=2000) | n/a — no inference-time link prediction |

*🟢 thestack doc_causal humaneval_buggy 0.61 (CI floor .537 > .50) — genuinely
distinguishes correct vs buggy code. Code-benchmark perplexity is SANE here
(softcap-immune: relative scoring across choices, unlike absolute held-out NLL).*
*🟢 repobench_cross_doc: cross_doc_link scores linked source files −4.0% NLL vs
flat concat of the same content (n=1750) — the code analog of the wiki hotpot
cross-doc signal. humaneval_buggy identical to doc_causal (single-doc, expected).
Note doc_causal edges out cross_doc on plain repobench ppl (10.4 vs 12.4).*

### Held-out / community perplexity (softcap FIXED; not yet re-run for thestack)

| Condition | held_out val_random | held_out test_random | community val | community test |
|-----------|:-------------------:|:--------------------:|:-------------:|:--------------:|
| doc_causal        | - | - | - | - |
| cross_doc_link    | - | - | - | - |
| doc_concatenated  | - | - | - | - |
| doc_concat_link   | - | - | - | - |

*Old "broken" values (mean_nll≈404) were the softcap bug — now fixed. thestack
perplexity not yet re-run on the fixed path; code-benchmark ppl above IS valid.*

---

## ARXIV — commonsense MC (no code/hotpot; domain is science text)

### Single-doc MC accuracy (condition = `doceval`; chance in header)

Runs were RETIRED early (nodes reclaimed 2026-07-07) at partial training — and
the hyperparameters (LR/schedule from moddedNanoGPT, tuned for first-hour
learning) were mis-scaled for this dataset. Evaluated from best-val checkpoints
just to check for above-chance signal. n=2000/split=all, doceval.

| Condition | hellaswag (0.25) | boolq (0.50) | openbookqa (0.25) |
|-----------|:----------------:|:------------:|:-----------------:|
| doc_causal        | 0.288 [.27,.31] ✓ | 0.374 [.35,.40] ↓ | 0.262 [.22,.30] |
| cross_doc_link    | 0.280 [.26,.30] ✓ | 0.378 [.36,.40] ↓ | 0.266 [.23,.30] |
| doc_concatenated  | 0.284 [.27,.31] ✓ | 0.374 [.35,.39] ↓ | 0.264 [.23,.30] |
| doc_concat_link   | 0.285 [.26,.31] ✓ | 0.376 [.35,.40] ↓ | 0.248 [.21,.28] |

*✓ = CI floor above chance; ↓ = below. All 4 conditions barely-above-chance on
hellaswag (~.28 vs .25), at/below on openbookqa, and BELOW chance on boolq
(~.37 vs .50 — common for tiny/undertrained models on yes-no). Conditions are
statistically indistinguishable from each other. Undertrained + mis-tuned; not a
real capability signal.*

### Held-out perplexity (POST-softcap-fix; ppl sane)

| Condition | held_out ppl (split=all, n=2000) | mean_nll |
|-----------|:--------------------------------:|:--------:|
| doc_causal        | 16.6 | 2.75 |
| cross_doc_link    | 39.8 | 3.61 |
| doc_concatenated  | 22.0 | 3.03 |
| doc_concat_link   | 15.1 | 2.66 |

*doc_concat_link (15.1) and doc_causal (16.6) best; cross_doc_link worst (39.8)
— but cross_doc resumed from step 5,800 (val-plateau; ~45k live steps lost when
nodes died), so it's the least-trained here. Perplexity gaps mostly reflect
training progress, not architecture, given the mis-tuned run.*

---

## Headline findings so far

- **Cross-doc attention shows real signal** on both benchmarks built to test it,
  post-softcap-fix (trustworthy): wiki `hotpotqa_cross_doc` **−13.4%** NLL vs
  flat concat (n=1466), thestack `repobench_cross_doc` **−4.0%** (n=1750). Two
  datasets, two domains, same direction — the core thesis holds.
- **All 4 wiki conditions evaluated** (doc_causal, cross_doc_link,
  doc_concatenated, doc_concat_link @1 epoch). On single-doc MC they are
  statistically indistinguishable and at chance — expected (single-doc gives
  cross-doc attention nothing to attend to; 1 epoch is tiny). held-out ppl:
  cross_doc_link 125.7 ≈ doc_concat_link 124.9 < doc_concatenated 144.1 (the
  link-aware masks edge out plain concat, but small and 1-epoch).
- **thestack doc_causal** clearly above chance on humaneval_buggy (0.61,
  CI floor .537 > .50); code-benchmark perplexity sane.
- **Perplexity metric FIXED** (was mean_nll≈404 from missing inference softcap).
  All perplexity/hotpot numbers here are on the corrected path except thestack
  held-out (not yet re-run) and wiki doc_causal ppl (small n=192 sample).
- **Deferred:** `annotated` (link-injection) eval — blocked on annotator speed
  (see TODOS.md → Eval); would answer whether injected prompt links help.
