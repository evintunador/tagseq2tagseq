<!-- PROVENANCE
Written against commit 6134163 (main @ 2026-08-07). This brief describes the code as of
that commit; it is NOT authoritative if the source has changed since. Before trusting it,
check whether the covered sources have drifted:
    git diff --stat 6134163..HEAD -- eval/scoring.py eval/perplexity.py eval/nlp_benchmarks.py eval/title_index.py eval/link_annotator.py eval/benchmark_harness/ eval_checkpoints.py
Empty output = brief still current. Non-empty = re-verify the changed parts against source.
Covered sources: eval/scoring.py eval/perplexity.py eval/nlp_benchmarks.py eval/title_index.py eval/link_annotator.py eval/benchmark_harness/ eval_checkpoints.py
-->

# CODE BRIEF: eval harness & cross-doc benchmarks (agent a4ae4e54)
Files: eval/scoring.py, perplexity.py, nlp_benchmarks.py, title_index.py, link_annotator.py, benchmark_harness/*, eval_checkpoints.py. All = teacher-forced mean-NLL over designated region from log_softmax(forward_inference logits), logit@t predicts token@t+1.

## Core primitives (scoring.py)
- score_completion_with_context_docs (:617-821) = CROSS-DOC ENGINE: aux snippets as leading DocSpans, primary context+completion last, detector over flat seq, keep links landing in primary, explicit link_to_target → forward_inference(mask_type='cross_doc_link'). precise (per-import path→doc_id + eval-only relative-import recovery) vs coarse (last import grants all aux). Score completion tokens only. None on no-aux/empty/no-import/no-grant.
- score_completions_independent_batched (:500-614) = PAIRED FLAT baseline: each (ctx,completion) isolated doc_causal, bin-packed. = the "flat" arm.
- score_doc_with_context (:867-999): scores only docs w/ INCOMING cross-doc edges (context-only excluded, mask-invariant would dilute). Option-B graph-edge grants.
- link_to_target_from_graph_edges (:824-864) = Option B from span.outgoing_identifiers. **span.start+1 key hack (:856)**: keying at span.start gives max|grant-dc|=0 (body never gains grant under causality) → key one token in so whole body attends. LOAD-BEARING, reviewer-relevant.

## Headline: paired same-token cross-vs-flat Δnll
Token-accounting parity at single tokenization point (schema.py:70-95, tier0 re-check).
- run_repobench_cross_doc: cross arm = cross-file snippets as aux, import_stmt+cropped_code context, next_line completion, precise match, _repobench_aux_identifier (py=path, java=FQN→source-root-rel). flat arm = SAME (ctx,completion) doc_causal no aux. Δ = flat_nll − cross_nll over PAIRED FIRED subset (n_cross_doc); non-fire → flat, only in with_fallback.
- run_hotpotqa_cross_doc: bridge-type, article B support→aux, article A→context w/ HTML→[text](Title) markdown, PRE-FILTER (≥1 A sentence contains ](B_title) else skip), context=A_md+Question+Answer:, completion=answer. 2 documented structural non-fires (paren titles, quoted titles) kept honest, NOT force-fired. Leakage arg: contrastive on identical text cancels memorization.

## Scopes (benchmark_harness/scopes.py) + Tiers
Scopes: native/use_line/use_block/rest_of_doc. use_line = single logical stmt at FIRST USE of imported symbol (import-syntax-FREE: takes top-level names DECLARED in granted aux via tree-sitter, finds first completion line w/ intersecting identifiers). Non-native rebuild context = full-file prefix up to use site → all use-scopes SAME context, vary target width = clean ablation "where cross-doc signal lives". native→use_line FLIP (go/java ~0/neg → clearly positive) compelling.
Tiers (certify PORTED external benchmarks legit vs py/java RepoBench originals): T0 CPU schema+invariants+token-parity+determinism (MAX_NO_IMPORT_FRAC 0.10); T1 CPU link-resolution audit vs tree-sitter ORACLE (grant precision≥0.95, fire-rate parity≥0.90×oracle); T2 GPU trained ckpt end-to-end (cross+flat+placebo, n≥200, fire-rate≥0.5, Δnll_real bootstrap 95%CI excl 0, placebo separation CI excl 0).

## PLACEBO control (tier2.py:169-234) — STRONGEST methodological contribution
DERANGEMENT of indices; keep own aux_raw_identifiers (grants still fire by id) but swap donor j's aux CONTENT → attend to WRONG-but-plausible in-language code. delta_real=flat−cross, placebo_separation=placebo−cross, both bootstrap CI. Proves benefit = RIGHT imported code not ANY extra context. Fire-rate preserved by construction. **Placebo is TIER-2 ONLY — headline hotpotqa/repobench have NO placebo arm (fairness gap).**

## Conditions (eval_checkpoints.py:240-259)
doceval (doc_causal+eos, ALL models, std cross-model column); baseline (doc_causal+eos, cross-doc models only, the floor); experimental (model's own mask+inference layout, auto-skip single-doc benchmarks for multi-doc-mask models); annotated (links INJECTED into flat prompts, cross_doc_link+markdown/arxiv detector, separate dispatcher, calibration curve over p25/p50/p75 link-opener-prob thresholds). detector_for_benchmark routes per-benchmark detector (merged model rebuilt per benchmark).

## community_pack + held_out
run_community_pack_perplexity: live packs from held-out community split, BFS outgoing, cross (mask None + grants_from_graph_edges=Option B) vs baseline doc_causal, mean_delta=baseline−cross bootstrap CI. **32k BUDGET BUG**: TS2TSModel no top-level max_seq_len → chain fell to 2048 default → collapsed long-doc (arxiv n=5). Fix = backbone.max_seq_len. Related **max_grants 64-vs-256 bug** (py Δ 0.135→0.09). Both = eval-config-mismatch caveats.
run_held_out_perplexity: per-doc isolated doc_causal, mask_type_override NO EFFECT (grants can't fire), single-doc → mask-independent.

## Novel/publishable
- Paired same-token cross-vs-flat Δnll (memorization-canceling, token-parity enforced).
- DERANGEMENT placebo (identifiers kept→fire-rate preserved, content swapped) — strongest.
- use-site re-anchoring (import-syntax-free tree-sitter-declaration, context held identical).
- Independent tree-sitter oracle audit (Tier1) w/ precision+fire-rate-parity gates.
- Hard train/eval dedup (repo-name + normalized-content SHA1).
- Option-B graph-edge grants for merged models.

## REVIEWER-ATTACKABLE (fairness)
1. **Headline benchmarks LACK placebo** → cross arm sees strictly MORE tokens than flat (no-aux, not wrong-aux); is gain from right doc or any tokens? Biggest concern.
2. Firing-CONDITIONED subset selection (Δ over fired examples only; fire correlates w/ parseable imports/clean titles; hotpotqa pre-filters to link-bearing).
3. Option-B span.start+1 key hack (effect literally 0 if keyed at boundary) — looks like tuning until effect appears + not matching training keying.
4. Eval/train config mismatches bit TWICE (max_grants, 2048 budget) → "re-eval" caveats.
5. In-distribution/leakage (HotpotQA 2017 wiki overlaps training).
6. Underpowered/undertrained pervasive (zig 59M neg).
7. RoPE-cap example dropping (kotlin ~73% packs >32k) biases toward short-aux.

FLAGS: didn't open benchmark_harness/ports/ adapters; _repobench_aux_identifier defined elsewhere; annotated aggregation grep-only; forward_inference mask semantics from docstrings.

## → LIT REVIEW IMPLICATIONS
- Q3: likelihood-based MC eval methodology (lm-eval-harness, byte/length normalization, Brown GPT-3 scoring).
- Placebo/counterfactual + bootstrap CI eval methodology; contrastive eval.
- Multi-hop QA (HotpotQA), cross-file code eval (RepoBench/CrossCodeEval/CoLT/ASE ports).
- Train-test contamination/decontamination in eval.
