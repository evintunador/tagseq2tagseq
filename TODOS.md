# TODO

Remaining work, organized by area. All completed items stripped.

---

## Paper — cluster-reliant items (filed 2026-08-23)

Everything here needs `/fss-data` (run dirs, artifacts, or GPUs). The paper's prose is
written against these being resolved; each maps to a `\fillin{}` / LaTeX comment in
`paper/sections/`. Run `python scripts/check_grounding.py` after any of them.

### Audit the wiki community_pack negative (suspected corruption) — BLOCKS §7 wiki paragraph
`RESULTS_graph_sparsity.md` §4 / `docs/handoff_wiki_crossdoc.md` report solo-wiki
cross-doc Δnll on `val_community` as small but consistently NEGATIVE (−0.02, tight CIs),
while HotpotQA-cross-doc on the same ckpts is +1.29. Human suspects the community_pack
finding is a harness/corruption artifact, not real. Re-derive from scratch: (a) rebuild
the wiki_merged val_community packs and confirm grants are the RIGHT edges (not stale
schedule/normalization drift — see the "schedule staleness" memory); (b) check the
Option-B `span.start+1` grant keying and the `max_seq_len` 2048-default bug do not apply;
(c) score the identical packs with the HotpotQA-style paired scorer as a cross-check;
(d) widen to seeds 0–2 / 500 packs. Until resolved the paper makes NO claim about
generic held-out wiki text (§7 "The Wikipedia community-pack measurement"). If it is
real, the fit-dependence hypothesis in the handoff becomes an appendix result.

### arXiv is scoped OUT until sequence-parallel training exists
arXiv papers are long relative to the 32k window, so packs hold 1–2 docs, targets are
rarely co-packed, and the cross-doc mask has ~nothing to grant (sparsity eff.
grants/pack = 1.5, Δ flat). The arXiv sweep (cdl val 2.471 vs dc 2.156) is therefore
uninformative about the method. §4 states this; no arXiv claim is made. Re-enable only
after implementing sequence-parallel / context-parallel training (ring/striped attention
over the BIM kernels) so a citation neighbourhood fits in one logical sequence. Also
note the `\cite`-in-`%`-comment detector gap (App. C) before re-running.

### Measure the actual thesis: native corpus-fetching generation from pretraining alone
The design's purpose (§1, §7, §8) is that `cross_doc_link` teaches, via plain NTP under
maintained causality, that EMITTING a reference is what opens access to the referenced
doc — so a trained model can fetch from its corpus by generating a citation, with no
retriever/fusion/SFT/RL. This is UNMEASURED. The density result (dc-trained model gets
nearly the full cross-doc benefit at eval) is about *reading* a handed target, not
*asking* for one, and must not stand in for it. Design a measurement that works at
~350M params where free generation is incoherent. Candidates:
- **Link-emission rate under teacher forcing**: at positions where the reference doc
  contains a link, compare P(link-opener) and P(correct target title | opener) for cdl vs
  dc vs concat_link models. Cheap, no free generation, uses existing annotator openers.
- **Constrained-decoding resolution rate**: force the opener, decode the target with the
  trie/title index, measure fraction resolving to a corpus doc and to the RIGHT doc.
- **Counterfactual use**: after a resolved fetch, Δnll of the continuation with the
  fetched doc vs a deranged doc (placebo) — does the model USE what it asked for?
- **Scale ladder**: the honest test needs a model that generates coherently; plan the
  first size at which free-generation link-following becomes measurable.
Fold into the eval harness as a first-class benchmark; report cdl vs dc vs concat_link.

### HotpotQA / RepoBench headline placebo (TENTATIVE — may already be done)
Human believes the derangement placebo for the headline arms (`TODOS` "Placebo/
derangement on the headline arms", filed 08-10) may have been implemented in a local
worktree on fss (`/fss/evin_t/tagseq2tagseq-*`) and NOT pushed. Check `git -C
/fss/evin_t/tagseq2tagseq* status` / unpushed branches before re-implementing. Once
landed: run on the wiki bfs headline ckpt + Java/Python RepoBench arms, add ledger keys
`crossdoc.wiki_hotpotqa.placebo_sep` etc., cite in §6.1 / §7 Limitations.

### Wiki concat-control HotpotQA eval (missing) — `\fillin` in §6.3
`wiki_docconcatlink_best` (run_20260717_234605_156456, wd0.3, best_model.pt) has
single-doc evals but NO `hotpotqa_cross_doc` eval in any record; `wiki_docconcat_best`
(run_20260717_234603_647704) has no eval at all. Run `hotpotqa_cross_doc` on the
concat_link ckpt (mask has links, so the cross arm is well-defined) and the single-doc
panel + held-out ppl on the concat ckpt; distill; add `compute.wiki_hotpotqa.*` keys.
Then §6.3's "matching contrast on Wikipedia" and Table Z.3 get a `doc_concatenated`
column.

### Provenance: `kind: log` source for the two val-loss literals
`traversal.wiki_val_loss.{graph,random}` are literals. Implement a `log` source in
`scripts/provenance_lib.py` that parses `runs/<id>/logs/metrics_rank_0.jsonl`
(`val_loss_mean`, last/min row) and have `distill_runs.py` snapshot the needed rows into
the record so it survives run-dir deletion. Same source unlocks the LR/WD U-curve tables
and the per-language sweep val-loss table for App. Z.

### Provenance: `kind: artifact` source for the sparsity-scaling JSONs — BLOCKS §6.5 numbers
§6.5 (link-density dose–response) is written with `\fillin` magnitudes; the values are
in `/fss-data/evin_t/tagseq2tagseq_artifacts/sparsity_scaling/{phase1_eval,
phase2_traintime,grid2d,effective_density.json,regression.json}` and listed in a LaTeX
comment under §6.5. Add an `artifact` source (path + JSON pointer, content-hashed, copied
into `provenance/artifacts/`) and ground: eval-time code-only r/slope/intercept/n,
train-time r, out-degree r, per-dataset Δ@κ=1 + eff. grants/pack, 2D-grid axis swings,
keep0-row TS dc-trained vs cdl-trained. Also copy `fig_sparsity.png`, `fig_traintime.png`,
`fig_grid2d.png`, `regression.png` into `paper/figures/` and wire them into §6.5. Finish
the full 6-dataset keep0 row (SLURM 81104) and the traversal-time TS spot-check (81106–08)
first so the section reports final numbers.

### Density-aware timing: locate the originating run — `\fillin` in §6.7
The 1.45×/1.14× speedup and CoV numbers in README.md come from `step_timing_rank*.csv`
files that are not in-repo and carry no run_id (2 nodes × 2 GPUs, The Stack 10M, 32k).
Find the run on `/fss-data` (or re-time: one live vs one precomputed short run at the
same world_size/node layout), archive the CSVs under `provenance/artifacts/`, ground via
the `artifact` source above.

### Diversity section — blocked on the corrected merged_v2 re-run
§6.6 is fully `\fillin`. All `merged_all_v2` models were retracted 2026-08-14
(sequential-not-interleaved loader). Re-run at 3.9B/8B/16B (corrected LR from the 16B
sweep) with all four masks (configs `merged_v2_*`), then ports (use_line, Tier-2 placebo)
+ specialist re-pull for an exact table, distill, ground. Note in the paper: rungs are
independent samples, not nested; packs are within-source (multi-task mixing, not
cross-source attention).

### Dataset table cells — `\fillin` in §4 Table 1
Missing: wiki_merged edges + mean out-degree (+ token count: RESULTS says 4 epochs ≈
3.8B tok); JS / Dart / Zig edge counts + out-degree; Go / Java out-degree; arXiv
nodes/edges (README says 2.20M nodes ~10B tok; framing notes say ~1.98M — reconcile).
Read each dataset's `metadata.json` / audit log under
`/fss-data/evin_t/tagseq2tagseq_artifacts/pretokenized_datasets/<ds>/`.

### Link-reliance probe: synthetic cross-doc versions of standard benchmarks (filed 2026-08-26)
How heavily does the model actually rely on the links it references? Build
cross-doc-link versions of regular, non-HotpotQA-like benchmarks (e.g. HellaSwag):
inject a synthetic link into the question body and score the completion with an aux doc
attached via the normal grant machinery. Full design is a 2×3 grid:
- **Link placement** (who picks where in the question body the link goes):
  1. a smarter model (Claude) picks the location;
  2. the trained (dumber) model picks its own location.
- **Aux-doc author** (what the granted reference contains), for each placement:
  (a) smarter model writes a genuinely helpful synthetic reference doc;
  (b) the trained model generates its own aux doc (its generation-fallback path);
  (c) smarter model writes a bad/placebo reference (plausible but unhelpful/wrong).
Reliance = Δnll sensitivity across (a)/(b)/(c) — a model that leans on its references
should gain from (a), gain less (or self-consistently) from (b), and be hurt or
unmoved by (c); placement row contrasts whether link POSITION quality matters.
Reuses the `annotated` link-injection machinery (`eval/link_annotator.py` opener
placement + `score_completion_with_context_docs`); the new parts are synthetic aux-doc
authoring, the model-picks-placement condition, and the 2×3 report. Not yet done as of
2026-08-26 (idea predates; never implemented). Relates to the derangement placebo
(content-swap) but tests *synthetic* references on benchmarks with no natural links.

### Related but pre-existing (see below): leakage-stratified Δnll (RETRO bpb(α)),
eval-time max_seq_len extension, RoPE link-utilization diagnostic, concat controls across
non-bfs traversals — all cited as future work in §7 and unchanged.

---

## Java dataset — cross-repo framework-class mock resolution

In the Java code dataset, framework/stdlib imports (`android.content.Context`,
`java.util.List`, etc.) can RESOLVE at generation/eval time to *another repo's*
mock/stub reimplementation of that class, because Java FQNs are a global namespace
and several repos in the corpus define their own `android.*` / `java.*` stubs.
Only ~0.8% of nodes are framework-namespaced, and the TRAINING graph is unaffected
(its edges are intra-repo by construction — see `build_java_graph.build_repo_nodes`);
this only touches the `PretokCorpus` generation/eval resolver, which matches an
emitted import against ALL nodes. Look into whether to (a) drop framework-prefixed
nodes from the corpus, (b) scope generation-time resolution to the active repo, or
(c) leave it. Surfaced 2026-07-19 via `run_sample_dump` on the Go/Java datasets;
see `docs/multilang_code_datasets_DESIGN.md` §13 (Java quality nuance).

**Verified 2026-07-20 (two adversarial reviewers of `01_sample_dump.txt`):** in the
sample-dump, ~71% of resolved links are framework/stdlib FQNs matching a foreign
repo's mock/stub (0% resolved to the WRONG class — it's exact-FQN, so no
mis-resolution, just semantically-empty "stub magnet" edges); only ~29% are genuine
intra-project deps. BUT this is a `PretokCorpus`-resolver artifact — spot-checked
the STORED training graph and `java.util.List`/`Map`/`android.content.Context` each
have their few in-edges all from the SINGLE repo that vendored the stub (intra-repo
by construction), so TRAINING edges are NOT contaminated. Fix is only needed for the
generation/eval path. Also: wildcard imports (`import a.b.*;`) are intentionally
dropped by the detector (a package has no single file node) — confirm that's the
desired behavior or add package-info handling.

---

## Data

### Wikipedia redirect map
The Wikipedia dump ships a `redirect.sql` table mapping stub redirect titles to
their canonical targets (e.g. "UK" → "United Kingdom"). Fix at graph construction
time: rewrite in-text `[anchor](RedirectTitle)` links to the canonical node's title
and drop redirect stub nodes entirely. Downstream benefit: `HashNormTitleIndex`
hits these titles directly, fixing the class of eval misses where the model generates
a redirect title that isn't a first-class node.

### Merge all datasets → diversity-scaling experiment (REDESIGNED 2026-07-29)
**Supersedes the old wiki+stack+arxiv+fineweb merge.** First-principles rethink:
the claim is "cross-doc attention helps graph-structured text, and the benefit
survives/strengthens when many link types are learned jointly (diversity) and as
data scales." Fineweb is DROPPED everywhere — it is edgeless, so (a) a fineweb-only
run has no cross-doc A/B (it's just a different corpus, uninterpretable vs the
merge) and (b) inside the merge its tokens are seen identically by both masks, so
they only DILUTE the measurable Δ. "General intelligence" was never the claim.

**Design:** 11 linked sources (wiki, arxiv, python/thestack, typescript, javascript,
kotlin, rust, go, java, dart, zig), NO fineweb. Equal-ish token split (the balance
IS the variable under test; every source now has an internal cross-doc benchmark
that match/exceed the external ports, so no signal-weighting needed). Each rung is
a cross_doc_link vs doc_causal PAIR (within-model Δ = the interpretable metric,
robust to LR/batch since both arms share them). Fixed recipe carried from the small
runs (muon_lr=0.003, muon_wd=0.1, adamw_lr=3e-5, 262144 tok/step) — only the step
budget changes across rungs.

**Rungs (token-scaling line):**
- 3.9B — exact COMPUTE-match to the small single-source runs (15k steps × 262k tok);
  the clean diversity control. Equal split ≈ 11809 packs/source (zig gives all 921).
  No extra precompute (every source has ≥1 epoch at this size).
- 8B — scaling point. ~27256 packs for the 7 big sources; go/java/dart/zig give all.
  No extra precompute.
- 16B ×2 — (a) perfectly-balanced multi-epoch and (b) take-what-it's-got (big
  sources absorb the remainder, small give 1 epoch). Bonus ablation: isolates
  whether BALANCE per se matters at fixed total tokens.

**Multi-epoch = DISTINCT-SEED epochs, NOT epoch_0 replay (confirmed 2026-07-30).**
precompute_epochs.py writes epoch_i with seed=base+i; the seed drives worker
partition + BFS re-seed + traversal order, so each epoch is a genuinely different
packing over the full graph (verified: zig epoch_0/1/2 = seeds 42/43/44, different
pack counts + doc_id fingerprints). So repeating epochs adds new co-occurrence
structure — real signal, not duplication. Multi-epoch schedules ALREADY EXIST on
disk from the traversal-ablation work: zig 16ep, dart 9, java 7, go 6, wiki/rust 4,
kotlin 2; thestack/arxiv/typescript/javascript have only epoch_0 (fine — they're
capped, never repeated).

**16B perfectly-balanced allocation (SAFE=4 epoch-repeat cap; NO top-up precompute
needed — all reachable from on-disk epochs):** equal-ish share, but zig can't reach
1/11 (would need ~48 epochs of a 59M-tok corpus) so it's capped at 4 epochs = 3,618
packs = 0.12B; its shortfall redistributes to the other 10 → 48,466 packs = 1.59B
each. Total 488,278 packs = 16.00B. Epochs used per source: arxiv/stack/ts/js 0.1-0.3
(capped, epoch_0 only), kotlin 0.6, rust 1.3, wiki 1.7, go 2.2, java 2.9, dart 3.5
(tightest, <4), zig 4 (capped). SAFE=4 per Muennighoff-style 2-4-epochs-near-lossless.

**Rungs are INDEPENDENT, NOT nested (decided 2026-07-30).** Each rung
balance-samples its own target (like the 3.9B build). Dropped strict nesting
(3.9B⊄8B⊄16B) — it would need a prefix-stable selector (deterministic per-bucket
order + proportional prefixes) and a 3.9B rebuild. ACCEPTED CONFOUND: a Δ-vs-budget
change across rungs could partly reflect pack RESAMPLING, not pure scale. Note this
in the paper. The 3.9B rung (epoch_3p9b, already built + training as jobs 50016/50017)
is NOT rebuilt.

**merge_packs.py still needs a small change for 16B:** accept MULTIPLE epoch dirs per
--source (comma-sep), union them (each epoch's packs are distinct; keep all, reassign
pack_id across the union), then _select_balanced the target across the union. Current
code takes one dir/source. Not needed for 3.9B/8B (single epoch each), only the
perfectly-balanced 16B. take-what-it's-got 16B also single-epoch (epoch_0 + caps).

Build: merge_datasets over 11 sources' splits/train → merged_all_v2/splits/*;
merge_packs.py (already source-generic: tag=train_dir=schedule_dir=target, auto-reads
per-source layout+detector from schedule metadata) with per-source targets → one
epoch dir per rung. Per-source val schedules for the 8 new langs via
scripts/precompute_merged_v2_val.sh. Eval per-source on 11 internal benchmarks +
5 external ports. See memory [[merged-corpus-build]].

### Epochs-to-degradation: does cross-doc tolerate MORE epochs than doc-causal? (filed 2026-07-30)
Hypothesis: doc_causal repetition replays near-identical 32k windows (same doc, same
accidental neighbors) → memorizes fast (Muennighoff ~2-4 epochs then degrade). But
cross_doc_link re-samples the DAG TRAVERSAL each epoch (distinct precompute seed →
each epoch packs a doc with a DIFFERENT subset/order of its linked neighbors), so the
CONDITIONING CONTEXT for the same tokens changes every epoch — combinatorially many
"views" scaling with graph connectivity, not 1. Prediction: cdl's val-loss upturn
(overfit onset) happens at a HIGHER epoch count than dc's on the identical corpus =
a shifted/shallower degradation curve (NOT infinite epochs — finite tokens still
memorize eventually). Experiment: one dense-graph single-source corpus (kotlin or go
— dense import graph, go already has 6 epochs on disk), {cdl, dc} × {1,2,4,8,16}
epochs, SAME schedule seeds so ONLY the mask differs (fair — the seed reshuffles dc
packs too). Plot held-out val loss + a memorization probe (train-val gap or train
verbatim recall) vs epochs per mask; the divergence point IS the result. Distinct
from the diversity ladder + edge-dropout line. Uses on-disk multi-epoch schedules.

### Edge-dropout density line (cross-doc Δ vs graph connectivity) — filed 2026-07-29
The principled version of "how much better could this get with a denser corpus?"
Hold the corpus COMPLETELY fixed (same docs/tokens/order) and vary ONLY the fraction
of resolved links the mask may use: keep 100/75/50/25/0% of grants, seeded. At 0% it
degenerates to doc_causal (sanity: the line should hit the doc_causal baseline). This
is the clean instrument — NOT interpolating %fineweb, which confounds density with
content/quality/distribution (fineweb is different, higher-quality text; adding it
changes content, not just connectivity, and %-fineweb isn't even monotone in density).
The resulting line "cross-doc Δ vs connectivity" extrapolates the payoff of a denser,
better-connected corpus than the ones we have. Impl: subsample each pack's
`link_to_target` grants with a seeded RNG.
- EVAL-TIME dropping (cheap: 1 trained cross_doc_link ckpt, N evals) answers "how much
  does the model USE density."
- TRAIN-TIME dropping (N runs) answers "how much does density HELP LEARNING" — the
  stronger paper claim, N× cost.
Cheapest at mask-build time (subsample grants before the backend split), not graph
build. Do after the merged run lands.

### Short LR check at the 16B rung (before cross-rung Δ claims) — filed 2026-07-29
Within a rung the cross_doc-vs-doc_causal Δ is LR-robust (both arms share LR). But
comparing Δ ACROSS rungs (3.9B vs 16B) could be biased if a longer schedule wants a
different peak LR. Model size is FIXED across rungs (no μP/width-transfer issue), so
sensitivity is mild — plan is fixed muon_lr=0.003 + extend schedule. Insurance: a
quick 2–3 value muon_lr check (~{0.002,0.003,0.004}) at 16B, pick by subset val_loss,
before trusting cross-rung Δ comparisons.

### Batch-size sweep (absolute efficiency, NOT the diversity experiment) — filed 2026-07-29
262144 tok/step (8×32k×accum1) was inherited from ../moddednanogpt (tuned for a 124M
GPT-2 on FineWeb), never tuned for the ~350M model here. It does NOT bias the
within-model cross-doc Δ (both arms share it) and re-tuning would break the 3.9B
compute-match to the small runs, so it stays fixed for the scaling experiment. File a
SEPARATE batch-size sweep only if absolute token-efficiency becomes a goal.

### Preprocess code data to make imports lazy
Rewrite The Stack code so imports are lazy, saving compute (fewer/later-resolved
import edges to traverse and attend to). Investigate whether this meaningfully
shrinks the link closure for cross_doc_link packing.

### Make thestack datasets of other programming languages
expand TheStack beyond Python to include all available 
coding languages. Grab more languages and build link detectors + import-graph
extractors for each (JS/TS `import`/`require`, Ruby `require`, Go imports, etc.),
extending the Python-only import graph in `model/graph_traversal/link_detector.py`
— or use a language-agnostic call-graph approach.

---

## Model

### RoPE length extrapolation past the 32k training window (later — consider)
The rotary cos/sin buffer is sized to max_seq_len=32768 and
`flex_self_attention` hard-asserts `cos.size(0) >= T`, so any pack over 32k
aborts (hit during cross-doc benchmark scoring on whole-file Kotlin aux; the
Tier-2 harness now skips+counts oversized packs as a workaround, and
`rest_of_doc`-scope benchmark packs are the ones that bump this). Consider
editing the RoPE code/config to let the model EXTRAPOLATE beyond its training
length — e.g. NTK-aware / YaRN / linear position-interpolation scaling of the
rotary frequencies, or just building a longer cos/sin table at inference — so
long-context eval (and generation) isn't clipped at the training window. Would
let benchmarks with large whole-file aux score without truncation and is a
prerequisite for any >32k context experiments.

### Link-utilization vs. packing-distance diagnostic (RoPE-offset concern — filed 2026-08-10)
We use global RoPE positions [0,T) with no per-doc reset, so a granted target sits at
a packing-order-dependent relative offset. zhao2024analysing resets RoPE per document,
but only because its IntraDoc *bans* cross-doc attention — a clean reset is ill-defined
once grants cross boundaries (one key, one absolute position, read by linkers at many
distances). So assess the concern empirically instead of ablating a reset. On existing
checkpoints (no retrain): measure attention mass on the granted target (link
utilization) as a function of link->target packing distance, per source. If utilization
is flat across distance, the concern is empirically dead and it becomes one appendix
sentence. If it decays, that motivates the per-grant embedding arm below. Appendix
result. See `paper/notes/synthesis_framing_notes.md` (RoPE section) for the framing.

### Per-grant additive position embedding (only if the diagnostic shows decay — filed 2026-08-10)
Larger follow-up to the utilization diagnostic above, gated on it showing distance
decay. Vanilla RoPE can't re-base a granted key to a canonical offset per querying edge.
A second, additive embedding tagging "this is a granted cross-doc token" (NoPE-style /
learned per-grant tag) sidesteps RoPE entirely. This is a real research arm (new
embedding, retrain), not a quick ablation — only pursue if the cheap diagnostic
justifies it. Appendix result.

---

## Generation / Inference

### Prompt-link resolution can fabricate cited docs (design decision — consider)
With defaults (`process_prompt_links=True` and a `link_retrieval_mode` that allows
generation), a link merely *cited* in the prompt that isn't in the corpus (or with
`corpus=None`) falls through `_handle_link` to the recursive-generation branch and
*hallucinates* a whole document for that identifier, inserting it before the root
so root generation attends to fabricated content. If the intent is "resolve
existing citations, don't fabricate them", prompt-link processing should force
`corpus_only` semantics. Confirm intended behavior before changing.

### Per-token detector routing within one mixed-syntax document ("Tier 2") — OUT OF SCOPE (future direction)
Deferred and explicitly **out of scope for this project** — recorded as an interesting
future direction, not a work item here. `CompositeLinkDetector` (built 2026-08-01,
`model/graph_traversal/composite_link_detector.py`) dispatches ONE sub-detector *per
document*, chosen by identifier/content sniff. A single document that genuinely **mixes
link syntaxes** — e.g. a markdown article embedding a `\cite{}`, a literate-programming
file interleaving prose links and code imports, or a README with both `[text](url)` and
fenced code `import`s — is classified by its dominant language and detected with that one
sub-detector, so the minority syntax's links are missed. A proper fix is **per-token (or
per-span) detector routing within one sequence**: segment the document by language region
(fenced code blocks, LaTeX environments, etc.) and run the matching detector on each
region, merging the results. This is the deferred "Tier 2" from the merged-corpus design
([[merged-corpus-build]]). No current use case needs it — qualitative single-root
generation and the single-language benchmark ports are all dominant-language — and it
would add real complexity (region segmentation, offset bookkeeping across regions,
cross-firing between adjacent regions). Revisit only if a downstream task requires
faithful multi-syntax detection inside one document.

### TheStack (Python) link resolution in generation is unsupported
`generate.py` / `model/generation_loop.py` resolve a detected link to a corpus doc
via `corpus.has_document(target)` (exact → detector-key → optional fuzzy cascade).
This works for Wikipedia (`[text](Title)`) and ArXiv (`\cite{Title}`) because the
detector's `target_str` equals the corpus `raw_identifier`. Fuzzy matching does not
help here — the mismatch is a structural key-format difference, not a near-miss.
It does **not** work for TheStack: the
`PythonImportDetector` emits *relative* import paths (e.g. `"Phaedra/Notebook.py"`)
while corpus `raw_identifier`s are repo-qualified (`"000alen/Phaedra:Phaedra/Notebook.py"`),
so corpus hits never fire on a multi-repo dataset. See the NOTE in
`generate.py::PretokCorpus.has_document`. Fix options: (a) build a single-repo
corpus so identifiers match, or (b) make the import detector emit repo-qualified
identifiers when a repo context is available. Until then, Python-link generation
falls back to generate/skip per `link_retrieval_mode` (no corpus fetch).

### Merged-model (composite) link RESOLUTION across all code sources (filed 2026-08-01)
`CompositeLinkDetector` (built 2026-08-01) closes link *detection* for merged-model
generation — it routes per-document to the right sub-detector, so links in wiki /
arxiv / all 9 code languages are all detected (verified: cross-distribution smoke,
10/11 sources fire in their own syntax). But *resolution* — turning a detected
`target_str` into an actual corpus fetch via `corpus.has_document(target)` — is a
SEPARATE, still-open step, and it generalizes the Python-specific note above to every
code source. The same structural key-format mismatch applies: the code detectors emit
relative / repo-*un*qualified targets (python `chess/board.py`, ts `src/util/helper`,
rust `crate::net::tcp`, dart `lib/models/user.dart`, …) while a multi-repo corpus's
`raw_identifier`s are repo-qualified (`owner/repo:...` / `owner/repo@...`). Wiki and
arxiv resolve fine (target == identifier); code sources never hit on a multi-repo
corpus. So merged-model generation today can DETECT a code link but not FETCH its
target — it falls back to generate/skip per `link_retrieval_mode`.
Fix options (same shape as the Python note, applied corpus-wide): (a) generate against
a SINGLE-repo corpus per language (`data/make_repo_corpus.py`) so identifiers match the
detectors' relative keys; or (b) thread a repo context into `detect_links_for_doc` /
the composite so it can emit repo-qualified targets; or (c) add a source-aware
resolution index in `PretokCorpus` that matches on the detector-key form
(`index_doc_span`) per source rather than the raw identifier. Option (c) is the most
general and mirrors how the training/eval match already works (`_match_links_to_docs`
uses `index_doc_span`). Note the composite already implements per-source
`index_doc_span`, so a resolution index keyed on it is the natural hook. Until done,
use `--link-retrieval-mode link_but_skip` (detect-only) for merged-model smokes, or a
single-repo corpus for real code retrieval. Detection is DONE; this is resolution only.

---

## Training

### Retune LR / schedule for this dataset scale (before next ablation run)
Current optimizer/schedule values (muon_lr, adamw_lr, warmup, cooldown_frac,
total_steps) are inherited from ../ModdedNanoGPT, which tunes for how much a
model learns in ~the first hour of training — mis-scaled for our dataset sizes
and multi-day runs. The 2026-07-01..07 arxiv/thestack/wiki ablation runs were all
undertrained/poorly-tuned as a result (see RESULTS.md — barely-above-chance).
Retune before spending GPU-time on the next matrix.

### Train the ablation matrix
Actually train reasonable-sized models for each ablation:
(random, random-walk, dfs, bfs) × (doc-causal, cross-doc-link). NOTE: `random`
strategy was previously introduced without approval and its runs deleted — confirm
the intended strategy set before committing GPU-time (BFS is the established one).

### Automate the compile-cache warmup (TODO)
Multi-rank/multi-node runs require a pre-warmed shared compile cache to avoid the
concurrent-compilation segfault (see `launch_slurm.py`: `TS2TS_SHARED_COMPILE_CACHE`
+ `TORCHINDUCTOR_COMPILE_THREADS=1`). Today this is a manual two-step: warm the
cache once with a short run at the target world_size, then point the real run at
the same `TS2TS_SHARED_COMPILE_CACHE`. Fold this into `launch_slurm.py` as an
automatic `--warmup-compile` pre-step (submit a brief warmup job at the target
world_size, wait, then launch the real job against the warmed cache). The warmup
must match world_size: the distributed Muon optimizer compiles shard-shape kernels
a single-GPU warmup never produces.

### Live PackedSequenceDataset: dedup docs within an epoch (TODO — consider)
The live `PackedSequenceDataset` / `PackBatchSampler` path samples WITH
replacement: seeds are drawn via `self._rng.randrange(num_nodes)` and dedup is
only *within* a pack (`pack_doc_ids`), with no cross-pack/epoch visited set. So a
doc can appear in many packs and some may never appear in a given pass. The
precomputed path already dedups per epoch (`epoch_precompute.py` `epoch_visited`
set → visited docs read as tok_len=0). Consider giving the live path the same
"each doc at most once per epoch" guarantee (a persistent visited set on the
sampler, reset per epoch) so the two paths match and data usage is even. Note the
interaction with truncation: a doc dropped/partially-used by pack-level trimming
should arguably remain eligible until its body is actually consumed.

### Parallelized eval in main.py
`run_benchmarks_on_model` runs serially. Naïve thread-pool parallelism is risky:
benchmarks vary widely in runtime (HellaSwag ~30s vs. community_pack_perplexity
~20min), so fast workers block waiting for slow ones — net win near zero, timeout
risk real. Better design: shared job queue (`queue.Queue`) where each worker pulls
the next unstarted benchmark, so fast workers don't sit idle. All workers share the
same compiled model (no re-compile). Implement only if eval wall-time becomes a
bottleneck.

---

## Eval

### Leakage-stratified cross-doc Δnll (RETRO bpb(α) protocol — filed 2026-08-10)
Our grant makes verbatim copying from the target more direct than retrieval baselines,
and dedup is sampling-only, so raw perplexity gains risk being re-exposure of
memorized/duplicated text (HotpotQA's 2017 Wikipedia and The Stack both overlap
training data). Measure FIRST on the already-completed sweep (pure eval, no retrain):
stratify the cross-doc Δnll by target<->context n-gram overlap α and show the effect
survives at low overlap. Only if the low-overlap effect is weak do we go back to the
datasets, filter, and re-run training — a HARD BLOCKER on final numbers *conditional*
on the measurement demanding it, NOT a pre-committed retrain. Expect a steeper leakage
slope than RETRO. (Related but distinct from the deferred dataset-side dedup blacklist
item below, which is graph-construction dedup, not eval-side α-stratification.)

### Placebo/derangement on the headline arms + fired-subset Δnll (filed 2026-08-10)
Tier-2 ports already have the derangement placebo (`eval/benchmark_harness/tier2.py`,
`placebo_separation` + bootstrap CI). The headline HotpotQA-cross-doc and
RepoBench-cross-doc arms do NOT — they report only cross-vs-flat, where the cross arm
sees strictly more tokens, so "is it the right doc or just more context?" is unanswered
(acute given HotpotQA single-hop solvability). Extend the existing derangement machinery
to the headline arms, or caveat prominently. Also report Δnll over the *fired* subset
honestly (Repoformer's ~20/60/20 help/neutral/hurt split makes averaging over non-fired
items misleading).

### Eval-time max_seq_len extension (long-context generalization probe — filed 2026-08-10)
Cheap probe, no retrain: eval a 32k-trained checkpoint at larger max_seq_len (64k+) and
check whether grants keep helping as context grows past the training window. Direct
long-context-generalization signal for cross-doc attention. NOTE: blocked by the same
`cos.size(0) >= T` RoPE-buffer assert as the "RoPE length extrapolation" Model item —
either build a longer cos/sin table at inference or land the extrapolation work first.

### Finish the concat controls across non-bfs traversal (filed 2026-08-10)
The traversal ablation grid is `{bfs, dfs, random_walk, random} x {dc, cdl}`, but the
concat controls (`doc_concatenated`, `doc_concat_link`) exist ONLY at bfs (every concat
config pins `strategy: 'bfs'`). `doc_concatenated` has no links, so traversal strategy
still legitimately varies *which docs co-pack* — running concat across dfs/random_walk/
random is a real control ("does co-packing selection matter independent of the mask?").
Either finish those cells or document why bfs-only concat is sufficient. Folds into the
"Train the ablation matrix" item under Training.

### `annotated` (link-injection) eval speed — PARTIALLY DONE, remaining levers
The `annotated` condition (inject `[text](Title)` / `\cite{Title}` links into
benchmark prompts, then let cross-doc attention pull the linked corpus doc) was
~1s+/item. Two fixes landed (2026-07, commits da0f530..bbd1eac):
- **C (done):** dropped `edit_distance` from `annotator_strategies` in all configs.
  It did an O(9.6M) rapidfuzz scan per prompt (~338ms → ~5ms/lookup, 71×). Only
  typo-recall lost (91%→4% on synthetic typos; a trained model rarely emits those).
  Still available in code + `link_fuzzy_strategies` (generation-time, per-link).
- **B (done):** removed a redundant forward pass in `MarkdownPromptAnnotator.annotate`.

**REMAINING (the dominant cost is now autoregressive title generation, ~60
sequential forwards/item on arxiv, no KV cache — NOT the lookup anymore):**
- **A — batch the Step-1 opener scan forwards** across benchmark items. Modest
  win (~5–10%); the scan is a small slice of total forwards.
- **D — bound/​batch title generation** (the real remaining cost): lower
  `max_title_tokens` (50/60 is generous; real titles are short) and/or batch the
  autoregressive title gen. `forward_inference` is hardwired to B=1 (assert in
  attention.py); the batching pattern to mirror is `eval/scoring.py`
  `score_completions_batched` (packs K seqs into one `[1, total_T]` doc_causal
  forward). Biggest remaining speedup but most invasive.

**NOT YET VERIFIED end-to-end:** the C+B wins were measured component-level
(lookup latency, forward counts) but a full `annotated` benchmark at realistic
`n` was never re-run to confirm wall-clock actually dropped as projected. Do this
before assuming the "annotated eval is slow" problem is closed.

### Integrate easy LLM benchmarks
Wire in easy LLM benchmarks — likely specific sub-tasks from larger suites (e.g.
MMLU sub-tasks) that these model sizes can handle, preferably tasks that benefit
from cross-document understanding. (Some already wired: hellaswag, boolq,
openbookqa in the arxiv config's eval block.)

### Multi-hop QA beyond 2-hop (future / low priority)
HotpotQA is strictly 2-hop. Deeper graph traversal (BFS depth ≥ 3) is a core
claim of the system but is untested. Candidates:
- **MuSiQue** (`datasets` id: `musique`) — up to 4-hop, ~20k items, English Wikipedia.
- **2WikiMultiHopQA** (`datasets` id: `locuslab/2WikiMultiHopQA`) — up to 5-hop.

Before implementing: check dataset availability on cluster and leakage vs. training
data (Wikipedia models).

### Synthetic intra-repo cross-doc benchmark for Stack models (designed 2026-07)
RepoBench cross-doc shows only ~0.4% NLL improvement at early training — good
signal but small headline number. **The feasible path for code is a synthetic
intra-repo benchmark, NOT a link annotator** (link *injection* is the wrong
abstraction for code: an `import` can't be spliced mid-snippet the way
`[text](Title)` can, and it must be positional + semantically used). Design:
take file B that imports file A, score B's tokens (specifically the spans that
*use* A's symbols) under two conditions — (1) B alone (doc_causal), (2) A
provided as a cross-doc aux (the real import edge). NLL delta measures whether
the model exploits the dependency. No annotator, no injection, no labelling —
the edge already exists in the repo graph; you present or withhold the real
neighbor. Reuses `score_completion_with_context_docs` (already used by the
annotated path) but skips all `annotate()` machinery. Depends on a single-repo
corpus so identifiers match (see `data/make_repo_corpus.py`). Alt candidate:
CodeSearchNet with cross-file call graphs.

### Create + re-run Go/Java (multi-language) cross-doc benchmarks (RAISED 2026-07-21 — action item)
**Why now:** the 2026-07-21 code cross-doc sweep (see `RESULTS_code_crossdoc.md`) left
the Go and Java cross-doc claims **INCONCLUSIVE**. Python confirmed the thesis via
`repobench_cross_doc` (Δnll +0.135), but that benchmark is Python-hardcoded — it asserts
`isinstance(model.link_detector, PythonImportDetector)` and loads `tianyang/repobench_python_v1.1`
(see `eval/nlp_benchmarks.py::run_repobench_cross_doc`, ~L966-982). Go/Java had to fall back
to `community_pack_perplexity`, which is **near-noise for code** (deltas 0.0002–0.03; Java
sparsest graph ≈0) because import-graph neighborhoods are too predictable. So we have NO
discriminating cross-doc signal for Go/Java yet.

**To do:**
- ~~Split `run_repobench_cross_doc` by language and dispatch to the matching
  `<Lang>ImportDetector`.~~ **DONE 2026-07-23** (commit 91cb33f): `language` param +
  `_REPOBENCH_LANGUAGES` (python, java) + `--repobench-language`. Java fix: strip the
  build source root from snippet paths so import FQNs resolve (`_repobench_aux_identifier`).
  Provisional Java results in `RESULTS_code_crossdoc.md` — cross_doc_link beats flat on
  every traversal (Δnll +0.065..+0.194), the discriminating signal community_pack lacked.
- ~~re-run all Java cross_doc_link runs on their FINAL `best_model.pt`.~~ **DONE 2026-07-23**:
  all 4 traversals (bfs/dfs/rw/random) evaluated on final ckpts (step 14000–14750, val_loss≈1.05).
  Final Δnll +0.0646/+0.0738/+0.0788/+0.0314 — all positive, graph traversals cluster, random
  weakest. Supersedes the provisional (undertrained-ckpt) numbers. See RESULTS.md + RESULTS_code_crossdoc.md;
  per-run `runs/<id>/eval_java_repobench_final.json`.
- **Go** has no RepoBench variant → survey the internet for a RepoBench-analogous Go
  cross-file dataset that can be hacked to expose import edges as cross-doc aux DocSpans;
  else fall back to the self-built test_community benchmark (filed below).
- Fold in TypeScript too — RepoBench has no TS variant either; same survey/self-built path.

### Self-built cross-doc code benchmark from test_community splits (future — filed 2026-07-23)
External RepoBench only exists for Python + Java. For the other languages (Go, TS, Rust,
Kotlin, Dart, Zig, JS) with no upstream cross-file benchmark, build one from our OWN
held-out `test_community` splits (the import-graph neighborhoods we already carve). Two
token-scope variants considered (human is most interested in these two):
  1. **Import-dependent tokens only** — score ONLY tokens that actually use an
     imported/cross-file symbol (identifiers resolved to the linked doc, or the line
     following an import reference), not all body tokens. This is why the current
     `community_pack_perplexity` is near-noise: it dilutes the cross-doc signal across
     every body token of dense/predictable import neighborhoods. Restricting to the
     import-consuming spans should recover a discriminating Δnll (this is the same token
     scope RepoBench's "next_line" achieves, just carved from our own graph instead of an
     external dataset).
  3. **Whole-body on sparser communities** — keep whole-doc scoring but curate to
     sparser / high-out-degree communities where each import carries more predictive
     weight. Simplest change to the existing metric; may still be diluted.
Both reuse `score_completion_with_context_docs` (already language-agnostic — takes any
`link_detector`) + the per-language `test_community` split; no external dataset, no
annotator/injection. Sequencing: do the external Java RepoBench port FIRST (below), then
survey the internet for RepoBench-analogous cross-file datasets for the other languages
that can be similarly hacked to expose import edges as cross-doc aux DocSpans; fall back
to this self-built path only where none exist.

### Dataset-side dedup blacklist + rebuild/retrain (deferred — filed 2026-07-24)
The external-benchmark ports (Go←CoLT-132K, TS←CrossCodeEval, Kotlin←ASE-2025;
see `docs/crossdoc_benchmark_port_harness_DESIGN.md`) enforce hard dedup on the
BENCHMARK side: repo-name intersection with our training corpora (a repo lives in
the dataset or the benchmark, never both) + file-hash exclusion of cross-repo
copy-pastes. If exclusion ever guts a benchmark below Tier-2 power (n_cross_doc≥200),
the fix is the inverse direction: add a repo blacklist stage to the dataset pipeline
(`data/stack_sharded_download.py` / graph builders), rebuild the affected language
dataset, regenerate schedules (see schedule-staleness memory), and RETRAIN its sweep
ckpts. Expensive — deferred until benchmark-side exclusion proves insufficient. Also
rerun the repo intersection against Stack-v2 repo lists before any v2 retrain
(CrossCodeEval's 2023 repos fall inside v2's crawl window).

### LLM conceptual-dependency audit of ALL cross-doc benchmarks (deferred — filed 2026-07-25)
The port harness deliberately has NO syntactic "target-uses-aux-symbol" gate (rejected:
an example's dependency on its cross-file snippet can be conceptual — patterns,
invariants, config values — with no in-line identifier match, so a tree-sitter check
is fragile in both directions). Instead, once ports exist, run a smarter LLM over the
ENTIRETY of every cross-doc benchmark — the new ports AND the existing python/java
RepoBench ones — asking per example: "does predicting this completion genuinely benefit
from the granted cross-file snippet(s), and how (symbol use / conceptual / not at all)?"
Report the per-benchmark dependency-type distribution and the no-benefit fraction;
compare against each benchmark's placebo gap (Tier 2). Cheap to run on a few hundred
examples per benchmark; gives the quality signal the syntactic check couldn't.

### Better cross-doc benchmark for multi-language Stack models (future)
Once TheStack is expanded to all languages, extend the synthetic intra-repo benchmark
above to non-Python languages (e.g. JS/TS `require`/`import`, Ruby `require`).

### Link injection eval for external benchmarks
For external benchmarks other than RepoBench, the path to cross-doc-link eval is
via prompt preprocessing using `eval/link_annotator.py` (`MarkdownPromptAnnotator`).
Score benchmark items with bare prompt vs. link-annotated prompt and report delta.

**Title-lookup miss recovery (deferred — implement when eval performance matters):**
- `display-text fallback` — when all `TitleIndex` strategies miss on the generated
  target_str, retry `lookup()` with the anchor text between `[` and `](`.
- `prefix_commit` strategy in `HashNormTitleIndex` — find corpus titles sharing the
  longest common word-level prefix with target_str. Covers early-halt and overshoot.
  Both described in `eval/title_index.py` module docstring.

### Validate arxiv c+ite opener refinement on a real checkpoint (2026-07)
`ArxivPromptAnnotator._refine_opener_position` (commit bbd1eac) re-ranks the
top-K opener positions by `P(\)·P(c|\)·P(ite|\c)` to avoid placing citations at
noise backslashes (`\alpha`, `\ref`, ...). It's implemented + unit-tested, but
its placement *quality* is UNVALIDATED: measured on the 6L/512D smoke-test arxiv
checkpoint it changes the chosen position in 63% of prompts, but that checkpoint
is too undertrained to have real `\cite`-in-context behavior (P(c|\)≈0 at every
position), so it's re-ranking noise. Re-measure `P(c|backslash)@chosen-pos` (raw
argmax vs refined) once a properly-trained arxiv checkpoint exists; if it doesn't
improve, reconsider or set `opener_refine_top_k<=1` (raw-argmax, one fewer fwd).

### Annotator factory + generalization (raised 2026-07, not done)
The `annotated` eval pipeline covers **markdown + arxiv only**. Dispatch is a
hardcoded isinstance ladder in `eval_checkpoints.py` (~L614: `_is_annotatable =
_is_markdown or _is_arxiv`); Python/Null models silently skip the condition.
There is no `make_annotator(detector)` factory mirroring `make_link_detector`,
and `annotate()` is duplicated per subclass. Cleanup: add a `make_annotator`
factory + a no-op `NullPromptAnnotator`, replacing the isinstance ladder. NOTE: a
Python *annotator* is likely the wrong goal (see the synthetic intra-repo code
benchmark under the cross-doc-benchmark item) — this is purely about cleaning up
dispatch for the two text annotators that exist.

### Opener token coverage — remaining sub-items (2026-07)
Shipped: markdown scans `{58, 685}`, arxiv scans `{59, 3467}` (both backslash
forms). See the TODO comment by `_CITE_OPENER_TOKENS` in `eval/link_annotator.py`.
Still open:
- **Traditional multi-head MTP shortcut:** this model's MTP is a training-only
  shared-`lm_head` aux loss (skipped at eval), so the high-precision c+ite signal
  needs an extra forward. A model trained with *separate persistent MTP heads*
  per offset could read P(c),P(ite) from ONE forward — exploit if such a
  checkpoint is ever trained.
- **Markdown merged-punctuation openers** (` ([` 29565, ` "[` 12878, ...) are
  deliberately excluded (<2% coverage; injecting them splices malformed markdown).
  Would need generalized splice logic to include as scan-but-not-inject targets.

### Cross-doc benchmark ports — follow-ups (filed 2026-07-25)
Harness in eval/benchmark_harness/ + 3 external ports built (see
docs/crossdoc_benchmark_port_harness_DESIGN.md verdict summary). Open items:
- **Kotlin/ASE-2025 Tier-2 — DIAGNOSED 2026-07-25, needs power**: the scope
  ablation (scopes.py, `--scope all`) settled cause (b): the native miss was
  arbitrary FIM spans burying the signal. Use-site anchoring recovers a
  significant placebo separation that strengthens toward the use line
  (rest_of_doc +0.051 → use_block +0.058 → use_line +0.106, all CI-exclude-0),
  while native placebo sep is −0.033. See docs/crossdoc_benchmark_port_harness_
  DESIGN.md "Kotlin scope-gradient result". REMAINING to make it a headline
  benchmark: (1) raise n above the 200 floor — build the port over the FULL
  ASE public+private split instead of the 500-cap subset (only 138/242 survive
  the use-site filter); (2) re-run on a stronger kotlin cross_doc_link ckpt to
  lift Δnll_real (cross-vs-flat) CIs above 0 — currently still include 0 even
  though placebo separation is solid. use_line or use_block is the scope to
  report.
- **TS/CrossCodeEval v2 whole-file aux**: current v1 uses retrieval chunks
  (fire 0.40, Δnll_real CI barely incl 0, but placebo sep CI EXCLUDES 0 —
  promising). Re-clone repos from metadata.repository for whole-file aux +
  runtime relative-import resolution; target fire ~0.67 and a clean Δnll_real
  pass. Also consider a Tier-1 runtime-resolution mode so the static gate stops
  under-firing on relative-import langs (advisory-only today).
- **Go/CoLT-132K unblock**: email aixcoder authors for the external `godata`
  dependency JSONs (dependency_file_path), OR build Go via the self-built repo-
  snapshot path (like Kotlin/ASE). Adapter ports/colt_go.py is ready for the
  former. NOT registered in ports/__init__.py until unblocked.
- Then wire the passing ports into eval_checkpoints.py as first-class benchmarks
  (like repobench_cross_doc) so sweeps report them automatically.

### Resume latent bugs (found 2026-08-07 by resume-corruption audit; NOT the 16B-degradation cause)
Two real latent resume bugs surfaced while diagnosing the 16B degradation (which was
actually LR-too-hot-at-length, not resume). Neither affected the merged_v2 runs
(explicit max_optimizer_steps + 0 skipped), but both are footguns:
1. **NULL-path cooldown never engages.** `main.py:1195` `total_steps_original =
   max_steps_for_cooldown + resumed_steps`. On `max_optimizer_steps: null` (auto-derive
   from n_packs) resume, the 552-560 `remaining = _max - resumed_steps` adjustment is
   skipped (it needs an explicit `_max`), so the auto-derive re-computes FULL and +R
   inflates total to FULL+R → cooldown_start pushed past run end → LR pinned at PEAK for
   the whole post-resume segment, compounding per resume. Untie can also never fire.
   Fix: apply `remaining = derived - resumed_steps` after auto-derive too, OR decouple
   the schedule-total (always FULL) from the loader-cap (FULL - resumed_steps).
2. **Silent optimizer partial-restore + bf16 mantissa truncation.** `main.py:1100`
   `load_state_dict_full` leaves name/shape-mismatched params cold and only logs at
   INFO; a skipped Muon bf16 param also gets its mantissa low-bits zeroed
   (`optimizers/muon.py:154-156`), truncating fp32-effective → bf16 that resume. Add a
   hard `assert skipped == []` on resume (currently only INFO-logged).
