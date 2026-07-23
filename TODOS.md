# TODO

Remaining work, organized by area. All completed items stripped.

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

### Merge all datasets into one combined corpus
Merge wiki + thestack + arxiv + fineweb into a single corpus for one bigger
model training run. Reuse `data/merge_datasets.py` + source-stratified splitting
(`split_graph.py --stratify-by-source`) so each source is proportionally
represented in train/val/test.

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

### nanochat RL & chat pipeline feasibility
Check feasibility of integrating `karpathy/nanochat/`'s RL + chat pipeline, and
how it might be edited to take advantage of this model's graph-aware features.

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
- **Still open:** re-run all Java cross_doc_link runs on their FINAL `best_model.pt` once
  the java ablations finish training (current numbers used available/early checkpoints; the
  random_walk ablation ckpt is undertrained, abs ppl ~1047).
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
