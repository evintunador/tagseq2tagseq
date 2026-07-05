# TODO

Remaining work, organized by area. All completed items stripped.

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

### Finish inference generation logic
Complete the generation feature (Stage 3 items): `find_evicted` / `restore_evicted`
re-eviction, `process_prompt_links`, and `GenerationTrace`. See the generation
memory / `GENERATION_WORK_BREAKDOWN.md`.

### TheStack (Python) link resolution in generation is unsupported
`generate.py` / `model/generation_loop.py` resolve a detected link to a corpus doc
via an exact `corpus.has_document(target)` lookup. This works for Wikipedia
(`[text](Title)`) and ArXiv (`\cite{Title}`) because the detector's `target_str`
equals the corpus `raw_identifier`. It does **not** work for TheStack: the
`PythonImportDetector` emits *relative* import paths (e.g. `"Phaedra/Notebook.py"`)
while corpus `raw_identifier`s are repo-qualified (`"000alen/Phaedra:Phaedra/Notebook.py"`),
so corpus hits never fire on a multi-repo dataset. See the NOTE in
`generate.py::PretokCorpus.has_document`. Fix options: (a) build a single-repo
corpus so identifiers match, or (b) make the import detector emit repo-qualified
identifiers when a repo context is available. Until then, Python-link generation
falls back to generate/skip per `link_retrieval_mode` (no corpus fetch).

### Corpus-match cascade is eval-only, not used in generation
The successive title-matching algorithms (`HashNormTitleIndex`: exact → norm →
word_overlap → edit_distance) are only used by the eval annotators
(`MarkdownPromptAnnotator` / `ArxivPromptAnnotator`). Generation
(`generation_loop._handle_link`) resolves links with a single exact
`corpus.has_document()` lookup and no fuzzy fallback, for all datasets. A model
that emits a near-miss title (casing/punctuation variant of a real corpus title)
will fail to fetch during generation even though eval would have matched it.
Consider threading a `TitleIndex` through `run_generation` so generation gets the
same recovery cascade. (Wiki/ArXiv both benefit; orthogonal to the Python-path
issue above.)

---

## Training

### Train the ablation matrix
Actually train reasonable-sized models for each ablation:
(random, random-walk, dfs, bfs) × (doc-causal, cross-doc-link). NOTE: `random`
strategy was previously introduced without approval and its runs deleted — confirm
the intended strategy set before committing GPU-time (BFS is the established one).

### Smoother Muon optimizer resume from checkpoints
Today `--resume-from` (main.py:836) restores **only AdamW** optimizer state; the
distributed Muon momentum buffers are deliberately dropped and reinitialized cold
(`main.py:864` "Muon momentum initialised cold"). Reason: `MuonWithAuxAdam` shards
the momentum buffers across ranks in a world_size-dependent layout, so the saved
state isn't portable across a different world_size (or even reliably to the same
one, given param-group ordering). Cold-restarting momentum on every resume costs
re-warmup steps and perturbs the optimization trajectory — bad for
resume-heavy/down-node workflows.
Goal: gather/reshard the Muon momentum on save/load so resume restores it exactly
(or maps it across world_size changes). Reference implementation:
`~/hopprai/python/ml/models/mic/helpers/checkpoint_utils.py` (how mic handles
optimizer-state checkpoint/resume) and PyTorch's official Muon at
`torch/optim/_muon.py` (torch 2.9) — check whether migrating to the upstream Muon
gets portable state_dict handling for free. Cross-ref the resume gotcha in
[[cli-and-launch]] memory.

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

### `annotated` (link-injection) eval is far too slow (TODO)
The `annotated` condition (inject `[text](Title)` links into benchmark prompts,
then let cross-doc attention pull the linked corpus doc) takes ~1s+/item just to
annotate — one benchmark at n≈10k ran ~57 min and didn't finish; a killed job at
n=2000 still took ~30 min. Bottleneck is the per-prompt corpus title lookup in
`eval/link_annotator.py` (`HashNormTitleIndex` / `MarkdownPromptAnnotator`) over
the full merged corpus (9.6M nodes): each prompt does string-normalization +
fuzzy matching (exact→norm→word_overlap→edit_distance) against all raw
identifiers. Options: precompute/persist the title index once (not per doc),
vectorize/batch the annotation phase, cap corpus size, or memoize matches.
Until fixed, run `annotated` at small `--max-docs` (≤2000) and expect it to be
the long pole of any eval sweep.

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

### Better cross-doc benchmark for Stack models (future / low priority)
RepoBench cross-doc shows only ~0.4% NLL improvement at early training — good
signal but small headline number. Candidates:
- CodeSearchNet with cross-file call graphs.
- Synthetic benchmark from The Stack: take a file that imports another, score the
  imported function's body under both conditions. No labelling needed, directly
  tests our training setup.

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
