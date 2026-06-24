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

---

## Model

### Additional datasets
- **EnWiki / full Wikipedia**: expand beyond SimpleWiki to the other available Wikipedia
  dumps (enwiki, etc.). Same markdown link graph pipeline; main cost is graph build +
  pretokenization at larger scale.
- **ArXiv LaTeX**: add an ArXiv dataset using LaTeX citation/reference links as graph
  edges. Requires implementing the LaTeX link detector sketched in
  `model/graph_traversal/link_detector.py` (the `# TODO(@jamesljr)` comment there).
- **TheStack all languages**: expand TheStack beyond Python to include all available
  coding languages. Requires extending the Python import graph extractor to handle
  other languages' import/require semantics, or using a language-agnostic call-graph
  approach.

---

## Training

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
