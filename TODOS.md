# TODO

Remaining work, organized by area. All completed items stripped.

---

## Data

### Epoch precompute for Wikipedia
`epoch_precompute.py` currently only supports TheStack (repo-partitioned identifiers).
Wikipedia needs a **graph-community partitioner** before it can use the precomputed
path. The TheStack partitioner groups all files in a repo onto one worker so BFS
traversal stays intra-shard; naive random chunking of Wikipedia would scatter linked
articles across workers and BFS would immediately hit boundaries, producing
effectively doc_causal packs with no cross-doc grants.

Design: multi-source BFS Voronoi — pick `n_workers` random seeds, expand round-robin
with a per-worker size cap (≈ 1.5 × `len(graph) / n_workers`) to prevent hub nodes
(e.g. "United States") from dominating; re-seed workers that exhaust their queue
before the cap; assign leftover isolated/overflow docs round-robin. O(n) like the
existing repo-prefix scan. Full design in `data/epoch_precompute.py` module docstring.

Until this is implemented, simplewiki / Wikipedia training uses the live
`PackedSequenceDataset` path (no density-aware bucketing).

### Wikipedia redirect map
The Wikipedia dump ships a `redirect.sql` table mapping stub redirect titles to
their canonical targets (e.g. "UK" → "United Kingdom"). Fix at graph construction
time: rewrite in-text `[anchor](RedirectTitle)` links to the canonical node's title
and drop redirect stub nodes entirely. Downstream benefit: `HashNormTitleIndex`
hits these titles directly, fixing the class of eval misses where the model generates
a redirect title that isn't a first-class node.

---

## Model

### Scale-up
36L/1280D is placeholder. For the paper, need at least one run at a meaningfully
larger scale. Decisions needed: target param count, num_layers/model_dim tradeoff,
whether to use the same BFS+cross_doc_link config as current best runs.

---

## Training

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
