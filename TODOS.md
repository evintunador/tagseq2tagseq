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

### 8-GPU (multi-rank DDP) training hangs — ROOT-CAUSED & FIXED 2026-06-17
**Root cause:** `PackBatchSampler._apply_pack_truncation` did not enforce
`token_budget` as a hard cap. Its `else` branch (when `effective_len < overshoot
< prefix+body+suffix`) skipped the doc and let `overshoot` leak, so a pack made
of many docs each carrying a non-trimmable prefix (the `*_prefix` layouts) could
ship ABOVE `max_seq_len`. One oversized pack (observed: 10185 > 8192) trips the
RoPE cos/sin cache assert (`cos.size(0) >= seq_len`) in the attention forward on
whichever DDP rank drew it; that AssertionError unwinds into the end-of-run
`ReproducibilityManager.__exit__` barrier, which hangs because the other ranks
never reach it — masking the traceback and presenting as a generic collective
hang. Manifests only at world_size ≥ 2 (a single GPU just raises visibly).

**Fix:** `_apply_pack_truncation` now sheds exactly `overshoot` tokens via body
trims (carrying residual to the next doc), only fully removing a doc when
overshoot covers its whole size — so the pack hits `token_budget` EXACTLY,
never over (which is the dangerous case that trips the RoPE assert) and never
under. See the "autotune stall" entry below for the full final design and tests;
the over-budget leak and the under-budget off-by-one were fixed together.
Verified end-to-end: 4-GPU arxiv smoke completes all 200 steps + 4 validations.

How it was diagnosed (the prior blocker — rank-0-only logs): added opt-in
`TS2TS_DEBUG=1` per-rank file logging + `faulthandler` watchdog
(`debug_instrumentation.py`) and a try/except around `smart_train` in `main.py`
that logs the per-rank traceback before the masking barrier.

### Per-rank Triton-kernel autotune stall at ≥2 ranks — FIXED 2026-06-18
**Root cause:** the v18 attention kernel (`_attn_fwd_cdb_bim_v10` et al.) takes the
sequence length as a `tl.constexpr` ``N`` and autotunes with `key=["N",...]`, so
EVERY distinct sequence length triggers a fresh ~140s 48-config
autotune+JIT-compile. Packs were occasionally NOT exactly `token_budget`: a
pack-level-truncation off-by-one trimmed a doc's body to exactly 0 then DROPPED
the doc, shedding its non-trimmable decoration (e.g. a 1-token eos suffix) too →
pack landed `decoration` tokens UNDER budget (8191 not 8192). That odd length
re-fired the autotune on whichever DDP rank drew it while peers raced ahead and
blocked at the next collective. (Confirmed via `TRITON_PRINT_AUTOTUNING`: key's
first element is N; measured frequency was ~0.25% on arxiv val_community, 0% on
train/simplewiki/thestack — rare, but one is enough to desync.)

**Fix (no padding — packs are made EXACTLY token_budget at the source):**
`_apply_pack_truncation` now sheds exactly `overshoot` tokens via body trims,
carrying any residual to the next doc and only fully removing a doc when overshoot
covers its entire size (body+decoration). A body-trimmed-to-0 doc is KEPT so its
decoration stays accounted for → pack always lands exactly on budget, every doc
preserved as far as possible, no padding tokens, no wasted attention FLOPs.
Verified on real arxiv: 600/600 packs exactly 8192 (was: one 8191). Tests:
`tests/data/test_pack_sampler.py::test_pack_truncation_hits_budget_exactly_with_prefixes`
(both trim sides), `::test_pack_truncation_body_trim_to_zero_keeps_decoration`
(the exact off-by-one repro), `::test_pack_truncation_no_overshoot_is_noop`.
(The earlier padding-based fix — `pad_to_length`/`decoration_trim`/`n_real_tokens`
— was REVERTED per user preference for dense, padding-free packs.)

**Also fixed — multi-rank Triton/inductor-compile SEGFAULT (the ≥4-rank / 2-node
blocker):** many ranks JIT-compiling the SAME kernels concurrently corrupts the
compiler and segfaults a rank during step-0/early-step compilation (crash frames
in `triton.language.semantic` / `compile_worker.subproc_pool` /
`static_cuda_launcher` / dynamo `symbolic_convert`). Probability rises with rank
count: single-GPU never, 4-GPU intermittent, 8-rank near-certain. TWO settings in
`launch_slurm.py` fix it:
  1. `TORCHINDUCTOR_COMPILE_THREADS=1` — default is 32 PER PROCESS, so N ranks
     fork up to 32*N concurrent compiler subprocs (256 at 8 ranks) → corruption.
     ``1`` = synchronous in-process compile, no subproc pool. This ALONE got
     single-node 8-GPU through all 200 steps clean.
  2. A shared, PRE-WARMED compile cache (`TS2TS_SHARED_COMPILE_CACHE` →
     inductor/ + triton/ subdirs on /fss-data). Because exact-fill makes every
     pack the same length, all ranks need IDENTICAL kernels — so warm the cache
     once (a prior run at the SAME world_size, so distributed-Muon shard shapes
     match), then every rank READS it (0 compilation → 0 concurrent-compile risk
     + fast startup). Single-GPU warmup is NOT sufficient: the distributed Muon
     optimizer (`MuonWithAuxAdam`, `@torch.compile(dynamic=False)`) compiles
     shard-shape-specific kernels a single-GPU (`SingleDeviceMuon`) warmup never
     produces → warm at the real world_size.
Also set `TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER=0` and per-rank `TRITON_CACHE_DIR`
(defensive). **VERIFIED: 2-node × 4-GPU (8 ranks, GPU-44+GPU-53) ran all 200
steps + 4 validations, train+val loss decreasing, NO segfault / NaN / NCCL error /
recompile (0 autotune misses, warmed cache).** See [[ddp-multinode-hang-bug]].

**Operational note:** for a NEW config/model-shape, warm the cache first with a
short run at the target world_size, then point real runs at the same
`TS2TS_SHARED_COMPILE_CACHE`. TODO: fold this into `launch_slurm.py` as an
automatic `--warmup-compile` pre-step (submit a 1-rank-per-distinct-shape warmup
job, wait, then the real job) so it's not a manual two-step.

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
