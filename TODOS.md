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

---

## Model

### Profile training throughput — even plain doc_causal may be leaving speed on the table
thestack_doc_causal runs ~6.6s/it at 262144 tok/step on 4 GPUs (~26 GPU-s/step);
the cross-doc-link mask is NOT the bottleneck (thestack_doc_concat_link is actually
cheaper per GPU-s, and arxiv's slowness is just its long docs filling the T×T matrix).
But 6s/it for this model size/context still feels high — plausibly 1-2s/it is
achievable. Profile the doc_causal path to find where time goes: dataloader / pack
build / collate (CPU-bound, starving GPU?), the varlen_bim Triton kernel, optimizer
(distributed Muon Newton-Schulz), grad-accum sync, or activation checkpointing / recompute.
Check GPU utilization (is it <90%? → input-bound). Profile via the
`profile_training` atomic feature: set `train_loop.profile.enabled: true` (plus a
small `train_loop.max_optimizer_steps`) on any config and launch normally — it
reports per-phase timing (data / mask / fwd / bwd / opt), an NCCL-isolation
estimate, a model-internal component breakdown (embedding / backbone / norm /
loss_fn), and an optional chrome trace (`train_loop.profile.trace: true`) for
per-kernel detail inside the compiled backbone. Flagged 2026-07-02; user suspects
headroom to 1-2s/it.

**First profile (job 43951, arxiv_cross_doc 1024d/24L/32k, 1 node × 8 A100,
2026-07-03):** wall=4444ms/step; **bwd=3190ms (72%)**, fwd=1221ms (27%),
mask=48ms (1%), data=10ms (0.2% — NOT input-bound), opt=23ms. Model-internal:
backbone=822ms, loss_fn=116ms (embedding/norm ~0). NCCL grad-sync est=635ms (via
no_sync isolation) — real but not dominant. The bottleneck is the attention
**backward** (bwd ≈ 2.6× fwd).

**Backend comparison (thestack, 1024d/24L/32k, 1×A100, 7 active steps,
2026-07-03)** — {doc_causal, cross_doc_link} × {triton, flex}, fwd / bwd / wall ms:

| config | fwd | bwd | wall |
|--------|-----|-----|------|
| doc_causal · triton (varlen_bim_v2) | 773 | **1694** | **2597** |
| doc_causal · flex                   | 778 | **2403** | 3263 |
| cross_doc · triton (triton_v18)     | 892 | **1582** | **2555** |
| cross_doc · flex                    | 1037| **2916** | 4035 |

Takeaways: (1) the custom triton kernels win entirely in the **backward** —
doc_causal varlen_bim bwd is 1.4× faster than flex (1694 vs 2403), cross_doc
triton_v18 bwd is 1.8× faster than flex (1582 vs 2916); forward is ~tied. (2)
bwd dominates every config (~2× fwd) — this is where any further speedup must
come from (bespoke FA2 backward, see the "Custom cross-doc-link FA2 kernel" TODO).
(3) data≈0 and mask≤42ms everywhere → NOT input- or mask-bound. (4) Even the best
config is ~2.5s/step at 1 GPU; the 6.6s/it figure was 4-GPU (DDP sync + smaller
per-GPU batch). Remaining knob: chrome trace (`profile.trace: true`) for
per-kernel attribution inside the backward, + the Muon Newton-Schulz cost
(opt=82ms single-GPU, but distributed MuonWithAuxAdam differs).

### Additional datasets
- **EnWiki / full Wikipedia**: expand beyond SimpleWiki to the other available Wikipedia
  dumps (enwiki, etc.). Same markdown link graph pipeline; main cost is graph build +
  pretokenization at larger scale.
- **ArXiv LaTeX**: add an ArXiv dataset using LaTeX citation/reference links as graph
  edges. Requires implementing the LaTeX link detector sketched in
  `model/graph_traversal/link_detector.py` (the `# TODO(@jamesljr)` comment there).
- **TheStack all languages**: expand TheStack beyond Python to include all available
  coding languages. Grab more languages and build link detectors + import-graph
  extractors for each (JS/TS `import`/`require`, Ruby `require`, Go imports, etc.),
  extending the Python-only import graph in `model/graph_traversal/link_detector.py`
  — or use a language-agnostic call-graph approach.

### Custom cross-doc-link FA2 kernel
Write a custom FA2-style forward+backward kernel for the cross_doc_link mask,
because FlexAttention's backward pass is absurdly slow. NOTE: partially addressed
— the triton BIM kernels (v12/v18) already beat flex fwd+bwd at 32k (see
kernels memory). This TODO now means: confirm whether the remaining backward cost
in the throughput profile (bwd ≈ 2.6× fwd, dominates the step) is inherent or has
more headroom vs. a bespoke FA2 backward. Cross-reference the profiling finding
in the Model section above.

### Linter-scoped Python mask
Build a more precise, python-linter-based mask that lets only the *relevant
scope* of the code in a given doc attend to what it imports (e.g. the specific
imported symbol's definition), rather than granting attention to the whole
imported document.

### Update to latest modded-nanogpt methods
Pull the newest techniques from `kellerjordan/modded-nanogpt/` into the model /
optimizer (the architecture already borrows from it: skip connections, x0
injection, value embeddings, bigram hash, Muon).

### nanochat RL & chat pipeline feasibility
Check feasibility of integrating `karpathy/nanochat/`'s RL + chat pipeline, and
how it might be edited to take advantage of this model's graph-aware features.

### Softmax flattening at long sequences (cross_doc_link only)
A real concern for `cross_doc_link` specifically: a well-connected node can attend
to hundreds of thousands of tokens (own doc + all linked docs), degrading
attention entropy. `doc_causal` is unaffected (attention is scoped per-document
regardless of total sequence length). **Do NOT implement anything until
empirically confirmed at the sequence length actually targeted.** Mitigations to
evaluate when needed:
- **NSA (Native Sparse Attention)** — compression attention (coarse global
  context) → LSE-based top-K block selection → selection attention on chosen
  blocks + sliding window. Composes on top of cross_doc_link (graph edges give
  document-level routing; NSA gives block-level content selection within the
  permitted region). Addresses flattening: compression reduces attended token
  count, selection operates on K×blocksize tokens (sharp weights). A simple Triton
  impl on A100 gets the algorithmic gain but misses Blackwell warp
  specialization/TMA (realistic ~2-3× vs. the paper's 9×; cuDNN NSA API needs
  SM100+, unavailable on A100). Refs: [cuDNN NSA](https://docs.nvidia.com/deeplearning/cudnn/frontend/v1.19.1/fe-oss-apis/nsa.html),
  [NSA paper 2502.11089](https://arxiv.org/abs/2502.11089).
- **Differential Attention** — subtracts two softmax maps to cancel uniform
  background noise; directly targets entropy inflation. Possibly simpler than NSA.

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

### 8-GPU (multi-rank DDP) training hangs — RESOLVED (2026-06-17, re-confirmed 2026-07-03)
Root cause (traceback captured, job 42245): a pack could exceed `max_seq_len`
(one rank built 10185 tokens > 8192), tripping the RoPE cache assert
(`assert self.cos.size(0) >= x_BTHD.size(-3)`) in that single rank's forward. The
exception unwound into `reproducibility.py __exit__ → barrier()`, which hung
waiting for peers that never arrive — masking the traceback so it *looked* like a
generic DDP collective-fingerprint hang. Manifests only at world_size≥2 (a single
GPU just raises visibly).

Fix: `data/pack_sampler.py::_apply_pack_truncation` now trims document bodies to
hit `token_budget` EXACTLY (never over budget; a fallback drops decoration-only
docs to stay ≤ budget). Regression tests:
`test_pack_truncation_hits_budget_exactly_with_prefixes` (both trim sides) +
`test_pack_truncation_body_trim_to_zero_keeps_decoration`. Multi-rank compile
segfault (separate ≥4-rank blocker) fixed by `launch_slurm.py`'s
`TORCHINDUCTOR_COMPILE_THREADS=1` + pre-warmed `TS2TS_SHARED_COMPILE_CACHE`.

Verified: 2-node×4-GPU clean run (2026-06-22); 1-node×8-GPU arxiv_cross_doc
(full 1024d/24L/32k) profiling run job 43951 completed all steps, zero rank
variance (all 8 ranks mean wall 4444–4445ms), no segfault/hang/RoPE-assert
(2026-07-03).

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

### Perplexity metric was broken — mean_nll ≈ 404 (RESOLVED 2026-07-03)
Root cause: the model trains with a **logit softcap** (`logit_softcap: 30.0`,
applied as `cap * tanh(logits/cap)` inside `FusedLinearCELoss`), but
`TS2TSModel.forward_inference` returned **raw uncapped logits** (range observed
[-156, 248]). `eval/scoring.py:score_doc` then took `log_softmax` over uncapped
logits → per-token NLL up to ~186, mean ~404. The MC/code benchmarks looked sane
only because relative scoring across choices partially cancels the scale error;
absolute per-doc NLL (perplexity) did not.

Fix (all in ts2, no tunalab change needed — softcap read from
`hyperparameters.json` which `load_inference_model` already parses):
`forward_inference` now replays `cap * tanh(logits/cap)` when `logit_softcap` is
set. Threaded via `TS2TSModel.__init__(logit_softcap=)` ←
`training_module.to_inference_model(logit_softcap=)` ←
`generate.py:load_inference_model` (`model_cfg["logit_softcap"]`).

Verified: wiki doc_causal held-out perplexity went from `mean_nll≈404 / ppl≈1e175`
to **`mean_nll=4.97 / ppl=143.5`** (n=192, split=all). Diagnostic in
`debug_perplexity_softcap.py` (temporary; delete when done).

Follow-ups:
- **Re-run every perplexity/hotpot eval** produced before this fix — their
  `*_perplexity` and `hotpotqa*` NLL numbers are from the broken path.
- The checkpoint's `metadata.config` is empty `{}` (softcap only survives in
  `hyperparameters.json`). Consider persisting the resolved config into the
  checkpoint so it's self-describing.
- Optional (reference): expose `softcap` as an attribute on tunalab's
  `FusedLinearCELoss` so inference can read it from the loss module directly
  rather than re-reading the config.

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
