# thestack cross_doc_link NaN Bug

## Symptom

`thestack/cross_doc_link` training consistently produces NaN loss at step ~100–150 during LR warmup (warmup_steps=300). The NaN is permanent once it appears. Loss looks healthy before that point (drops from ~63 → ~37 over ~100 steps, then suddenly NaN).

## Observations

- **doc_causal on identical setup**: no NaN, ran full epoch (13,986 steps) cleanly.
- **simplewiki/cross_doc_link**: no NaN in previous runs (up to 12,200 steps).
- **thestack/cross_doc_link**: has never successfully trained in this project.
- NaN step varies slightly across runs: 108, 132, 136, 146 — not deterministically fixed.
- NaN hits on **sparse** density buckets (kv_block_count ~5k–7k), not dense ones.
- Pattern: several consecutive dense-bucket steps (kv ~15k–25k) immediately precede the NaN step.
- Attempts to stabilize with aggressive gradient clipping (0.3) and LR÷3 delayed NaN by ~10–40 steps but did not prevent it.
- The `register_full_backward_hook` approach to capturing per-parameter grad norms during training failed — grads showed as 0 under `torch.compile`, likely because the hook fires before gradient accumulation completes in the compiled graph.

## Run history

| Job | world_size | NaN step | Notes |
|-----|-----------|----------|-------|
| 40249 | 8 | 136 | 1-node, precomputed epoch |
| 40295 | 16 | 108 | 2-node, precomputed epoch |
| 40297 | 8 | 150 | clip=0.3 |
| 40298 | 8 | crash | lr÷3 |
| 40299 | 8 | 146 | clip=0.3 + lr÷3 |
| 40301 | 8 | 146 | clip=0.3 + lr÷3, 500-step run |
| 41792 | 16 | 154 | data pipeline fixed (token_budget=32768, correct truncation accounting) |
| 41802 | 16 | 110 | ablation: logit_softcap=null — loss 1000–1500 from step 1, NaN at 110. Softcap is load-bearing but not the NaN cause. |
| 41803 | 16 | 110 | ablation: mtp_extra_weights=[] — loss healthy (30→19), NaN at 110. MTP is not the cause. |
| 41804 | 16 | N/A | ablation: max_grants=9999 — OOM/hung on compile warmup (bitmask too large), no result |
| 41805 | 16 | 121 | ablation: max_grants=64 — NaN at step 121, slight delay vs baseline 110 |
| 41806 | 16 | N/A | ablation: max_grants=1024 — hung on compile warmup (same as 9999); warmup OOMs at >256 grants |

## Setup at time of bug

- Model: 24L/1024D, `mask_type=cross_doc_link`, `link_detector=python`, `max_grants=256`, `logit_softcap=30.0`
- Data: `schedules/thestack_bfs/epoch_0` (precomputed, BFS, ~221k packs, 32 buckets, `token_budget=32768`)
- Optimizer: Muon+AdamW, `muon_lr=0.001`, `warmup_steps=300`
- `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800`, compile warmup pass enabled

## What was NOT the cause

- Variable sequence length recompiles — fixed separately by padding to `max_seq_len+1`.
- NCCL compile-time hang — fixed separately by compile warmup pass + longer heartbeat timeout.
- Attention output NaN from padding positions — `triton_v18` docstring states forward output is 0 for empty-row positions (sentinel LSE=-1e6 → O=0). Confirmed: adding `nan_to_num` on forward output made no difference.
- Dense packs directly — NaN triggers on sparse bucket after dense buckets, not on the dense buckets themselves.
- `logit_softcap=30.0` interaction — job 41802: disabling softcap produced rubbish losses (1000–1500) from step 1 but NaN still hit at step 110. Softcap is load-bearing for numerical scale but not the NaN cause.
- `mtp_extra_weights` interaction — job 41803: removing MTP entirely made no difference, NaN at step 110.
- BIM bitmask truncation — job 41805 (max_grants=64, more aggressive truncation): NaN at step 121, slightly later but same range. max_grants=1024/9999 OOM on compile warmup and couldn't be tested. Step shift from 64→256 is too small to implicate truncation as cause.

## Probe results (2026-06-09)

Three probes written and run against `schedules/thestack_bfs/epoch_0/packs.parquet`:

### `benchmarks/thestack_nan_probe.py` — kernel correctness sweep
Iterates the exact rank-0/world_size-16 pack sequence, steps 0–199. At each step
builds MaskInputs from the real parquet record and runs cdb_bim_v18 vs flex on
random q/k/v (no model, no optimizer). fwd_max_err ~8e-3, bwd_max_err ~6e-2
throughout — matching the established flex-vs-naive error floor. **Zero NaN/Inf
across all 200 steps, including the full NaN window (108–154). Rules out
hypothesis 1 (kernel bug on thestack pack structures).**

### `benchmarks/thestack_training_probe.py` — 1-layer training loop
Real Muon+AdamW training loop, real token IDs, real pack sequence. At each step:
cdb_bim_v18 used for training forward; flex runs on the same stashed q/k/v
(detached) for comparison; activations, gradients, and optimizer state buffers all
checked for NaN/Inf. 300 steps, 0 NaN/Inf, attn_err flatlined ~5e-4 to 1e-3.
Grad norms 1–28 (climbing late but stable). **Rules out Muon momentum corruption
as a single-process mechanism. Also confirms kernel produces no NaN on real data
with real weight dynamics at 1 layer.**

### 24-layer run (in progress)
Same probe with `--num-layers 24`. Grad norms immediately ~300–900 (vs 1–28 at 1
layer) — qualitatively different regime. Awaiting completion.

## Remaining hypotheses

2. **Gradient explosion through the cross_doc grant path, amplified across depth.**
The gradient norm clip operates on the global grad norm after accumulation. With
24 layers the residual stream can amplify a per-layer spike to a globally
catastrophic level before clipping takes effect. The 24-layer single-GPU probe
shows grad norms 300–900 even without NaN — consistent with depth being the
load-bearing factor. Next: per-parameter grad norm logging in a real SLURM run
(1 node × 8 GPUs) to identify which parameter explodes first.

6. **Optimizer state corruption from dense-bucket gradient spikes (multi-rank).**
Muon momentum corruption ruled out in single-process. May still manifest under
multi-rank all-gather: sharded parameter updates interact across ranks in a way
the single-GPU probe can't replicate. Test jointly with hypothesis 2 via per-step
logging in the production run.

## Files relevant to investigation

- `kernels/cross_doc_bitmask_bim_v18.py` — the training attention kernel for cross_doc_link
- `model/graph_traversal/cross_doc_mask.py` — `CrossDocLinkMaskCreator.__call__`, `_collect_links_per_doc`
- `benchmarks/attention_harness.py` — correctness/benchmark harness; add thestack fixtures
- `schedules/thestack_bfs/epoch_0/packs.parquet` — precomputed packs with `kv_block_count` and `link_target_doc_ids`
- `data/bucketed_pack_dataset.py` — `_make_bucket_sequence`, step→bucket mapping
