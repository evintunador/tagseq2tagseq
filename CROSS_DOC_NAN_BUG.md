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

## Probe results (2026-06-09 / 2026-06-10)

### `benchmarks/thestack_nan_probe.py` — kernel correctness sweep
Iterates the exact rank-0/world_size-16 pack sequence, steps 0–199. At each step
builds MaskInputs from the real parquet record and runs cdb_bim_v18 vs flex on
random q/k/v (no model, no optimizer). fwd_max_err ~8e-3, bwd_max_err ~6e-2
throughout — matching the established flex-vs-naive error floor. **Zero NaN/Inf
across all 200 steps, including the full NaN window (108–154).**

### `benchmarks/thestack_training_probe.py` — 1-layer training loop
Real Muon+AdamW training loop, real token IDs, real pack sequence. 300 steps, 0 NaN/Inf.
Grad norms 1–28 (climbing late but stable). **No NaN at 1 layer with real weight dynamics.**

### SLURM job 41830 — per-parameter gradient norm logging (rank 0, world_size=8)

`main.py` feature: `_GradNormLogOptimizer` wrapper (gated by `train_loop.log_grad_norms: true`)
writes `runs/<run_dir>/grad_norms.jsonl` per optimizer step (rank 0, post-`clip_grad_norm_`).

**Key observations from `runs/run_20260609_202318_076868/grad_norms.jsonl`:**

1. **Steps 64, 67, 80, 83–85, 88–90, 97**: ALL 149 param norms are exactly 0.0 (post-clip).
   These are the sparse-bucket steps from the precomputed schedule.

2. **Step 98**: `layers.0.attn.Wqkv.grad` is NaN; all other 148 params are 0.0.
   `clip_grad_norm_()` did not clear the NaN (CUDA `fminf(NaN, 1.0) = 1.0`).

3. **Step 99**: all 149 params go NaN simultaneously (NaN weight → NaN forward → NaN everywhere).

### SLURM job 41835 — flex backend control run (2026-06-09)

Same config/data/scale, `--model.attention_backend flex`. **200 steps, no NaN, no zero-norm steps.**

| | triton_v18 (job 41830) | flex (job 41835) |
|---|---|---|
| Zero-norm steps | 10 | 0 |
| First NaN step | 98 (Wqkv) | never |
| Steps completed | 200 | 200 |

Flex produces nonzero gradients on all the sparse-bucket steps where triton returns zero.
This establishes that `triton_v18`'s backward is behaving differently from flex's on these packs.

### SLURM job 41841 — layer-0 q/k/v capture (triton-trained, 2026-06-09)

`main.py` feature: `_AttnCapture` class (gated by `train_loop.capture_attn_steps: [...]`)
intercepts `TS2TSAttention.forward` at specified optimizer steps and saves post-projection,
post-norm, post-RoPE q/k/v plus block_mask metadata to `runs/<run_dir>/attn_captures/step_N_rank0.pt`.

Captured steps 97, 98, 99 from a triton_v18 training run.
Saved permanently: `tests/fixtures/thestack_packs/nan_0_qkv_pre.pt` (step 97) and
`nan_0_qkv_nan.pt` (step 98).

- Step 97 q/k/v: clean (no NaN/Inf), std ~1.0
- Step 98 q/k/v: clean input (no NaN/Inf in q/k/v entering the kernel)
- Step 99 v: NaN present (weight contamination propagated from step 98)

### SLURM job 41851 — layer-0 q/k/v capture (flex-trained, 2026-06-10)

Same setup but `--model.attention_backend flex`. Captured steps 64, 67, 80, 83–85, 88–90, 97, 98, 99.
All captures clean (no NaN/Inf anywhere through step 99).

Saved permanently: `tests/fixtures/thestack_packs/flex_step_<N>.pt` for each step.

### `benchmarks/thestack_bwd_probe.py` — backward correctness test (2026-06-10)

New script testing `cdb_bim_v18` (and future versions) against flex reference on three fixture types:

1. **Pack-structure fixtures with random q/k/v** (`zero_0`–`zero_9`, `nan_0.pt`):
   - All PASS. The backward bugs do not reproduce with random activations.

2. **Triton-trained activations** (`nan_0_qkv_pre.pt` = step 97, `nan_0_qkv_nan.pt` = step 98):
   - `step97_pre`: **FAIL** — bwd_dq_max = 1.27e+04 (spike at token 1920, which is at the start of BIM block 15; nearest doc boundary is at token 2014, mid-block)
   - `step98_nan`: **FAIL** — bwd_max_err = 5.06 (dK and dV errors well above tolerance)

3. **Flex-trained activations** (`flex_step_*.pt`, all 12 steps):
   - 11/12 PASS
   - `flex:step84`: **FAIL** — bwd_dk_max = 2.50e-01 (just above the 2e-1 atol)
   - All zero-gradient steps (64–90) and NaN-adjacent steps (97–99): PASS

Run:
```bash
CUDA_VISIBLE_DEVICES=1 python benchmarks/thestack_bwd_probe.py --impls cdb_bim_v18
CUDA_VISIBLE_DEVICES=1 python benchmarks/thestack_bwd_probe.py --impls cdb_bim_v18 cdb_bim_v19
```

## Raw data summary

| Step | Pack kv_block_count | triton bwd dQ norm | flex bwd dQ norm | Notes |
|------|--------------------|--------------------|-----------------|-------|
| 64 | 12309 | 0 (all-zero) | nonzero | zero_0 pack |
| 67 | 20042 | 0 | nonzero | zero_1 pack |
| 80 | 13581 | 0 | nonzero | zero_2 pack |
| 83 | 8405 | 0 | nonzero | zero_3 pack |
| 84 | 15682 | 0 | nonzero | zero_4 pack; flex-trained: dk_max=2.50e-01 |
| 85 | 16494 | 0 | nonzero | zero_5 pack |
| 88 | 3608 | 0 | nonzero | zero_6 pack |
| 89 | 22741 | 0 | nonzero | zero_7 pack |
| 90 | 3736 | 0 | nonzero | zero_8 pack |
| 97 | 20457 | 0 | nonzero | zero_9 pack |
| 98 | 29530 | NaN (Wqkv) | nonzero | nan_0 pack; 5 grants, 3 docs; triton-trained: dQ spike at token 1920 |
| 99 | — | all NaN | nonzero | model already contaminated |

## Fixtures and tooling

```
tests/fixtures/thestack_packs/
  zero_0.pt .. zero_9.pt        — pack structure only (random q/k/v in probe)
  nan_0.pt                      — pack structure for step 98 (random q/k/v in probe)
  nan_0_qkv_pre.pt              — real q/k/v from step 97 (triton-trained)
  nan_0_qkv_nan.pt              — real q/k/v from step 98 (triton-trained)
  flex_step_64.pt .. flex_step_99.pt  — real q/k/v from corresponding steps (flex-trained)

scripts/generate_thestack_fixtures.py   — regenerates zero_*.pt / nan_0.pt from parquet
benchmarks/thestack_bwd_probe.py        — backward correctness test harness
```

Pack fixtures use the same schema as `tests/fixtures/real_packs/` (simplewiki fixtures).
q/k/v captures are saved as `{step, q, k, v, block_mask}` dicts; flex captures have an
empty `block_mask` dict and require the corresponding pack fixture to supply the mask.

## Files relevant to investigation

- `kernels/cross_doc_bitmask_bim_v18.py` — the training attention kernel
- `main.py` — `_GradNormLogOptimizer`, `_AttnCapture` (debug tooling, gated by config flags)
- `model/modules/attention.py` — `TS2TSAttention.forward`, kernel dispatch
- `schedules/thestack_bfs/epoch_0/packs.parquet` — precomputed packs
- `data/bucketed_pack_dataset.py` — `_make_bucket_sequence`, step→bucket mapping
