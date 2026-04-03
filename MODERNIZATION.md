# Modernization Plan

Differences vs. modded-nanogpt ordered by implementation priority.
Each item gets its own commit. Check off as done.

---

## Smoke Test Protocol

After every item below, run all three conditions to catch runtime regressions,
OOMs, and confirm loss is still declining.

All conditions use `CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2`
(exercises the distributed `MuonWithAuxAdam` all_gather path, not just
single-device).  Single-process `python main.py` is NOT sufficient — it only
hits `SingleDeviceMuonWithAuxAdam` and misses the rank-sharding logic entirely.

`--dataset-dir` and `--strategy` are named argparse args in main.py; all other
overrides use dotted-key notation via `compose_config`.

Condition A: baseline (doc_causal + random, `stack_100m_32k_baseline.yaml`):
```bash
CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 main.py \
    --config configs/stack_100m_32k_baseline.yaml \
    --dataset-dir /fss/evin_t/tagseq2tagseq/data/pretokenized_datasets/stack_100m \
    --train_loop.max_optimizer_steps 5 \
    --train_loop.val_steps 2
```

Condition B: experimental (cross_doc_link + BFS, `stack_100m_32k.yaml`):
```bash
CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 main.py \
    --config configs/stack_100m_32k.yaml \
    --dataset-dir /fss/evin_t/tagseq2tagseq/data/pretokenized_datasets/stack_100m \
    --train_loop.max_optimizer_steps 5 \
    --train_loop.val_steps 2
```

Pass criteria:
1. No CUDA OOM
2. No NaN/Inf loss at any step
3. Loss is not diverging by step 5 (not strictly monotone over 5 steps, but
   should not be trending upward relative to step 1)
4. Wall-clock time per step not significantly regressed vs. before the change
   (run condition A both before and after the implementation to compare)

If an item causes an OOM or significant slowdown, note it and evaluate whether
stepping down model size (e.g. from 24L/1024D to 12L/768D) restores
acceptable performance before deciding whether to keep it for the final run.

---

## Tier 1 — Trivial, extremely well validated

- [x] **1. Remove MLP dropout (0.1 → 0.0)**
  All configs: `dropout: 0.1 → 0.0`. Dropout universally hurts modern pretraining —
  it regularizes the wrong thing once you have enough data and weight decay.

- [x] **2. LR cooldown schedule**
  Decay both Muon LR and AdamW LR linearly in the final 40–60% of training down
  to 0.15× base rate. Add `cooldown_frac` and `min_lr_ratio` to the train_loop
  config section. Apply via a step-count multiplier in `main.py`'s optimizer step.

- [x] **3. Muon momentum warmup (0.85 → 0.95 over first 300 steps)**
  Prevents large early updates from uninitialized model. Linear ramp from 0.85
  to 0.95. A few lines in `main.py` alongside the cooldown logic.

---

## Tier 2 — Easy, meaningful payoff

- [x] **4. Vocab size alignment: 50257 → 50304**
  `50304 = 393 × 128`. Aligns every vocab-dimension GEMM to tensor core tile
  boundaries. New runs only (old checkpoints stay at 50257). Config + model
  change only.

- [x] **5. Adam β1=0.5 for embeddings/lm_head**
  Embedding rows get sparse gradients; the default β1=0.9 accumulates stale
  near-zero estimates. β1=0.5 discounts them aggressively. Requires either
  per-group beta1 support in `MuonWithAuxAdam` or splitting the AdamW group.

- [x] **6. Higher Muon weight decay (0.1 → 1.2) + lower AdamW WD (0.1 → 0.005)**
  Muon operates post-orthogonalization in spectral space — WD of 1.2 is
  appropriate there. AdamW params (embeddings, norms) should be lightly decayed
  (0.005). Requires splitting the single `wd` config into two fields.

---

## Tier 3 — Moderate code changes

- [x] **7. Logit softcapping**
  `logits → 23 * sigmoid((logits + 5) / 7.5)` before cross-entropy (from
  Gemma 2). Stabilizes the loss landscape by bounding logit magnitude.
  Currently using Liger `FusedLinearCELoss` with no cap — need to add a
  pre-cap op or swap to a kernel that supports softcap.

- [x] **8. Head dim 128 instead of 64**
  12H/64D → 6H/128D (baseline 768), 20H/64D → 10H/128D (large 1280),
  16H/64D → 8H/128D (stack_100m 1024). Better tensor core utilization.
  Verify Triton kernels are head-dim-agnostic (likely fine, BIM_BS=128).
  Breaks checkpoint layout — new runs only.

- [x] **9. Per-layer learnable residual scaling**
  `resid_lambdas` initialized to `sqrt(1.1)` per sublayer (attention + MLP),
  replacing fixed skip_weights=1.0. Compounds to ~1.1× amplification per
  layer. Modify `backbone.py` and `layer.py`.

---

## Tier 4 — Real but complex, consider for a dedicated run

- [x] **10. Fused ReLU² MLP (replace SwiGLU)**
  ~1–2% over GELU, but only worthwhile paired with a custom Triton kernel
  fusing the squaring into the matmul. Without the kernel, SwiGLU wins.
  Port/adapt `FusedLinearReLUSquareFunction` from modded-nanogpt.

- [x] **11. Multi-token prediction (MTP)**
  Predict 2–3 tokens ahead with tapering weights in early training, decaying
  to single-token prediction. Good convergence efficiency. Requires training
  loop changes and a new loss accumulation path.

- [x] **12. Embed/LM head weight untying at 2/3 of training**
  Start tied (already do this), then split at `step = 2/3 * max_steps`. The
  two matrices can specialize post-warmup. Medium effort; requires passing
  step to model and adding a split operation.

---

## Tier 5 — NorMuon optimizer overhaul (no architecture risk)

These four items share a prerequisite: the Muon optimizer moves out of tunalab
and into the repo as `optimizers/muon.py`. The tunalab `MuonWithAuxAdam` /
`SingleDeviceMuonWithAuxAdam` imports in `main.py` are replaced with the new
in-repo classes. This is also the right time to add the beta2 config knob
(see item 13b). Items 14–16 then layer on top of the new in-repo base.

Target world sizes: 1–4 nodes (4–32 GPUs). Nothing in these items is
world-size-specific, so behavior should be identical across that range.

- [x] **13. In-repo Muon + cautious weight decay**
  Copy `MuonWithAuxAdam` and `SingleDeviceMuonWithAuxAdam` (and their helpers
  `muon_update`, `zeropower_via_newtonschulz5`, `adam_update`) from
  `tunalab/catalogs/common/src/tunalab/optimizers/muon.py` into a new
  `optimizers/muon.py`. Remove the tunalab import from `main.py`.

  While doing this, add **cautious weight decay** to both the Muon and Adam
  update paths. The change is: before applying weight decay, mask it so it
  only fires when the gradient and the parameter are sign-aligned:
  ```python
  mask = (grad * p) >= 0          # True where update and current weight agree
  p.mul_(1 - lr * wd * mask)      # decay only the aligned components
  p.add_(update, alpha=-lr)
  ```
  For the Muon path `update` is the Newton-Schulz-orthogonalized gradient;
  for the Adam path it is the bias-corrected Adam step. Both masks use the
  post-update direction (not raw gradient) — this matches the reference code.

  Note: the modded-nanogpt reference uses `lr² * wd` for the weight-decay
  magnitude (calibrated to their large LR values 0.04–0.6). We use `lr * wd`
  because our AdamW LR (~3e-4) would make `lr² * wd ≈ 5e-10` — effectively
  zero. Muon LR (0.02) is comparable, so `lr * wd` is used there too.

  `muon_beta2` config key deferred to item 14 (unused until variance reduction
  second-moment buffer exists).

  Files: new `optimizers/muon.py`, `main.py` (swap import).

- [x] **14. NorMuon variance reduction**
  Add an Adafactor-style low-rank second moment to the Muon update. After the
  Newton-Schulz orthogonalization step, scale the update by the inverse square
  root of a running mean of its squared row/column norms. This gives Muon
  per-row adaptive step sizes without the full O(n²) second-moment matrix.

  Concretely, for a gradient matrix `v` (post-orthogonalization, shape [M, N]):
  - Reduce along the shorter dimension: `v_mean = (v² ).mean(dim=red_dim, keepdim=True)`
    where `red_dim = -1` if M ≥ N else `-2`.
  - Maintain `second_momentum_buffer` (same shape as `v_mean`) with EMA:
    `second_momentum_buffer = beta2 * buf + (1-beta2) * v_mean`
  - Scale: `v = v * (second_momentum_buffer.clamp_min(1e-10).rsqrt() * renorm_factor)`
    where `renorm_factor` preserves the Frobenius norm of `v` before/after scaling.

  Reference implementation: `NorMuonAndAdam._apply_normuon_variance_reduction`
  in `/fss/evin_t/modded-nanogpt/train_gpt.py:929`.

  New optimizer state key: `second_momentum_buffer` per Muon param. New config
  key: `muon_beta2` (shared with item 13 if done together). Checkpoint
  compatibility: new state key, so resumed checkpoints cold-start the second
  moment (fine, recovers quickly).

  Files: `optimizers/muon.py` (Muon update path only).

- [x] **15. Polar Express orthogonalization (replace Newton-Schulz)**
  Newton-Schulz is already stable in BF16 and good for our use, but Polar
  Express ([arxiv 2505.16932](https://arxiv.org/pdf/2505.16932)) converges
  faster per iteration and has better theoretical guarantees. This gives a
  small training-speed win (fewer NS steps needed) and potentially better
  gradient conditioning.

  The algorithm is structurally identical to Newton-Schulz: iterative
  polynomial refinement of X toward its polar factor. It uses a different
  set of precomputed coefficients (5 iterations) and requires two custom
  Triton helper kernels:
  - `XXT(X, out)` — computes `X @ X.T` into a pre-allocated buffer (symmetric,
    so only the lower triangle is computed; ~2× faster than `torch.mm`)
  - `XTX(X, out)` — same but `X.T @ X`
  - `ba_plus_cAA(A, alpha, beta, out)` — fused `beta*A + alpha*(A@A)`, avoids
    materializing the intermediate

  For tall matrices (M > N) use XTX + right-multiply; for wide use XXT +
  left-multiply. The modded-nanogpt reference implementation also has a
  `split_baddbmm` flag for matrices larger than 1024 rows to avoid PyTorch's
  defensive copy in `baddbmm`.

  Place the three Triton kernels in `kernels/polar_express.py`. Add a
  `polar_express(grad_chunk, momentum_buffer, momentum_t)` function (same
  signature as the current `zeropower_via_newtonschulz5` call site but with
  the new kernels and coefficients). Swap the call in `optimizers/muon.py`.

  Precomputed coefficients (for num_iters=5, safety_factor=2e-2):
  ```python
  polar_express_coeffs = [
      (8.156554524902461,  -22.48329292557795,  15.878769915207462),
      (4.042929935166739,   -2.808917465908714,  0.5000178451051316),
      (3.8916678022926607,  -2.772484153217685,  0.5060648178503393),
      (3.285753657755655,   -2.3681294933425376, 0.46449024233003106),
      (2.3465413258596377,  -1.7097828382687081, 0.42323551169305323),
  ]
  ```

  Files: new `kernels/polar_express.py`, `optimizers/muon.py`.

- [x] **16. Mantissa tracking for BF16 Muon params**
  BF16 has only 7 mantissa bits. When the Muon LR is small (late training,
  cooldown phase), the update magnitude can be smaller than a BF16 ULP, so
  the update is silently lost. Fix: maintain a separate `uint16` mantissa
  buffer per Muon param and reconstruct FP32 precision before each update:
  ```python
  # p and mantissa are both uint16 views of the param storage
  p32 = ((p.to(uint32) << 16) | mantissa.to(uint32)).view(float32)
  mask = (grad * p32) >= 0                          # cautious WD mask
  p32 -= p32 * mask * wd_factor * lr_factor         # weight decay
  p32 -= grad * lr_factor                           # parameter update
  p.copy_((p32.view(uint32) >> 16).to(uint16))      # store high bits back
  mantissa.copy_(p32.view(uint32).to(uint16))       # store low bits
  ```
  This effectively gives the optimizer FP32 accumulation precision for Muon
  params stored in BF16, at the cost of one extra uint16 buffer (half the
  size of the param itself).

  New optimizer state key: `mantissa` per Muon param (uint16, same shape as
  param). Implemented together with item 13 (cautious WD) since both operate
  on the FP32-reconstructed param. Non-BF16 params (e.g. FP32 in tests) fall
  back to a simple cautious update without mantissa tracking.

  Reference: `NorMuonAndAdam._cautious_wd_and_update_inplace`
  in `/fss/evin_t/modded-nanogpt/train_gpt.py:909`.

  Files: `optimizers/muon.py`.

---

## Tier 6 — Architecture additions (more experimental, optional)

These were previously marked "not applicable" as competition-specific tricks.
As of Record 78 (March 2026) they have been baked into the speedrun architecture
for 3+ months and show up in every run. Still, they were tuned on short-context
dense text (FineWeb, ≤2k tokens) and our setting is long-context
graph-structured text (32k, sparse cross-doc attention). Treat these as
hypotheses to test, not proven wins.

- [x] **17. x0 injection into every layer (embedding residual highway)**
  In modded-nanogpt each layer receives a small injection of the raw input
  embedding (before any transformer processing). A per-layer scalar
  `x0_lambda[i]` gates how much of `x0` is added to the residual stream at
  layer i. Zero-initialized → no-op at start.

  tagseq2tagseq already has skip connections from the first half of layers to
  the second half, but nothing that propagates the original embedding all the
  way through. This creates an information highway for the raw token signal.

  Implementation: add `x0_lambdas = nn.Parameter(torch.zeros(num_layers))` to
  `TS2TSBackbone`; in the layer loop, before or after the layer call, add
  `x = x + x0_lambdas[i] * x0` (where `x0` is the embedding output saved
  before the first layer). `x0_lambdas` go into the AdamW scalar group.

  Files: `model/modules/backbone.py`.

- [x] **19. Value embeddings**
  A `num_banks × vocab_size × model_dim` parameter bank (init: `0.01 * randn`,
  BF16). Before the attention kernel, per-token value vectors are looked up and
  added to V through a learned per-head gate — no kernel changes required (the
  gated ve is mixed through the attention distribution exactly like ordinary
  value vectors).

  Config keys (both optional; omitting `ve_layers` disables the feature):
  - `ve_layers: [1, 2, 21, 22, 23]` — which layers receive ve injection
  - `shared_ve_bank: false` — true = one bank shared by all ve_layers (~98 MB);
    false = one bank per ve_layer (~490 MB for 5 layers at 1024D)

  The gate is a `(H, 12)` parameter per bank using first 6 dims of the
  (pre-norm) input and first 6 dims of ve: `2*σ(linear([x[:6], ve[:6]], W))`.
  Gate zero-init → no-op at start.

  value_embeds use a dedicated Adam group: β1=0.75 (sparse gradients; only
  rows for tokens in the current pack are updated), wd=0.0 (unseen rows must
  not shrink). ve_gate_bank goes in other_adamw_params.

  Files: `model/modules/attention.py`, `model/modules/layer.py`,
  `model/modules/backbone.py`, `model/modules/training_module.py`,
  `model/model.py`, `main.py`.

- [ ] **20. Bigram hash embedding**
  For each position, compute a hash of the (prev_token, curr_token) pair and
  look it up in a small embedding table. The hash is a simple XOR-with-random-
  multiplier scheme: `hash = (tokens[i-1] * A) XOR (tokens[i] * B) mod vocab_size_bigram`.
  The bigram embedding is injected into the residual stream at each layer (like
  x0_lambdas in item 18) with its own per-layer scalar.

  Bigram vocab size in modded-nanogpt: `50304 * 5` (much larger than the
  unigram vocab). The table is zero-initialized so it starts as a no-op.

  This is probably the least likely to transfer: FineWeb has many repetitive
  bigrams (common English phrases), while Wikipedia/code have more diverse
  token patterns. Worth testing but lower priority.

  Files: `data/collate.py` or `main.py` (bigram hash computation), new
  embedding in `model/modules/backbone.py`, new AdamW group in `main.py`.

---

## Not applicable

- Multi-stage sequence length scheduling (they sprint at ≤2k context; we're at 32k)
- RoPE base 1/1024 (wrong for 32k context)
- Paired-head attention (risky, not validated outside speedrun, conflicts with
  doc-level masking semantics)
- Backout pattern (x -= backout_lambda × x_backout) — adds a non-monotone
  forward path, unclear motivation outside the speedrun setting
- Skip gate / sparse attention gate on layer 6 — too architecture-specific
- Adam on odd steps only — minor compute savings, complicates training loop
  and checkpoint logic significantly for unclear benefit at our scale
- FP8 matmul (efficiency only, not training dynamics; worth revisiting
  separately if memory becomes the binding constraint)
- YaRN / dynamic window warmup — we use fixed 32k context throughout
