# Modernization Plan

Differences vs. modded-nanogpt ordered by implementation priority.
Each item gets its own commit. Check off as done.

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

- [ ] **4. Vocab size alignment: 50257 → 50304**
  `50304 = 393 × 128`. Aligns every vocab-dimension GEMM to tensor core tile
  boundaries. New runs only (old checkpoints stay at 50257). Config + model
  change only.

- [ ] **5. Adam β1=0.5 for embeddings/lm_head**
  Embedding rows get sparse gradients; the default β1=0.9 accumulates stale
  near-zero estimates. β1=0.5 discounts them aggressively. Requires either
  per-group beta1 support in `MuonWithAuxAdam` or splitting the AdamW group.

- [ ] **6. Higher Muon weight decay (0.1 → 1.2) + lower AdamW WD (0.1 → 0.005)**
  Muon operates post-orthogonalization in spectral space — WD of 1.2 is
  appropriate there. AdamW params (embeddings, norms) should be lightly decayed
  (0.005). Requires splitting the single `wd` config into two fields.

---

## Tier 3 — Moderate code changes

- [ ] **7. Logit softcapping**
  `logits → 23 * sigmoid((logits + 5) / 7.5)` before cross-entropy (from
  Gemma 2). Stabilizes the loss landscape by bounding logit magnitude.
  Currently using Liger `FusedLinearCELoss` with no cap — need to add a
  pre-cap op or swap to a kernel that supports softcap.

- [ ] **8. Head dim 128 instead of 64**
  12H/64D → 6H/128D (baseline 768), 20H/64D → 10H/128D (large 1280),
  16H/64D → 8H/128D (stack_100m 1024). Better tensor core utilization.
  Verify Triton kernels are head-dim-agnostic (likely fine, BIM_BS=128).
  Breaks checkpoint layout — new runs only.

- [ ] **9. Per-layer learnable residual scaling**
  `resid_lambdas` initialized to `sqrt(1.1)` per sublayer (attention + MLP),
  replacing fixed skip_weights=1.0. Compounds to ~1.1× amplification per
  layer. Modify `backbone.py` and `layer.py`.

---

## Tier 4 — Real but complex, consider for a dedicated run

- [ ] **10. Fused ReLU² MLP (replace SwiGLU)**
  ~1–2% over GELU, but only worthwhile paired with a custom Triton kernel
  fusing the squaring into the matmul. Without the kernel, SwiGLU wins.
  Port/adapt `FusedLinearReLUSquareFunction` from modded-nanogpt.

- [ ] **11. Multi-token prediction (MTP)**
  Predict 2–3 tokens ahead with tapering weights in early training, decaying
  to single-token prediction. Good convergence efficiency. Requires training
  loop changes and a new loss accumulation path.

- [ ] **12. Embed/LM head weight untying at 2/3 of training**
  Start tied (already do this), then split at `step = 2/3 * max_steps`. The
  two matrices can specialize post-warmup. Medium effort; requires passing
  step to model and adding a split operation.

---

## Not applicable

- Multi-stage sequence length scheduling (they sprint at ≤2k context; we're at 32k)
- RoPE base 1/1024 (wrong for 32k context)
- Bigram/value embeddings, smear gate, skip gate, paired-head attention, backout —
  competition-specific tricks not validated outside their narrow setting
- Adam on odd steps only (unclear motivation)
- FP8 (efficiency only, not training dynamics)
