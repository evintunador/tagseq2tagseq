<!-- PROVENANCE
Written against commit 6134163 (main @ 2026-08-07). This brief describes the code as of
that commit; it is NOT authoritative if the source has changed since. Before trusting it,
check whether the covered sources have drifted:
    git diff --stat 6134163..HEAD -- optimizers/muon.py kernels/polar_express.py main.py
Empty output = brief still current. Non-empty = re-verify the changed parts against source.
Covered sources: optimizers/muon.py kernels/polar_express.py main.py
-->

# CODE BRIEF: Muon optimizer (agent abb68cda)

Files: optimizers/muon.py, kernels/polar_express.py, main.py. **NorMuon variant** (self-described muon.py:1-2) = Muon + 4 enhancements + aux AdamW, distributed via round-robin param shard.

## Muon update (muon.py:347-435 dist; 481-556 single)
(a) Nesterov momentum fused into polar_express (polar_express.py:300-302); μ passed as 0-D CPU tensor to avoid torch.compile recompiles (muon.py:329,356). FP32 momentum, orthogonalize casts to BF16.
(b) **Polar Express** 5-iter orthogonalization REPLACES Newton-Schulz (polar_express.py:281-348). Spectral-normalize input (X/(‖X‖_F·(1+2e-2)+1e-6)), then 5 quintics X←a·X+(b·A+c·A²)·X, A=XXᵀ(wide)/XᵀX(tall). Custom Triton kernels for symmetric Gram (XXT/XTX skip ~half via symmetry, :53-186) + fused β·A+α·A·A (ba_plus_cAA :193-264). split_baddbmm for M>1024.
(c) **NorMuon variance reduction** (muon.py:216-234): Adafactor-style rank-1 (per-row OR per-col, red_dim=-1 if M≥N else -2) 2nd moment; EMA 1-beta2 (muon_beta2=0.95), rsqrt scale, then RENORMALIZE globally to preserve Frobenius norm. Gives Muon adaptive per-axis steps w/o full O(n²) 2nd moment.
(d) Shape scaling v*max(1,M/N)**0.5 (muon.py:399-400) = standard Muon spectral-norm LR correction.
(e) **Cautious weight decay + BF16 mantissa tracking** (muon.py:237-257): BF16 params reconstruct FP32 from BF16 hi bits + stored uint16 low-mantissa buffer so sub-ULP late updates aren't dropped; WD fires only where grad*p>=0 (sign-aligned). Uses lr×wd not lr²×wd (their low AdamW LR ~1e-5 makes lr²·wd≈0).

## Polar Express = CITEABLE METHOD, arXiv:2505.16932 (polar_express.py:5, muon.py:10-11)
Computes polar factor via optimized odd matrix polynomials w/ PER-ITERATION non-stationary coefficients (vs standard Muon's fixed (3.4445,-4.7750,2.0315) quintic all iters). 5 coeff triples (polar_express.py:271-278), iter1 aggressive (8.16,-22.48,15.88) → relax toward NS fixed pt. safety_factor=2e-2. → CITE Polar Express as orthogonalization backend, DISTINCT from Bernstein/Jordan Muon Newton-Schulz-5.

## Distributed Muon (muon.py:347-413)
Round-robin shard across DDP ranks: params size-sorted desc (:308), pos i owned by rank i%world_size; each rank does full Muon update only for its params, single dist.all_gather broadcasts (:410-413). AdamW NOT sharded (every rank steps every AdamW param). Polar-Express Triton kernels torch.compile(dynamic=False,fullgraph=True) shape-specialized → shard shapes depend on world_size → CLAUDE.md warning "compiles shard-shape kernels a single-GPU warmup never produces" → shared compile cache must warm at SAME world_size. Bespoke name-keyed collective checkpoint state_dict_full/load_state_dict_full (muon.py:39-209) survives world_size changes + lm_head untie.

## Param split (main.py:1003-1066, by name + ndim)
- Muon: 'backbone' in name and ndim>=2 → lr=muon_lr, momentum 0.95, wd=muon_wd(0.1), beta2=0.95
- embed/lm_head ('embedding'/'loss_fn') → AdamW lr=adamw_lr, betas=(0.5,0.95) [low β1 for sparse grads]
- **value_embeds / bigram_embed** → AdamW betas=(0.75,0.95), **wd=0** (main.py:1008-1011,1056-1066)  ← THE VE BANKS
- other/norms/scalars/ve_gate_bank → AdamW betas=(0.9,0.95)
AdamW eps=1e-10 (unusually small). **adamw_lr ≈ muon_lr × 0.01** in all configs. Shared WSD schedule; muon momentum separate warmup (muon_momentum_warmup_steps=300). Deferred embed/lm_head untie at untie_at_frac (main.py:64-72,934).

## Novel/publishable vs reference (Keller Jordan modded-nanogpt Muon, named muon.py:6,22)
1. Polar Express backend (arXiv:2505.16932) replacing NS-5.
2. NorMuon variance reduction (rank-1 2nd moment, Frobenius-preserving). FLAG: verify "NorMuon" is published or in-repo coinage.
3. BF16 mantissa tracking (uint16 shadow) — unusual, likely novel.
4. Custom fused Triton Gram/quintic kernels.
5. Cautious WD w/ deliberate lr×wd.
6. Distributed round-robin + name-keyed world-size-portable checkpoint.

FLAGS: NorMuon provenance unclear (verify citeable vs coinage); Polar Express coeff generator not in repo (cite arXiv:2505.16932); mantissa reconstruction assumes truncation storage (unverified); momentum target 0.95 via default not explicit config key.

## → LIT REVIEW IMPLICATIONS
- O1 slice MUST include Polar Express (2505.16932) + NorMuon (search!) + orthogonalization-polynomial lineage.
- value_embeds + bigram_embed = the "VE banks" → dedicated value-embedding slice (modded-nanoGPT value embeddings, "learning to skip", residual/value embeddings speedrun tricks).
- Cautious optimizers (C-Optim / cautious weight decay) → find citeable source.
