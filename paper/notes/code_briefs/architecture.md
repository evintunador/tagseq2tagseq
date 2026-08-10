<!-- PROVENANCE
Written against commit 6134163 (main @ 2026-08-07). This brief describes the code as of
that commit; it is NOT authoritative if the source has changed since. Before trusting it,
check whether the covered sources have drifted:
    git diff --stat 6134163..HEAD -- model/modules/backbone.py model/modules/layer.py model/modules/attention.py model/modules/training_module.py model/model.py
Empty output = brief still current. Non-empty = re-verify the changed parts against source.
Covered sources: model/modules/backbone.py model/modules/layer.py model/modules/attention.py model/modules/training_module.py model/model.py
-->

# CODE BRIEF: model architecture (agent ada964f9)
modded-nanoGPT-derived GPT decoder (YouJiacheng/Grad62304977/KellerJordan speedrun lineage), batch=1 packed, doc-graph attention. tunalab installed editable from /fss/evin_t/tunalab/.

## Architecture (ref configs/arxiv_cross_doc.yaml)
d_model 1024, 24 layers, 8 heads (head_dim 128), max_seq_len 32768, vocab 50304 (GPT-2 padded), bf16.
- Norm: **RMSNorm PARAMETER-FREE** (F.rms_norm no learnable gain). Pre-norm ln_1/ln_2, final norm, + QK-norm.
- MLP: **ReLU² (squared-ReLU) 4× expansion, NO gating** (backbone docstring says "GLU" — STALE, actual = FusedReLUSquaredMLP). relu(x@W1ᵀ)²@W2, W1 Kaiming, **W2 ZERO-init** (muP). W2 stored transposed.
- Attention: **MHA not GQA** (Q,K,V all num_heads). Fused Wqkv (3,hdim,dim). scale 1/sqrt(head_dim). **QK-norm** RMSNorm on q,k before RoPE. Wout ZERO-init, FP8Linear (fp8 off in configs).
- **RoPE half-truncated** (HalfTruncatedRotary): freqs (1/1024)^linspace → **base/theta=1024 (unusually small vs 10000)**, UPPER QUARTER of freqs ZEROED (only half head_dim rotated). YouJiacheng.
- Residual (modded-nanoGPT tricks): per-sublayer learnable resid_lambdas(√1.1)/post_lambdas(1.0); **U-Net skip connections** first→second half (skip_weights ones(L/2)); **x0 embedding highway** x0_lambdas zeros(L) (zero-init no-op start, adds raw embedding per layer). DropPath avail rate 0.
- Backbone EXCLUDES embedding+head (in TS2TSTrainingModule). Optional bigram hash embedding (off default).

## VE BANKS = modded-nanoGPT VALUE EMBEDDINGS (the undocumented feature!)
Config ve_layers:[1,2,21,22,23], shared_ve_bank:false.
- value_embeds param (num_banks, vocab, model_dim) bf16, init 0.01*randn. ONE bank per VE layer (or shared). **~98MB/bank, 5 banks = 258M params = 42% of model** (TUNING_LESSONS.md:27).
- ve_gate_bank (num_banks, num_heads, 12) ZERO-init → gate no-op at start.
- Wiring: prepare_ve(input_ids) looks up per-VE-layer (T,D) slab by token id + gate; injected INSIDE ATTENTION on VALUE tensor before kernel (_inject_ve attention.py:48-65): gate = 2*sigmoid(Linear(concat(x_norm[...:6], ve[...:6]), ve_gate_w)) range 0-2, then v = v + gate*ve. Token-identity-keyed value vector added to attention values, mixed through attention weights. "no kernel changes."
- **VE-off vs VE-on axis**: data-gated. Wiki (~2 tok/param) VE-off BEAT VE-on + ~4% faster → kept off. Arxiv (38B) converged but VE-on did NOT overtake even at 38B (predicted flip didn't materialize). "VE needs data-rich regime to pay for its params" (TUNING_LESSONS.md:27-32). Gate 0-init + 0.01 bank init = near-no-op, must earn contribution.
- Optimizer: value_embeds/bigram_embed → AdamW β1=0.75 NO weight decay ("rows not updated must not shrink"); ve_gate_bank → standard scalar AdamW.
FLAG: gating on first 6 dims (12-wide gate input) = fixed magic number, no in-repo rationale (inherited from reference).

## Weight tying + untie_at_frac (mid-training untie)
weight_tying:true → embedding.weight aliases FusedLinearCELoss.weight (one storage). untie_at_frac:0.667 → split at round(total_steps*0.667) fired by LRCooldownScheduler split_fn: clone embed→new Parameter for loss_fn.weight, add to AdamW group, **TRANSFER Adam moments** (exp_avg/exp_avg_sq so head not cold), register name-keyed for portable resume, DDP pre_step_fn manually all-reduces new grad. Resume: detects post-split ckpt, breaks aliasing before load, resumed_split flag prevents re-fire. Rationale: tied trains faster/stabler early, untie+cooldown late lets head specialize. Both need max_optimizer_steps set (pre-2026-07-15 wiki silently skipped).

## Logit soft-cap (logit_softcap 30)
Training: fused inside Liger CE kernel (LigerFusedLinearCrossEntropyLoss softcap, logits never materialized). Inference: explicit cap*tanh(logits/cap) after head. Gemma-2 style. FLAG: to_inference_model logit_softcap defaults None → inference could silently run WITHOUT cap unless caller re-passes 30.

## Attention plumbing / train vs inference forward
TS2TSAttention subclasses FlexSelfAttention only for init, never parent forward. Dispatch on mutable self.backend: flex (torch.compile flex_attention + BlockMask) | triton (kernel from mask type: TritonMaskInputs→v12 BIM, DocCausalTritonMaskInputs→varlen BIM v1) | triton_v17/v18 (v18=v17+nan_to_num, CLAUDE.md default) | varlen_bim_v2. Triton wrappers torch._dynamo.disable. **Backend swappable at runtime zero-copy** (layer.attn.backend='flex' switches train-triton→inference-flex). batch=1 required.
Training forward (TS2TSTrainingModule): batch in loss out; shift tokens; build mask; embed + VE + bigram; backbone + norm; **fused linear+CE (Liger, dynamo-disabled)** logits never materialized. **MTP (multi-token prediction)**: offset-2/3 losses same head, weights decay over mtp_decay_micro_steps, SKIPPED in eval.
Inference forward (TS2TSModel.forward_inference): no_grad, tokens→logits; (mask_type,backend) overridable per call (A/B eval same model cross_doc_link vs doc_causal); plain F.linear head + explicit tanh softcap; materializes [1,T,V]; no MTP.

## Novel/publishable
From modded-nanoGPT (attributed): half-trunc RoPE base=1024, QK-norm, zero-init Wout, ReLU²+zero-W2, resid/post lambdas, U-Net skips, x0 highway, bigram, VALUE EMBEDDINGS, Muon.
This-project: (1) graph-aware cross-doc attention (4 mask types + LinkDetectors) = THE thesis; (2) custom BIM Triton kernels @32k A100 + swappable flex reference; (3) fused ReLU² MLP transposed-W2; (4) deferred mid-training untie w/ Adam-moment transfer + portable resume; (5) data-gated VE-bank empirical finding.
FLAGS: "GLU" docstring stale; inference softcap depends on caller re-passing 30; ve_gate 6-dim magic; RESULTS 3.75 vs 83 diff runs.

## → LIT REVIEW IMPLICATIONS
- O5/O6: modded-nanoGPT tricks each need citation — value embeddings (find source), QK-norm (Henry/Query-Key Norm), U-Net skips in transformers, x0/embedding-highway/residual-scaling, half-truncated RoPE, zero-init output/muP, bigram hash embedding.
- MTP: Multi-Token Prediction (Gloeckle/Meta, DeepSeek-V3 MTP).
- Liger kernels, fused cross-entropy, FP8 training.
- Parameter-free RMSNorm; squared-ReLU (Primer) [have].
