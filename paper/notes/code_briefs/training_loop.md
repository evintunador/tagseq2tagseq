<!-- PROVENANCE
Written against commit 6134163 (main @ 2026-08-07). This brief describes the code as of
that commit; it is NOT authoritative if the source has changed since. Before trusting it,
check whether the covered sources have drifted:
    git diff --stat 6134163..HEAD -- main.py launch_slurm.py
Empty output = brief still current. Non-empty = re-verify the changed parts against source.
Covered sources: main.py launch_slurm.py
-->

# CODE BRIEF: training loop & distributed infra (agent ac90664e)
main.py = 1633 LINES (80764 BYTES, not 80k lines). Training STEP not in main.py — hands to LLM-composed loop.

## Training step
main.py:1399 smart_train() = LLM-COMPILER: selects atomic-feature files by kwarg-set, emits cached run_training. Production loop committed at artifacts/train_loops/llm_compiled/device-grad_accum-grad_norm_clip-logging-multi_epoch-multi_val_bucketed-tqdm.py.
Per micro-batch: to_device, loss=model(batch), loss/=accum_steps, backward.
TS2TSTrainingModule.forward: tokens (B,T+1) → input=tokens[:,:-1] target=tokens[:,1:]; mask from INPUT slice; embed→backbone(+ve_map+bigram)→RMSNorm; loss=torch._dynamo.disable(loss_fn)(x,target) — Dynamo disabled around Liger CE (inductor buffer-overread → CUDA illegal-memory). 
CE: FusedLinearCELoss wraps LigerFusedLinearCrossEntropyLoss(ignore_index=-100, softcap), lm_head FUSED into CE (no materialized logits). 
**MASKING: packing produces NO padding, NO explicit label masking.** build_packed_batch = dense (1,T) no pad, no labels/loss_mask tensor. ignore_index=-100 EFFECTIVELY INERT on main loss (every shifted target real). EOS/doc-boundary handling = STRUCTURAL in attention mask NOT loss. FLAG: no place writes -100 into targets; if paper claims boundary/EOS excluded from loss, NOT what code does.
MTP (training-only): +w_k·CE(x[:,:-k], tokens[:,k+1:]) weights decay linear over mtp_decay_micro_steps; eval plain single-offset (comparable val).
Grad accum: loss/accum, backward every micro, clip+step+zero every accum-th, trailing flush. clip_grad_norm 1.0. NO DDP no_sync (moot at accum_steps=1 production).
Density-bucket draw: BucketedPackDataset __iter__ — same bucket_seq every rank (seed=epoch_idx), rank r draws bucket_consumed[B]+r, consumed+=world_size → ALL RANKS SAME DENSITY BUCKET per step.

## LR schedule = WSD (LRCooldownScheduler main.py:60-206)
Absolute-step piecewise: warmup step/warmup_steps (300); stable 1.0; decay linear to min_lr_ratio(0.1), progress=(step-cooldown_start)/(total-cooldown_start). cooldown_start=int(total*(1-cooldown_frac)), cooldown_frac 0.4 → e.g. 9000/15000. Co-scheduled: muon momentum warmup 0.85→max, deferred untie.
**max_optimizer_steps:null auto-derive**: sum n_packs across epoch metadata // (world_size×accum). Chinchilla NOT computed in code — configs PIN max_optimizer_steps manually for ~20 tok/param. Guard: null → cooldown_frac forced 0 + untie_at_frac nulled (the 2026-07-03 constant-LR-no-untie footgun).

## Checkpoint/resume
Absolute-step resume (resumed_steps from metadata, start_step, records absolute; fixes resume-of-resume rewind). max_optimizer_steps→remaining; total_steps_original=remaining+resumed reconstructs WSD clock. Bucket resumes from BucketState. Optimizer name-keyed world-size-portable (muon_full_v1).
**Host-OOM barrier** (_save_ckpt_full): full optimizer save COLLECTIVE (all_gather every rank), rank0 writes, **non-rank0 del full_state BEFORE cpu_barrier** — else ~model-size host RAM/rank × world_size spike → host-OOM SIGKILL rank0 → peers die on gloo barrier "connection closed by peer". "5/6 merged_v2 runs died this way." PUBLISHABLE reliability finding.
Latent bug1 NULL-path cooldown inflation (GUARDED: store remaining not full derived, else double-count resumed → hold peak LR too long). Latent bug2 silent optimizer partial-restore (GUARDED: raises on skipped param else cold momentum for some params). Untie-aware weight-load guard (break tie before load, resumed_split prevents re-fire).
Periodic latest.pt (save_latest_interval) + best_model.pt; both full portable state + bucket_state. Must pass save_latest_interval/start_step/bucket_state_fn as int/present else smart_train deselects multi_val_bucketed feature → silently drops bucket-resume.

## torch.compile / TorchInductor
torch.compile(backbone, dynamic=True) BEFORE DDP wrap; optimize_ddp default True (all-reduce graph-breaks overlap). DDP static_graph=True, find_unused_parameters=False, bucket_cap_mb=256.
launch_slurm.py: TORCHINDUCTOR_COMPILE_THREADS=1 (default 32/proc → 32×N subprocs oversubscribe → corrupt compile-worker pool); TS2TS_SHARED_COMPILE_CACHE prewarmed on /fss-data warm at SAME world_size (distributed-Muon shard-shape kernels single-GPU warmup never produces); disable static CUDA launcher.
Concurrent cold compile 2 failure modes: (1) compiler corruption/segfault (rare@4 near-certain@8 GPU); (2) NCCL collective-count desync (optimize_ddp inserts collectives depending on graph shape; non-identical graphs → different #all-reduces → step-1 hang). Fix _compile_warmup: rank0 broadcasts one batch, synchronized fwd+bwd w/ barrier between, side-effect-free (snapshot MTP+dataset state). Honest note: original 8-GPU hang was actually rank0 CUDA-OOM from co-tenant, warmup kept as correct invariant.

## DDP density-balancing + distributed optimizer
Claim: each accum step draws one density bucket across all ranks → per-rank backward cost (∝ kv_block_count) matched → no straggler all-reduce stall. world_size NOT baked at precompute (BucketState reload changes world_size/accum).
MuonWithAuxAdam shards Muon round-robin (i%world_size), orthogonalize slice then all_gather; AdamW UNSHARDED/replicated → plain state_dict incomplete → name-keyed state_dict_full/load_state_dict_full.
CAVEAT on claim: approximate during max_grants warmup (bucket kv_block_count reflects FINAL max_grants); outward-scan fallback near epoch tail breaks equal-density.

## Novel/publishable
1. Density-bucketed DDP data scheduling for attention-cost balancing, world-size-agnostic resume.
2. World-size-portable name-keyed Muon+AdamW state (muon_full_v1) — exact resume across changed world_size + param reorder + untie.
3. Deferred embed/lm_head untie mid-run + Adam-state transfer + manual grad all-reduce + resume-safety.
4. NorMuon: Adafactor low-rank 2nd moment + BF16 mantissa tracking (uint16 low-bits, sub-ULP late updates).
5. Systems-reliability findings: synchronized compile warmup, shared-cache/COMPILE_THREADS=1, pre-barrier del full_state host-OOM fix (5/6 runs, quantified). Publishable "lessons from multi-node speedrun training."
6. smart_train LLM-compiled training loop from cached atomic features (methods footnote).
Provenance modded-nanoGPT: cautious WD (dev: lr×wd not lr²×wd), resid scaling, bigram hash, VE, Muon+aux-Adam, spectral-norm LR ×max(1,M/N)^0.5, Polar Express.
FLAGS: loss masking (no -100 in packed targets, ignore_index inert, boundaries only in attn mask); didn't deeply read precompute_epochs; params_pad over-alloc when len%world_size==0; config values from thestack_veoff_dc.yaml.

## → LIT REVIEW IMPLICATIONS
- O3 WSD (warmup-stable-decay MiniCPM/DeepSeek, cosine, cooldown/annealing).
- O4 Chinchilla [have] + ~20 tok/param.
- O7 systems: DDP (Li) [have], NCCL, gradient clipping, fault tolerance/checkpointing in large-scale training, elastic/portable checkpointing.
- Multi-node LLM training reliability lessons (OPT/BLOOM chronicles, Gemini/Llama infra papers).
- Grad accumulation, mixed-precision [have].
