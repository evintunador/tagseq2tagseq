# TAGSeq2TAGSeq

## Custom kernels are the default — don't revert to flex

`attention_backend: triton` (default) automatically selects the fastest custom Triton kernel per mask type: `cross_doc_link` → `triton_v18`, `doc_causal` → `varlen_bim_v2`. Don't set it to `flex` unless benchmarking against FlexAttention.

## Never disable torch.compile

`--model.compile false` / `model.compile: false` causes FlexAttention to fall back to dense O(T²) math attention, which immediately OOMs at T=32k. All configs have it on. Don't override it — just accept the ~2–3 min Triton compile on first step.

## Smoke tests use real-use-case parameters only

Never change `model.compile`, `attention_backend`, or any other setting that alters the compute graph just to make a smoke test faster or avoid a compile wait. Limiting `max_optimizer_steps`, `val_steps`, or dataset size is fine. Disabling compile or switching to `flex` silently tests a completely different execution path and masks real failures.

## launch_slurm.py argparse has no defaults

The launcher's `_training_worker` parser is bare (`add_help=False`, no argument definitions). All overrides must use dotted keys (`--data.dataset_dir`, `--model.mask_type`, etc.). Argparse defaults silently clobber YAML — this burned a 3h run at 2k context instead of 32k.

## Multi-rank runs need a pre-warmed compile cache

Many ranks JIT-compiling the same Triton/inductor kernels concurrently corrupts the compiler and segfaults a rank during first-step compilation (rare at 4 GPUs, near-certain at 8). `launch_slurm.py` mitigates this with `TORCHINDUCTOR_COMPILE_THREADS=1` and an optional shared compile cache: set `TS2TS_SHARED_COMPILE_CACHE` to a dir on `/fss-data`, warm it once with a short run **at the same world_size** (the distributed Muon optimizer compiles shard-shape kernels a single-GPU warmup never produces), then point the real run at the same dir so every rank reads the cache instead of compiling.

## Never launch jobs simultaneously — always stagger

Start training jobs **one at a time**, waiting for each to reach its first training step (past `torch.compile`) before launching the next. Two independent failure modes both come from launching together:

1. **Concurrent cold compilation** thrashes/deadlocks the Triton/inductor compiler (see the compile-cache section above) — worst when jobs share a node or a compile-cache dir.
2. **Run-directory collision**: `main.py` derives its run dir from `datetime.now()` at **second** precision (`runs/YYYYMMDD_HHMMSS`). Two jobs started in the same second get the same dir, and `ReproducibilityManager` aborts the second with *"Output directory ... already contains a 'reproducibility' folder"*.

This applies to SLURM submissions, down-node `launch_downnode.sh` runs, and especially multiple jobs sharing one node (e.g. thestack + arxiv + wiki on disjoint GPU sets). Launch, confirm "Training: Nit" appears in the log, then launch the next.
