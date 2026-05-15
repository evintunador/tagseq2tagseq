# TAGSeq2TAGSeq

## Custom kernels are the default — don't revert to flex

`attention_backend: triton` (default) automatically selects the fastest custom Triton kernel per mask type: `cross_doc_link` → `triton_v18`, `doc_causal` → `varlen_bim_v2`. Don't set it to `flex` unless benchmarking against FlexAttention.

## Never disable torch.compile

`--model.compile false` / `model.compile: false` causes FlexAttention to fall back to dense O(T²) math attention, which immediately OOMs at T=32k. All configs have it on. Don't override it — just accept the ~2–3 min Triton compile on first step.

## Smoke tests use real-use-case parameters only

Never change `model.compile`, `attention_backend`, or any other setting that alters the compute graph just to make a smoke test faster or avoid a compile wait. Limiting `max_optimizer_steps`, `val_steps`, or dataset size is fine. Disabling compile or switching to `flex` silently tests a completely different execution path and masks real failures.

## launch_slurm.py argparse has no defaults

The launcher's `_training_worker` parser is bare (`add_help=False`, no argument definitions). All overrides must use dotted keys (`--data.dataset_dir`, `--model.mask_type`, etc.). Argparse defaults silently clobber YAML — this burned a 3h run at 2k context instead of 32k.
