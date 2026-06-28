#!/usr/bin/env python
"""Pre-compute packed epochs for density-aware batch scheduling.

Each epoch pre-computes all packs from a pre-tokenized dataset, assigns them
kv_block_count density metrics, and groups them into density buckets for
load-balanced DDP training.  Works for TheStack (repo-partitioned) and for
flat-identifier datasets (Wikipedia, ArXiv) via the graph-community partitioner;
the dataset type is auto-detected from the identifier format.

Usage
-----
    python precompute_epochs.py \\
        --dataset-dir  /fss/evin_t/tagseq2tagseq_artifacts/pretokenized_datasets/thestack/splits/train \\
        --output-dir   schedules/thestack_bfs \\
        --n-epochs     5 \\
        --strategy     bfs \\
        --local-seq-len 32768 \\
        --n-buckets    32 \\
        --n-workers    16 \\
        --seed         42 \\
        --link-detector python \\
        --device       cuda:0

Epoch i is written to {output-dir}/epoch_{i}/ and uses seed+i.
Already-completed epochs are skipped (resume-safe).
"""

import argparse
import logging
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-compute packed epochs for density-aware batch scheduling.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-dir",    type=str, required=True,
                        help="Pre-tokenized TheStack dataset directory.")
    parser.add_argument("--output-dir",     type=str, required=True,
                        help="Root output dir; epoch i → {output-dir}/epoch_{i}/.")
    parser.add_argument("--n-epochs",       type=int, default=1,
                        help="Number of epochs to pre-compute.")
    parser.add_argument("--strategy",       type=str, default="bfs",
                        choices=["bfs", "dfs", "random_walk", "random"],
                        help="Graph traversal strategy.")
    parser.add_argument("--local-seq-len",  type=int, default=32768,
                        help="Token budget per pack (== model.max_seq_len).")
    parser.add_argument("--n-buckets",      type=int, default=32,
                        help="Number of density buckets.")
    parser.add_argument("--n-workers",      type=int, default=8,
                        help="Subprocess workers for pack generation.")
    parser.add_argument("--seed",           type=int, default=42,
                        help="Base seed; epoch i uses seed+i.")
    parser.add_argument("--link-detector",  type=str, default="python",
                        choices=["python", "markdown", "arxiv"],
                        help="Link detector type (python=TheStack, markdown=Wikipedia, arxiv=ArXiv).")
    parser.add_argument("--layout-policy",  type=str, default="null",
                        help="Layout policy name (null, bos_eos, etc.).")
    parser.add_argument("--max-grants",     type=int, default=64,
                        help="max_grants for CrossDocLinkMaskCreator GPU pass.")
    parser.add_argument("--order-mode",     type=str, default="prefer_targets_first",
                        help="Document ordering mode for PackBatchSampler.")
    parser.add_argument("--device",         type=str, default="cuda:0",
                        help="CUDA device for kv_block_count GPU pass.")
    parser.add_argument("--gpu-kv-pass", action="store_true",
                        help="Compute kv_block_count via GPU BlockMask (Method B, ~36ms/pack "
                             "sequential) instead of the default CPU analytical method "
                             "(Method C, ~1ms/pack parallel-in-workers). Only useful for "
                             "post-hoc verification that C==B on a real dataset.")
    parser.add_argument("--log-level",      type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from data.epoch_precompute import EpochPrecomputer

    precomputer = EpochPrecomputer(
        dataset_dir=args.dataset_dir,
        token_budget=args.local_seq_len,
        n_buckets=args.n_buckets,
        n_workers=args.n_workers,
        strategy=args.strategy,
        link_detector=args.link_detector,
        layout_policy=args.layout_policy,
        max_grants=args.max_grants,
        order_mode=args.order_mode,
        device=device,
        use_analytical=not args.gpu_kv_pass,
    )

    for i in range(args.n_epochs):
        epoch_seed = args.seed + i
        epoch_dir = str(output_dir / f"epoch_{i}")
        logger.info("=== Epoch %d / %d (seed=%d) ===", i, args.n_epochs, epoch_seed)
        precomputer.run(epoch_dir=epoch_dir, epoch_idx=i, seed=epoch_seed)

    logger.info("All %d epochs pre-computed → %s", args.n_epochs, args.output_dir)


if __name__ == "__main__":
    main()
