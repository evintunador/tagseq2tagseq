import argparse
import itertools
import logging
import os
import datetime
from pathlib import Path
from typing import Dict, Any
import json

# Set before any CUDA allocation so the memory allocator picks it up.
# Expandable segments dramatically reduce fragmentation when sequence lengths
# vary across steps (which they do for packed graph batches).
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from tunalab.configuration import compose_config
from tunalab.distributed import DistributedManager, setup_signal_handlers
from tunalab.reproducibility import ReproducibilityManager
from tunalab import tracking
from tunalab.smart_train import smart_train
from tunalab.optimizers.muon import MuonWithAuxAdam, SingleDeviceMuonWithAuxAdam
from tunalab.llm_compilers.auto import get_default_llm_client

# Local imports
from model import TS2TSTrainingModule
import tiktoken

from model.graph_traversal.block_mask_creator import (
    make_mask_creator_callable,
    make_mask_creator_callable_from,
)
from model.graph_traversal.cross_doc_mask import CrossDocLinkMaskCreator
from model.graph_traversal.markdown_link_detector import MarkdownLinkDetector
from model.graph_traversal.python_import_detector import PythonImportDetector
from data.dataset import GraphIndex, PretokShardedBackend
from data.packed_dataset import PackedSequenceDataset
from data.layout import make_layout_policy
from data.pack_sampler import PackBatchSampler
from data.traversal import (
    BFSStrategy,
    DFSStrategy,
    RandomSelectionStrategy,
    RandomWalkStrategy,
)


logger = logging.getLogger(__name__)


class LimitedDataLoader:
    """Wraps a DataLoader to yield at most ``max_batches`` items per iteration.

    Creates a fresh islice on each call to ``__iter__``, so the underlying
    loader can be iterated more than once (e.g. repeated validation passes).
    """
    def __init__(self, loader: DataLoader, max_batches: int) -> None:
        self.loader = loader
        self.max_batches = max_batches

    def __iter__(self):
        return itertools.islice(iter(self.loader), self.max_batches)


def main(cfg: Dict[str, Any], dist: DistributedManager, rep: ReproducibilityManager):
    """Main training entry point for tagseq2tagseq."""

    # Register SIGTERM/SIGINT handlers so SLURM job cancellation doesn't
    # leave ranks blocked in a collective operation.
    setup_signal_handlers()

    # -------------------------------------------------------------------------
    # 1. Setup Logging & Reproducibility
    # -------------------------------------------------------------------------
    if rep.output_dir:
        log_dir = os.path.join(rep.output_dir, "logs")
        tracking.init(log_dir, dist.rank)

    dist.set_seed(cfg.get("seed", 42))

    # Only rank 0 writes the hyperparameter dump; no point writing N copies.
    if rep.output_dir and dist.is_main_process:
        json_path = os.path.join(rep.output_dir, "hyperparameters.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)

    # -------------------------------------------------------------------------
    # 1b. Resume: read checkpoint metadata early so max_optimizer_steps can be
    #     adjusted before the LimitedDataLoader is built.
    # -------------------------------------------------------------------------
    # compose_config preserves CLI hyphens when called via the bare SLURM launcher
    # parser, so --resume-from becomes cfg['resume-from'] there, but dest="resume_from"
    # produces cfg['resume_from'] when main.py is invoked directly.  Handle both.
    resume_from   = cfg.get('resume_from') or cfg.get('resume-from')
    resume_ckpt   = None   # loaded lazily below; freed after state is restored
    resumed_steps = 0

    if resume_from:
        if not os.path.exists(resume_from):
            raise FileNotFoundError(f"--resume-from checkpoint not found: {resume_from}")
        logger.info("Loading resume checkpoint: %s", resume_from)
        resume_ckpt   = torch.load(resume_from, map_location='cpu', weights_only=False)
        resumed_steps = int(resume_ckpt.get('metadata', {}).get('step', 0))
        resumed_val   = resume_ckpt.get('metadata', {}).get('val_loss', float('nan'))
        logger.info("Checkpoint: step=%d  val_loss=%.4f", resumed_steps, resumed_val)

        _max = cfg.get('train_loop', {}).get('max_optimizer_steps')
        if _max is not None:
            remaining = _max - resumed_steps
            if remaining <= 0:
                raise ValueError(
                    f"Checkpoint step ({resumed_steps}) >= max_optimizer_steps ({_max}); "
                    "nothing left to train."
                )
            cfg['train_loop']['max_optimizer_steps'] = remaining
            logger.info(
                "max_optimizer_steps adjusted: %d total − %d done = %d remaining",
                _max, resumed_steps, remaining,
            )

    # -------------------------------------------------------------------------
    # 2. Data Loading Setup
    # -------------------------------------------------------------------------
    dataset_dir_str = cfg.get('data', {}).get('dataset_dir')
    if not dataset_dir_str:
        logger.error("No dataset_dir specified in config.")
        return

    dataset_dir = Path(dataset_dir_str)
    if not dataset_dir.is_dir():
        logger.error("Dataset directory not found: %s", dataset_dir)
        return

    logger.info("Initializing GraphIndex from %s", dataset_dir)
    graph_index = GraphIndex(dataset_dir)

    # The backend handles memory-mapping of token shards
    backend = PretokShardedBackend(graph_index)

    # Configure Layout Policy
    # Options: null | bos_eos | identifier_prefix | identifier_prefix_bos_eos
    layout_policy_name = cfg.get('data', {}).get('layout_policy', 'null')
    enc = tiktoken.get_encoding(graph_index.metadata.get('tokenizer', 'gpt2'))
    layout_policy = make_layout_policy(
        name=layout_policy_name,
        encode_fn=enc.encode_ordinary,
    )

    # Configure Traversal Strategy
    strategy_name = cfg.get('data', {}).get('strategy', 'random')
    if strategy_name == "random":
        strategy_factory = lambda: RandomSelectionStrategy()
    elif strategy_name == "random_walk":
        strategy_factory = lambda: RandomWalkStrategy(edge_mode="outgoing", restart_prob=0.05)
    elif strategy_name == "bfs":
        strategy_factory = lambda: BFSStrategy(edge_mode="outgoing")
    elif strategy_name == "dfs":
        strategy_factory = lambda: DFSStrategy(edge_mode="outgoing")
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    # Each rank gets a unique sampler seed so they traverse different parts of
    # the graph simultaneously.  Using base_seed + rank ensures repeatability
    # while guaranteeing per-rank diversity.
    base_seed = cfg.get("seed", 42)
    rank_seed = base_seed + dist.rank

    pack_sampler = PackBatchSampler(
        graph=graph_index,
        strategy_factory=strategy_factory,
        token_budget=cfg.get('model', {}).get('max_seq_len', 2048),
        doc_budget=cfg.get('data', {}).get('doc_budget'),
        overflow_policy="truncate",
        doc_level_trim_side="tail",
        pack_level_trim_side="head",
        max_candidates_per_component=1000,
        seed=rank_seed,
        order_mode=cfg.get('data', {}).get('order_mode', 'prefer_targets_first'),
        layout_policy=layout_policy,
    )

    dataset = PackedSequenceDataset(
        graph=graph_index,
        backend=backend,
        pack_sampler=pack_sampler,
        layout_policy=layout_policy,
        as_2d=True,
    )

    train_loader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=0,
    )
    max_optimizer_steps = cfg.get('train_loop', {}).get('max_optimizer_steps')
    if max_optimizer_steps is not None:
        accum_steps = cfg.get('train_loop', {}).get('atomic_feature_kwargs', {}).get('accum_steps', 1)
        train_loader = LimitedDataLoader(train_loader, max_batches=max_optimizer_steps * accum_steps)

    # Validation loader — same dataset/graph but with a different seed so the
    # sampler draws different packs.  We cap it at val_steps batches per pass
    # since PackedSequenceDataset is an infinite iterable.
    #
    # TODO(@jamesljr): Replace this stopgap with proper held-out validation splits per dataset.
    # The right strategy differs by dataset type:
    #   - Stack (code repos): hold out some % of repositories entirely — clean since
    #     repos are largely self-contained subgraphs with minimal cross-repo links.
    #   - Wikipedia/SimpleWiki: random article splits are problematic because the
    #     hyperlink graph is dense and nearly every article links to something in
    #     train. Need to identify a densely-connected sub-graph (e.g., a topical
    #     cluster) to use as val, so val packs contain enough internal links to be
    #     representative of the training distribution.
    # Until then, val is drawn from the same graph with a different RNG seed, which
    # leaks data but is fine for loss tracking during initial development.
    val_pack_sampler = PackBatchSampler(
        graph=graph_index,
        strategy_factory=strategy_factory,
        token_budget=cfg.get('model', {}).get('max_seq_len', 2048),
        doc_budget=cfg.get('data', {}).get('doc_budget'),
        overflow_policy="truncate",
        doc_level_trim_side="tail",
        pack_level_trim_side="head",
        max_candidates_per_component=1000,
        seed=rank_seed + 1,
        order_mode=cfg.get('data', {}).get('order_mode', 'prefer_targets_first'),
        layout_policy=layout_policy,
    )
    val_dataset = PackedSequenceDataset(
        graph=graph_index,
        backend=backend,
        pack_sampler=val_pack_sampler,
        layout_policy=layout_policy,
        as_2d=True,
    )
    val_steps = cfg.get('train_loop', {}).get('val_steps', 10)
    val_loader = LimitedDataLoader(
        DataLoader(val_dataset, batch_size=None, num_workers=0),
        max_batches=val_steps,
    )

    # -------------------------------------------------------------------------
    # 3. Model & Optimizer Setup
    # -------------------------------------------------------------------------
    logger.info("Initializing Model...")

    tokenizer_name = graph_index.metadata.get('tokenizer', 'gpt2')
    vocab_size = 50257 if tokenizer_name == 'gpt2' else cfg['model'].get('vocab_size', 50257)

    # Create block mask creator
    mask_type = cfg.get('model', {}).get('mask_type', 'doc_causal')
    if mask_type == 'cross_doc_link':
        link_detector_name = cfg.get('model', {}).get('link_detector')
        if not link_detector_name:
            raise ValueError(
                "model.link_detector must be set to 'markdown' or 'python' "
                "when model.mask_type is 'cross_doc_link'"
            )
        enc = tiktoken.get_encoding('gpt2')
        if link_detector_name == 'markdown':
            detector = MarkdownLinkDetector(decode_fn=enc.decode)
        elif link_detector_name == 'python':
            detector = PythonImportDetector(decode_fn=enc.decode)
        else:
            raise ValueError(
                f"Unknown model.link_detector '{link_detector_name}'. "
                "Use 'markdown' (Wikipedia) or 'python' (TheStack)."
            )
        block_mask_creator = make_mask_creator_callable_from(
            CrossDocLinkMaskCreator(
                link_detector=detector,
                max_grants=cfg.get('model', {}).get('max_grants', 64),
            )
        )
    else:
        block_mask_creator = make_mask_creator_callable(mask_type)

    model = TS2TSTrainingModule.from_config(
        vocab_size=vocab_size,
        num_layers=cfg['model']['num_layers'],
        model_dim=cfg['model']['model_dim'],
        num_heads=cfg['model']['num_heads'],
        max_seq_len=cfg['model']['max_seq_len'],
        dropout=cfg['model'].get('dropout', 0.0),
        drop_path_rate=cfg['model'].get('drop_path_rate', 0.0),
        block_mask_creator=block_mask_creator,
        fp8=cfg['model'].get('fp8', False),
        weight_tying=cfg['model'].get('weight_tying', True),
        ignore_index=cfg['model'].get('ignore_index', -100),
        dtype=getattr(torch, cfg['model'].get('dtype', 'bfloat16')),
    ).to(dist.device)

    # Build optimizer param groups BEFORE compile/DDP so that named_parameters()
    # gives clean names and weight-tied tensors are only counted once.
    logger.info("Initializing Optimizer...")
    muon_params, adamw_params = [], []
    seen_ids: set = set()
    for name, param in model.named_parameters():
        if id(param) in seen_ids:
            continue
        seen_ids.add(id(param))
        # Backbone 2-D weights use Muon; embedding, norms, biases use AdamW.
        if 'backbone' in name and param.ndim >= 2:
            muon_params.append(param)
        else:
            adamw_params.append(param)

    param_groups = [
        dict(
            params=muon_params,
            use_muon=True,
            lr=cfg['optimizer']['muon_lr'],
            momentum=cfg['optimizer'].get('momentum', 0.95),
            weight_decay=cfg['optimizer']['wd'],
        ),
        dict(
            params=adamw_params,
            use_muon=False,
            lr=cfg['optimizer']['adamw_lr'],
            betas=(cfg['optimizer'].get('beta1', 0.9), cfg['optimizer'].get('beta2', 0.95)),
            weight_decay=cfg['optimizer']['wd'],
        ),
    ]

    # MuonWithAuxAdam handles distributed Muon parameter-sharding via all_gather.
    # SingleDeviceMuon is used when there is no process group (single GPU / CPU).
    if dist.is_distributed:
        optimizer = MuonWithAuxAdam(param_groups)
        logger.info("Using distributed MuonWithAuxAdam (world_size=%d)", dist.world_size)
    else:
        optimizer = SingleDeviceMuonWithAuxAdam(param_groups)
        logger.info("Using SingleDeviceMuonWithAuxAdam (single process)")

    # -------------------------------------------------------------------------
    # 3b. Restore weights and AdamW state from checkpoint (if resuming).
    #
    # Muon momentum buffers are world_size-dependent: each rank only saves its
    # own shard, so they cannot be remapped when world_size changes.  We restore
    # the AdamW state (embedding + skip_weights) which IS portable, and let Muon
    # restart with cold momentum (recovers within ~100 steps).
    # -------------------------------------------------------------------------
    if resume_ckpt is not None:
        # --- model weights ---
        model_sd = resume_ckpt['model']
        # Strip 'module.' prefix produced by DDP-wrapped saves, if present.
        model_sd = {
            (k[len('module.'):] if k.startswith('module.') else k): v
            for k, v in model_sd.items()
        }
        model.load_state_dict(model_sd, strict=True)
        logger.info("Resume: model weights restored.")

        # --- AdamW optimizer state ---
        saved_opt = resume_ckpt.get('optimizer', {})
        saved_state  = saved_opt.get('state', {})
        saved_groups = saved_opt.get('param_groups', [])

        adamw_indices = set()
        for g in saved_groups:
            if not g.get('use_muon', False):
                adamw_indices.update(g['params'])

        portable_state = {k: v for k, v in saved_state.items() if k in adamw_indices}
        if portable_state:
            cur_sd = optimizer.state_dict()
            cur_sd['state'].update(portable_state)
            optimizer.load_state_dict(cur_sd)
            logger.info(
                "Resume: AdamW state restored for %d param(s); "
                "Muon momentum initialised cold (world_size changed: %d → %d).",
                len(portable_state),
                len(saved_groups[0].get('params', [])) + len(adamw_indices),  # old world total
                dist.world_size,
            )

        del resume_ckpt   # free ~1.8 GB
        resume_ckpt = None

    # Compile backbone BEFORE DDP wrapping.  torch.compile operates on the
    # backbone nn.Module; DDP adds communication hooks on top without
    # interfering with the compiled graph.
    if cfg['model']['compile']:
        logger.info("Compiling model backbone with torch.compile...")
        # optimize_ddp=True lets dynamo insert all-reduce graph breaks at the
        # right points during backbone backward, enabling overlap between
        # gradient compute and DDP bucket all-reduces.
        torch._dynamo.config.optimize_ddp = True
        model.backbone = torch.compile(
            model.backbone,
            dynamic=True,
            mode=cfg['model']['compile_mode'],
        )

    # Wrap in DDP for multi-GPU / multi-node training.
    # static_graph=True is safe because our forward graph is identical every
    # step (same mask type, same model structure).
    # bucket_cap_mb=256 reduces the number of all-reduce calls (default is
    # 25 MB, which creates ~36 buckets for a 900 MB gradient blob).
    if dist.is_distributed:
        logger.info(
            "Wrapping model in DDP (rank=%d, local_rank=%d, world_size=%d)",
            dist.rank, dist.local_rank, dist.world_size,
        )
        model = DDP(
            model,
            device_ids=[dist.local_rank],
            static_graph=True,
            find_unused_parameters=False,
            bucket_cap_mb=256,
        )

    # -------------------------------------------------------------------------
    # 4. Training Loop
    # -------------------------------------------------------------------------
    logger.info("Starting Training...")

    atomic_feature_kwargs = cfg.get('train_loop', {}).get('atomic_feature_kwargs', {})
    atomic_feature_kwargs.update({
        'enable_logging': True,
        'save_best_model': True,
        'val_loader': val_loader,
        'val_interval': cfg['train_loop'].get('val_interval', 50),
        'output_dir': rep.output_dir,
        'device': str(dist.device),
        'use_tqdm': dist.is_main_process,  # only rank 0 shows the progress bar
        'num_epochs': cfg['train_loop'].get('epochs', 1),
    })

    result = smart_train(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        llm_client=get_default_llm_client(),
        **atomic_feature_kwargs
    )

    logger.info("Training complete!")

    # Cleanup
    backend.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a TAGSeq2TAGSeq model on the TAGWiki dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--dataset-dir", dest="data.dataset_dir", type=str,
                        help="Path to the pre-tokenized dataset directory.")
    parser.add_argument("--strategy", dest="data.strategy", type=str, default="random",
                        choices=["random", "random_walk", "bfs", "dfs"],
                        help="Graph traversal strategy.")
    parser.add_argument("--max-seq-len", dest="model.max_seq_len", type=int, default=2048,
                        help="Maximum sequence length (token budget per pack).")
    parser.add_argument("--seed", dest="seed", type=int, default=42,
                        help="Random seed.")
    parser.add_argument("--resume-from", dest="resume_from", type=str, default=None,
                        help="Path to a best_model.pt checkpoint to resume training from. "
                             "Restores model weights and AdamW state; Muon momentum is "
                             "restarted cold (world_size-dependent, cannot be remapped). "
                             "max_optimizer_steps is automatically reduced by the checkpoint step.")

    config = compose_config(parser)

    # Run directory is created only by rank 0 (ReproducibilityManager handles this).
    run_dir = os.path.join(
        os.path.dirname(__file__), "runs",
        datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
    )

    dist_mgr = DistributedManager()
    rep = ReproducibilityManager(output_dir=run_dir, is_main_process=dist_mgr.is_main)

    with dist_mgr, rep:
        main(config, dist_mgr, rep)
