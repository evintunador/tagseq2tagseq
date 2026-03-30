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
from data.bucketed_pack_dataset import BucketedPackDataset, BucketState
from data.layout import make_layout_policy
from data.pack_sampler import PackBatchSampler
from data.traversal import (
    BFSStrategy,
    DFSStrategy,
    RandomSelectionStrategy,
    RandomWalkStrategy,
)


logger = logging.getLogger(__name__)


class LRCooldownScheduler:
    """Wraps an optimizer to apply linear LR cooldown in the final portion of training.

    LR is held at 1× for the first (1 - cooldown_frac) of total_steps, then
    decays linearly to min_lr_ratio× over the remaining steps.

    Optionally handles deferred embedding/lm_head weight untying (item 12 of the
    modernization plan).  During the tied phase the embedding grad is folded into
    the lm_head grad so Adam sees the combined signal on a single parameter.  At
    split_step the two matrices are given independent parameters with Adam state
    transferred from the lm_head, and subsequent lm_head grad all-reduces are
    performed manually (DDP only registered a hook on the original shared param).

    Designed to be transparent to smart_train: exposes the same interface as a
    normal optimizer (step, zero_grad, param_groups, state_dict, load_state_dict).

    Args:
        optimizer: The base optimizer to wrap.
        total_steps: Total number of optimizer steps for the full training run
            (not adjusted for resume — use the *original* schedule total).
        start_step: Step number at which this training run begins.  0 for a fresh
            run; ``resumed_steps`` when resuming from a checkpoint.
        cooldown_frac: Fraction of total_steps over which to decay the LR.
            E.g. 0.4 starts cooldown at step int(total_steps * 0.6).
        min_lr_ratio: Final LR as a fraction of the base LR.  0.1 means the LR
            decays to 10% of the configured value.
        pre_step_fn: If provided, called before each optimizer.step().  Used for
            manual gradient all-reduce of the newly-independent lm_head param
            after the split.
        split_step: Absolute step at which to perform the embedding/lm_head split.
            0 disables splitting.
        split_fn: Callable(optimizer) invoked once at split_step (after the last
            tied optimizer step) to create the independent lm_head parameter and
            transfer Adam state.
    """

    def __init__(
        self,
        optimizer,
        total_steps: int,
        start_step: int = 0,
        cooldown_frac: float = 0.4,
        min_lr_ratio: float = 0.1,
        muon_momentum_warmup_steps: int = 0,
        pre_step_fn=None,
        split_step: int = 0,
        split_fn=None,
    ):
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.cooldown_start = int(total_steps * (1.0 - cooldown_frac))
        self.min_lr_ratio = min_lr_ratio
        self._step_count = start_step
        # Capture base LRs from optimizer param_groups at construction time.
        # Callers should reset param_groups['lr'] to config values before
        # constructing this (so a resumed checkpoint's saved LR doesn't pollute
        # the base).
        self._base_lrs = [float(pg['lr']) for pg in optimizer.param_groups]
        # Momentum warmup: ramp from momentum_min to the configured max over
        # muon_momentum_warmup_steps steps.  Only applies to param groups with
        # use_muon=True.  0 disables the warmup.
        self.muon_momentum_warmup_steps = muon_momentum_warmup_steps
        self._momentum_max = next(
            (float(pg['momentum']) for pg in optimizer.param_groups if pg.get('use_muon')),
            0.95,
        )
        self._momentum_min = 0.85
        self._pre_step_fn = pre_step_fn
        self.split_step = split_step
        self._split_fn = split_fn
        self._split_done = (split_step == 0)

    def _lr_mul(self) -> float:
        step = self._step_count
        if step < self.cooldown_start or self.cooldown_start >= self.total_steps:
            return 1.0
        progress = (step - self.cooldown_start) / max(1, self.total_steps - self.cooldown_start)
        return 1.0 - (1.0 - self.min_lr_ratio) * min(progress, 1.0)

    def _muon_momentum(self) -> float:
        if self.muon_momentum_warmup_steps <= 0:
            return self._momentum_max
        step = self._step_count
        if step >= self.muon_momentum_warmup_steps:
            return self._momentum_max
        frac = step / self.muon_momentum_warmup_steps
        return self._momentum_min + (self._momentum_max - self._momentum_min) * frac

    def step(self, *args, **kwargs):
        # Pre-step hook: manual gradient all-reduce for newly-independent lm_head
        # param after the split (DDP didn't register a hook for it).
        if self._pre_step_fn is not None:
            self._pre_step_fn()
        mul = self._lr_mul()
        for pg, base_lr in zip(self.optimizer.param_groups, self._base_lrs):
            pg['lr'] = base_lr * mul
        if self.muon_momentum_warmup_steps > 0:
            momentum = self._muon_momentum()
            for pg in self.optimizer.param_groups:
                if pg.get('use_muon'):
                    pg['momentum'] = momentum
        self.optimizer.step(*args, **kwargs)
        self._step_count += 1
        # Perform embedding/lm_head split after the last tied optimizer step.
        if not self._split_done and self._split_fn is not None and self._step_count >= self.split_step:
            self._split_fn(self.optimizer)
            self._split_done = True

    def zero_grad(self, *args, **kwargs):
        self.optimizer.zero_grad(*args, **kwargs)

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    @property
    def param_groups(self):
        return self.optimizer.param_groups


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

    @property
    def dataset(self):
        return self.loader.dataset


def _run_generation_demo(training_module, tokenizer, link_detector, layout_policy, mask_type):
    """
    Quick generation sanity check at the end of training.

    Runs two short generation calls with hardcoded Python prompts containing
    import statements so the cross-doc link machinery is exercised. Results are
    printed to the training log via logger.info.
    """
    from model.generation_config import GenerationConfig
    from model.model import TS2TSModel

    logger.info("=" * 60)
    logger.info("End-of-training generation demo")
    logger.info("=" * 60)

    try:
        inference_model = training_module.to_inference_model(
            tokenizer=tokenizer,
            link_detector=link_detector,
            layout_policy=layout_policy,
        )
        # Move inference model to the same device as training module.
        device = next(training_module.parameters()).device
        inference_model.to(device)
    except Exception as e:
        logger.warning("Generation demo skipped — could not build inference model: %s", e)
        return

    # Select prompts that match the actual link detector syntax so links fire.
    from model.graph_traversal.python_import_detector import PythonImportDetector
    from model.graph_traversal.markdown_link_detector import MarkdownLinkDetector
    if isinstance(link_detector, PythonImportDetector):
        prompts = [
            '"""Sorting utilities."""\nimport heapq\nfrom utils import validate_input\n\ndef quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n',
            '"""Data processing pipeline."""\nimport numpy as np\nfrom transforms import normalize\n\ndef process(data):\n',
        ]
    elif isinstance(link_detector, MarkdownLinkDetector):
        prompts = [
            'The [quicksort](Quicksort) algorithm is a divide-and-conquer sorting method. It was developed by [Tony Hoare](Tony_Hoare) in 1959.',
            '[Python](Python_(programming_language)) is a high-level programming language known for its readability. See also [Java](Java_(programming_language)).',
        ]
    else:
        # Fallback: no link detection (doc_causal or unknown detector).
        prompts = [
            'def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n',
            'class DataLoader:\n    def __init__(self, dataset):\n',
        ]

    config = GenerationConfig(
        max_new_tokens=200,
        max_tokens_per_document=200,
        max_context_length=2048,
        max_link_depth=1,
        allow_generation_fallback=True,
        max_auxiliary_documents=4,
        temperature=1.0,
        repetition_penalty=1.3,
        device=str(device),
    )

    for i, prompt in enumerate(prompts, 1):
        logger.info("--- Demo %d/%d ---", i, len(prompts))
        logger.info("Prompt: %r", prompt[:80])
        try:
            result = inference_model.generate(prompt, config=config)
            root = result.root_document
            logger.info(
                "Root (%d tokens): %s",
                len(root.tokens) if root.tokens is not None else 0,
                (root.text or "")[:400],
            )
            for doc in result.auxiliary_documents:
                logger.info(
                    "  Aux '%s' depth=%d source=%s (%d tokens): %s",
                    doc.raw_identifier, doc.depth, doc.source,
                    len(doc.tokens) if doc.tokens is not None else 0,
                    (doc.text or "")[:200],
                )
            if result.trace is not None:
                logger.info(
                    "  Trace: %d fwd passes, %d links detected, %d resolved, "
                    "%d corpus fetches, %d docs generated",
                    result.trace.total_forward_passes,
                    result.trace.links_detected,
                    result.trace.links_resolved,
                    result.trace.corpus_fetches,
                    result.trace.docs_generated,
                )
        except Exception as e:
            logger.warning("Demo %d failed: %s", i, e)

    logger.info("=" * 60)


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
    strategy_name = cfg.get('data', {}).get('strategy', 'bfs')
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

    epoch_dirs = cfg.get('data', {}).get('epoch_dirs')
    # CLI passes epoch_dirs as a string ("[dir1,dir2]" or "dir1,dir2"); parse to list.
    if isinstance(epoch_dirs, str):
        epoch_dirs = [p.strip() for p in epoch_dirs.strip('[]').split(',') if p.strip()]
    if epoch_dirs:
        # Density-aware path: BucketedPackDataset from pre-computed epoch dirs.
        # Each accum step draws from the same density bucket on all ranks,
        # eliminating FlexAttention backward variance across DDP ranks.
        bucket_state_dict = None
        if resume_ckpt is not None:
            bucket_state_dict = resume_ckpt.get('metadata', {}).get('bucket_state')
        if bucket_state_dict is None and resume_from:
            # Also check for bucket_state in a legacy/separate checkpoint load
            try:
                _tmp = torch.load(resume_from, map_location='cpu', weights_only=False)
                bucket_state_dict = _tmp.get('metadata', {}).get('bucket_state')
                del _tmp
            except Exception:
                pass
        start_state = BucketState(**bucket_state_dict) if bucket_state_dict else None
        if start_state:
            logger.info(
                "Resuming BucketedPackDataset from epoch=%d, accum_step=%d",
                start_state.epoch_idx, start_state.global_accum_step,
            )
        # Warn if max_grants warmup is active (bucketing is approximate during warmup)
        _mgstart = cfg.get('model', {}).get('max_grants_start')
        _mg = cfg.get('model', {}).get('max_grants', 64)
        _mgwarm = int(cfg.get('model', {}).get('max_grants_warmup_steps', 0))
        if _mgstart is not None and _mgstart < _mg:
            logger.warning(
                "max_grants warmup active: kv_block_count bucketing reflects final "
                "max_grants (%d); density balance is approximate during warmup "
                "(%d→%d over %d steps).",
                _mg, _mgstart, _mg, _mgwarm,
            )
        dataset = BucketedPackDataset(
            epoch_dirs=epoch_dirs,
            graph=graph_index,
            backend=backend,
            layout=layout_policy,
            rank=dist.rank,
            world_size=dist.world_size,
            start_state=start_state,
        )
    else:
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
    vocab_size = 50304 if tokenizer_name == 'gpt2' else cfg['model'].get('vocab_size', 50304)

    # Create block mask creator
    mask_type = cfg.get('model', {}).get('mask_type', 'doc_causal')
    detector = None  # populated below for cross_doc_link; stays None otherwise
    model_cfg = cfg.get('model', {})
    use_triton = model_cfg.get('attention_backend', 'triton') != 'flex'

    if use_triton and not model_cfg.get('compile', True):
        raise AssertionError(
            "Custom Triton attention kernels (attention_backend='triton') require "
            "torch.compile (model.compile: true). Their backward passes are not "
            "supported in eager mode. Use --model.attention_backend flex if you "
            "need to run without compile (e.g. for quick smoke tests)."
        )

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
        attention_backend = 'triton_v17' if use_triton else 'flex'
        block_mask_creator = make_mask_creator_callable_from(
            CrossDocLinkMaskCreator(
                link_detector=detector,
                max_grants=model_cfg.get('max_grants', 64),
                max_grants_start=model_cfg.get('max_grants_start'),
                max_grants_warmup_steps=int(model_cfg.get('max_grants_warmup_steps', 0)),
                backend=attention_backend,
            )
        )
    else:
        if use_triton and mask_type == 'doc_causal':
            attention_backend = 'varlen_bim_v1'
            from model.graph_traversal.block_mask_creator import create_doc_causal_triton_mask
            block_mask_creator = make_mask_creator_callable_from(create_doc_causal_triton_mask)
        else:
            attention_backend = 'flex'
            block_mask_creator = make_mask_creator_callable(mask_type)

    mtp_extra_weights = cfg['model'].get('mtp_extra_weights') or []
    # compose_config may deliver CLI list overrides as a raw string (e.g. "[0.3,0.1]").
    if isinstance(mtp_extra_weights, str):
        import ast
        mtp_extra_weights = ast.literal_eval(mtp_extra_weights)
    mtp_decay_steps = int(cfg['model'].get('mtp_decay_steps', 0))
    accum_steps_val = int(
        cfg.get('train_loop', {}).get('atomic_feature_kwargs', {}).get('accum_steps', 1)
    )
    # Convert optimizer-step units (user-facing) to micro-step units (model-internal).
    mtp_decay_micro_steps = mtp_decay_steps * accum_steps_val

    untie_at_frac = cfg['model'].get('untie_at_frac')

    model = TS2TSTrainingModule.from_config(
        vocab_size=vocab_size,
        num_layers=cfg['model']['num_layers'],
        model_dim=cfg['model']['model_dim'],
        num_heads=cfg['model']['num_heads'],
        max_seq_len=cfg['model']['max_seq_len'],
        drop_path_rate=cfg['model'].get('drop_path_rate', 0.0),
        block_mask_creator=block_mask_creator,
        fp8=cfg['model'].get('fp8', False),
        weight_tying=cfg['model'].get('weight_tying', True),
        ignore_index=cfg['model'].get('ignore_index', -100),
        dtype=getattr(torch, cfg['model'].get('dtype', 'bfloat16')),
        activation_checkpointing=cfg['model'].get('activation_checkpointing', False),
        attention_backend=attention_backend,
        logit_softcap=cfg['model'].get('logit_softcap'),
        mtp_extra_weights=mtp_extra_weights,
        mtp_decay_micro_steps=mtp_decay_micro_steps,
    ).to(dist.device)

    # Build optimizer param groups BEFORE compile/DDP so that named_parameters()
    # gives clean names and weight-tied tensors are only counted once.
    logger.info("Initializing Optimizer...")
    muon_params, embed_adamw_params, other_adamw_params = [], [], []
    seen_ids: set = set()
    for name, param in model.named_parameters():
        if id(param) in seen_ids:
            continue
        seen_ids.add(id(param))
        if 'backbone' in name and param.ndim >= 2:
            # Backbone 2-D weights → Muon (spectral-norm optimizer)
            muon_params.append(param)
        elif 'embedding' in name or 'loss_fn' in name:
            # Embedding / lm_head get lower β1: sparse gradients make the
            # running mean stale for rows not touched this step.
            embed_adamw_params.append(param)
        else:
            other_adamw_params.append(param)

    adamw_beta2   = cfg['optimizer'].get('beta2', 0.95)
    embed_beta1   = cfg['optimizer'].get('adamw_embed_beta1', 0.5)
    other_beta1   = cfg['optimizer'].get('beta1', 0.9)
    muon_wd       = cfg['optimizer'].get('muon_wd', cfg['optimizer'].get('wd', 0.1))
    adamw_wd      = cfg['optimizer'].get('adamw_wd', cfg['optimizer'].get('wd', 0.1))

    param_groups = [
        dict(
            params=muon_params,
            use_muon=True,
            lr=cfg['optimizer']['muon_lr'],
            momentum=cfg['optimizer'].get('momentum', 0.95),
            weight_decay=muon_wd,
        ),
        dict(
            params=embed_adamw_params,
            use_muon=False,
            lr=cfg['optimizer']['adamw_lr'],
            betas=(embed_beta1, adamw_beta2),
            weight_decay=adamw_wd,
        ),
        dict(
            params=other_adamw_params,
            use_muon=False,
            lr=cfg['optimizer']['adamw_lr'],
            betas=(other_beta1, adamw_beta2),
            weight_decay=adamw_wd,
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

    # -------------------------------------------------------------------------
    # 3c. LR cooldown / weight-untying schedule config (parsed before compile;
    #     LRCooldownScheduler is constructed after DDP so that embed_sync_fn
    #     can reference model.module when DDP is active).
    # -------------------------------------------------------------------------
    cooldown_frac = cfg.get('train_loop', {}).get('cooldown_frac', 0.0)
    min_lr_ratio = cfg.get('train_loop', {}).get('min_lr_ratio', 0.1)
    muon_momentum_warmup_steps = int(cfg.get('optimizer', {}).get('muon_momentum_warmup_steps', 0))
    max_steps_for_cooldown = cfg.get('train_loop', {}).get('max_optimizer_steps')

    if cooldown_frac > 0.0 and max_steps_for_cooldown is None:
        logger.warning(
            "cooldown_frac=%.2f requested but train_loop.max_optimizer_steps is not set; "
            "skipping LR cooldown (cannot determine schedule length).",
            cooldown_frac,
        )
        cooldown_frac = 0.0

    if untie_at_frac is not None and max_steps_for_cooldown is None:
        logger.warning(
            "model.untie_at_frac=%.3f requested but train_loop.max_optimizer_steps is not set; "
            "skipping deferred weight untying (cannot determine split_step).",
            untie_at_frac,
        )
        untie_at_frac = None

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

    # Construct LRCooldownScheduler now that model is in its final form (after
    # DDP wrapping), so split closures can reference model.module correctly.
    needs_scheduler = (
        cooldown_frac > 0.0
        or muon_momentum_warmup_steps > 0
        or untie_at_frac is not None
    )
    if needs_scheduler:
        total_steps_original = (max_steps_for_cooldown + resumed_steps) if max_steps_for_cooldown is not None else 0
        # Reset param_group LRs to the config values so a resume checkpoint's
        # (potentially already-cooled) LR doesn't corrupt the base.
        for pg in optimizer.param_groups:
            pg['lr'] = cfg['optimizer']['muon_lr'] if pg.get('use_muon') else cfg['optimizer']['adamw_lr']

        pre_step_fn = None
        split_step = 0
        split_fn = None
        if untie_at_frac is not None:
            split_step = round(total_steps_original * untie_at_frac)
            _tm = model.module if hasattr(model, 'module') else model
            _embed_adamw_beta = (
                cfg['optimizer'].get('adamw_embed_beta1', 0.5),
                cfg['optimizer'].get('beta2', 0.95),
            )

            def split_fn(opt):
                # Create an independent lm_head weight, initialized from the current
                # (shared) embedding weight.
                new_lm_head = torch.nn.Parameter(_tm.embedding.weight.data.clone())
                _tm.loss_fn.weight = new_lm_head

                # Add to the embed AdamW param group (same betas / wd as embedding).
                # Find the group by checking which contains the current embedding.weight.
                embed_weight = _tm.embedding.weight
                for pg in opt.param_groups:
                    if not pg.get('use_muon', False) and embed_weight in pg['params']:
                        pg['params'].append(new_lm_head)
                        break
                else:
                    # Fallback: add a new group matching the embed betas
                    opt.param_groups.append(dict(
                        params=[new_lm_head],
                        use_muon=False,
                        lr=cfg['optimizer']['adamw_lr'],
                        betas=_embed_adamw_beta,
                        weight_decay=cfg['optimizer'].get('adamw_wd', cfg['optimizer'].get('wd', 0.1)),
                    ))

                # Transfer Adam state from the embedding to the new lm_head so the
                # optimizer doesn't start cold (copies exp_avg and exp_avg_sq).
                embed_state = opt.state.get(embed_weight)
                if embed_state:
                    opt.state[new_lm_head] = {
                        k: v.clone() if isinstance(v, torch.Tensor) else v
                        for k, v in embed_state.items()
                    }

                logger.info(
                    "Weight untying: lm_head split from embedding at step %d "
                    "(Adam state transferred: %s).",
                    split_step,
                    "yes" if embed_state else "no (cold start)",
                )

            if dist.is_distributed:
                def pre_step_fn():
                    # After the split, lm_head is a new param DDP doesn't know about.
                    # Manually all-reduce its gradient before the optimizer step.
                    w = _tm.loss_fn.weight
                    if w.grad is not None:
                        torch.distributed.all_reduce(w.grad)
                        w.grad.div_(dist.world_size)

        optimizer = LRCooldownScheduler(
            optimizer,
            total_steps=total_steps_original,
            start_step=resumed_steps,
            cooldown_frac=cooldown_frac,
            min_lr_ratio=min_lr_ratio,
            muon_momentum_warmup_steps=muon_momentum_warmup_steps,
            pre_step_fn=pre_step_fn,
            split_step=split_step,
            split_fn=split_fn,
        )
        logger.info(
            "Training scheduler: cooldown_frac=%.2f, min_lr_ratio=%.2f, "
            "total_steps=%d, cooldown_starts_at_step=%d; "
            "muon_momentum_warmup_steps=%d; "
            "untie_at_frac=%s (split_step=%d); "
            "(resumed_steps=%d).",
            cooldown_frac, min_lr_ratio,
            total_steps_original, optimizer.cooldown_start,
            muon_momentum_warmup_steps,
            f"{untie_at_frac:.3f}" if untie_at_frac is not None else "None", split_step,
            resumed_steps,
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
    # When using BucketedPackDataset, inject bucket_state_fn so the
    # bucket_state_checkpoint atomic feature saves dataset position alongside
    # every best-val-loss checkpoint for exact resume capability.
    if epoch_dirs:
        atomic_feature_kwargs['bucket_state_fn'] = dataset.get_state

    result = smart_train(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        llm_client=get_default_llm_client(),
        **atomic_feature_kwargs
    )

    logger.info("Training complete!")

    # -------------------------------------------------------------------------
    # 5. End-of-training generation demo (main process only)
    # -------------------------------------------------------------------------
    if dist.is_main_process:
        _run_generation_demo(
            training_module=model.module if dist.is_distributed else model,
            tokenizer=enc,
            link_detector=detector,
            layout_policy=layout_policy,
            mask_type=mask_type,
        )

    # Cleanup
    backend.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a TAGSeq2TAGSeq model on the TAGWiki dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--dataset-dir", dest="data.dataset_dir", type=str,
                        help="Path to the pre-tokenized dataset directory.")
    parser.add_argument("--strategy", dest="data.strategy", type=str, default=None,
                        choices=["random", "random_walk", "bfs", "dfs"],
                        help="Graph traversal strategy. If not set, uses the config file value.")
    parser.add_argument("--max-seq-len", dest="model.max_seq_len", type=int, default=None,
                        help="Maximum sequence length (token budget per pack). "
                             "If not set, uses the value from the config file.")
    parser.add_argument("--seed", dest="seed", type=int, default=None,
                        help="Random seed. If not set, uses the config file value.")
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
