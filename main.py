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


def _run_generation_demo(training_module, tokenizer, link_detector, layout_policy, mask_type,
                         inference_model=None):
    """
    Quick generation sanity check at the end of training.

    Runs two short generation calls with hardcoded Python prompts containing
    import statements so the cross-doc link machinery is exercised. Results are
    printed to the training log via logger.info.

    Args:
        inference_model: Optional pre-built TS2TSModel (e.g. the flex-backend model
            from _build_flex_inference_model). If None, builds from training_module.
    """
    from model.generation_config import GenerationConfig
    from model.model import TS2TSModel

    logger.info("=" * 60)
    logger.info("End-of-training generation demo")
    logger.info("=" * 60)

    if inference_model is None:
        try:
            inference_model = training_module.to_inference_model(
                tokenizer=tokenizer,
                mask_type=mask_type,
                link_detector=link_detector,
                inference_layout_policy=layout_policy,
            )
            inference_model.to(next(training_module.parameters()).device)
        except Exception as e:
            logger.warning("Generation demo skipped — could not build inference model: %s", e)
            return

    device = next(training_module.parameters()).device

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
        link_retrieval_mode="corpus_then_generate",
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


def _build_inference_model(
    training_module_unwrapped, cfg, enc, detector,
    training_layout_policy, inference_layout_policy, device,
):
    """Build a post-training inference model using the configured inference backend.

    Calls to_inference_model which does a zero-copy __class__ swap on each
    attention layer so the uncompiled backbone uses FlexSelfAttention for
    inference without duplicating weights.

    Used for the generation demo and benchmark eval at the end of training.
    Returns None on failure (caller should fall back gracefully).
    """
    try:
        model_cfg = cfg.get('model', {})
        inference_backend = model_cfg.get('inference_attention_backend', 'flex')
        mask_type = model_cfg.get('mask_type', 'doc_causal')
        use_triton = model_cfg.get('attention_backend', 'triton') != 'flex'
        training_backend = 'triton' if use_triton else 'flex'

        inference_model = training_module_unwrapped.to_inference_model(
            tokenizer=enc,
            mask_type=mask_type,
            link_detector=detector,
            training_backend=training_backend,
            inference_backend=inference_backend,
            training_layout_policy=training_layout_policy,
            inference_layout_policy=inference_layout_policy,
        )
        inference_model.to(torch.device(device), torch.bfloat16)

        if inference_backend == 'flex' and torch.cuda.is_available():
            import tunalab.modules.sequence_mixing.flex_self_attention as _fa_mod
            from torch.nn.attention.flex_attention import flex_attention as _raw_fa
            _fa_mod.flex_attention = torch.compile(_raw_fa, dynamic=True, mode='default')

        return inference_model

    except Exception as e:
        logger.warning("Could not build inference model for post-training steps: %s", e)
        return None


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

    # data.train_dir — explicit path to the training graph (typically splits/train/).
    # If absent, falls back to dataset_dir (full graph, no split exclusion).
    train_dir_str = cfg.get('data', {}).get('train_dir')
    train_graph_dir = Path(train_dir_str) if train_dir_str else dataset_dir
    if not train_graph_dir.is_dir():
        logger.error("train_dir not found: %s", train_graph_dir)
        return
    logger.info("Initializing GraphIndex from %s", train_graph_dir)
    graph_index = GraphIndex(train_graph_dir)

    # The backend handles memory-mapping of token shards
    backend = PretokShardedBackend(graph_index)

    # Configure Layout Policy
    # Options: null | eos | identifier_prefix | identifier_prefix_eos | stochastic_identifier_prefix
    #          | stochastic_identifier_prefix
    layout_policy_name = cfg.get('data', {}).get('layout_policy', 'null')
    enc = tiktoken.get_encoding(graph_index.metadata.get('tokenizer', 'gpt2'))
    layout_policy = make_layout_policy(
        name=layout_policy_name,
        encode_fn=enc.encode_ordinary,
    )

    # Inference layout policy — defaults to training policy, but can be
    # overridden via data.inference_layout_policy.  Needed when training with
    # 'stochastic_identifier_prefix' so inference always uses a deterministic
    # policy (e.g. 'identifier_prefix') for stable aux-doc generation.
    inference_layout_policy_name = cfg.get('data', {}).get(
        'inference_layout_policy', layout_policy_name
    )
    if inference_layout_policy_name == layout_policy_name:
        inference_layout_policy = layout_policy
    else:
        inference_layout_policy = make_layout_policy(
            name=inference_layout_policy_name,
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

    # ── Validation loaders ────────────────────────────────────────────────────
    # data.val_dirs  — dict of {name: path} for live-packed val loaders.
    # data.val_epoch_dirs — dict of {name: [epoch_dir, ...]} for precomputed val.
    # If neither is set, falls back to a single live loader over the train graph
    # with an offset seed (no split exclusion — for backward compat / pre-split use).
    val_steps   = cfg.get('train_loop', {}).get('val_steps', 10)
    _seq_len    = cfg.get('model', {}).get('max_seq_len', 2048)
    _doc_budget = cfg.get('data', {}).get('doc_budget')
    _order_mode = cfg.get('data', {}).get('order_mode', 'prefer_targets_first')

    _extra_backends: list = []  # separately-opened backends; closed at cleanup

    def _make_live_val_loader(graph_idx, bknd, seed_offset):
        sampler = PackBatchSampler(
            graph=graph_idx,
            strategy_factory=strategy_factory,
            token_budget=_seq_len,
            doc_budget=_doc_budget,
            overflow_policy="truncate",
            doc_level_trim_side="tail",
            pack_level_trim_side="head",
            max_candidates_per_component=1000,
            seed=rank_seed + seed_offset,
            order_mode=_order_mode,
            layout_policy=layout_policy,
        )
        ds = PackedSequenceDataset(
            graph=graph_idx, backend=bknd,
            pack_sampler=sampler, layout_policy=layout_policy, as_2d=True,
        )
        return LimitedDataLoader(DataLoader(ds, batch_size=None, num_workers=0),
                                 max_batches=val_steps)

    def _make_precomputed_val_loader(epoch_dirs_list, graph_dir: Path):
        _dirs = epoch_dirs_list
        if isinstance(_dirs, str):
            _dirs = [p.strip() for p in _dirs.strip('[]').split(',') if p.strip()]
        _g = GraphIndex(graph_dir)
        _b = PretokShardedBackend(_g)
        _extra_backends.append(_b)
        _ds = BucketedPackDataset(
            epoch_dirs=_dirs,
            graph=_g,
            backend=_b,
            layout=layout_policy,
            rank=0,
            world_size=1,
        )
        return LimitedDataLoader(DataLoader(_ds, batch_size=None, num_workers=0),
                                 max_batches=val_steps)

    cfg_val_dirs       = cfg.get('data', {}).get('val_dirs', {})       # {name: path}
    cfg_val_epoch_dirs = cfg.get('data', {}).get('val_epoch_dirs', {}) # {name: [dir,...]}

    val_loaders: Dict[str, Any] = {}
    seed_counter = 1

    for name, path in cfg_val_dirs.items():
        p = Path(path)
        if not p.is_dir():
            logger.warning("val_dirs[%r] not found: %s — skipping", name, p)
            continue
        _g = GraphIndex(p)
        _b = PretokShardedBackend(_g)
        _extra_backends.extend([_b])
        val_loaders[name] = _make_live_val_loader(_g, _b, seed_offset=seed_counter)
        seed_counter += 1

    for name, epoch_dirs_list in cfg_val_epoch_dirs.items():
        # Graph dir for precomputed val: must be in val_dirs[name] or fall back to train_graph_dir.
        _graph_dir = Path(cfg_val_dirs[name]) if name in cfg_val_dirs else train_graph_dir
        try:
            val_loaders[name] = _make_precomputed_val_loader(epoch_dirs_list, _graph_dir)
        except Exception as _exc:
            logger.warning("val_epoch_dirs[%r] failed to load: %s — skipping", name, _exc)

    if not val_loaders:
        # No splits configured — fall back to live loader over train graph.
        logger.info("No val_dirs/val_epoch_dirs configured — val uses train graph with offset seed")
        val_loaders["train_dist"] = _make_live_val_loader(graph_index, backend, seed_offset=1)

    # -------------------------------------------------------------------------
    # 3. Model & Optimizer Setup
    # -------------------------------------------------------------------------
    logger.info("Initializing Model...")

    tokenizer_name = graph_index.metadata.get('tokenizer', 'gpt2')
    vocab_size = 50257 if tokenizer_name == 'gpt2' else cfg['model'].get('vocab_size', 50257)

    # Create block mask creator
    mask_type = cfg.get('model', {}).get('mask_type', 'doc_causal')
    detector = None  # populated below for cross_doc_link; stays None otherwise
    model_cfg = cfg.get('model', {})
    use_triton = model_cfg.get('attention_backend', 'triton') != 'flex'

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
        attention_backend = 'triton_v12' if use_triton else 'flex'
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
        activation_checkpointing=cfg['model'].get('activation_checkpointing', False),
        attention_backend=attention_backend,
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
        'val_loaders': val_loaders,
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
    # 5. End-of-training generation demo + post-training eval (main process)
    #
    # Both use a fresh inference model with attention_backend='flex' + torch.compile
    # so they run fast regardless of the training attention backend.
    # -------------------------------------------------------------------------
    if dist.is_main_process:
        training_module_unwrapped = model.module if dist.is_distributed else model
        _flex_inference_model = _build_inference_model(
            training_module_unwrapped=training_module_unwrapped,
            cfg=cfg, enc=enc, detector=detector,
            training_layout_policy=layout_policy,
            inference_layout_policy=inference_layout_policy,
            device=str(dist.device),
        )

        _run_generation_demo(
            training_module=training_module_unwrapped,
            tokenizer=enc,
            link_detector=detector,
            layout_policy=inference_layout_policy,
            mask_type=mask_type,
            inference_model=_flex_inference_model,
        )

    # -------------------------------------------------------------------------
    # 6. Post-training benchmark evaluation (main process only)
    #
    # Runs community_pack_perplexity and held_out_perplexity on all four
    # val/test splits when splits exist, plus any additional benchmarks
    # configured under eval.benchmarks in the YAML.
    # -------------------------------------------------------------------------
    if dist.is_main_process:
        eval_cfg = cfg.get("eval", {})
        if eval_cfg.get("run_on_completion", False):
            dataset_dir_str = cfg.get("data", {}).get("dataset_dir", "")
            if _flex_inference_model is not None and dataset_dir_str:
                from eval_checkpoints import run_benchmarks_on_model
                logger.info("Running post-training benchmark evaluation...")
                _flex_inference_model.eval()

                # data.val_dirs / data.test_dirs drive the post-training split evals.
                # community dirs  → community_pack_perplexity
                # random dirs     → held_out_perplexity (split="all" since dir is already filtered)
                _conditions = ["doceval"]
                if mask_type == "cross_doc_link":
                    _conditions = ["baseline", "experimental"]

                _split_results: Dict[str, Any] = {}

                def _run_split_eval(split_name: str, split_path: str, bench: str) -> None:
                    """Run one post-training benchmark on a split directory.

                    community_pack_perplexity navigates dataset_dir/splits/{split}/
                    internally, so it receives the parent dataset_dir.
                    held_out_perplexity scores all docs in its dataset_dir directly,
                    so it receives the split path and uses split='all'.
                    """
                    logger.info("Post-training eval: %s on %s", bench, split_name)
                    try:
                        if bench == "community_pack_perplexity":
                            # Function appends splits/{split_name}/ itself.
                            _ddir = dataset_dir_str
                            _eval_split = split_name
                        else:
                            # held_out_perplexity: score all docs in the split dir.
                            _ddir = split_path
                            _eval_split = "all"
                        _scfg = {
                            "benchmarks": [{"name": bench, "conditions": _conditions,
                                            "split": _eval_split}],
                            "split": _eval_split,
                            "max_docs": eval_cfg.get("max_docs", 500),
                        }
                        _r = run_benchmarks_on_model(
                            model=_flex_inference_model,
                            dataset_dir=_ddir,
                            eval_cfg=_scfg,
                            device=str(dist.device),
                        )
                        _split_results.update({f"{k}__{split_name}": v for k, v in _r.items()})
                    except Exception as _exc:
                        logger.error("Split eval %s/%s failed: %s", bench, split_name, _exc)
                        _split_results[f"{bench}/{split_name}"] = {"error": str(_exc)}

                # val splits
                for _name, _path in cfg.get('data', {}).get('val_dirs', {}).items():
                    _bench = "community_pack_perplexity" if "community" in _name else "held_out_perplexity"
                    _run_split_eval(_name, _path, _bench)

                # test splits
                for _name, _path in cfg.get('data', {}).get('test_dirs', {}).items():
                    _bench = "community_pack_perplexity" if "community" in _name else "held_out_perplexity"
                    _run_split_eval(_name, _path, _bench)

                # Additional benchmarks from eval.benchmarks in YAML.
                eval_results = run_benchmarks_on_model(
                    model=_flex_inference_model,
                    dataset_dir=dataset_dir_str,
                    eval_cfg=eval_cfg,
                    device=str(dist.device),
                )
                eval_results.update(_split_results)

                eval_path = os.path.join(rep.output_dir, "eval_results.json")
                with open(eval_path, "w", encoding="utf-8") as f:
                    json.dump(eval_results, f, ensure_ascii=False, indent=2)
                logger.info("Eval results written to %s", eval_path)
            else:
                logger.warning(
                    "Post-training eval skipped: flex inference model unavailable "
                    "or dataset_dir not set in config."
                )

    # Cleanup
    backend.close()
    for _b in _extra_backends:
        _b.close()


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
