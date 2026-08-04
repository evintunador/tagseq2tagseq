import argparse
import itertools
import logging
import traceback
import os
import datetime
from pathlib import Path
from typing import Dict, Any
import json

# Set before any CUDA allocation so the memory allocator picks it up.
# Expandable segments dramatically reduce fragmentation when sequence lengths
# vary across steps (which they do for packed graph batches).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from tunalab.configuration import compose_config
from tunalab.distributed import DistributedManager, setup_signal_handlers
from tunalab.reproducibility import ReproducibilityManager
from tunalab import tracking
from tunalab.smart_train import smart_train
from optimizers.muon import MuonWithAuxAdam, SingleDeviceMuonWithAuxAdam
from tunalab.llm_compilers.auto import get_default_llm_client

# Local imports
from model import TS2TSTrainingModule
import tiktoken

from model.graph_traversal.block_mask_creator import (
    make_mask_creator_callable,
    make_mask_creator_callable_from,
    create_doc_causal_triton_mask,
    create_doc_concat_triton_mask,
)
from model.graph_traversal.cross_doc_mask import CrossDocLinkMaskCreator
from model.graph_traversal.link_detector import make_link_detector
from model.graph_traversal.markdown_link_detector import MarkdownLinkDetector
from model.graph_traversal.python_import_detector import PythonImportDetector
from model.graph_traversal.arxiv_cite_detector import ArxivCiteDetector
from data.dataset import GraphIndex, PretokShardedBackend
from data.packed_dataset import PackedSequenceDataset
from data.bucketed_pack_dataset import BucketedPackDataset, BucketState
from tunalab.device import to_device
from data.layout import make_layout_policy, inference_layout_for_detector
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
        warmup_steps: Number of steps to linearly ramp LR from 0 to 1×.  Applied
            at the start of the *full* run (absolute step numbers), so a resumed
            run that starts after warmup_steps simply skips the warmup.
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
        warmup_steps: int = 0,
        cooldown_frac: float = 0.4,
        min_lr_ratio: float = 0.1,
        muon_momentum_warmup_steps: int = 0,
        pre_step_fn=None,
        split_step: int = 0,
        split_fn=None,
    ):
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
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
        if self.warmup_steps > 0 and step < self.warmup_steps:
            return step / self.warmup_steps
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

    def state_dict_full(self):
        # Delegate the name-keyed, world-size-portable optimizer state to the
        # wrapped optimizer so checkpoints saved while the scheduler is active
        # (warmup/cooldown/untie — i.e. essentially always) still capture the
        # full Muon+AdamW state. Without this delegation the training loop's
        # getattr(optimizer, "state_dict_full", None) would miss it and silently
        # fall back to a checkpoint with no optimizer state.
        return self.optimizer.state_dict_full()

    def load_state_dict_full(self, full_sd):
        return self.optimizer.load_state_dict_full(full_sd)

    @property
    def _id_to_name(self):
        return self.optimizer._id_to_name

    @_id_to_name.setter
    def _id_to_name(self, value):
        self.optimizer._id_to_name = value

    @property
    def param_groups(self):
        return self.optimizer.param_groups


class LimitedDataLoader:
    """Wraps a DataLoader to yield at most ``max_batches`` items per iteration.

    Creates a fresh islice on each call to ``__iter__``, so the underlying
    loader can be iterated more than once (e.g. repeated validation passes).

    ``rewind_each_iter`` (VAL ONLY): a BucketedPackDataset is a *stateful*
    generator — it mutates ``_bucket_consumed``/``_global_accum_step`` as it
    yields and does NOT give those packs back. Used as a val loader, each pass
    (every ``val_interval`` steps) would resume where the previous pass stopped
    and, once a small val schedule (e.g. zig val = 25 packs) is drained, fall
    through to the "advance to next epoch" branch and RAISE "All pre-computed
    epoch dirs exhausted" — the step-500 crash that killed the merged_v2 runs.
    Rewinding to the captured initial state before every pass fixes that AND
    makes each pass score the SAME deterministic subset (comparable across
    checkpoints). Do NOT set this for the training loader: training relies on
    the persistent state to walk the full epoch across many __iter__ calls and
    to resume mid-epoch from latest.pt.
    """
    def __init__(self, loader: DataLoader, max_batches: int,
                 rewind_each_iter: bool = False) -> None:
        self.loader = loader
        self.max_batches = max_batches
        self._rewind = rewind_each_iter
        self._init_state = None
        if rewind_each_iter:
            ds = getattr(loader, "dataset", None)
            if ds is not None and hasattr(ds, "get_state"):
                self._init_state = ds.get_state()

    def __iter__(self):
        if self._rewind and self._init_state is not None:
            ds = self.loader.dataset
            if hasattr(ds, "set_state"):
                ds.set_state(self._init_state)
        return itertools.islice(iter(self.loader), self.max_batches)

    @property
    def dataset(self):
        return self.loader.dataset


def _compile_warmup(model, train_loader, dist, num_steps: int) -> None:
    """Synchronized torch.compile warmup to prevent cross-rank NCCL desync.

    THE PROBLEM: with torch.compile + DDP, the first real step triggers a lazy,
    per-rank compilation of the backbone graph. If ranks compile *non-identical*
    graphs — which happens when each rank fetches a *different* batch (packed
    sequences vary in shape/sparsity, so the FlexAttention BlockMask / Triton
    mask tensors differ) or when concurrent cold-compile under submitit/srun
    resolves the compile-cache race differently per rank — dynamo can insert a
    different number of collectives (or DDP assigns mismatched gradient buckets),
    so step 1 desyncs and hangs on an NCCL all-reduce. (This is the same
    first-step hang tracked in the submitit investigation; disabling
    optimize_ddp removes the compiler-inserted collectives, and this warmup
    removes the graph-divergence source itself.)

    THE FIX (mirrors the mic reference orchestrator): before the real loop, rank
    0 fetches ONE batch and broadcasts it to every rank, so all ranks trace the
    IDENTICAL shape and compile the same graph. We run ``num_steps`` fwd+bwd
    passes with a ``barrier()`` between each so compilation completes
    synchronously across ranks, then zero the grads. We NEVER call
    optimizer.step() — this must not advance training.

    SIDE-EFFECT-FREE: the model forward mutates the MTP step counter and reading
    a batch advances the (stateful, IterableDataset) schedule position; both are
    snapshotted and restored so the warmup batch is re-yielded by the real loop
    and the schedule/dedup accounting is unperturbed.

    HISTORICAL NOTE: this warmup was written while chasing an intermittent
    step-0 8-GPU hang. That hang was ultimately root-caused to a rank-0 CUDA OOM
    from a foreign GPU-0 co-tenant (a self-inflicted service that autostarted on
    SSH into a node, since fixed), NOT a genuine graph-divergence desync — 9
    clean overnight 8-GPU runs after the co-tenant fix showed no warmup hangs.
    The warmup is kept because identical-graph compilation is still the right
    invariant and it makes any residual first-step failure loud at step 0.
    """
    if num_steps <= 0 or not dist.is_distributed:
        return

    device = dist.device
    raw_model = model.module if hasattr(model, "module") else model

    # Snapshot the dataset schedule position so the warmup read is invisible to
    # the real loop (only meaningful for the stateful BucketedPackDataset path).
    dataset = train_loader.dataset
    ds_state = dataset.get_state() if hasattr(dataset, "get_state") else None

    # Rank 0 fetches one real batch; broadcast it so every rank traces the same
    # shape. broadcast_object_list moves the (CPU) batch dict across ranks.
    batch = None
    if dist.is_main_process:
        batch = next(iter(train_loader))
        batch = to_device(batch, "cpu")
    obj = [batch]
    torch.distributed.broadcast_object_list(obj, src=0)
    batch = to_device(obj[0], device)

    # Snapshot MTP counter (raw_model.forward increments it in training mode).
    mtp_step = None
    if hasattr(raw_model, "_mtp_step"):
        mtp_step = raw_model._mtp_step.clone()

    logger.info("[compile warmup] Running %d synchronized fwd+bwd pass(es) on a "
                "broadcast batch to compile identical graphs across %d ranks...",
                num_steps, dist.world_size)
    was_training = model.training
    model.train()
    # If ANY rank raises here (compile error, OOM, kernel assert), the exception
    # would otherwise unwind straight into reproducibility.__exit__'s barrier(),
    # which hangs forever (peers never arrive) and SWALLOWS the traceback — the
    # classic "DDP collective hang" that's really a masked single-rank error.
    # Log the full traceback on the failing rank FIRST so it's always visible,
    # then re-raise (the process will still abort; at least we see why).
    try:
        for i in range(num_steps):
            loss = model(batch)
            loss.backward()
            torch.cuda.synchronize()
            dist.barrier()
            logger.info("[compile warmup] step %d/%d done (loss=%.4f)", i + 1, num_steps, loss.item())
    except Exception:
        logger.error("[compile warmup] rank %d raised during warmup — full traceback:\n%s",
                     dist.rank, traceback.format_exc())
        raise

    # Discard warmup grads so the real first step starts clean; do NOT step().
    model.zero_grad(set_to_none=True)
    if not was_training:
        model.eval()

    # Restore mutated state so warmup leaves no trace.
    if mtp_step is not None:
        raw_model._mtp_step.copy_(mtp_step)
    if ds_state is not None and hasattr(dataset, "set_state"):
        dataset.set_state(ds_state)
    logger.info("[compile warmup] Complete — graphs compiled, state restored.")


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
    elif isinstance(link_detector, ArxivCiteDetector):
        prompts = [
            '% Title: Attention Mechanisms for Sequence Modeling\n\nThe Transformer architecture \\cite{Attention Is All You Need} replaced recurrence with self-attention. ',
            '% Title: A Survey of Deep Residual Networks\n\nDeep residual learning \\cite{Deep Residual Learning for Image Recognition} enabled training of very deep networks. ',
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

        # TODO: move this handler into tunalab/tracking.py and install it automatically
        # in tracking.init() so that logger.info(extra={"metrics": {...}}) calls from
        # compiled training loops are captured without any main.py glue.
        class _MetricsHandler(logging.Handler):
            def emit(self, record):
                metrics = getattr(record, "metrics", None)
                if metrics and isinstance(metrics, dict):
                    tracking.log(metrics)

        _mh = _MetricsHandler()
        _mh.setLevel(logging.INFO)
        logging.getLogger().addHandler(_mh)

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
        # Resume from an explicit checkpoint FILE — the user chooses latest.pt
        # (where the run was at the kill) vs best_model.pt (last val improvement).
        logger.info("Loading resume checkpoint: %s", resume_from)
        resume_ckpt = torch.load(resume_from, map_location='cpu', weights_only=False)
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
    # Auto-derive max_optimizer_steps from the pre-computed schedules when it is
    # not set explicitly.  BucketedPackDataset walks every epoch_dir in a single
    # pass (data-repeat = number of dirs), so the total step count is the sum of
    # each dir's n_packs, spread across DDP ranks and grad-accum micro-steps.
    # This is REQUIRED for LR cooldown and the deferred weight-untie split, both
    # of which are silently disabled when max_optimizer_steps is None (see 3c).
    # Leaving it None with epoch_dirs set was the latent footgun that trained the
    # 2026-07-03 wiki runs at constant LR with no untie.
    if epoch_dirs and cfg.get('train_loop', {}).get('max_optimizer_steps') is None:
        _accum = int(cfg.get('train_loop', {}).get('atomic_feature_kwargs', {}).get('accum_steps', 1))
        _total_packs = 0
        _ok = True
        for _ed in epoch_dirs:
            _meta_p = os.path.join(_ed, "metadata.json")
            try:
                with open(_meta_p) as _f:
                    _total_packs += int(json.load(_f)["n_packs"])
            except (OSError, KeyError, ValueError) as _exc:
                logger.warning("Could not read n_packs from %s (%s); "
                               "cannot auto-derive max_optimizer_steps.", _meta_p, _exc)
                _ok = False
                break
        if _ok and _total_packs > 0:
            _derived = _total_packs // (max(1, dist.world_size) * max(1, _accum))
            cfg.setdefault('train_loop', {})['max_optimizer_steps'] = _derived
            logger.info(
                "Auto-derived max_optimizer_steps=%d from %d epoch_dir(s) "
                "(%d total packs / world_size=%d / accum=%d). Enables LR cooldown + untie.",
                _derived, len(epoch_dirs), _total_packs, dist.world_size, _accum,
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
            # Source-unbiased, deterministic within-bucket shuffle: packs are
            # pack_id-sorted = source-sequential, so a capped val otherwise always
            # scores the earliest-source head. Fixed seed → density-stratified,
            # source-mixed, and identical across checkpoints.
            shuffle_within_bucket_seed=1234,
        )
        # rewind_each_iter: BucketedPackDataset is stateful; without a per-pass
        # rewind, repeated val passes advance through DIFFERENT packs each pass
        # (drift misread as overfitting) and eventually raise "epoch dirs
        # exhausted". Reset to initial state each pass → same deterministic subset
        # every checkpoint. Together with the shuffle + a larger val_steps this
        # gives an unbiased, stable, representative val sample.
        return LimitedDataLoader(DataLoader(_ds, batch_size=None, num_workers=0),
                                 max_batches=val_steps, rewind_each_iter=True)

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

    # cross_doc_link and doc_concat_link both use the link-grant kernel; the
    # latter sets whole_doc_grant=True so a detected link concatenates the entire
    # source doc onto the target (no link-position gate, strict FLOP-superset).
    if mask_type in ('cross_doc_link', 'doc_concat_link'):
        link_detector_name = cfg.get('model', {}).get('link_detector')
        if not link_detector_name:
            raise ValueError(
                "model.link_detector must be set to 'markdown', 'python' or 'arxiv' "
                f"when model.mask_type is '{mask_type}'"
            )
        enc = tiktoken.get_encoding('gpt2')
        detector = make_link_detector(link_detector_name, enc.decode)
        attention_backend = 'triton_v18' if use_triton else 'flex'
        block_mask_creator = make_mask_creator_callable_from(
            CrossDocLinkMaskCreator(
                link_detector=detector,
                max_grants=model_cfg.get('max_grants', 64),
                max_grants_start=model_cfg.get('max_grants_start'),
                max_grants_warmup_steps=int(model_cfg.get('max_grants_warmup_steps', 0)),
                backend=attention_backend,
                whole_doc_grant=(mask_type == 'doc_concat_link'),
            )
        )
    elif mask_type == 'doc_concatenated':
        # Merge each connected document component into one causally-concatenated
        # super-doc; reuses the doc-causal varlen kernel with component-keyed ids.
        if use_triton:
            attention_backend = 'varlen_bim_v2'
            block_mask_creator = make_mask_creator_callable_from(create_doc_concat_triton_mask)
        else:
            attention_backend = 'flex'
            block_mask_creator = make_mask_creator_callable(mask_type)
    else:
        if use_triton and mask_type == 'doc_causal':
            attention_backend = 'varlen_bim_v2'
            block_mask_creator = make_mask_creator_callable_from(create_doc_causal_triton_mask)
        else:
            attention_backend = 'flex'
            block_mask_creator = make_mask_creator_callable(mask_type)

    mtp_extra_weights = cfg['model'].get('mtp_extra_weights') or []

    ve_layers = cfg['model'].get('ve_layers') or []
    ve_layers = [int(x) for x in ve_layers]
    shared_ve_bank = bool(cfg['model'].get('shared_ve_bank', False))
    use_bigram = bool(cfg['model'].get('use_bigram', False))
    bigram_vocab_size = int(cfg['model'].get('bigram_vocab_size', 50304 * 5))
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
        ve_layers=ve_layers,
        shared_ve_bank=shared_ve_bank,
        use_bigram=use_bigram,
        bigram_vocab_size=bigram_vocab_size,
    ).to(dist.device)

    # Resume-aware weight untying: if we are resuming from a checkpoint that was
    # saved AFTER the deferred lm_head split (model.untie_at_frac), its state_dict
    # holds a `loss_fn.weight` that genuinely differs from `embedding.weight`. The
    # model was just built with weight_tying=True, so those two keys alias ONE
    # storage; loading the distinct on-disk tensors into it is order-dependent
    # (last key wins) and silently collapses the input embedding into the output
    # projection. Break the tie NOW — before optimizer param-groups are built —
    # so (a) load_state_dict below restores both matrices independently, (b) the
    # untied lm_head flows into the embed AdamW group via the 'loss_fn' name
    # match, and (c) DDP registers its own gradient hook for it (no manual
    # all-reduce needed). resumed_split records that the split already happened so
    # the LRCooldownScheduler does not re-fire split_fn and clobber the loaded
    # lm_head. See generate.load_inference_model for the inference-path twin.
    resumed_split = False
    if resume_ckpt is not None:
        _msd = resume_ckpt.get('model', {})
        _lm = _msd.get('loss_fn.weight')
        _em = _msd.get('embedding.weight')
        _tied_now = getattr(model.loss_fn, 'weight', None) is model.embedding.weight
        if (
            _tied_now
            and _lm is not None and _em is not None
            and _lm.shape == _em.shape
            and not torch.equal(_lm, _em)
        ):
            model.loss_fn.weight = torch.nn.Parameter(
                torch.empty_like(model.embedding.weight)
            )
            resumed_split = True
            logger.info(
                "Resume: checkpoint has an untied lm_head (loss_fn.weight != "
                "embedding.weight on disk); broke the tied-weight aliasing before "
                "load and marked the deferred split as already-done."
            )

    # Build optimizer param groups BEFORE compile/DDP so that named_parameters()
    # gives clean names and weight-tied tensors are only counted once.
    logger.info("Initializing Optimizer...")
    muon_params, embed_adamw_params, other_adamw_params, ve_embed_params = [], [], [], []
    seen_ids: set = set()
    # id(param) -> parameter name, for name-keyed (world-size-portable) optimizer
    # state save/restore. Captured here (pre-compile/DDP) where named_parameters()
    # gives clean names and each tied tensor appears once.
    id_to_name: dict = {}
    for name, param in model.named_parameters():
        if id(param) in seen_ids:
            continue
        seen_ids.add(id(param))
        id_to_name[id(param)] = name
        if 'value_embeds' in name or 'bigram_embed' in name:
            # Sparse lookup tables: lower β1 (0.75) for sparse gradients,
            # no weight decay (rows not seen this step must not shrink).
            ve_embed_params.append(param)
        elif 've_gate_bank' in name:
            # Small gating tensor — standard AdamW scalar group.
            other_adamw_params.append(param)
        elif 'backbone' in name and param.ndim >= 2:
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
    muon_beta2    = cfg['optimizer'].get('muon_beta2', 0.95)

    param_groups = [
        dict(
            params=muon_params,
            use_muon=True,
            lr=cfg['optimizer']['muon_lr'],
            momentum=cfg['optimizer'].get('momentum', 0.95),
            weight_decay=muon_wd,
            beta2=muon_beta2,
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
    if ve_embed_params:
        # value_embeds: sparse lookup table — lower β1=0.75 (reference choice),
        # no weight decay (rows not updated this step must not shrink).
        ve_beta1 = cfg['optimizer'].get('ve_beta1', 0.75)
        param_groups.append(dict(
            params=ve_embed_params,
            use_muon=False,
            lr=cfg['optimizer']['adamw_lr'],
            betas=(ve_beta1, adamw_beta2),
            weight_decay=0.0,
        ))

    # MuonWithAuxAdam handles distributed Muon parameter-sharding via all_gather.
    # SingleDeviceMuon is used when there is no process group (single GPU / CPU).
    if dist.is_distributed:
        optimizer = MuonWithAuxAdam(param_groups)
        logger.info("Using distributed MuonWithAuxAdam (world_size=%d)", dist.world_size)
    else:
        optimizer = SingleDeviceMuonWithAuxAdam(param_groups)
        logger.info("Using SingleDeviceMuonWithAuxAdam (single process)")

    # Give the optimizer the param-name map so state_dict_full/load_state_dict_full
    # can (de)serialize state keyed by name — the basis for exact, world-size-
    # portable Muon+AdamW resume (including the untied lm_head across the split).
    optimizer._id_to_name = id_to_name

    # -------------------------------------------------------------------------
    # 3b. Restore weights and optimizer state from checkpoint (if resuming).
    #
    # The checkpoint carries a name-keyed, world-size-portable full optimizer
    # state under metadata['optimizer_state'] (format 'muon_full_v1', written by
    # the training loop via optimizer.state_dict_full). load_state_dict_full
    # restores BOTH Muon momentum and AdamW state exactly — at any world_size and
    # across the deferred lm_head untie — by matching saved→current params on NAME
    # rather than positional index.
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

        # Full, name-keyed optimizer state — restores Muon momentum + AdamW
        # exactly, at any world_size and across the deferred lm_head untie split
        # (matched by parameter name, not positional index). All checkpoints
        # written by the current training loops carry this under
        # metadata['optimizer_state'] (format 'muon_full_v1').
        full_opt_state = resume_ckpt.get('metadata', {}).get('optimizer_state')
        if not (isinstance(full_opt_state, dict) and full_opt_state.get('format') == 'muon_full_v1'):
            raise ValueError(
                f"--resume-from {resume_from}: checkpoint has no name-keyed "
                "optimizer state (metadata['optimizer_state'] with format "
                "'muon_full_v1'). This checkpoint predates the portable-resume "
                "format and is no longer supported — resume from a checkpoint "
                "saved by the current training code."
            )
        restored, skipped = optimizer.load_state_dict_full(full_opt_state)
        logger.info(
            "Resume: full optimizer state restored by name for %d param(s) "
            "(%d skipped: absent/shape-mismatch). Muon momentum + AdamW "
            "restored exactly across world_size and the untie split.",
            len(restored), len(skipped),
        )
        if skipped:
            logger.info("Resume: optimizer state NOT restored for: %s", sorted(skipped)[:20])

        del resume_ckpt   # free ~1.8 GB
        resume_ckpt = None

    # -------------------------------------------------------------------------
    # 3c. LR cooldown / weight-untying schedule config (parsed before compile;
    #     LRCooldownScheduler is constructed after DDP so that embed_sync_fn
    #     can reference model.module when DDP is active).
    # -------------------------------------------------------------------------
    cooldown_frac = cfg.get('train_loop', {}).get('cooldown_frac', 0.0)
    min_lr_ratio = cfg.get('train_loop', {}).get('min_lr_ratio', 0.1)
    warmup_steps = int(cfg.get('train_loop', {}).get('warmup_steps', 0))
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
        # optimize_ddp lets dynamo insert all-reduce graph breaks during backbone
        # backward, overlapping gradient compute with DDP bucket all-reduces
        # (a real throughput win, so True is the default).
        #
        # The hazard it USED to create: the number of inserted collectives depends
        # on the compiled graph shape, so if ranks compile non-identical graphs
        # (concurrent cold-compile under submitit/srun with per-task cache timing),
        # they emit a DIFFERENT number of all-reduces per step → NCCL
        # collective-count desync → first-step hang (rank 0 one ALLREDUCE behind
        # peers). This is now prevented at the source by the synchronized compile
        # warmup below (`_compile_warmup`): it compiles the backbone on a
        # rank-0-broadcast batch before the real loop, so every rank traces the
        # IDENTICAL graph and the collective counts match. Keep both together — if
        # you disable the warmup (train_loop.compile_warmup_steps: 0) on a
        # multi-rank run, set model.optimize_ddp: false to stay safe.
        torch._dynamo.config.optimize_ddp = bool(cfg['model'].get('optimize_ddp', True))
        logger.info("torch._dynamo.config.optimize_ddp = %s", torch._dynamo.config.optimize_ddp)
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
        warmup_steps > 0
        or cooldown_frac > 0.0
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
        # Only arm the deferred split when it has NOT already happened. On a
        # resume from a post-split checkpoint (resumed_split), the lm_head is
        # already an independent Parameter — loaded above, folded into the embed
        # AdamW group by name, and DDP-hooked (tie broken before DDP wrapping).
        # Re-firing split_fn would clone embedding.weight over the trained
        # lm_head; a manual pre_step_fn all-reduce would double-reduce a gradient
        # DDP already handles. Leaving split_step=0 makes the scheduler treat the
        # split as done (_split_done=True at construction).
        if untie_at_frac is not None and not resumed_split:
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

                # Register the new lm_head in the optimizer's name map so a
                # checkpoint saved AFTER this split captures its optimizer state
                # (state_dict_full is name-keyed). Without this the freshly-untied
                # lm_head would be silently omitted, defeating exact resume across
                # a fresh run that crosses the untie point. 'loss_fn.weight'
                # matches the resume-side model.named_parameters() name.
                id_map = getattr(opt, '_id_to_name', None)
                if id_map is not None:
                    id_map[id(new_lm_head)] = 'loss_fn.weight'

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
            warmup_steps=warmup_steps,
            cooldown_frac=cooldown_frac,
            min_lr_ratio=min_lr_ratio,
            muon_momentum_warmup_steps=muon_momentum_warmup_steps,
            pre_step_fn=pre_step_fn,
            split_step=split_step,
            split_fn=split_fn,
        )
        logger.info(
            "Training scheduler: warmup_steps=%d, cooldown_frac=%.2f, min_lr_ratio=%.2f, "
            "total_steps=%d, cooldown_starts_at_step=%d; "
            "muon_momentum_warmup_steps=%d; "
            "untie_at_frac=%s (split_step=%d); "
            "(resumed_steps=%d).",
            warmup_steps, cooldown_frac, min_lr_ratio,
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
        'val_loaders': val_loaders,
        'val_interval': cfg['train_loop'].get('val_interval', 50),
        'output_dir': rep.output_dir,
        'device': str(dist.device),
        'use_tqdm': dist.is_main_process,  # only rank 0 shows the progress bar
        'num_epochs': cfg['train_loop'].get('epochs', 1),
    })
    # Periodic latest.pt: overwrite a single checkpoint every N optimizer steps
    # (independent of val improvement) so a kill/preemption loses at most N steps
    # instead of everything since the last val gain. Carries the full,
    # world-size-portable optimizer state for exact Muon+AdamW resume.
    #
    # ALWAYS pass this kwarg (as an int): it is a declared kwarg of the multi_val
    # feature, and smart_train only selects a feature when its ENTIRE kwarg set is
    # present — omitting it would deselect multi_val_bucketed and silently drop
    # the bucket-resume feature. 0 (falsy) disables periodic saving; the training
    # loop treats it as off, preserving the old best-only behaviour by default.
    atomic_feature_kwargs['save_latest_interval'] = int(
        cfg['train_loop'].get('save_latest_interval') or 0
    )
    # Absolute base step (resumed_steps on --resume-from, 0 fresh) so the loop records
    # ABSOLUTE step in checkpoint metadata. Without it a resumed run saves its local
    # step count and a resume-of-that (yield-watcher relaunch) rewinds to ~0. This is a
    # DECLARED kwonly arg of the multi_val_bucketed feature (so it passes smart_train's
    # allowlist AND is part of that feature's kwarg-set — selected only when provided,
    # which we always do here).
    atomic_feature_kwargs['start_step'] = int(resumed_steps)
    # Profiling: inject profile_* knobs into the shared kwargs so profile_training
    # is selected and composed alongside the other atomic features (grad_accum,
    # val, checkpointing, etc.) by the LLM compiler.  profile_run=False is the
    # default and is a no-op, so this is safe to include unconditionally.
    _profile_cfg = cfg.get('train_loop', {}).get('profile', {}) or {}
    if _profile_cfg.get('enabled', False):
        logger.info("Profiling mode enabled (world=%d) — will compose with other features.",
                    dist.world_size)
        atomic_feature_kwargs.update({
            'profile_run':              True,
            'profile_warmup_steps':     _profile_cfg.get('warmup_steps', 3),
            'profile_no_sync_steps':    _profile_cfg.get('no_sync_steps', 0),
            'profile_active_steps':     _profile_cfg.get('active_steps', 20),
            'profile_model_internals':  _profile_cfg.get('model_internals', True),
            'profile_trace':            _profile_cfg.get('trace', False),
            'profile_trace_steps':      _profile_cfg.get('trace_steps', 6),
            'profile_nccl_bench_steps': _profile_cfg.get('nccl_bench_steps', 0),
            'profile_all_ranks':        _profile_cfg.get('all_ranks', False),
        })
    # When using BucketedPackDataset, inject bucket_state_fn.  With plural
    # val_loaders this selects the ts2-local multi_val_bucketed feature, which
    # saves the dataset schedule position alongside every best-val-loss
    # checkpoint for exact resume capability.
    if epoch_dirs:
        atomic_feature_kwargs['bucket_state_fn'] = dataset.get_state

    # Synchronized torch.compile warmup (multi-rank only): compile the backbone
    # graph on a broadcast batch BEFORE the real loop so every rank traces the
    # identical shape and step 1 can't desync on an NCCL collective. Runs on both
    # fresh and resumed runs (every process cold-compiles regardless of resume).
    # Configurable via train_loop.compile_warmup_steps (default 2); 0 disables.
    if cfg['model']['compile']:
        _warmup_steps = int(cfg.get('train_loop', {}).get('compile_warmup_steps', 2))
        _compile_warmup(model, train_loader, dist, _warmup_steps)

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
    # Tier-1 per-benchmark inference: a merged corpus has no single link
    # detector, and each cross-doc benchmark asserts its own detector type. Build
    # (and cache) one inference model per detector name; its inference layout is
    # derived from the detector via inference_layout_for_detector. detector_name
    # None reuses the config's default detector + inference layout (the single-
    # source case, fully back-compatible).
    training_module_unwrapped = model.module if dist.is_distributed else model
    _inference_model_cache: Dict[Any, Any] = {}

    def _get_inference_model(detector_name):
        """Return a cached flex inference model wired for ``detector_name``.

        None → the config's default detector + inference layout. A named detector
        gets a freshly-built detector and the deterministic layout that matches it.
        """
        if detector_name in _inference_model_cache:
            return _inference_model_cache[detector_name]
        if detector_name is None:
            _det = detector
            _inf_layout = inference_layout_policy
        else:
            _det = make_link_detector(detector_name, enc.decode)
            _inf_layout = make_layout_policy(
                inference_layout_for_detector(detector_name), encode_fn=enc.encode_ordinary
            )
        _m = _build_inference_model(
            training_module_unwrapped=training_module_unwrapped,
            cfg=cfg, enc=enc, detector=_det,
            training_layout_policy=layout_policy,
            inference_layout_policy=_inf_layout,
            device=str(dist.device),
        )
        _inference_model_cache[detector_name] = _m
        return _m

    if dist.is_main_process:
        # Default inference model (config detector) — the fallback used by
        # perplexity/split evals and by benchmarks with no implied detector.
        _flex_inference_model = _get_inference_model(None)

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
                from eval_checkpoints import run_benchmarks_on_model, detector_for_benchmark
                logger.info("Running post-training benchmark evaluation...")
                _flex_inference_model.eval()

                # data.val_dirs / data.test_dirs drive the post-training split evals.
                # community dirs  → community_pack_perplexity
                # random dirs     → held_out_perplexity (split="all" since dir is already filtered)
                _conditions = ["doceval"]
                # Multi-doc masks get the dual-mask eval: doc_causal 'baseline'
                # (apples-to-apples) + the model's own mask 'experimental'.
                if mask_type in ("cross_doc_link", "doc_concat_link", "doc_concatenated"):
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

                # Additional benchmarks from eval.benchmarks in YAML. Group by the
                # detector each benchmark implies (Tier-1), and run each group on
                # an inference model wired for that detector + its layout. Groups
                # with no implied detector (None) use the config default model.
                _bench_groups: Dict[Any, list] = {}
                for _spec in eval_cfg.get("benchmarks", []):
                    _det_name = detector_for_benchmark(_spec)
                    _bench_groups.setdefault(_det_name, []).append(_spec)

                eval_results: Dict[str, Any] = {}
                for _det_name, _specs in _bench_groups.items():
                    _grp_model = _get_inference_model(_det_name)
                    if _grp_model is None:
                        logger.warning(
                            "Skipping %d benchmark(s) for detector=%r: inference "
                            "model unavailable.", len(_specs), _det_name,
                        )
                        continue
                    _grp_model.eval()
                    _grp_cfg = {**eval_cfg, "benchmarks": _specs}
                    logger.info("Post-training eval: %d benchmark(s) with detector=%r",
                                len(_specs), _det_name)
                    _grp_results = run_benchmarks_on_model(
                        model=_grp_model,
                        dataset_dir=dataset_dir_str,
                        eval_cfg=_grp_cfg,
                        device=str(dist.device),
                    )
                    eval_results.update(_grp_results)
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
                        help="Path to a checkpoint FILE to resume from — e.g. "
                             ".../checkpoints/latest.pt (where the run was at the kill) or "
                             ".../checkpoints/best_model.pt (last val improvement); you pick "
                             "which. Restores model weights AND the full optimizer state "
                             "(Muon momentum + AdamW) exactly, keyed by parameter name so "
                             "resume is correct at any world_size and across the lm_head "
                             "untie split. Checkpoints must carry the name-keyed "
                             "optimizer_state written by the current training code (older "
                             "formats are not supported). max_optimizer_steps is "
                             "automatically reduced by the checkpoint step.")

    config = compose_config(parser)

    dist_mgr = DistributedManager()

    with dist_mgr:
        # Rank 0 picks the run directory name; all ranks receive it via broadcast
        # so they agree on one name even when launched within the same second.
        if dist_mgr.is_main:
            run_dir = os.path.join(
                os.path.dirname(__file__), "runs",
                datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
            )
        else:
            run_dir = None

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            run_dir_list = [run_dir]
            torch.distributed.broadcast_object_list(run_dir_list, src=0)
            run_dir = run_dir_list[0]

        rep = ReproducibilityManager(output_dir=run_dir, is_main_process=dist_mgr.is_main)
        with rep:
            main(config, dist_mgr, rep)
