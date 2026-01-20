import argparse
import logging
import os
import datetime
from pathlib import Path
from typing import Dict, Any
import json

import torch
from torch.utils.data import DataLoader

from tunalab.configuration import compose_config
from tunalab.distributed import DistributedManager
from tunalab.reproducibility import ReproducibilityManager
from tunalab import tracking
from tunalab.smart_train import smart_train

# Local imports
from model import DS2DSTrainingModule
from model.graph_traversal.block_mask_creator import make_mask_creator_callable
from data.dataset import GraphIndex, PretokShardedBackend, PackedSequenceDataset
from data.layout import BOSEOSLayoutPolicy, NullLayoutPolicy
from data.pack_sampler import PackBatchSampler
from data.traversal import (
    BFSStrategy,
    DFSStrategy,
    RandomSelectionStrategy,
    RandomWalkStrategy,
)


logger = logging.getLogger(__name__)


def main(cfg: Dict[str, Any], dist: DistributedManager, rep: ReproducibilityManager):
    """
    Main training entry point for dagseq2dagseq.
    
    This script is currently a template. It sets up the data pipeline using the
    graph traversal logic but has the model training loop commented out.
    """
    
    # -------------------------------------------------------------------------
    # 1. Setup Logging & Reproducibility
    # -------------------------------------------------------------------------
    if rep.output_dir:
        log_dir = os.path.join(rep.output_dir, "logs")
        tracking.init(log_dir, dist.rank)

    dist.set_seed(cfg.get("seed", 42))

    if rep.output_dir:
        json_path = os.path.join(rep.output_dir, "hyperparameters.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)

    # -------------------------------------------------------------------------
    # 2. Data Loading Setup
    # -------------------------------------------------------------------------
    # The dataset directory is expected to be passed via config or CLI
    # e.g. --data.dataset_dir /path/to/data
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

    # Configure Layout Policy (BOS/EOS wrapping)
    if cfg.get('data', {}).get('use_bos_eos', False):
        # GPT-2 uses 50256 as <|endoftext|> for both BOS and EOS
        tokenizer_name = graph_index.metadata.get('tokenizer', 'gpt2')
        if tokenizer_name == 'gpt2':
            bos_id = 50256  # GPT-2 <|endoftext|>
            eos_id = 50256
        else:
            bos_id = cfg.get('data', {}).get('bos_token_id', 1)
            eos_id = cfg.get('data', {}).get('eos_token_id', 2)
        layout_policy = BOSEOSLayoutPolicy(bos_token_id=bos_id, eos_token_id=eos_id)
    else:
        layout_policy = NullLayoutPolicy()

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

    # Configure Pack Sampler
    # This component is responsible for selecting documents to fill the context window
    pack_sampler = PackBatchSampler(
        graph=graph_index,
        strategy_factory=strategy_factory,
        token_budget=cfg.get('model', {}).get('max_seq_len', 2048),
        doc_budget=cfg.get('data', {}).get('doc_budget'),
        overflow_policy="truncate",
        doc_level_trim_side="tail",
        pack_level_trim_side="head",
        max_candidates_per_component=1000,
        seed=cfg.get("seed", 42),
        order_mode=cfg.get('data', {}).get('order_mode', 'prefer_targets_first'),
        layout_policy=layout_policy,
    )

    # Create the Dataset
    # This yields dictionaries containing 'tokens', 'doc_spans', etc.
    dataset = PackedSequenceDataset(
        graph=graph_index,
        backend=backend,
        pack_sampler=pack_sampler,
        layout_policy=layout_policy,
        as_2d=True,
    )

    # Create DataLoader
    # Since PackedSequenceDataset yields full batches (packed sequences), batch_size is None
    # or handled upstream if we want to stack multiple packed sequences.
    # For now, we assume 1 packed sequence per step.
    train_loader = DataLoader(
        dataset,
        batch_size=None, 
        num_workers=0,  # Can experiment with multiprocessing later
    )

    # -------------------------------------------------------------------------
    # 3. Model & Optimizer Setup
    # -------------------------------------------------------------------------
    logger.info("Initializing Model...")
    
    # Get vocab size from tokenizer metadata
    tokenizer_name = graph_index.metadata.get('tokenizer', 'gpt2')
    vocab_size = 50257 if tokenizer_name == 'gpt2' else cfg['model'].get('vocab_size', 50257)
    
    # Create block mask creator
    mask_type = cfg.get('model', {}).get('mask_type', 'doc_causal')
    block_mask_creator = make_mask_creator_callable(mask_type)
    
    # Build model using from_config factory
    model = DS2DSTrainingModule.from_config(
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
    
    if cfg['model']['compile']:
        logger.info("Compiling model with torch.compile...")
        model = torch.compile(
            model,
            dynamic=False,
            mode=cfg['model']['compile_mode'],
        )
    
    logger.info("Initializing Optimizer...")
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=cfg['optimizer']['max_lr'],
        betas=(cfg['optimizer'].get('beta1', 0.9), cfg['optimizer'].get('beta2', 0.95)),
        weight_decay=cfg['optimizer']['wd']
    )

    # -------------------------------------------------------------------------
    # 4. Training Loop
    # -------------------------------------------------------------------------
    logger.info("Starting Training...")
    
    atomic_feature_kwargs = cfg.get('train_loop', {}).get('atomic_feature_kwargs', {})
    atomic_feature_kwargs.update({
        'enable_logging': True,
        'output_dir': rep.output_dir,
        'device': dist.device,
        'use_tqdm': True,
        'max_epochs': cfg['train_loop'].get('epochs', 1),
    })

    result = smart_train(
        model=model, 
        optimizer=optimizer, 
        train_loader=train_loader, 
        **atomic_feature_kwargs
    )
    
    logger.info("Training complete!")
    
    # Cleanup
    backend.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a DAGSeq2DAGSeq model on the TAGWiki dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Define CLI arguments that map to config keys
    # These allow overriding config.yaml values from the command line
    parser.add_argument("--dataset-dir", dest="data.dataset_dir", type=str, 
                        help="Path to the pre-tokenized dataset directory.")
    parser.add_argument("--strategy", dest="data.strategy", type=str, default="random",
                        choices=["random", "random_walk", "bfs", "dfs"],
                        help="Graph traversal strategy.")
    parser.add_argument("--max-seq-len", dest="model.max_seq_len", type=int, default=2048,
                        help="Maximum sequence length (token budget per pack).")
    parser.add_argument("--seed", dest="seed", type=int, default=42,
                        help="Random seed.")

    # Load configuration
    config = compose_config(parser)
    
    # Setup run directory for logs and checkpoints
    run_dir = os.path.join(os.path.dirname(__file__), "runs", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    
    # Initialize Managers
    dist = DistributedManager()
    rep = ReproducibilityManager(output_dir=run_dir, is_main_process=dist.is_main)

    with dist, rep:
        main(config, dist, rep)
