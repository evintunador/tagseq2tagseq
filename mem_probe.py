"""
Quick VRAM probe: runs 3 training steps and reports peak GPU memory.
Called by launch_slurm.py exactly like main.py (same signature).
"""
import os
import itertools
import torch
from pathlib import Path
from typing import Dict, Any

from tunalab.distributed import DistributedManager
from tunalab.reproducibility import ReproducibilityManager

from model import TS2TSTrainingModule
from model.graph_traversal.block_mask_creator import (
    make_mask_creator_callable, make_mask_creator_callable_from,
)
from model.graph_traversal.cross_doc_mask import CrossDocLinkMaskCreator
from model.graph_traversal.markdown_link_detector import MarkdownLinkDetector
from model.graph_traversal.python_import_detector import PythonImportDetector
from data.dataset import GraphIndex, PretokShardedBackend
from data.packed_dataset import PackedSequenceDataset
from data.layout import NullLayoutPolicy
from data.pack_sampler import PackBatchSampler
from data.traversal import DFSStrategy, RandomSelectionStrategy
from tunalab.optimizers.muon import SingleDeviceMuonWithAuxAdam, MuonWithAuxAdam
from torch.nn.parallel import DistributedDataParallel as DDP
import tiktoken


def main(cfg: Dict[str, Any], dist: DistributedManager, rep: ReproducibilityManager):
    dataset_dir = Path(cfg["data"]["dataset_dir"])
    graph = GraphIndex(dataset_dir)
    backend = PretokShardedBackend(graph)

    strategy_name = cfg.get("data", {}).get("strategy", "random")
    sf = (lambda: DFSStrategy(edge_mode="outgoing")) if strategy_name == "dfs" \
         else (lambda: RandomSelectionStrategy())

    seq_len = cfg["model"]["max_seq_len"]
    sampler = PackBatchSampler(
        graph=graph, strategy_factory=sf,
        token_budget=seq_len, overflow_policy="truncate",
        doc_level_trim_side="tail", pack_level_trim_side="head",
        max_candidates_per_component=1000,
        seed=cfg.get("seed", 42) + dist.rank,
        layout_policy=NullLayoutPolicy(),
    )
    dataset = PackedSequenceDataset(graph=graph, backend=backend,
                                    pack_sampler=sampler,
                                    layout_policy=NullLayoutPolicy(), as_2d=True)

    mask_type = cfg["model"]["mask_type"]
    if mask_type == "cross_doc_link":
        enc = tiktoken.get_encoding("gpt2")
        det_name = cfg["model"]["link_detector"]
        detector = PythonImportDetector(decode_fn=enc.decode) if det_name == "python" \
                   else MarkdownLinkDetector(decode_fn=enc.decode)
        bmc = make_mask_creator_callable_from(CrossDocLinkMaskCreator(link_detector=detector))
    else:
        bmc = make_mask_creator_callable(mask_type)

    model = TS2TSTrainingModule.from_config(
        vocab_size=50257,
        num_layers=cfg["model"]["num_layers"],
        model_dim=cfg["model"]["model_dim"],
        num_heads=cfg["model"]["num_heads"],
        max_seq_len=seq_len,
        dropout=cfg["model"].get("dropout", 0.0),
        drop_path_rate=0.0,
        block_mask_creator=bmc,
        weight_tying=cfg["model"].get("weight_tying", True),
        dtype=getattr(torch, cfg["model"].get("dtype", "bfloat16")),
    ).to(dist.device)

    muon_p, adam_p, seen = [], [], set()
    for name, p in model.named_parameters():
        if id(p) in seen: continue
        seen.add(id(p))
        if "backbone" in name and p.ndim >= 2: muon_p.append(p)
        else: adam_p.append(p)

    OptCls = MuonWithAuxAdam if dist.is_distributed else SingleDeviceMuonWithAuxAdam
    opt = OptCls([
        dict(params=muon_p, use_muon=True,
             lr=cfg["optimizer"]["muon_lr"],
             momentum=cfg["optimizer"].get("momentum", 0.95),
             weight_decay=cfg["optimizer"]["wd"]),
        dict(params=adam_p, use_muon=False,
             lr=cfg["optimizer"]["adamw_lr"],
             betas=(cfg["optimizer"].get("beta1", 0.9), cfg["optimizer"].get("beta2", 0.95)),
             weight_decay=cfg["optimizer"]["wd"]),
    ])

    if cfg["model"]["compile"]:
        model.backbone = torch.compile(model.backbone, dynamic=False,
                                       mode=cfg["model"]["compile_mode"])

    if dist.is_distributed:
        model = DDP(model, device_ids=[dist.local_rank], static_graph=True,
                    find_unused_parameters=False)

    model.train()
    opt.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(dist.device)

    total_mem = torch.cuda.get_device_properties(dist.device).total_memory / 1e9

    for step, batch in enumerate(itertools.islice(iter(dataset), 3)):
        loss = model(batch)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

        peak = torch.cuda.max_memory_allocated(dist.device) / 1e9
        if dist.is_main_process:
            L = cfg["model"]["num_layers"]
            D = cfg["model"]["model_dim"]
            print(f"[{L}L/{D}D seq={seq_len}]  "
                  f"step={step}  loss={loss.item():.3f}  "
                  f"peak={peak:.1f}/{total_mem:.0f}GB  "
                  f"({peak/total_mem*100:.0f}%)", flush=True)

    backend.close()
