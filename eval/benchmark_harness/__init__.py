"""Frozen verification harness for cross-doc benchmark ports.

Judges whether a ported external benchmark (Go←CoLT-132K, TS←CrossCodeEval,
Kotlin←ASE-2025, ...) is as legitimate as the existing python/java RepoBench
cross-doc benchmarks. Same philosophy as data/graph_harness: port implementers
author only a thin adapter (schema mapping + aux-identifier shaping); all
scoring, matching audits, and gates live here and are FROZEN — builder agents
must not modify this package.

Tiers (see docs/crossdoc_benchmark_port_harness_DESIGN.md):
  * Tier 0 (tier0.py, CPU): schema + mechanical invariants.
  * Tier 1 (tier1.py, CPU): link-resolution audit against the tree-sitter
    oracle from data/graph_harness — precision + fire-rate-parity gates.
  * Tier 2 (tier2.py, GPU): paired Δnll with bootstrap CI + shuffled-aux
    placebo control on a trained cross_doc_link checkpoint.
  * Tier C (dedup.py, CPU): hard dedup gate vs the training corpus
    (repo-name intersection, then file-hash for cross-repo copy-pastes).

Calibration: run every tier on the python/java RepoBench ports
(ports/repobench.py) first; their numbers define the legitimacy band.
"""
from .schema import AuxDoc, CrossDocExample, PortAdapter

__all__ = ["AuxDoc", "CrossDocExample", "PortAdapter"]
