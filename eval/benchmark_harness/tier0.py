"""Tier 0 — schema + mechanical invariants. CPU-only, no model, no oracle.

Hard gates a port must pass before Tier 1/2 are worth running:
  * non-empty context/target; ≥1 non-empty aux doc
  * aux paths repo-relative (no leading '/', no '..' traversal); no fully
    identical (path, content) aux duplicates per example (same path with
    DIFFERENT content is legitimate — RepoBench ships multiple snippets from
    one file)
  * imports visible: the language's import construct appears in `context`,
    checked by TREE-SITTER parse via the graph_harness oracle (not regex) —
    an example whose oracle key set is empty has had its import block cropped
    upstream (report; gated on the fraction)
  * token-accounting parity: the flat pair and the cross-doc pack encode the
    identical completion token ids (re-derived here independently of
    schema.encode_example)
  * determinism: two examples_fn() runs produce identical example lists
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from .schema import CrossDocExample, PortAdapter, encode_example

logger = logging.getLogger(__name__)

# An example whose context yields ZERO oracle import keys has lost its import
# block upstream. python/java RepoBench sit near 0; gate leaves headroom for
# upstream noise without letting a port degenerate.
MAX_NO_IMPORT_FRAC = 0.10

# Upstream sets ship occasional rows with no cross-file snippets (RepoBench
# python: 1/500); runtime falls back to flat scoring for them. Tolerate a
# trace amount — more means the port is not a cross-doc benchmark.
MAX_NO_AUX_FRAC = 0.02


@dataclass
class Tier0Report:
    port: str
    n_examples: int
    n_empty_context: int = 0
    n_empty_target: int = 0
    n_no_aux: int = 0
    n_bad_aux_path: int = 0
    n_dup_aux_path: int = 0
    n_no_import_in_context: int = 0
    n_token_parity_fail: int = 0
    deterministic: bool = True
    failures: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.failures


def run_tier0(
    port: PortAdapter,
    enc: Callable[[str], List[int]],
    max_examples: Optional[int] = None,
    check_determinism: bool = True,
) -> Tier0Report:
    examples = port.load(max_examples)
    rep = Tier0Report(port=port.name, n_examples=len(examples))
    if not examples:
        rep.failures.append("port produced zero examples")
        return rep

    from data.graph_harness.specs import get_spec
    from data.graph_harness.oracle import TreeSitterOracle
    oracle = TreeSitterOracle(get_spec(port.language))

    for i, ex in enumerate(examples):
        if not ex.context.strip():
            rep.n_empty_context += 1
        if not ex.target.strip():
            rep.n_empty_target += 1
        if not any(d.content.strip() for d in ex.aux):
            rep.n_no_aux += 1
        paths = [d.path for d in ex.aux]
        if any(p.startswith("/") or ".." in p.split("/") for p in paths):
            rep.n_bad_aux_path += 1
        pairs = [(d.path, d.content) for d in ex.aux]
        if len(set(pairs)) != len(pairs):
            rep.n_dup_aux_path += 1
        if not oracle.import_keys(ex.context):
            rep.n_no_import_in_context += 1

        # Token-accounting parity: completion ids identical in flat and packed
        # form. encode_example is the shared tokenization point; here we
        # re-derive the flat pair directly from the raw text.
        packed = encode_example(ex, enc, port.identifier_fn)
        if packed["completion_tokens"] != enc(ex.target):
            rep.n_token_parity_fail += 1

    if check_determinism:
        second = port.load(max_examples)
        rep.deterministic = second == examples
        if not rep.deterministic:
            rep.failures.append("examples_fn is not deterministic across runs")

    n = rep.n_examples
    if rep.n_empty_context:
        rep.failures.append(f"{rep.n_empty_context}/{n} empty context")
    if rep.n_empty_target:
        rep.failures.append(f"{rep.n_empty_target}/{n} empty target")
    if rep.n_no_aux > MAX_NO_AUX_FRAC * n:
        rep.failures.append(
            f"{rep.n_no_aux}/{n} examples with no usable aux docs "
            f"(gate {MAX_NO_AUX_FRAC:.0%})")
    if rep.n_bad_aux_path:
        rep.failures.append(f"{rep.n_bad_aux_path}/{n} non-repo-relative aux paths")
    if rep.n_dup_aux_path:
        rep.failures.append(
            f"{rep.n_dup_aux_path}/{n} examples with fully duplicate (path, content) aux docs")
    if rep.n_no_import_in_context > MAX_NO_IMPORT_FRAC * n:
        rep.failures.append(
            f"{rep.n_no_import_in_context}/{n} contexts contain NO parseable import "
            f"(gate {MAX_NO_IMPORT_FRAC:.0%}) — import block cropped upstream?")
    if rep.n_token_parity_fail:
        rep.failures.append(f"{rep.n_token_parity_fail}/{n} token-accounting parity failures")

    return rep
