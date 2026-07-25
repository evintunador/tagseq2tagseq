"""Tier 1 — link-resolution audit against the tree-sitter oracle. CPU-only.

Ground truth comes from data/graph_harness's frozen LanguageSpec (independent
of both the import detectors and the port adapters). Per example:

  * oracle keys   = spec-licensed canonical import keys parsed from `context`
  * reachable aux = aux docs whose path projects into the oracle key set.
    Projection is HARNESS-owned and port-independent: canonical_target over
    every path suffix (progressively stripping leading components). Suffix
    stripping makes reachability an upper bound that absorbs build-layout
    prefixes (e.g. java/kotlin source roots) without trusting the port's
    identifier_fn — the component under test.
  * port matches  = what the real runtime pipeline grants: detector.detect_links
    over the packed token sequence, target_str looked up via index_doc_span of
    identifier_fn-shaped DocSpans (mirrors score_completion_with_context_docs
    precise mode; relative-import recovery is runtime-only and excluded on BOTH
    sides — the oracle skips relative imports too).

Gates:
  * match precision ≥ 0.95 — every granted aux is oracle-licensed (import key
    in oracle set AND aux path suffix-projects to that key). A grant to a
    snippet the file does not import corrupts the benchmark.
  * fire-rate parity — port fire-rate ≥ 0.9 × oracle reachable-rate. Losing
    resolvable links to identifier-shaping bugs is exactly the java
    source-root failure class.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple

from data.collate import DocSpan

from .schema import CrossDocExample, PortAdapter, encode_example

logger = logging.getLogger(__name__)

PRECISION_GATE = 0.95
FIRE_RATE_PARITY = 0.90


def _suffix_keys(path: str, canonical_target) -> Set[str]:
    """All canonical keys reachable by stripping leading path components."""
    parts = path.replace("\\", "/").split("/")
    keys: Set[str] = set()
    for i in range(len(parts)):
        k = canonical_target("/".join(parts[i:]))
        if k:
            keys.add(k)
    return keys


@dataclass
class Tier1Report:
    port: str
    n_examples: int
    n_oracle_reachable: int = 0
    n_port_fired: int = 0
    n_grants: int = 0
    n_grants_licensed: int = 0
    failures: List[str] = field(default_factory=list)

    @property
    def oracle_reachable_rate(self) -> float:
        return self.n_oracle_reachable / self.n_examples if self.n_examples else 0.0

    @property
    def port_fire_rate(self) -> float:
        return self.n_port_fired / self.n_examples if self.n_examples else 0.0

    @property
    def precision(self) -> float:
        return self.n_grants_licensed / self.n_grants if self.n_grants else 1.0

    @property
    def passed(self) -> bool:
        return not self.failures


def run_tier1(
    port: PortAdapter,
    enc: Callable[[str], List[int]],
    decode_fn: Callable[[List[int]], str],
    max_examples: Optional[int] = None,
) -> Tier1Report:
    import torch
    from data.graph_harness.specs import get_spec
    from data.graph_harness.oracle import TreeSitterOracle

    spec = get_spec(port.language)
    oracle = TreeSitterOracle(spec)
    detector = port.detector_factory(decode_fn)

    examples = port.load(max_examples)
    rep = Tier1Report(port=port.name, n_examples=len(examples))

    for ex in examples:
        oracle_keys = oracle.import_keys(ex.context)
        packed = encode_example(ex, enc, port.identifier_fn)
        aux_tok = packed["aux_token_lists"]
        aux_ids = packed["aux_raw_identifiers"]
        ctx_tok = packed["context_tokens"]
        if not aux_tok or not ctx_tok:
            continue

        # Aux paths that survive encode_example's empty-content skip, aligned
        # with aux_ids. Reachability per aux via suffix projection.
        aux_paths = [d.path for d in ex.aux if d.content.strip()]
        aux_suffix_keys = [_suffix_keys(p, spec.canonical_target) for p in aux_paths]
        reachable = [bool(sk & oracle_keys) for sk in aux_suffix_keys]
        if any(reachable):
            rep.n_oracle_reachable += 1

        # Port-side matching: mirror score_completion_with_context_docs precise
        # mode (absolute matches only — relative recovery is runtime-only).
        path_to_idx: Dict[str, int] = {}
        offset = 0
        for i, toks in enumerate(aux_tok):
            span = DocSpan(doc_id=i, normed_identifier="",
                           raw_identifier=aux_ids[i],
                           start=offset, end=offset + len(toks),
                           truncated=False, outgoing_identifiers=[])
            path_to_idx[detector.index_doc_span(span)] = i
            offset += len(toks)
        primary_start = offset
        all_tokens = [t for toks in aux_tok for t in toks] + ctx_tok
        tensor = torch.tensor(all_tokens, dtype=torch.long)
        links = [lk for lk in detector.detect_links(tensor)
                 if primary_start <= lk.link_end_pos <= len(all_tokens)]

        granted: List[Tuple[str, int]] = []   # (target_str, aux index)
        for lk in links:
            idx = path_to_idx.get(lk.target_str)
            if idx is not None:
                granted.append((lk.target_str, idx))
        if granted:
            rep.n_port_fired += 1

        for target_str, idx in granted:
            rep.n_grants += 1
            k = spec.canonical_target(target_str)
            if k is not None and k in oracle_keys and k in aux_suffix_keys[idx]:
                rep.n_grants_licensed += 1

    if rep.precision < PRECISION_GATE:
        rep.failures.append(
            f"grant precision {rep.precision:.3f} < {PRECISION_GATE} "
            f"({rep.n_grants_licensed}/{rep.n_grants} grants oracle-licensed)")
    if rep.port_fire_rate < FIRE_RATE_PARITY * rep.oracle_reachable_rate:
        rep.failures.append(
            f"fire-rate {rep.port_fire_rate:.3f} < {FIRE_RATE_PARITY} × oracle "
            f"reachable-rate {rep.oracle_reachable_rate:.3f} — identifier "
            f"shaping is losing resolvable links")

    return rep
