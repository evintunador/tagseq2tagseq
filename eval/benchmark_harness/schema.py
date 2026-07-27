"""Canonical port schema — the ONLY interface between a port and the harness.

A port adapter converts one upstream dataset (RepoBench, CoLT-132K,
CrossCodeEval, ASE-2025, ...) into a deterministic list of CrossDocExample.
Everything downstream — the harness tiers AND the eventual eval runner —
consumes only this shape, so a port that passes the harness is by construction
runnable through score_completion_with_context_docs the same way
run_repobench_cross_doc runs python/java today.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class AuxDoc:
    """One cross-file context document.

    path is the repo-relative file path of the snippet's source file; content
    is the snippet text (whole file, chunk, or signature skeleton — whatever
    the upstream provides). The port's identifier_fn shapes (repo, path) into
    the raw_identifier whose index_doc_span key matches what the language's
    import detector emits as target_str.
    """
    path: str
    content: str


@dataclass(frozen=True)
class CrossDocExample:
    repo: str
    file_path: str            # repo-relative path of the primary file
    context: str              # primary-file prefix INCLUDING the import block
    target: str               # completion text to score (the port's NATIVE target)
    aux: Tuple[AuxDoc, ...]   # cross-file context docs, pack order
    meta: Dict[str, Any] = field(default_factory=dict)
    # Optional: the FULL primary file text (context + native target + whatever
    # follows). Required only for re-scoping the target to use-site spans
    # (scopes.py); ports that can't supply it (e.g. CCEval ships only left
    # context + groundtruth) leave it None and support the 'native' scope only.
    full_file: Optional[str] = None


@dataclass(frozen=True)
class PortAdapter:
    """Everything the harness needs to know about one benchmark port.

    Port implementers construct one of these; the harness never imports
    port-specific code paths beyond it.

    identifier_fn(repo, path, content) -> raw_identifier for an AuxDoc — the
    per-language shaping that makes index_doc_span(raw_identifier) equal the
    detector's emitted target_str (e.g. python: "repo:path" verbatim; java:
    "repo:" + source-root-stripped path). This is the port's ONLY logic beside
    schema mapping, and the component Tier 1 audits hardest.

    detector_factory(decode_fn) -> LinkDetector instance for the language.
    """
    name: str                 # e.g. "repobench_python", "colt_go"
    language: str             # graph_harness spec name: python/java/go/...
    examples_fn: Callable[[Optional[int]], List[CrossDocExample]]
    identifier_fn: Callable[[str, str, str], str]
    detector_factory: Callable[[Callable[[List[int]], str]], Any]

    def load(self, max_examples: Optional[int] = None) -> List[CrossDocExample]:
        return self.examples_fn(max_examples)


def encode_example(
    ex: CrossDocExample,
    enc: Callable[[str], List[int]],
    identifier_fn: Callable[[str, str, str], str],
) -> Dict[str, Any]:
    """Tokenize one example into score_completion_with_context_docs kwargs.

    Single tokenization point shared by every tier and the runner, so flat and
    cross-doc conditions provably score identical completion token ids
    (Tier 0's token-accounting parity is enforced by construction here and
    re-checked independently in tier0).
    """
    aux_token_lists: List[List[int]] = []
    aux_raw_identifiers: List[str] = []
    for doc in ex.aux:
        if not doc.content.strip():
            continue
        aux_token_lists.append(enc(doc.content))
        aux_raw_identifiers.append(identifier_fn(ex.repo, doc.path, doc.content))
    return {
        "aux_token_lists": aux_token_lists,
        "aux_raw_identifiers": aux_raw_identifiers,
        "context_tokens": enc(ex.context),
        "completion_tokens": enc(ex.target),
        "source_file_path": ex.file_path,
    }
