#!/usr/bin/env python
"""
Visualize EXACTLY what text goes into the LLM for the ArXiv (unarXive) dataset.

This script reproduces the *training* data path faithfully — it uses the same
GraphIndex / PretokShardedBackend / PackBatchSampler / layout-policy / collate
code that ``main.py`` wires up — then decodes the resulting token tensor back to
text so you can read, character-for-character, what the model is trained on.

For each packed sequence it shows:

  * the pack-level summary (token budget, #docs, #connected-components);
  * a per-document breakdown of the three segments the layout policy emits:
        [ prefix card ]  +  [ body ]  +  [ EOS suffix ]
    with the prefix/suffix shown as both decoded text AND raw token ids, and the
    body shown as a head/tail snippet (or in full with --full-body);
  * every cross-doc link the configured detector finds (arxiv ``\\cite{Title}``,
    wikipedia/markdown ``[text](Target)``, thestack ``import``), and whether its
    target document is *in this pack* (a real cross-doc attention grant) or not;
  * the input_ids / labels causal shift the model actually sees.

The defaults mirror ``configs/arxiv_cross_doc.yaml`` (BFS traversal,
``stochastic_latex_comment_prefix`` layout, arxiv link detector) but every knob
is overridable on the CLI.

Run it (must use the mic2 env + data_registry PYTHONPATH, like training):

    python visualize_llm_input.py \\
        --dataset-dir /fss-data/evin_t/tagseq2tagseq_artifacts/pretokenized_datasets/arxiv/splits/val_random \\
        --num-packs 2 --token-budget 8192

By default it reads the small ``val_random`` split (loads in a few seconds vs.
~24s for the full corpus) so you get output fast; point --dataset-dir at the
full dataset or another split for the real thing.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict

import tiktoken

from data.collate import build_packed_batch
from data.dataset import GraphIndex, PretokShardedBackend
from data.layout import DocLayoutInfo, make_layout_policy
from data.pack_sampler import PackBatchSampler
from data.traversal import (
    BFSStrategy, DFSStrategy, RandomSelectionStrategy, RandomWalkStrategy,
)
from model.graph_traversal.link_detector import make_link_detector

# ---------------------------------------------------------------------------
# Defaults (mirror configs/arxiv_cross_doc.yaml)
# ---------------------------------------------------------------------------

DEFAULT_DATASET = (
    "/fss-data/evin_t/tagseq2tagseq_artifacts/"
    "pretokenized_datasets/arxiv/splits/val_random"
)


# ---------------------------------------------------------------------------
# Pretty-printing helpers
# ---------------------------------------------------------------------------

def _c(text: str, code: str, use_color: bool) -> str:
    return f"\033[{code}m{text}\033[0m" if use_color else text


def _rule(char: str = "=", width: int = 100) -> str:
    return char * width


def _short(text: str, n: int) -> str:
    """Collapse newlines and clip to n chars with an ellipsis marker."""
    flat = text.replace("\n", "\\n")
    if len(flat) <= n:
        return flat
    return flat[:n] + f" … (+{len(flat) - n} chars)"


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset-dir", default=DEFAULT_DATASET,
        help="Pretokenized dataset dir (must have metadata.json + "
             "tokenized_graph.jsonl). Default: the arxiv val_random split.",
    )
    parser.add_argument(
        "--num-packs", type=int, default=2,
        help="How many packed sequences to render.",
    )
    parser.add_argument(
        "--token-budget", type=int, default=8192,
        help="Tokens per pack (== model.max_seq_len in training; the real arxiv "
             "config uses 32768 — smaller is more readable here).",
    )
    parser.add_argument(
        "--strategy", default="bfs", choices=["bfs", "dfs", "random", "random_walk"],
        help="Graph traversal strategy (config: data.strategy).",
    )
    parser.add_argument(
        "--layout-policy", default="stochastic_latex_comment_prefix",
        help="Layout policy name (config: data.layout_policy). Use "
             "'latex_comment_prefix' for a deterministic always-on card.",
    )
    parser.add_argument(
        "--order-mode", default="prefer_targets_first",
        choices=["as_traversed", "prefer_targets_first"],
        help="In-pack document ordering (config: data.order_mode).",
    )
    parser.add_argument(
        "--doc-budget", type=int, default=None,
        help="Max body tokens drawn from a single doc (config: data.doc_budget).",
    )
    parser.add_argument(
        "--link-detector", default="arxiv",
        help="Link detector for citation/grant detection (config: model.link_detector).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Sampler seed (training uses cfg.seed + rank).",
    )
    parser.add_argument(
        "--epoch", type=int, default=0,
        help="Epoch index — drives the stochastic layout coin-flip (md5(id:epoch)).",
    )
    parser.add_argument(
        "--full-body", action="store_true",
        help="Print each document body in full instead of a head/tail snippet.",
    )
    parser.add_argument(
        "--snippet-chars", type=int, default=600,
        help="Chars of body head/tail to show when not using --full-body.",
    )
    parser.add_argument(
        "--no-color", action="store_true", help="Disable ANSI colour.",
    )
    parser.add_argument(
        "--dump-raw", action="store_true",
        help="Also print the entire decoded pack as one raw stream (what the "
             "tokenizer would round-trip), with doc boundaries marked.",
    )
    args = parser.parse_args()

    use_color = not args.no_color and sys.stdout.isatty()

    dataset_dir = Path(args.dataset_dir)
    print(_rule())
    print(f"Loading GraphIndex from {dataset_dir} ...")
    graph = GraphIndex(dataset_dir)
    backend = PretokShardedBackend(graph)

    # Tokenizer — exactly as main.py: read the name from dataset metadata.
    enc = tiktoken.get_encoding(graph.metadata.get("tokenizer", "gpt2"))
    eos_id = enc.eot_token  # 50256 for gpt2 (<|endoftext|>)

    # Layout policy — same factory + same encode_fn (encode_ordinary) as training.
    layout = make_layout_policy(
        name=args.layout_policy,
        encode_fn=enc.encode_ordinary,
        eos_token_id=eos_id,
    )
    # Stochastic policies pick the card per (doc, epoch); pin the epoch so the
    # render is reproducible and matches a specific training epoch.
    if hasattr(layout, "set_epoch"):
        layout.set_epoch(args.epoch)

    # Traversal strategy — mirrors main.py's strategy_factory switch.
    if args.strategy == "bfs":
        strategy_factory = lambda: BFSStrategy(edge_mode="outgoing")
    elif args.strategy == "dfs":
        strategy_factory = lambda: DFSStrategy(edge_mode="outgoing")
    elif args.strategy == "random":
        strategy_factory = lambda: RandomSelectionStrategy()
    else:  # random_walk
        strategy_factory = lambda: RandomWalkStrategy(edge_mode="outgoing", restart_prob=0.05)

    sampler = PackBatchSampler(
        graph=graph,
        strategy_factory=strategy_factory,
        token_budget=args.token_budget,
        doc_budget=args.doc_budget,
        overflow_policy="truncate",
        doc_level_trim_side="tail",
        pack_level_trim_side="head",
        max_candidates_per_component=1000,
        seed=args.seed,
        order_mode=args.order_mode,
        layout_policy=layout,
    )

    # Link detector — the same object CrossDocLinkMaskCreator uses to find the
    # \cite{...} citations that become cross-document attention grants.
    detector = make_link_detector(args.link_detector, enc.decode)

    print(
        f"  nodes={len(graph):,}  tokenizer={graph.metadata.get('tokenizer','gpt2')}  "
        f"eos={eos_id}  dtype={graph.token_dtype}"
    )
    print(
        f"  strategy={args.strategy}  layout={args.layout_policy}  "
        f"order_mode={args.order_mode}  token_budget={args.token_budget}  "
        f"doc_budget={args.doc_budget}  epoch={args.epoch}"
    )
    print(_rule())

    pack_iter = iter(sampler)
    for pack_no in range(args.num_packs):
        try:
            placements = next(pack_iter)
        except StopIteration:
            print("\n(no more packs)")
            break
        if not placements:
            continue

        batch = build_packed_batch(graph, backend, layout, placements, as_2d=True)
        tokens = batch["tokens"]            # shape [1, T]
        spans = batch["doc_spans"]
        flat = tokens.view(-1)
        T = flat.shape[0]

        # Causal shift the training_module applies: input = tokens[:-1], target = tokens[1:].
        n_components = len({s.component_id for s in spans})

        print()
        print(_c(_rule("#"), "1;36", use_color))
        print(_c(
            f"PACK {pack_no}   |   {T:,} tokens   |   {len(spans)} docs   |   "
            f"{n_components} connected component(s)",
            "1;36", use_color,
        ))
        print(_c(
            f"  (model sees input_ids = tokens[:-1] = {T-1:,} tokens, "
            f"predicts labels = tokens[1:])",
            "36", use_color,
        ))
        print(_c(_rule("#"), "1;36", use_color))

        # ---- per-document breakdown ---------------------------------------
        # Key spans by the SAME string CrossDocLinkMaskCreator matches against:
        # detector.index_doc_span(span). This is raw_identifier for the arxiv /
        # markdown detectors but a sub-component (e.g. a file path) for python,
        # so using it keeps the in-pack grant check correct across datasets.
        target_key_to_span: Dict[str, list] = {}
        for span in spans:
            target_key_to_span.setdefault(detector.index_doc_span(span), []).append(span)

        for i, span in enumerate(spans):
            normed = span.normed_identifier
            # Recompute the three segments the layout policy emitted for this doc
            # so we can label which tokens are decoration vs body. (build_packed_batch
            # already concatenated them; here we re-derive the lengths.)
            info = DocLayoutInfo(
                raw_identifier=span.raw_identifier,
                normed_identifier=normed,
                outgoing_identifiers=span.outgoing_identifiers,
                incoming_identifiers=graph.get_incoming_links(normed),
                body_tokens=None,
                categories=graph.get_categories(normed),
            )
            prefix_ids = layout.prefix_tokens(info)
            suffix_ids = layout.suffix_tokens(info)
            n_pre, n_suf = len(prefix_ids), len(suffix_ids)
            doc_token_ids = flat[span.start:span.end].tolist()
            body_ids = doc_token_ids[n_pre: len(doc_token_ids) - n_suf] if n_suf else doc_token_ids[n_pre:]

            header = (
                f"── doc[{i}]  pos[{span.start}:{span.end}]  "
                f"({span.end - span.start} tok)  "
                f"doc_id={span.doc_id}  component={span.component_id}"
                + ("  TRUNCATED" if span.truncated else "")
            )
            print()
            print(_c(header, "1;33", use_color))
            print(f"   normed_id : {normed}")
            print(f"   title     : {span.raw_identifier!r}")
            print(f"   categories: {graph.get_categories(normed)!r}")

            # PREFIX (the LaTeX-comment card, or empty on a stochastic 'no-card' flip)
            if n_pre:
                print(_c(f"   ┌ PREFIX  ({n_pre} tok)  ids={prefix_ids}", "32", use_color))
                pre_text = enc.decode(prefix_ids)
                for line in pre_text.splitlines():
                    print(_c(f"   │   {line}", "32", use_color))
            else:
                reason = (
                    " (stochastic coin-flip: no card this epoch)"
                    if "stochastic" in args.layout_policy else ""
                )
                print(_c(f"   ┌ PREFIX  (none){reason}", "32", use_color))

            # BODY (the rehydrated LaTeX paper text from the pretokenized shard)
            body_text = enc.decode(body_ids)
            print(_c(f"   ├ BODY    ({len(body_ids)} tok)", "0", use_color))
            if args.full_body:
                for line in body_text.splitlines():
                    print(f"   │   {line}")
            else:
                half = args.snippet_chars // 2
                print(f"   │   head: {_short(body_text[:half], half)!r}")
                if len(body_text) > args.snippet_chars:
                    print(f"   │   tail: {_short(body_text[-half:], half)!r}")

            # SUFFIX (the EOS doc-boundary token)
            if n_suf:
                print(_c(
                    f"   └ SUFFIX  ({n_suf} tok)  ids={suffix_ids}  "
                    f"decoded={enc.decode(suffix_ids)!r}  (EOS / doc boundary)",
                    "35", use_color,
                ))
            else:
                print(_c("   └ SUFFIX  (none)", "35", use_color))

        # ---- cross-doc links ----------------------------------------------
        # Render the detected link in the surface syntax of the active detector
        # so the display matches the dataset (arxiv \cite{}, wikipedia/markdown
        # [](), thestack `import`); detection itself is detector-agnostic.
        def _fmt_link(target: str) -> str:
            t = _short(target, 70)
            if args.link_detector == "arxiv":
                return f"\\cite{{{t}}}"
            if args.link_detector == "markdown":
                return f"[...]({t})"
            if args.link_detector == "python":
                return f"import {t}"
            return f"→ {t!r}"

        print()
        print(_c(
            f"── CROSS-DOC LINKS ({args.link_detector} detector; targets matched "
            f"against in-pack docs)", "1;34", use_color,
        ))
        links = detector.detect_links(flat)
        if not links:
            print("   (none detected)")
        else:
            for ln in links:
                # Which doc emitted the link (link_end_pos falls in its span)?
                src = next(
                    (s for s in spans if s.start < ln.link_end_pos <= s.end), None
                )
                src_idx = spans.index(src) if src in spans else "?"
                # Is the target present in this pack? CrossDocLinkMaskCreator
                # matches target_str against detector.index_doc_span(span). If so,
                # the model is granted attention to it — a real cross-doc edge.
                target_spans = target_key_to_span.get(ln.target_str)
                if target_spans:
                    tgt_idxs = [spans.index(t) for t in target_spans]
                    status = _c(
                        f"IN-PACK → grants attention to doc{tgt_idxs}", "1;32", use_color
                    )
                else:
                    status = _c("not in pack (no grant)", "90", use_color)
                print(
                    f"   @tok {ln.link_end_pos:>6}  from doc[{src_idx}]  "
                    f"{_fmt_link(ln.target_str)}  {status}"
                )

        # ---- optional raw stream dump -------------------------------------
        if args.dump_raw:
            print()
            print(_c("── RAW DECODED STREAM (full pack, doc boundaries marked ⟦…⟧)", "1;37", use_color))
            parts = []
            for i, span in enumerate(spans):
                seg = enc.decode(flat[span.start:span.end].tolist())
                parts.append(_c(f"⟦doc{i}⟧", "1;31", use_color) + seg)
            print("".join(parts))

    backend.close()
    print()
    print(_rule())
    print("done.")


if __name__ == "__main__":
    main()
