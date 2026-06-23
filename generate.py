"""
generate.py — standalone generation CLI for TS2TS models.

Usage:
    python generate.py \\
        --checkpoint runs/YYYYMMDD/checkpoints/best_model.pt \\
        --prompt "Python is a programming language" \\
        [--dataset data/pretokenized_datasets/simplewiki] \\
        [--max-new-tokens 300] \\
        [--max-link-depth 2] \\
        [--link-retrieval-mode {corpus_only,generate_only,corpus_then_generate,link_but_skip,full_skip}] \\
        [--temperature 0.8] \\
        [--top-k 50] \\
        [--max-display-tokens 200] \\
        [--no-color]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tiktoken
import torch
import torch.nn.functional as F

from data.collate import DocSpan
from data.dataset import GraphIndex, PretokShardedBackend
from data.layout import (
    EOSLayoutPolicy,
    IdentifierPrefixEOSLayoutPolicy,
    IdentifierPrefixLayoutPolicy,
    NullLayoutPolicy,
)
from model.generation_config import GenerationConfig
from model.generation_result import GeneratedDocument, GenerationResult
from model.graph_traversal.block_mask_creator import (
    make_mask_creator_callable,
    make_mask_creator_callable_from,
)
from model.graph_traversal.cross_doc_mask import CrossDocLinkMaskCreator
from model.graph_traversal.link_detector import make_link_detector
from data.normalization import normalize_wiki_title
from model.modules.training_module import TS2TSTrainingModule


# ─────────────────────────────────────────────────────────────────────────────
# ANSI helpers
# ─────────────────────────────────────────────────────────────────────────────

CYAN  = "\033[96m"
BOLD  = "\033[1m"
DIM   = "\033[2m"
RESET = "\033[0m"

def _c(text: str, code: str, use_color: bool) -> str:
    return f"{code}{text}{RESET}" if use_color else text


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint loading
# ─────────────────────────────────────────────────────────────────────────────

# Maps training layout policy names to their deterministic inference equivalents.
# Stochastic policies must never be used at inference — they produce different
# prefix tokens each call, breaking stable generation and eval comparisons.
_STOCHASTIC_TO_DETERMINISTIC = {
    'stochastic_identifier_prefix': 'identifier_prefix_eos',
}


def _build_layout_policy(name: str, enc):
    """Construct a DocLayoutPolicy from a config name string.

    TODO: unify with data.layout.make_layout_policy. This is a second, smaller
    reimplementation of that factory and only handles the identifier_prefix
    family — it does NOT know the latex_comment_prefix policies, so a checkpoint
    trained with 'stochastic_latex_comment_prefix' (without an explicit, also-
    supported inference_layout_policy) raises ValueError here. Replacing this
    with a call to make_layout_policy (passing enc.encode_ordinary) gives one
    source of truth and closes that gap.
    """
    if name == "null":
        return NullLayoutPolicy()
    elif name == "eos":
        return EOSLayoutPolicy(eos_token_id=50256)
    elif name == "identifier_prefix":
        return IdentifierPrefixLayoutPolicy(encode_fn=enc.encode_ordinary)
    elif name in ("identifier_prefix_eos", "identifier_prefix_bos_eos"):
        # 'identifier_prefix_bos_eos' is a legacy alias from before BOS was removed.
        return IdentifierPrefixEOSLayoutPolicy(
            encode_fn=enc.encode_ordinary, eos_token_id=50256
        )
    else:
        raise ValueError(
            f"Unknown layout_policy {name!r}. "
            "Expected 'null', 'eos', 'identifier_prefix', "
            "'identifier_prefix_eos', or 'stochastic_identifier_prefix'."
        )


def load_inference_model(
    checkpoint_path: str | Path,
    device: str = "cuda",
    inference_attention_backend: str = "flex",
):
    """
    Load a trained checkpoint and return (inference_model, hyperparams_dict).

    Reads hyperparameters.json from the run directory adjacent to the
    checkpoint, reconstructs the architecture, then converts to inference mode.

    Args:
        checkpoint_path: Path to best_model.pt.
        device: Target device.
        inference_attention_backend: Attention backend for the returned inference
            model. Defaults to 'flex' — FlexSelfAttention + torch.compile is
            ~40× faster than varlen_bim_v1 for single-doc forward passes (37ms
            vs 1.6s) because it compiles a single dynamic kernel rather than
            JIT-compiling per unique sequence length. Use 'triton' to keep the
            training backend (e.g. for batched multi-doc eval pipelines).
    """
    checkpoint_path = Path(checkpoint_path)
    run_dir = checkpoint_path.parent.parent   # .../runs/YYYYMMDD/checkpoints/best.pt
    hp_path = run_dir / "hyperparameters.json"
    if not hp_path.exists():
        raise FileNotFoundError(f"hyperparameters.json not found at {hp_path}")

    with open(hp_path) as f:
        hp = json.load(f)

    model_cfg = hp["model"]
    data_cfg  = hp.get("data", {})

    # Tokenizer (GPT-2 only for now)
    enc = tiktoken.get_encoding("gpt2")

    # Mask type and link detector
    mask_type          = model_cfg.get("mask_type", "doc_causal")
    link_detector_name = model_cfg.get("link_detector")
    link_detector      = None

    if mask_type == "cross_doc_link":
        link_detector = make_link_detector(link_detector_name, enc.decode)

    # Training backend — determines which attention kernel was used during training
    use_triton = model_cfg.get("attention_backend", "triton") != "flex"
    if mask_type == "cross_doc_link":
        training_attention_backend = "triton" if use_triton else "flex"
        training_attn_class_backend = "triton_v12" if use_triton else "flex"
    elif mask_type == "doc_causal" and use_triton:
        training_attention_backend = "triton"
        training_attn_class_backend = "varlen_bim_v1"
    else:
        training_attention_backend = "flex"
        training_attn_class_backend = "flex"

    # Build the block_mask_creator for weight reconstruction only.
    # The resulting TS2TSModel will build its own _creators dict.
    from model.graph_traversal.block_mask_creator import create_doc_causal_triton_mask
    if mask_type == "cross_doc_link":
        training_bmc = make_mask_creator_callable_from(
            CrossDocLinkMaskCreator(
                link_detector=link_detector,
                max_grants=model_cfg.get('max_grants', 64),
                backend=training_attn_class_backend,
            )
        )
    elif training_attn_class_backend == "varlen_bim_v1":
        training_bmc = make_mask_creator_callable_from(create_doc_causal_triton_mask)
    else:
        training_bmc = make_mask_creator_callable(mask_type)

    # Layout policies — training name from config, inference name may differ.
    training_layout_name = data_cfg.get("layout_policy") or "null"

    # inference_layout_policy key wins; stochastic policies map to deterministic equivalents.
    inference_layout_name = data_cfg.get(
        "inference_layout_policy", training_layout_name
    )
    inference_layout_name = _STOCHASTIC_TO_DETERMINISTIC.get(
        inference_layout_name, inference_layout_name
    )
    # Also resolve stochastic training name (safety net for legacy checkpoints)
    training_layout_name_resolved = _STOCHASTIC_TO_DETERMINISTIC.get(
        training_layout_name, training_layout_name
    )

    training_layout_policy  = _build_layout_policy(training_layout_name_resolved, enc)
    inference_layout_policy = _build_layout_policy(inference_layout_name, enc)

    # Reconstruct architecture (dropout=0 at inference).
    # Infer vocab_size from the checkpoint embedding weight so that both
    # old (50257) and new (50304) checkpoints load correctly.
    ckpt       = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model"]
    vocab_size = state_dict["embedding.weight"].shape[0]

    ve_layers = model_cfg.get("ve_layers") or []
    ve_layers = [int(x) for x in ve_layers]

    mtp_extra_weights = model_cfg.get("mtp_extra_weights") or []

    training_module = TS2TSTrainingModule.from_config(
        vocab_size=vocab_size,
        num_layers=model_cfg["num_layers"],
        model_dim=model_cfg["model_dim"],
        num_heads=model_cfg["num_heads"],
        max_seq_len=model_cfg["max_seq_len"],
        drop_path_rate=0.0,
        block_mask_creator=training_bmc,
        fp8=model_cfg.get("fp8", False),
        weight_tying=model_cfg.get("weight_tying", True),
        ignore_index=model_cfg.get("ignore_index", -100),
        dtype=torch.bfloat16,
        attention_backend=training_attn_class_backend,
        logit_softcap=model_cfg.get("logit_softcap"),
        mtp_extra_weights=mtp_extra_weights,
        ve_layers=ve_layers,
        shared_ve_bank=bool(model_cfg.get("shared_ve_bank", False)),
        use_bigram=bool(model_cfg.get("use_bigram", False)),
        bigram_vocab_size=int(model_cfg.get("bigram_vocab_size", 50304 * 5)),
    )

    # Load weights (ckpt/state_dict already loaded above)
    training_module.load_state_dict(state_dict)

    # Convert to inference model. to_inference_model handles the backbone
    # class-swap (triton → flex) and builds _creators internally.
    inference_model = training_module.to_inference_model(
        tokenizer=enc,
        mask_type=mask_type,
        link_detector=link_detector,
        training_backend=training_attention_backend,
        inference_backend=inference_attention_backend,
        training_layout_policy=training_layout_policy,
        inference_layout_policy=inference_layout_policy,
    )
    inference_model.to(torch.device(device), torch.bfloat16)

    # Set a stable per-project inductor cache so compiled kernels survive between runs.
    # flex_attention is compiled at module load in attention.py and flex_self_attention.py.
    if torch.cuda.is_available():
        _cache_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), ".torch_compile_cache"
        )
        os.makedirs(_cache_dir, exist_ok=True)
        os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", _cache_dir)

    return inference_model, hp


# ─────────────────────────────────────────────────────────────────────────────
# Corpus wrapper
# ─────────────────────────────────────────────────────────────────────────────

class PretokCorpus:
    """
    Thin wrapper around GraphIndex + PretokShardedBackend that satisfies the
    corpus protocol expected by the generation loop:
        has_document(raw_identifier) -> bool
        get_document(raw_identifier) -> Iterator[int]
    """

    def __init__(self, dataset_dir: str | Path):
        dataset_dir   = Path(dataset_dir)
        self._graph   = GraphIndex(dataset_dir)
        self._backend = PretokShardedBackend(self._graph)
        # Build a raw_identifier → normed_identifier map from the graph for O(1)
        # lookup in has_document / get_document. Falls back to normalize_wiki_title
        # for titles not in the graph (e.g. free-generated titles that missed the corpus).
        self._raw_to_normed: dict[str, str] = {
            node["raw_identifier"]: node["normed_identifier"]
            for node in self._graph.nodes.values()
            if "raw_identifier" in node and "normed_identifier" in node
        }

    def has_document(self, raw_identifier: str) -> bool:
        # NOTE: Python import detector emits relative paths (e.g. "Phaedra/Notebook.py")
        # but corpus identifiers are repo-qualified ("000alen/Phaedra:Phaedra/Notebook.py").
        # This means corpus hits will never fire for Python imports when using a multi-repo
        # dataset like thestack. Fix: either (a) build a single-repo corpus so identifiers
        # match, or (b) make the import detector emit repo-qualified identifiers when a repo
        # context is available.
        normed = self._raw_to_normed.get(raw_identifier) or normalize_wiki_title(raw_identifier)
        return normed in self._graph

    def get_document(self, raw_identifier: str):
        normed = self._raw_to_normed.get(raw_identifier) or normalize_wiki_title(raw_identifier)
        tokens = self._backend.get_tokens(normed)
        if tokens is None:
            return iter([])
        return iter(tokens.tolist())

    def close(self):
        self._backend.close()


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def _root_prefix_length(layout_policy, doc: GeneratedDocument) -> int:
    """Return the number of layout prefix tokens for the root document."""
    if layout_policy is None:
        return 0
    from data.layout import DocLayoutInfo
    info = DocLayoutInfo(
        raw_identifier=doc.raw_identifier,
        normed_identifier=doc.normed_identifier,
    )
    return layout_policy.prefix_length(info)


def compute_metrics(
    model,
    result: GenerationResult,
    prompt_token_len: int,
) -> Dict[str, Optional[Dict[str, float]]]:
    """
    Run a single no-grad forward pass over the complete generated sequence
    and return per-document metrics for generated documents only.

    Returns a dict mapping raw_identifier -> {"entropy_frac": float, "bits_per_tok": float,
    "num_generated_tokens": int}, or None for corpus docs.

    Metric positions:
    - Root doc: only the generated tokens (after the prompt)
    - Aux generated docs: all body tokens
    - Corpus docs: skipped (None)
    """
    vocab_size  = model.vocab_size
    log_V       = math.log(vocab_size)

    # Build packed sequence: aux docs (topological order) then root
    aux_docs  = result.auxiliary_documents
    root_doc  = result.root_document
    all_in_order = list(aux_docs) + [root_doc]   # matches generation-time order

    all_tokens_list: List[int] = []
    spans: List[Tuple[int, int]] = []   # (start, end) inclusive/exclusive for each doc
    for doc in all_in_order:
        start = len(all_tokens_list)
        all_tokens_list.extend(doc.tokens.tolist())
        spans.append((start, len(all_tokens_list)))

    if not all_tokens_list:
        return {}

    tokens_tensor = torch.tensor(
        all_tokens_list, dtype=torch.long, device=next(model.backbone.parameters()).device
    ).unsqueeze(0)

    # Rebuild DocSpans
    doc_spans = [
        DocSpan(
            doc_id=i,
            normed_identifier=doc.normed_identifier,
            raw_identifier=doc.raw_identifier,
            start=spans[i][0],
            end=spans[i][1],
            truncated=doc.truncated,
            outgoing_identifiers=[],
        )
        for i, doc in enumerate(all_in_order)
    ]

    with torch.no_grad():
        logits = model.forward_inference(tokens_tensor, doc_spans)   # [1, T, V]

    log_probs = F.log_softmax(logits[0].float(), dim=-1)   # [T, V]

    metrics: Dict[str, Optional[Dict[str, float]]] = {}

    for i, doc in enumerate(all_in_order):
        if doc.source == "corpus":
            metrics[doc.raw_identifier] = None
            continue

        start, end = spans[i]
        # For root: skip prefix + prompt tokens; for others: use all tokens.
        # doc.tokens is prefix + body + suffix.  prompt_token_len counts only
        # the prompt body tokens, so we must also skip the layout prefix.
        if doc.is_root:
            prefix_len = _root_prefix_length(model.inference_layout_policy, doc)
            body_start = start + prefix_len + prompt_token_len
        else:
            body_start = start

        # Logit at position t predicts token t+1, so p(tokens[t]) = log_probs[t-1]
        # We need t in [body_start, end), so we use logit positions [body_start-1, end-1)
        logit_start = max(body_start - 1, 0)
        logit_end   = end - 1
        token_start = logit_start + 1
        token_end   = end

        if logit_end <= logit_start or token_end <= token_start:
            metrics[doc.raw_identifier] = None
            continue

        lp_slice  = log_probs[logit_start:logit_end]          # [N, V]
        tok_slice = tokens_tensor[0, token_start:token_end]   # [N]

        # bits/tok: -log2(p(sampled_token))
        sampled_lp  = lp_slice[torch.arange(len(tok_slice)), tok_slice]   # [N]
        bits_per_tok = (-sampled_lp / math.log(2)).mean().item()

        # entropy_frac: H(p) / log(V), averaged over positions
        p        = lp_slice.exp()
        entropy  = -(p * lp_slice).sum(dim=-1)   # [N], nats
        ent_frac = (entropy / log_V).mean().item()

        metrics[doc.raw_identifier] = {
            "entropy_frac":       round(ent_frac, 3),
            "bits_per_tok":       round(bits_per_tok, 2),
            "num_generated_tokens": int(token_end - token_start),
        }

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Rendering
# ─────────────────────────────────────────────────────────────────────────────

_LINK_RE = re.compile(r'(\[[^\]\n]{1,80}\]\([^)\n]{1,100}\))')


def _highlight_links(text: str, use_color: bool) -> str:
    if not use_color:
        return text
    return _LINK_RE.sub(lambda m: _c(m.group(1), CYAN, True), text)


def _extract_links(text: str) -> List[str]:
    """Return list of full '[text](target)' strings found in text."""
    return _LINK_RE.findall(text)


def _render_doc(
    doc: GeneratedDocument,
    label: str,
    enc,
    metrics: Optional[Dict[str, float]],
    max_display_tokens: int,
    use_color: bool,
    width: int = 68,
) -> str:
    lines = []
    sep   = "─" * width
    thick = "═" * width

    lines.append(_c(thick if doc.is_root else sep, BOLD, use_color))
    lines.append(_c(f" {label}", BOLD, use_color))
    lines.append(_c(thick if doc.is_root else sep, BOLD, use_color))

    # Decoded text
    token_ids = doc.tokens.tolist() if doc.tokens is not None else []
    full_text = doc.text if doc.text is not None else (enc.decode(token_ids) if token_ids else "")

    # Truncate display
    if len(token_ids) > max_display_tokens:
        display_text = enc.decode(token_ids[:max_display_tokens])
        truncated_suffix = (
            f"\n{_c(f'[...truncated at {max_display_tokens} tokens'
                     f' — full doc: {len(token_ids)} tokens]', DIM, use_color)}"
        )
        # Link list from full doc text
        all_links = _extract_links(full_text)
        if all_links:
            link_str = " · ".join(all_links[:10])
            if len(all_links) > 10:
                link_str += f" · (+{len(all_links)-10} more)"
            truncated_suffix += f"\n {_c('Links:', BOLD, use_color)} {link_str}"
    else:
        display_text     = full_text
        truncated_suffix = ""

    lines.append(_highlight_links(display_text, use_color) + truncated_suffix)

    # Metrics footer
    if metrics is not None:
        n   = metrics["num_generated_tokens"]
        ef  = metrics["entropy_frac"]
        bpt = metrics["bits_per_tok"]
        lines.append(
            _c(f"\n generated tokens: {n} | entropy_frac: {ef:.2f} | bits/tok: {bpt:.1f}", DIM, use_color)
        )
    elif doc.source == "corpus":
        lines.append(_c("\n (corpus — no generation metrics)", DIM, use_color))
    else:
        lines.append(_c("\n (metrics unavailable)", DIM, use_color))

    return "\n".join(lines)


def render_result(
    result: GenerationResult,
    hp: dict,
    prompt: str,
    metrics: Dict,
    enc,
    max_display_tokens: int,
    use_color: bool,
) -> str:
    model_cfg  = hp["model"]
    data_cfg   = hp.get("data", {})
    ckpt_meta  = {}   # filled in by caller if available

    nl   = model_cfg.get("num_layers", "?")
    md   = model_cfg.get("model_dim", "?")
    mask = model_cfg.get("mask_type", "?")
    ds   = data_cfg.get("dataset_dir", "?").split("/")[-1]

    lines = []

    # Root
    root = result.root_document
    root_label = (
        f"ROOT  [{ds} | {mask} | {nl}L/{md}D]"
    )
    root_metrics = metrics.get(root.raw_identifier)
    lines.append(_render_doc(root, root_label, enc, root_metrics, max_display_tokens, use_color))

    # Aux docs
    aux_docs = result.auxiliary_documents
    for idx, doc in enumerate(aux_docs):
        source_tag = doc.source
        parent     = doc.parent_raw_identifier or "ROOT"
        parent_tag = "ROOT" if (parent == "" or parent is None) else f'"{parent}"'
        label      = (
            f'AUX {idx+1}/{len(aux_docs)}  "{doc.raw_identifier}"'
            f'  [{source_tag} | depth {doc.depth} | parent: {parent_tag}]'
        )
        doc_metrics = metrics.get(doc.raw_identifier)
        lines.append(_render_doc(doc, label, enc, doc_metrics, max_display_tokens, use_color))

    # Summary
    width = 68
    n_gen    = len(result.get_generated_documents())
    n_corpus = len(result.get_corpus_documents())
    n_aux    = len(aux_docs)
    cfg_dict = result.generation_config
    trace    = result.trace

    lines.append(_c("═" * width, BOLD, use_color))
    lines.append(_c(" SUMMARY", BOLD, use_color))
    doc_breakdown = f"{result.get_document_count()} docs ({n_gen} generated, {n_corpus} corpus)"
    lines.append(f"   prompt:   \"{prompt}\"")
    lines.append(f"   docs:     {doc_breakdown}")
    lines.append(f"   depth:    {cfg_dict.get('max_link_depth', '?')}")
    if trace is not None:
        lines.append(
            f"   trace:    {trace.total_forward_passes} fwd passes, "
            f"{trace.total_tokens_generated} tokens, "
            f"{trace.links_detected} links detected ({trace.links_resolved} resolved), "
            f"{trace.corpus_fetches} corpus fetches, "
            f"{trace.docs_generated} docs generated, "
            f"{trace.docs_evicted} evicted"
        )
    lines.append(_c("═" * width, BOLD, use_color))

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate text from a trained TS2TS checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Path to best_model.pt checkpoint.")
    parser.add_argument("--prompt", required=True,
                        help="Text prompt to generate from.")
    parser.add_argument("--root-identifier", default="",
                        help="Filename / identifier prefix for the root document "
                             "(e.g. 'attention.py'). Used as the '# <id>' header "
                             "that the identifier_prefix_bos_eos layout policy adds.")
    parser.add_argument("--dataset", default=None,
                        help="Path to pretokenized dataset dir (corpus for link resolution).")
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--max-link-depth", type=int, default=2)
    parser.add_argument(
        "--link-retrieval-mode",
        choices=["corpus_only", "generate_only", "corpus_then_generate",
                 "link_but_skip", "full_skip"],
        default=None,
        help="How to resolve links during generation. Default: 'corpus_only' when "
             "--dataset is provided, 'corpus_then_generate' otherwise.",
    )
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--repetition-penalty", type=float, default=1.3,
                        help="Penalise tokens already generated in the current doc. "
                             "1.0 = disabled; 1.3 is a reasonable default.")
    parser.add_argument("--max-display-tokens", type=int, default=200,
                        help="Truncate displayed text per doc to this many tokens.")
    parser.add_argument("--no-color", action="store_true",
                        help="Disable ANSI color output.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    use_color = not args.no_color and sys.stdout.isatty()

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"Loading checkpoint: {args.checkpoint} ...", file=sys.stderr)
    model, hp = load_inference_model(args.checkpoint, device=args.device)
    enc        = model.tokenizer

    # ── Load corpus ───────────────────────────────────────────────────────────
    corpus = None
    if args.dataset:
        print(f"Loading corpus: {args.dataset} ...", file=sys.stderr)
        corpus = PretokCorpus(args.dataset)

    # ── Generation config ─────────────────────────────────────────────────────
    # Default mode: corpus_only when a dataset is provided (no generation fallback),
    # corpus_then_generate otherwise (no corpus → generation is the only option).
    link_retrieval_mode = args.link_retrieval_mode or (
        "corpus_only" if corpus is not None else "corpus_then_generate"
    )
    # max_tokens_per_document must be >= max_new_tokens (GenerationConfig validates this).
    # Give a small 256-token headroom above max_new_tokens for layout prefix/suffix,
    # but cap at half the model's context window.
    max_tokens_per_document = min(args.max_new_tokens + 256, hp["model"]["max_seq_len"] // 2)
    max_tokens_per_document = max(max_tokens_per_document, args.max_new_tokens)
    config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        max_link_depth=args.max_link_depth,
        link_retrieval_mode=link_retrieval_mode,
        max_context_length=hp["model"]["max_seq_len"],
        max_tokens_per_document=max_tokens_per_document,
        device=args.device,
    )

    # ── Generate ──────────────────────────────────────────────────────────────
    print("Generating ...", file=sys.stderr)
    result = model.generate(args.prompt, corpus=corpus, config=config,
                            root_identifier=args.root_identifier)

    # ── Metrics ───────────────────────────────────────────────────────────────
    prompt_token_len = len(enc.encode(args.prompt))
    metrics = compute_metrics(model, result, prompt_token_len)

    # ── Render ────────────────────────────────────────────────────────────────
    output = render_result(
        result=result,
        hp=hp,
        prompt=args.prompt,
        metrics=metrics,
        enc=enc,
        max_display_tokens=args.max_display_tokens,
        use_color=use_color,
    )
    print(output)

    if corpus is not None:
        corpus.close()


if __name__ == "__main__":
    main()
