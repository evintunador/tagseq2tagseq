"""
eval/link_annotator.py — prompt link injection for external benchmarks.

MarkdownPromptAnnotator finds the most likely location in a prompt for a
Wikipedia-style [display](Title) link, injects the link syntax by splicing
tokens, autoregressively generates the target title, and fetches or generates
an aux document — all using the model's own next-token probabilities to guide
placement.

The resulting AnnotatedPrompt feeds directly into score_completion_with_context_docs
so the cross-doc attention grant fires on the injected link, enabling standard
NLP benchmarks (HellaSwag, BoolQ, etc.) to be evaluated as cross-doc benchmarks
without any structural reformatting of the original examples.

Algorithm overview
------------------
1. Forward pass 1 (doc_causal, original tokens)
   → find position i with highest P('[' or ' [')
   → record link_opener_prob for threshold calibration

2. Insert the higher-P opener token at position i.

3. Forward pass 2 (doc_causal, tokens with '[' spliced in)
   → find position j > i within max_display_tokens window with
     highest P('](') × decay_factor^(j-i)

4. Insert '](' at position j.
   display text = decode(tokens[i+1 : j]) — from existing tokens.

5. Autoregressive title generation from position j+1.
   Stop when ')' or EOS sampled, or max_title_tokens reached.
   target_str = decode(generated title tokens).

6. Aux doc acquisition per link_retrieval_mode.

Entry points
------------
  MarkdownPromptAnnotator.scan_prob(model, context_tokens, device) -> float
      Phase-1-only scan for threshold calibration.

  MarkdownPromptAnnotator.annotate(model, context_tokens, device) -> AnnotatedPrompt
      Full pipeline.

  PromptAnnotator (Protocol)
      Minimal interface for future annotators (LaTeX citations, etc.).

  render_annotated_example(annotated, tokenizer, choices, completion, use_color) -> str
      Pretty-print an annotated benchmark item showing the original context with
      the injected link highlighted, plus the aux doc (if any) and answer choices.
      Suitable for visual verification and paper appendices.
"""

import logging
import math
import re as _re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Protocol, Set, Tuple, runtime_checkable

import torch
import torch.nn.functional as F

from data.collate import DocSpan
from eval.title_index import TitleIndex
from model.generation_config import GenerationConfig
from model.sampling import sample_token

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Link-retrieval vocabulary (shared with model/generation_config.py)
# ---------------------------------------------------------------------------
# The annotator speaks the same five-value vocabulary as
# GenerationConfig.link_retrieval_mode so a single condition name selects the
# same behaviour in generation and in eval:
#
#   full_skip            — no link at all: skip injection entirely (no-link baseline)
#   link_but_skip        — inject link syntax, but acquire no aux doc
#   corpus_only          — inject link, fetch aux from corpus, no-op on miss
#   generate_only        — inject link, always generate the aux doc
#   corpus_then_generate — inject link, corpus first, generate on miss
VALID_LINK_RETRIEVAL_MODES = frozenset({
    "full_skip", "link_but_skip", "corpus_only", "generate_only", "corpus_then_generate",
})


def validate_link_retrieval_mode(mode: str) -> str:
    """Return mode if it is a valid link-retrieval mode, else raise ValueError."""
    if mode not in VALID_LINK_RETRIEVAL_MODES:
        raise ValueError(
            f"link_retrieval_mode must be one of {sorted(VALID_LINK_RETRIEVAL_MODES)}, "
            f"got {mode!r}"
        )
    return mode


# ---------------------------------------------------------------------------
# AnnotatedPrompt
# ---------------------------------------------------------------------------

@dataclass
class AnnotatedPrompt:
    """Result of annotating one prompt with an injected link."""
    context_tokens: List[int]
    aux_token_lists: List[List[int]]
    aux_raw_identifiers: List[str]
    target_str: str
    link_opener_pos: int
    link_mid_pos: int
    link_opener_prob: float
    link_fired: bool
    # Populated only when TrieTitleIndex is used with return_candidates=True.
    # List of (raw_identifier, length_normalized_score) for all completed beam
    # paths, sorted best-first. The first entry is the selected title.
    beam_candidates: Optional[List[Tuple[str, float]]] = None


# ---------------------------------------------------------------------------
# PromptAnnotator protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class PromptAnnotator(Protocol):
    """Minimal interface shared by all annotator variants."""

    def scan_prob(
        self,
        model,
        context_tokens: List[int],
        device: str = "cuda",
    ) -> float:
        """Return link_opener_prob without doing title generation or aux fetch."""
        ...

    def annotate(
        self,
        model,
        context_tokens: List[int],
        device: str = "cuda",
    ) -> AnnotatedPrompt:
        """Full annotation pipeline."""
        ...


# ---------------------------------------------------------------------------
# TrieTitleIndex
# ---------------------------------------------------------------------------

@dataclass
class _TrieNode:
    children: Dict[int, "_TrieNode"] = field(default_factory=dict)
    raw_identifier: Optional[str] = None


class TrieTitleIndex:
    """
    Token-level prefix trie over corpus titles for constrained title generation.

    At construction time tokenizes every raw_identifier and inserts the resulting
    token-id sequence as a trie path.  During generation, beam search keeps the
    top-beam_width active paths alive simultaneously, scores each completed path
    by its cumulative joint log-prob under the unconstrained distribution, and
    returns the highest-scoring completed title.

    beam_width=1 is greedy single-path traversal (original behaviour).
    beam_width>1 allows shorter high-first-token titles (e.g. '25') to be beaten
    by longer titles whose total log-prob is higher (e.g. 'New Hampshire').

    Implements generate_title(model, prefix_tokens, device) which
    MarkdownPromptAnnotator._generate_title delegates to when present.

    lookup() delegates to fallback_index when provided (for post-hoc recovery
    when generate_title returns None).

    Args:
        raw_identifiers: Corpus title strings.
        tokenizer: Must expose .encode(str) -> List[int] and .decode(List[int]) -> str.
        beam_width: Number of active paths to maintain during generation. Default 1.
        length_penalty: Exponent alpha in the Wu et al. length normalization:
            score = joint_log_prob / (n_tokens ** alpha). 0.0 = no normalization
            (raw joint log-prob, shorter titles win); 1.0 = full per-token mean
            log-prob; 0.6 = recommended middle ground. Default 0.0.
        min_joint_logprob: If set, prune any path whose cumulative log P drops
            below this value.  None disables the threshold.
        fallback_index: Optional TitleIndex to delegate lookup() calls to.

    Notes:
        Collision policy: if two titles tokenize to identical token sequences,
            the first-inserted raw_identifier is stored; subsequent collisions
            are silently ignored.
        Sampling parameters (temperature, top_k, top_p) passed to generate_title
            are forwarded to the free-generation fallback path only. Beam search
            itself always selects children deterministically by descending
            unconstrained probability.
    """

    def __init__(
        self,
        raw_identifiers: Iterable[str],
        tokenizer,
        beam_width: int = 1,
        length_penalty: float = 0.0,
        min_joint_logprob: Optional[float] = None,
        fallback_index: Optional[TitleIndex] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.beam_width = beam_width
        self.length_penalty = length_penalty
        self.min_joint_logprob = min_joint_logprob
        self.fallback_index = fallback_index

        # Build trie from raw token sequences (no normalization).
        self._root = _TrieNode()
        for raw in raw_identifiers:
            try:
                token_ids: List[int] = list(tokenizer.encode(raw))
            except Exception:
                continue
            if not token_ids:
                continue
            node = self._root
            for tid in token_ids:
                if tid not in node.children:
                    node.children[tid] = _TrieNode()
                node = node.children[tid]
            if node.raw_identifier is None:  # first-inserted wins on collision
                node.raw_identifier = raw

        # Token ID for ')' — obtained once at construction.
        try:
            ids = tokenizer.encode(")")
            self._close_paren_id: Optional[int] = ids[0] if ids else None
        except Exception:
            self._close_paren_id = None

    # ------------------------------------------------------------------
    # TitleIndex protocol
    # ------------------------------------------------------------------

    def lookup(self, generated_str: str) -> Optional[str]:
        """Delegate to fallback_index, or None if none set."""
        if self.fallback_index is not None:
            return self.fallback_index.lookup(generated_str)
        return None

    # ------------------------------------------------------------------
    # Constrained generation
    # ------------------------------------------------------------------

    def generate_title(
        self,
        model,
        prefix_tokens: List[int],
        device: str,
        max_title_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        return_candidates: bool = False,
    ) -> Optional[Tuple]:
        """Trie-constrained title generation with beam search.

        Maintains up to beam_width active paths simultaneously.  Each path is
        scored by its cumulative sum of log P(token) under the *unconstrained*
        distribution, so a longer title can beat a short one that happened to
        have a high-probability first token.  All completed paths are collected
        and the highest-scoring one is returned.

        Returns (raw_identifier, title_token_ids) on success, or None when no
        path completes within max_title_tokens or all paths are pruned by
        min_joint_logprob.  None causes the caller to fall back to free generation.
        """
        def _fwd(tokens: List[int]) -> torch.Tensor:
            tok_t = torch.tensor(tokens, dtype=torch.long, device=device)
            span = DocSpan(
                doc_id=0, normed_identifier="", raw_identifier="",
                start=0, end=tok_t.shape[0], truncated=False,
                outgoing_identifiers=[],
            )
            logits = model.forward_inference(
                tok_t.unsqueeze(0).to(device), [span], mask_type="doc_causal"
            )
            return logits[0]  # [T, V]

        # Clamp max_steps to both caller budget and model's positional limit.
        _raw = getattr(getattr(model, "backbone", None), "max_seq_len", None)
        model_max_seq = _raw if isinstance(_raw, int) else None
        room = (model_max_seq - len(prefix_tokens) - 1) if model_max_seq else max_title_tokens
        max_steps = min(max_title_tokens, max(room, 0))

        # Each beam entry: (joint_logprob, title_tokens, trie_node)
        beam: List[Tuple[float, List[int], _TrieNode]] = [(0.0, [], self._root)]
        # Completed paths: (joint_logprob, title_tokens, raw_identifier)
        candidates: List[Tuple[float, List[int], str]] = []

        for _ in range(max_steps):
            if not beam:
                break
            next_beam: List[Tuple[float, List[int], _TrieNode]] = []

            for logprob, tokens, node in beam:
                logits = _fwd(list(prefix_tokens) + tokens)[-1]  # [V]
                probs = F.softmax(logits.float(), dim=-1)

                # Interior-leaf: compare P(")") vs best valid child.
                if node.raw_identifier is not None:
                    p_close = (
                        probs[self._close_paren_id].item()
                        if self._close_paren_id is not None else 0.0
                    )
                    p_best_child = (
                        max(probs[tid].item() for tid in node.children)
                        if node.children else 0.0
                    )
                    if p_close >= p_best_child or not node.children:
                        candidates.append((logprob, tokens, node.raw_identifier))
                        continue  # don't expand further

                if not node.children:
                    if node.raw_identifier:
                        candidates.append((logprob, tokens, node.raw_identifier))
                    continue

                # Expand: score every valid child by unconstrained P, keep top beam_width.
                child_scores = sorted(
                    ((probs[tid].item(), tid) for tid in node.children),
                    reverse=True,
                )
                for p, tid in child_scores[:self.beam_width]:
                    new_logprob = logprob + math.log(p + 1e-12)
                    if self.min_joint_logprob is None or new_logprob >= self.min_joint_logprob:
                        next_beam.append((new_logprob, tokens + [tid], node.children[tid]))

            # Global prune to beam_width best active paths.
            next_beam.sort(key=lambda x: x[0], reverse=True)
            beam = next_beam[:self.beam_width]

        # Collect any active paths that landed on a leaf node.
        for logprob, tokens, node in beam:
            if node.raw_identifier:
                candidates.append((logprob, tokens, node.raw_identifier))

        if not candidates:
            return None

        def _score(logprob: float, tokens: List[int]) -> float:
            n = len(tokens)
            if self.length_penalty == 0.0 or n == 0:
                return logprob
            return logprob / (n ** self.length_penalty)

        scored = sorted(candidates, key=lambda x: _score(x[0], x[1]), reverse=True)
        best_logprob, best_tokens, best_raw = scored[0]
        if return_candidates:
            sorted_candidates = [(raw, _score(lp, toks)) for lp, toks, raw in scored]
            return best_raw, best_tokens, sorted_candidates
        return best_raw, best_tokens


# ---------------------------------------------------------------------------
# _AnnotatorBase — shared helpers for all PromptAnnotator implementations
# ---------------------------------------------------------------------------

class _AnnotatorBase:
    """Shared helpers for MarkdownPromptAnnotator and ArxivPromptAnnotator.

    Subclasses supply the dataset-specific link syntax via three hooks:
      * ``_opener_probs(model, tokens, device) -> [T]`` — P(start-of-link-opener)
        at each position (markdown ' [' vs LaTeX '\\').
      * ``_title_stop_regex`` — compiled regex whose first match in the decoded
        title marks the end of the title (markdown ')' vs LaTeX '}').
      * ``max_title_tokens`` — generation budget for the title.
    The shared ``scan_prob`` and ``_generate_title`` (free, autoregressive) are
    defined here; subclasses override only where behaviour genuinely diverges
    (e.g. MarkdownPromptAnnotator adds trie-constrained title generation).
    """

    # Subclasses must set these in __init__
    corpus = None
    title_index = None
    link_retrieval_mode: str = "full_skip"
    generation_config: "GenerationConfig"
    layout_policy = None
    eos_token_id: int = 50256
    max_title_tokens: int = 50

    # Subclasses must set this to the compiled stop regex for title generation.
    _title_stop_regex = None

    def _run_fwd(self, model, tok_tensor: torch.Tensor, device: str) -> torch.Tensor:
        """Single doc_causal forward pass. Returns logits [T, V]."""
        span = DocSpan(
            doc_id=0,
            normed_identifier="",
            raw_identifier="",
            start=0,
            end=tok_tensor.shape[0],
            truncated=False,
            outgoing_identifiers=[],
        )
        logits = model.forward_inference(
            tok_tensor.unsqueeze(0).to(device),
            [span],
            mask_type='doc_causal',
        )
        return logits[0]   # [T, V]

    # --- link-opener scan (shared) ---

    def _opener_probs(self, model, context_tokens: List[int], device: str) -> torch.Tensor:
        """Return P(link opener) per position per opener token, shape [T, n_openers].

        The trailing opener dimension lets annotate() pick both the best position
        and the best opener token at that position from a single forward pass —
        do NOT collapse it here. Subclass hook.
        """
        raise NotImplementedError

    def scan_prob(self, model, context_tokens: List[int], device: str = "cuda") -> float:
        """Phase-1 scan: return max P(link opener) across all positions.

        Cheap — one forward pass, no generation. Called for all examples during
        threshold calibration before any full annotation.
        """
        if not context_tokens:
            return 0.0
        probs = self._opener_probs(model, context_tokens, device)   # [T, n_openers]
        return float(probs.max().item())

    # --- title generation (shared free-generation core) ---

    def _generate_title_free(
        self,
        model,
        prefix_tokens: List[int],
        device: str,
    ) -> Tuple[str, List[int]]:
        """Autoregressively generate a link-target title (free, unconstrained).

        Stops when EOS is sampled, ``_title_stop_regex`` matches the decoded
        accumulation (handles BPE merges that absorb the closing delimiter), or
        ``max_title_tokens`` is exhausted. Returns (target_str, title_token_ids)
        where title_token_ids is the clean re-encoding of target_str.
        """
        tokenizer = getattr(model, "tokenizer", None)
        current_tokens = list(prefix_tokens)
        title_tokens: List[int] = []
        cfg = self.generation_config
        stop_re = self._title_stop_regex

        for _ in range(self.max_title_tokens):
            tok_tensor = torch.tensor(current_tokens, dtype=torch.long, device=device)
            logits = self._run_fwd(model, tok_tensor, device)
            next_token = sample_token(
                logits[-1], temperature=cfg.temperature, top_k=cfg.top_k, top_p=cfg.top_p,
                allowed_vocab_size=cfg.allowed_vocab_size,
            )
            if next_token == self.eos_token_id:
                break
            title_tokens.append(next_token)
            current_tokens.append(next_token)
            if tokenizer is not None:
                try:
                    decoded_so_far = tokenizer.decode(title_tokens)
                    if "\n" in decoded_so_far or (stop_re is not None and stop_re.search(decoded_so_far)):
                        break
                except Exception:
                    pass

        if title_tokens:
            try:
                raw = tokenizer.decode(title_tokens) if tokenizer is not None else (
                    "".join(chr(t % 256) for t in title_tokens)
                )
            except Exception:
                raw = ""
        else:
            raw = ""

        # Trim at the first stop delimiter or newline, whichever comes first.
        trim_pat = r'\n' if stop_re is None else (stop_re.pattern + r'|\n')
        m = _re.search(trim_pat, raw)
        if m:
            raw = raw[:m.start()]
        target_str = raw.strip()

        # Re-encode so title_tokens exactly matches what goes into context_tokens.
        if target_str and tokenizer is not None:
            try:
                title_tokens = list(tokenizer.encode(target_str))
            except Exception:
                pass
        return target_str, title_tokens

    def _fetch_aux(
        self,
        model,
        target_str: str,
        device: str,
    ) -> Tuple[List[List[int]], List[str], bool]:
        """Fetch or generate the aux document according to link_retrieval_mode.

        Returns (aux_token_lists, aux_raw_identifiers, link_fired).
        """
        if not target_str or self.link_retrieval_mode in ("link_but_skip", "full_skip"):
            return [], [], False

        if self.link_retrieval_mode in ("corpus_only", "corpus_then_generate"):
            if self.corpus is not None:
                if self.title_index is not None:
                    resolved = self.title_index.lookup(target_str)
                    if resolved is None:
                        resolved = target_str if self.corpus.has_document(target_str) else None
                else:
                    resolved = target_str if self.corpus.has_document(target_str) else None
                if resolved is not None and self.corpus.has_document(resolved):
                    aux_tokens = list(self.corpus.get_document(resolved))
                    if aux_tokens:
                        logger.debug(
                            "Annotator corpus hit: %r -> %r (%d tokens)",
                            target_str, resolved, len(aux_tokens),
                        )
                        return [aux_tokens], [resolved], True
            if self.link_retrieval_mode == "corpus_only":
                logger.debug("Annotator corpus miss (corpus_only): %r", target_str)
                return [], [], False

        if self.link_retrieval_mode in ("generate_only", "corpus_then_generate"):
            aux_tokens = self._generate_aux_doc(model, target_str, device)
            if aux_tokens:
                logger.debug("Annotator generated aux doc: %r (%d tokens)", target_str, len(aux_tokens))
                return [aux_tokens], [target_str], True

        return [], [], False

    def _generate_aux_doc(
        self,
        model,
        target_str: str,
        device: str,
    ) -> List[int]:
        """Generate an aux document for target_str using the full generation loop."""
        from model.generation_loop import run_generation

        cfg = self.generation_config
        gen_cfg = GenerationConfig(
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_k=cfg.top_k,
            top_p=cfg.top_p,
            max_tokens_per_document=cfg.max_tokens_per_document,
            max_context_length=cfg.max_context_length,
            max_auxiliary_documents=0,
            max_link_depth=0,
            link_retrieval_mode="full_skip",
            eviction_policy="stop_new",
            process_prompt_links=False,
            repetition_penalty=cfg.repetition_penalty,
            eos_token_id=self.eos_token_id,
            record_trace=False,
            device=device,
        )

        try:
            tokenizer = getattr(model, "tokenizer", None)
            decode_fn = tokenizer.decode if tokenizer is not None else None

            _lp = self.layout_policy
            _has_prefix = False
            if _lp is not None and target_str:
                try:
                    from data.layout import DocLayoutInfo
                    _info = DocLayoutInfo(raw_identifier=target_str, normed_identifier=target_str)
                    _has_prefix = len(_lp.prefix_tokens(_info)) > 0
                except Exception:
                    pass
            if _has_prefix:
                prompt_tokens: List[int] = []
            else:
                prompt_tokens = list(tokenizer.encode(target_str)) if (tokenizer and target_str) else []

            result = run_generation(
                model=model,
                prompt_tokens=prompt_tokens,
                corpus=None,
                config=gen_cfg,
                link_detector=None,
                tokenizer_decode=decode_fn,
                layout_policy=self.layout_policy,
                root_identifier=target_str,
            )
            doc = result.root_document
            if doc.tokens is not None:
                return doc.tokens.tolist()
        except Exception as exc:
            logger.warning("Annotator aux doc generation failed for %r: %s", target_str, exc)

        return []


# ---------------------------------------------------------------------------
# MarkdownPromptAnnotator
# ---------------------------------------------------------------------------

class MarkdownPromptAnnotator(_AnnotatorBase):
    """
    Injects a Wikipedia-style [display](Title) link into a prompt.

    Uses the model's own next-token probabilities to select the most natural
    position for the link opener and mid-point, autoregressively generates the
    target title, then fetches or generates the auxiliary document.

    Args:
        corpus: Optional corpus object with has_document/get_document (e.g.
            PretokCorpus from model.document_corpus). Required for corpus_only and
            corpus_then_generate modes.
        link_retrieval_mode: How to handle the link (see VALID_LINK_RETRIEVAL_MODES).
            "full_skip"           — no link at all: skip injection entirely.
            "link_but_skip"       — inject link syntax only, no aux doc.
            "corpus_only"         — corpus lookup; no aux on miss.
            "generate_only"       — always generate aux doc autoregressively.
            "corpus_then_generate"— corpus first, generate on miss.
        generation_config: Sampling settings (temperature, top_k, top_p) used
            for title generation and (when mode includes generate) aux doc
            generation. If None, uses GenerationConfig defaults.
        layout_policy: Layout policy passed to run_generation when generating
            aux docs. None means no prefix/suffix decoration.
        max_display_tokens: Hard window (in tokens) after the '[' position
            within which to search for ']('. Default 100.
        decay_factor: Multiplicative distance penalty applied to P('](') at
            each position beyond the opener. P_effective(j) = P(j) *
            decay_factor^(j - link_opener_pos). Default 0.95.
        max_title_tokens: Maximum tokens to generate autoregressively for
            the target title. Default 50.
        link_opener_token_ids: Token IDs considered as valid link openers.
            Default (58, 685) — bare '[' and ' [' (space-bracket) in GPT-2,
            matching MarkdownLinkDetector.link_start_token_ids. A scan of 125k+
            real simplewiki links found ' [' (685) opens ~92% and bare '[' (58)
            ~8% (the latter after newlines, ']', list markers, and doc starts),
            so together they cover ~97% of real links. Both are clean single
            tokens, so annotate()'s [T, n_openers] scan-and-inject handles them
            unchanged. The remaining ~3% open with merged-punctuation tokens
            (' ([' 29565, ' "[' 12878, ...) that are deliberately NOT included:
            they'd add <2% coverage and, since annotate() *inserts* best_opener
            as a literal token, injecting e.g. ' ([' would splice malformed
            markdown (' ([display](Title)') into the prompt.
        link_mid_token_id: Token ID for the '](' bigram. Default 16151.
        eos_token_id: EOS token ID. Default 50256 (GPT-2).

    """

    def __init__(
        self,
        corpus=None,
        title_index: Optional[TitleIndex] = None,
        link_retrieval_mode: str = "full_skip",
        generation_config: Optional[GenerationConfig] = None,
        layout_policy=None,
        max_display_tokens: int = 100,
        decay_factor: float = 0.95,
        max_title_tokens: int = 50,
        link_opener_token_ids: Tuple[int, ...] = (58, 685),
        link_mid_token_id: int = 16151,
        eos_token_id: int = 50256,
        show_beam_candidates: bool = False,
    ):
        link_retrieval_mode = validate_link_retrieval_mode(link_retrieval_mode)
        self.corpus = corpus
        self.title_index = title_index
        self.link_retrieval_mode = link_retrieval_mode
        self.generation_config = generation_config or GenerationConfig(repetition_penalty=1.3)
        self.layout_policy = layout_policy
        self.max_display_tokens = max_display_tokens
        self.decay_factor = decay_factor
        self.max_title_tokens = max_title_tokens
        self.link_opener_token_ids = set(link_opener_token_ids)
        self.link_mid_token_id = link_mid_token_id
        self.eos_token_id = eos_token_id
        self.show_beam_candidates = show_beam_candidates

    # Title ends at a link-closing ')' — one NOT immediately followed by a word
    # character, so a mid-word paren like ")iani" or "U2)" doesn't trigger.
    _title_stop_regex = _re.compile(r'\)(?!\w)')

    # scan_prob is inherited from _AnnotatorBase (uses _opener_probs below).

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def annotate(
        self,
        model,
        context_tokens: List[int],
        device: str = "cuda",
    ) -> AnnotatedPrompt:
        """Full annotation pipeline.

        Returns an AnnotatedPrompt whose context_tokens has the link syntax
        spliced in. aux_token_lists / aux_raw_identifiers are populated iff
        link_fired is True.

        In full_skip mode (the no-link baseline) no link is injected at all: the
        original context is returned unchanged with link_fired=False.
        """
        if not context_tokens or self.link_retrieval_mode == "full_skip":
            return AnnotatedPrompt(
                context_tokens=list(context_tokens),
                aux_token_lists=[],
                aux_raw_identifiers=[],
                target_str="",
                link_opener_pos=0,
                link_mid_pos=0,
                link_opener_prob=0.0,
                link_fired=False,
            )

        # ── Step 1: forward pass 1, pick link opener position + token ──────
        # _opener_probs returns [T, n_openers] — the per-position probability of
        # each configured opener token. From this single forward we get both the
        # best position (argmax over the per-position maxima) and the best opener
        # token at that position (argmax over openers). This used to re-run
        # _run_fwd on the *identical* context_tokens purely to recover the second
        # value, which _opener_probs had already computed and then collapsed away
        # with a max(dim=-1). That second forward was pure waste and is gone.
        opener_probs = self._opener_probs(model, context_tokens, device)  # [T, n_openers]
        per_pos_max = opener_probs.max(dim=-1).values                      # [T]
        link_opener_prob = float(per_pos_max.max().item())
        link_opener_pos = int(per_pos_max.argmax().item())
        opener_ids = list(self.link_opener_token_ids)
        best_opener = opener_ids[int(opener_probs[link_opener_pos].argmax().item())]

        # ── Step 2: insert the chosen opener, forward pass 2, pick '](' position ──
        # best_opener is whichever configured opener token scored highest at
        # link_opener_pos — ' [' (685, the common case) or bare '[' (58).
        toks_with_opener = (
            context_tokens[:link_opener_pos]
            + [best_opener]
            + context_tokens[link_opener_pos:]
        )
        opener_tensor = torch.tensor(toks_with_opener, dtype=torch.long, device=device)
        fwd2_logits = self._run_fwd(model, opener_tensor, device)   # [T+1, V]

        # Search window for '](' — at least one display token required between
        # '[' and '](', so search starts at link_opener_pos + 2.
        search_start = link_opener_pos + 2
        search_end = min(len(toks_with_opener), link_opener_pos + 1 + self.max_display_tokens)
        if search_start >= search_end:
            link_mid_pos = link_opener_pos + 2
        else:
            best_j = search_start
            best_score = -math.inf
            for j in range(search_start, search_end):
                raw_prob = F.softmax(fwd2_logits[j].float(), dim=-1)[self.link_mid_token_id].item()
                distance = j - link_opener_pos
                score = raw_prob * (self.decay_factor ** distance)
                if score > best_score:
                    best_score = score
                    best_j = j
            link_mid_pos = best_j

        # ── Step 3: build display tokens with string-level space stripping ─
        # The tokens between opener and mid come from the original sequence
        # and may have a leading space from BPE tokenization. Work at string
        # level: decode → strip leading whitespace → re-tokenize. This keeps
        # the display text clean (e.g. "bmc software" not " bmc software").
        raw_display_toks = toks_with_opener[link_opener_pos + 1:link_mid_pos]
        tokenizer = getattr(model, "tokenizer", None)
        if tokenizer is not None and raw_display_toks:
            try:
                raw_display_str = tokenizer.decode(raw_display_toks)
                clean_display_str = raw_display_str.lstrip()
                display_tokens = list(tokenizer.encode(clean_display_str))
            except Exception:
                display_tokens = raw_display_toks
        else:
            display_tokens = raw_display_toks

        # ── Step 4: build prefix for title generation (truncated at '](') ─
        # Title generation sees: context_prefix + ' [' + display_tokens + '](
        # NOT the original suffix after link_mid_pos.
        prefix_before_opener = context_tokens[:link_opener_pos]
        title_prefix = (
            prefix_before_opener
            + [best_opener]
            + display_tokens
            + [self.link_mid_token_id]
        )

        # ── Step 5: autoregressive title generation ──────────────────────
        _title_result = self._generate_title(model, title_prefix, device)
        if len(_title_result) == 3:
            target_str, title_tokens, beam_candidates = _title_result
        else:
            target_str, title_tokens = _title_result
            beam_candidates = None

        # ── Step 6: build final context_tokens ───────────────────────────
        # Structure: prefix + ' [' + display + '](' + title + ')' + original_suffix
        # The display tokens span link_mid_pos - link_opener_pos - 1 positions
        # in the original sequence starting at link_opener_pos. The suffix must
        # skip those so they aren't duplicated after the closing ')'.
        n_display_original = link_mid_pos - link_opener_pos - 1
        original_suffix = context_tokens[link_opener_pos + n_display_original:]
        close_paren_id = self._get_close_paren_id(model)
        final_context = (
            title_prefix
            + title_tokens
            + ([close_paren_id] if close_paren_id is not None else [])
            + original_suffix
        )

        # Positions of injected tokens in the final sequence (for visualization).
        opener_pos_final = link_opener_pos                              # ' [' token
        mid_pos_final    = link_opener_pos + 1 + len(display_tokens)   # '](' token

        # ── Step 7: aux doc acquisition ─────────────────────────────────
        aux_token_lists, aux_raw_identifiers, link_fired = self._fetch_aux(
            model, target_str, device
        )

        return AnnotatedPrompt(
            context_tokens=final_context,
            aux_token_lists=aux_token_lists,
            aux_raw_identifiers=aux_raw_identifiers,
            target_str=target_str,
            link_opener_pos=opener_pos_final,
            link_mid_pos=mid_pos_final,
            link_opener_prob=link_opener_prob,
            link_fired=link_fired,
            beam_candidates=beam_candidates,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _opener_probs(
        self,
        model,
        context_tokens: List[int],
        device: str,
    ) -> torch.Tensor:
        """Run forward pass 1; return P(opener) per position per opener. Shape [T, n_openers].

        The opener dimension is preserved (not reduced with max) so annotate() can
        recover the best opener token at the chosen position without a second
        forward pass. scan_prob reduces it to a scalar itself.
        """
        tok_tensor = torch.tensor(context_tokens, dtype=torch.long, device=device)
        logits = self._run_fwd(model, tok_tensor, device)    # [T, V]
        probs = F.softmax(logits.float(), dim=-1)            # [T, V]
        opener_ids = list(self.link_opener_token_ids)
        return probs[:, opener_ids]                           # [T, n_openers]

    def _generate_title(
        self,
        model,
        prefix_tokens: List[int],
        device: str,
    ):
        """Generate the link target title.

        First tries trie-constrained generation when the title_index provides a
        ``generate_title`` (TrieTitleIndex) — this guarantees a valid corpus title
        and, with show_beam_candidates, returns (target_str, tokens, beam_candidates).
        Otherwise falls back to free autoregressive generation
        (``_generate_title_free`` in the base, stopping at the ')' link-close).
        """
        # Delegate to a constrained generation loop if the title_index provides one.
        if self.title_index is not None and hasattr(self.title_index, "generate_title"):
            cfg = self.generation_config
            result = self.title_index.generate_title(
                model, prefix_tokens, device,
                max_title_tokens=self.max_title_tokens,
                temperature=cfg.temperature,
                top_k=cfg.top_k,
                top_p=cfg.top_p,
                return_candidates=self.show_beam_candidates,
            )
            if result is not None:
                return result
            # None → fall through to free generation below.

        return self._generate_title_free(model, prefix_tokens, device)

    def _get_close_paren_id(self, model) -> Optional[int]:
        """Return the token ID for ')' using the model's tokenizer."""
        try:
            tokenizer = getattr(model, "tokenizer", None)
            if tokenizer is None:
                return None
            ids = tokenizer.encode(")")
            return ids[0] if ids else None
        except Exception:
            return None


# ---------------------------------------------------------------------------
# ArxivPromptAnnotator
# ---------------------------------------------------------------------------

# GPT-2 token IDs for the LaTeX \cite{ opener sequence and } closer.
# Two forms, keyed by the leading backslash token:
#   '\cite{'  = (59, 66, 578, 90)   → ['\\', 'c', 'ite', '{']  (line-start form)
#   ' \cite{' = (3467, 66, 578, 90) → [' \\', 'c', 'ite', '{']  (in-prose form, ~77%)
# A scan of 7k+ real arxiv cites found token 3467 (' \\') opens ~77% and bare
# '\\' (59) ~23%. annotate() scans BOTH first-tokens and injects the full
# sequence matching whichever scored highest at the chosen position.
_CITE_OPENER_TOKENS: Tuple[int, ...] = (59, 66, 578, 90)          # '\cite{'
_CITE_OPENER_TOKENS_SPACE: Tuple[int, ...] = (3467, 66, 578, 90)  # ' \cite{'
# first backslash token -> full inject sequence
_CITE_OPENER_BY_FIRST_TOKEN: Dict[int, Tuple[int, ...]] = {
    59: _CITE_OPENER_TOKENS,
    3467: _CITE_OPENER_TOKENS_SPACE,
}
_CITE_CLOSE_TOKEN: int = 92   # '}'

# TODO(arxiv-opener-coverage): _opener_probs currently scans P(token 59, '\\')
# to place the citation, but a scan of 7k+ real arxiv cites shows this is wrong
# on two counts:
#   1. Wrong dominant token: ~77% of real \cite occurrences begin with token
#      3467 (' \\', space+backslash), NOT bare '\\' (59). So the current scan
#      targets the ~23% minority form.
#   2. Noisy signal: '\\' (59) is ~2.5% of ALL arxiv tokens (it prefixes every
#      LaTeX macro: \alpha, \ref, \frac, ...). P('\\') at a position is a poor
#      proxy for "wants a citation" (~0.06% precision).
# Correct opener sequences: '\\cite{' = (59,66,578,90); ' \\cite{' =
# (3467,66,578,90). Since annotate() *injects* the opener, we must inject the
# WHOLE (' \\'/'\\')+c+ite+{ sequence matching whichever backslash-first-token
# scored highest — not just the c+ite bigram.
#
# PLANNED FIX (option B, multi-forward refinement): the high-precision signal is
# the c+ite (66,578) bigram (~800x better SNR than bare '\\'), but reading a
# 2-tokens-ahead signal needs additional forward passes. To bound cost: take the
# top-K positions by the cheap one-forward P('\\'/' \\') signal, then teacher-force
# the KNOWN fixed sequence '\\','c','ite' and read P at each step. This is LINEAR,
# not combinatorial — the opener is a fixed token sequence, so each refinement
# step has exactly one continuation to score (no branching/fan-out). Cost is a
# constant ~1 + (refinement_depth * K) forwards, batchable if forward_inference
# supports B>1. Acceptable multi-forward ugliness for a research repo; a from-
# scratch trainer would instead use a custom tokenizer where each link opener is
# a single token and this problem vanishes.
#
# Why NOT the MTP shortcut: this model's MTP reuses the single lm_head as a
# training-only auxiliary loss and is skipped at eval (see training_module.py) —
# there is no persistent +2/+3 head to query at inference, so P('ite') at
# position t cannot be read from one forward. NOTE for future work: a model
# trained with *traditional multi-head MTP* (separate persistent heads per
# offset) COULD read P(c),P(ite) from a single forward and get the high-precision
# signal for free — worth exploiting if such a checkpoint exists.
# (mtp_decay_steps: 0 in the configs is correct, not a bug — main.py maps it to
# mtp_decay_micro_steps=0, and training_module.py only decays when that is > 0,
# so 0 means constant MTP weights [0.5, 0.25] throughout training.)
#
# (\citep/\citet are absent from this corpus — the unarXive extractor normalises
# all citations to bare \cite{ — so no variant handling needed unless pointed at
# raw LaTeX.)


class ArxivPromptAnnotator(_AnnotatorBase):
    """
    Injects a LaTeX ``\\cite{Title}`` link into a prompt for arXiv-trained models.

    Unlike MarkdownPromptAnnotator (which has a display-text phase and two
    forward passes to find '[(display)]('), this annotator has a simpler two-step
    injection: find the best position to insert ``\\`` (the start of ``\\cite{``),
    then autoregressively generate the paper title until ``}`` is sampled or EOS.

    Algorithm
    ---------
    1. Forward pass (doc_causal, original tokens).
       Find position i with highest P('\\') — the first token of ``\\cite{``.
    2. Build prefix: ``context[:i] + ['\\', 'c', 'ite', '{']``.
    3. Autoregressively generate title tokens until ``}`` is sampled.
       target_str = decoded title tokens (strip leading/trailing whitespace).
    4. Build final context: ``context[:i] + cite_opener + title_tokens + ['}'] + context[i:]``.
       The original suffix starts at i (nothing consumed — pure insertion).
    5. Aux doc acquisition via _fetch_aux (shared with MarkdownPromptAnnotator).

    Args:
        corpus: Optional corpus with has_document/get_document.
        title_index: Optional TitleIndex for fuzzy title matching.
        link_retrieval_mode: full_skip / link_but_skip / corpus_only /
            generate_only / corpus_then_generate (see VALID_LINK_RETRIEVAL_MODES).
        generation_config: Sampling settings for title generation and aux doc generation.
        layout_policy: Layout policy for aux doc generation (None = NullLayoutPolicy).
        max_title_tokens: Maximum tokens to generate for the title. Default 60.
        eos_token_id: EOS token id. Default 50256.
        opener_refine_top_k: Number of top backslash-probability positions to
            re-rank by the high-precision citation signal
            P(bslash)*P(c|bslash)*P(ite|bslash,c) in one extra packed forward
            pass. Bare backslash (token 59) prefixes every LaTeX macro so raw
            P(backslash) is a noisy citation signal; conditioning on the c+ite
            continuation is ~800x more citation-specific. Default 8. Set to 0/1
            to disable refinement and place at the raw P(backslash) argmax
            (cheaper, one fewer forward).
    """

    def __init__(
        self,
        corpus=None,
        title_index: Optional[TitleIndex] = None,
        link_retrieval_mode: str = "full_skip",
        generation_config: Optional[GenerationConfig] = None,
        layout_policy=None,
        max_title_tokens: int = 60,
        eos_token_id: int = 50256,
        opener_refine_top_k: int = 8,
    ):
        link_retrieval_mode = validate_link_retrieval_mode(link_retrieval_mode)
        self.corpus = corpus
        self.title_index = title_index
        self.link_retrieval_mode = link_retrieval_mode
        self.generation_config = generation_config or GenerationConfig(repetition_penalty=1.3)
        self.layout_policy = layout_policy
        self.max_title_tokens = max_title_tokens
        self.eos_token_id = eos_token_id
        # >1 enables the top-K citation-intent refinement (one extra forward);
        # <=1 uses the raw P(backslash) argmax position (commit-1 behavior).
        self.opener_refine_top_k = opener_refine_top_k

    # Title ends at the '}' that closes \cite{...}.
    _title_stop_regex = _re.compile(r'\}')

    # scan_prob is inherited from _AnnotatorBase (uses _opener_probs below).

    # ------------------------------------------------------------------
    # Public API (PromptAnnotator protocol)
    # ------------------------------------------------------------------

    def annotate(
        self,
        model,
        context_tokens: List[int],
        device: str = "cuda",
    ) -> AnnotatedPrompt:
        """Full annotation pipeline: inject \\cite{Title} and fetch aux doc.

        In full_skip mode no citation is injected: the original context is
        returned unchanged with link_fired=False.
        """
        if not context_tokens or self.link_retrieval_mode == "full_skip":
            return AnnotatedPrompt(
                context_tokens=list(context_tokens),
                aux_token_lists=[],
                aux_raw_identifiers=[],
                target_str="",
                link_opener_pos=0,
                link_mid_pos=0,
                link_opener_prob=0.0,
                link_fired=False,
            )

        # ── Step 1: find best opener position AND which backslash form ─────
        # _opener_probs returns [T, n_openers] over self._opener_first_tokens
        # (bare '\' 59 and space '\' 3467): per position, P(each backslash form).
        opener_probs = self._opener_probs(model, context_tokens, device)  # [T, n_openers]
        per_pos_max = opener_probs.max(dim=-1).values                     # [T]

        # Raw P(backslash) is a noisy citation signal ('\' opens every LaTeX
        # macro). When opener_refine_top_k > 1, re-rank the top-K positions by
        # the high-precision joint signal P(\)·P(c|\)·P(ite|\c) using ONE extra
        # packed forward; otherwise place at the raw P(backslash) argmax.
        link_opener_pos, best_col = self._refine_opener_position(
            model, context_tokens, opener_probs, per_pos_max, device
        )
        link_opener_prob = float(per_pos_max[link_opener_pos].item())
        best_first = self._opener_first_tokens[best_col]

        # ── Step 2: build title-generation prefix ────────────────────────
        # Inject the full \cite{ opener sequence matching the chosen backslash
        # form at link_opener_pos. link_mid_pos points to the '{' (last token).
        cite_opener = list(_CITE_OPENER_BY_FIRST_TOKEN[best_first])
        title_prefix = list(context_tokens[:link_opener_pos]) + cite_opener
        link_mid_pos = link_opener_pos + len(cite_opener) - 1   # position of '{'

        # ── Step 3: autoregressive title generation ───────────────────────
        target_str, title_tokens = self._generate_title(model, title_prefix, device)

        # ── Step 4: build final context_tokens ───────────────────────────
        # Pure insertion — no original tokens consumed.
        # Structure: context[:i] + \cite{ + title_tokens + } + context[i:]
        final_context = (
            title_prefix
            + title_tokens
            + [_CITE_CLOSE_TOKEN]
            + list(context_tokens[link_opener_pos:])
        )

        # ── Step 5: aux doc acquisition ───────────────────────────────────
        aux_token_lists, aux_raw_identifiers, link_fired = self._fetch_aux(
            model, target_str, device
        )

        return AnnotatedPrompt(
            context_tokens=final_context,
            aux_token_lists=aux_token_lists,
            aux_raw_identifiers=aux_raw_identifiers,
            target_str=target_str,
            link_opener_pos=link_opener_pos,
            link_mid_pos=link_mid_pos,
            link_opener_prob=link_opener_prob,
            link_fired=link_fired,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    # First tokens of the two \cite{ opener forms, in a fixed order that
    # _opener_probs and annotate() share so the argmax index maps back correctly.
    _opener_first_tokens: Tuple[int, ...] = (59, 3467)   # '\', ' \'

    def _opener_probs(
        self,
        model,
        context_tokens: List[int],
        device: str,
    ) -> torch.Tensor:
        """Forward pass; return P(opener first token) per position. Shape [T, n_openers].

        Scans both backslash forms — bare '\\' (59) and space '\\' (3467) — since
        ~77% of real arxiv cites open with the space form. annotate() picks the
        best (position, form) from this one forward; scan_prob reduces to a scalar.
        """
        tok_tensor = torch.tensor(context_tokens, dtype=torch.long, device=device)
        logits = self._run_fwd(model, tok_tensor, device)   # [T, V]
        probs = F.softmax(logits.float(), dim=-1)            # [T, V]
        return probs[:, list(self._opener_first_tokens)]     # [T, n_openers]

    def _refine_opener_position(
        self,
        model,
        context_tokens: List[int],
        opener_probs: torch.Tensor,   # [T, n_openers]
        per_pos_max: torch.Tensor,    # [T]
        device: str,
    ) -> Tuple[int, int]:
        """Pick the (position, backslash-form) to inject at.

        Bare '\\' (59) opens every LaTeX macro, so the raw P(backslash) argmax
        is a noisy citation-placement signal. This re-ranks the top-K positions
        by the high-precision joint citation signal

            score = P(backslash) · P('c' | backslash) · P('ite' | backslash 'c')

        which is ~800x more citation-specific than P(backslash) alone. Both
        conditional factors come from ONE extra forward: for each candidate we
        pack ``context[:pos] + [backslash, 'c']`` as a doc_causal DocSpan (the
        score_completions_batched pattern), then read P('c') from the backslash
        token's logits and P('ite') from the 'c' token's logits. The opener is a
        FIXED token sequence, so there is exactly one continuation to score per
        candidate — linear in K, no combinatorial fan-out.

        Returns (position, column-into-self._opener_first_tokens). Falls back to
        the raw per_pos_max argmax when refinement is disabled (top_k <= 1), the
        context is empty, or the packed forward fails.
        """
        n_pos = per_pos_max.shape[0]
        # Column of the best backslash form at each position (for injection choice).
        best_cols = opener_probs.argmax(dim=-1)   # [T]

        def _raw_argmax() -> Tuple[int, int]:
            pos = int(per_pos_max.argmax().item())
            return pos, int(best_cols[pos].item())

        k = self.opener_refine_top_k
        if k is None or k <= 1 or n_pos == 0:
            return _raw_argmax()

        c_tok = _CITE_OPENER_TOKENS[1]     # 'c'  (66)
        ite_tok = _CITE_OPENER_TOKENS[2]   # 'ite' (578)

        k = min(k, n_pos)
        top_pos = torch.topk(per_pos_max, k).indices.tolist()   # K candidate positions

        # Pack one doc_causal span per candidate: context[:pos] + [backslash, 'c'].
        # backslash token = the winning opener form's first token at that pos.
        all_tokens: List[int] = []
        spans: List[DocSpan] = []
        meta: List[Tuple[int, int, int]] = []   # (pos, col, span_start)
        offset = 0
        for pos in top_pos:
            col = int(best_cols[pos].item())
            bslash = self._opener_first_tokens[col]
            seq = list(context_tokens[:pos]) + [bslash, c_tok]
            if len(seq) < 2:
                continue
            spans.append(DocSpan(
                doc_id=len(spans), normed_identifier="", raw_identifier="",
                start=offset, end=offset + len(seq),
                truncated=False, outgoing_identifiers=[],
            ))
            meta.append((pos, col, offset))
            all_tokens.extend(seq)
            offset += len(seq)

        if not spans:
            return _raw_argmax()

        try:
            tokens_tensor = torch.tensor(all_tokens, dtype=torch.long, device=device).unsqueeze(0)
            logits = model.forward_inference(tokens_tensor, spans, mask_type="doc_causal")  # [1, tot, V]
            log_probs = F.log_softmax(logits[0].float(), dim=-1)   # [tot, V]
        except Exception as exc:
            logger.warning("Arxiv opener refinement forward failed (%s); using raw argmax.", exc)
            return _raw_argmax()

        best_score = -math.inf
        best = _raw_argmax()
        for (pos, col, span_start) in meta:
            seq_len = len(context_tokens[:pos]) + 2   # ... + [backslash, 'c']
            bslash_logit_idx = span_start + seq_len - 2   # backslash token position → predicts 'c'
            c_logit_idx = span_start + seq_len - 1         # 'c' token position → predicts 'ite'
            log_p_backslash = math.log(max(float(per_pos_max[pos].item()), 1e-12))
            log_p_c = float(log_probs[bslash_logit_idx, c_tok].item())
            log_p_ite = float(log_probs[c_logit_idx, ite_tok].item())
            score = log_p_backslash + log_p_c + log_p_ite
            if score > best_score:
                best_score = score
                best = (pos, col)
        return best

    # Title generation (free, stop at '}') is inherited: _generate_title_free.
    _generate_title = _AnnotatorBase._generate_title_free


# ─── Visualization ────────────────────────────────────────────────────────────

# ANSI helpers (same scheme as generate.py)
_CYAN   = "\033[96m"
_YELLOW = "\033[93m"
_GREEN  = "\033[92m"
_BOLD   = "\033[1m"
_DIM    = "\033[2m"
_RESET  = "\033[0m"


def _c(text: str, code: str, use_color: bool) -> str:
    return f"{code}{text}{_RESET}" if use_color else text


def render_annotated_example(
    original_tokens: List[int],
    annotated: "AnnotatedPrompt",
    tokenizer,
    choices: Optional[List[List[int]]] = None,
    completion_tokens: Optional[List[int]] = None,
    label: Optional[int] = None,
    pred_flat: Optional[int] = None,
    pred_annotated: Optional[int] = None,
    use_color: bool = True,
    max_aux_tokens: int = 200,
    width: int = 72,
) -> str:
    """Pretty-print an annotated benchmark example for visual verification.

    Shows three panels:
      1. ORIGINAL CONTEXT — the unmodified context tokens decoded as plain text.
      2. ANNOTATED CONTEXT — same text with the injected link highlighted:
           - The '[' insertion point is marked in yellow.
           - The display text (existing tokens between '[' and '](') is shown plainly.
           - The '](' and generated title are shown in cyan.
           - The ')' closing the title is shown in cyan.
           - Text before and after the link is dim.
      3. AUX DOC (if link_fired) — the first aux_token_lists entry decoded, truncated.
      4. ANSWER — choices (MC) or completion (fill-in-the-blank), with correct answer
         marked if label is provided.

    Args:
        original_tokens: The unmodified context_tokens (before annotation).
        annotated: AnnotatedPrompt returned by annotator.annotate().
        tokenizer: Object with a .decode(List[int]) -> str method.
        choices: For MC benchmarks — list of per-choice token lists.
        completion_tokens: For fill-in-the-blank — the answer token list.
        label: Correct answer index (MC only). Marks the right choice with ✓.
        use_color: Whether to emit ANSI color codes. False for file output.
        max_aux_tokens: Truncate aux doc display at this many tokens.
        width: Width of separator lines.

    Returns:
        A multi-line string ready to print.
    """
    sep   = "─" * width
    thick = "═" * width
    lines: List[str] = []

    def _decode(toks: List[int]) -> str:
        try:
            return tokenizer.decode(toks)
        except Exception:
            return "".join(chr(t % 256) for t in toks)

    # ── Panel 1: original context ─────────────────────────────────────
    lines.append(_c(thick, _BOLD, use_color))
    lines.append(_c(" ORIGINAL CONTEXT", _BOLD, use_color))
    lines.append(_c(sep, _BOLD, use_color))
    lines.append(_c(_decode(original_tokens), _DIM, use_color))

    # ── Panel 2: annotated context with link highlighted ──────────────
    lines.append("")
    lines.append(_c(thick, _BOLD, use_color))
    lines.append(_c(" ANNOTATED CONTEXT", _BOLD, use_color))
    lines.append(
        _c(
            f"  link_opener_pos={annotated.link_opener_pos}  "
            f"link_mid_pos={annotated.link_mid_pos}  "
            f"P([)={annotated.link_opener_prob:.4f}  "
            f"link_fired={annotated.link_fired}  "
            f"target={annotated.target_str!r}",
            _DIM, use_color,
        )
    )
    lines.append(_c(sep, _BOLD, use_color))

    ctx = annotated.context_tokens
    # Locate the injected tokens in context_tokens.
    # Structure: ...original_prefix... [ display_text ]( title_tokens ) ...original_suffix...
    # link_opener_pos points to the '[' in context_tokens.
    # link_mid_pos points to the '](' in context_tokens.
    # After link_mid_pos come the title tokens, then ')', then the original suffix.
    i_open = annotated.link_opener_pos
    i_mid  = annotated.link_mid_pos

    # Find the close ')' by looking for it after i_mid
    i_close = None
    for k in range(i_mid + 1, min(len(ctx), i_mid + 1 + 60)):
        tok_str = _decode([ctx[k]])
        if ")" in tok_str:
            i_close = k
            break

    # Render segments with color highlighting
    prefix_text  = _decode(ctx[:i_open])
    opener_text  = _decode(ctx[i_open:i_open + 1])            # '[' or ' ['
    display_text = _decode(ctx[i_open + 1:i_mid])             # original tokens between [ and ](
    mid_text     = _decode(ctx[i_mid:i_mid + 1])              # ']('
    if i_close is not None:
        title_text  = _decode(ctx[i_mid + 1:i_close])
        close_text  = _decode(ctx[i_close:i_close + 1])
        suffix_text = _decode(ctx[i_close + 1:])
    else:
        title_text  = _decode(ctx[i_mid + 1:])
        close_text  = ""
        suffix_text = ""

    annotated_line = (
        _c(prefix_text, _DIM, use_color)
        + _c(opener_text, _YELLOW, use_color)
        + display_text
        + _c(mid_text + title_text + close_text, _CYAN, use_color)
        + _c(suffix_text, _DIM, use_color)
    )
    lines.append(annotated_line)

    # ── Panel 3: aux doc ──────────────────────────────────────────────
    lines.append("")
    lines.append(_c(sep, _BOLD, use_color))
    if annotated.link_fired and annotated.aux_token_lists:
        id_str = annotated.aux_raw_identifiers[0] if annotated.aux_raw_identifiers else "?"
        lines.append(_c(f' AUX DOC  "{id_str}"  [FIRED]', _BOLD, use_color))
        lines.append(_c(sep, _BOLD, use_color))
        aux_toks = annotated.aux_token_lists[0]
        aux_text = _decode(aux_toks[:max_aux_tokens])
        if len(aux_toks) > max_aux_tokens:
            aux_text += _c(f"\n[...truncated — {len(aux_toks)} tokens total]", _DIM, use_color)
        lines.append(aux_text)
    else:
        status = "corpus miss" if annotated.target_str else "no title generated"
        lines.append(
            _c(f' AUX DOC  target={annotated.target_str!r}  [{status}]', _DIM, use_color)
        )
        lines.append(_c(sep, _BOLD, use_color))

    # ── Panel 4: beam candidates (TrieTitleIndex only) ───────────────
    if annotated.beam_candidates:
        lines.append("")
        lines.append(_c(sep, _BOLD, use_color))
        lines.append(_c(" BEAM CANDIDATES", _BOLD, use_color))
        lines.append(_c(sep, _BOLD, use_color))
        for rank, (raw_id, score) in enumerate(annotated.beam_candidates):
            marker = " ←" if rank == 0 else ""
            lines.append(
                _c(f"  [{rank + 1}]", _CYAN if rank == 0 else _DIM, use_color)
                + f" {raw_id!r:<50} score={score:.4f}{marker}"
            )

    # ── Panel 5: answer ───────────────────────────────────────────────
    if choices is not None:
        lines.append("")
        lines.append(_c(sep, _BOLD, use_color))
        legend = " CHOICES   ✓=correct"
        if pred_flat is not None:
            legend += "  →=model(flat)"
        if pred_annotated is not None:
            legend += "  ★=model(annotated)"
        lines.append(_c(legend, _BOLD, use_color))
        for idx, ch_toks in enumerate(choices):
            ch_text = _decode(ch_toks)
            markers = ""
            if label is not None and idx == label:
                markers += _c(" ✓", _GREEN, use_color)
            if pred_flat is not None and idx == pred_flat:
                markers += _c(" →", _YELLOW, use_color)
            if pred_annotated is not None and idx == pred_annotated:
                markers += _c(" ★", _CYAN, use_color)
            lines.append(f"  [{idx}] {ch_text}{markers}")
    elif completion_tokens is not None:
        lines.append("")
        lines.append(_c(sep, _BOLD, use_color))
        lines.append(_c(" ANSWER", _BOLD, use_color))
        lines.append(f"  {_decode(completion_tokens)}")

    lines.append(_c(thick, _BOLD, use_color))
    return "\n".join(lines)
