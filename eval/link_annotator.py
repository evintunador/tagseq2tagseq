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
from typing import Any, Dict, List, Optional, Protocol, Set, Tuple, runtime_checkable

import torch
import torch.nn.functional as F

from data.collate import DocSpan
from eval.title_index import TitleIndex
from model.generation_config import GenerationConfig
from model.sampling import sample_token

logger = logging.getLogger(__name__)

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
# MarkdownPromptAnnotator
# ---------------------------------------------------------------------------

class MarkdownPromptAnnotator:
    """
    Injects a Wikipedia-style [display](Title) link into a prompt.

    Uses the model's own next-token probabilities to select the most natural
    position for the link opener and mid-point, autoregressively generates the
    target title, then fetches or generates the auxiliary document.

    Args:
        corpus: Optional corpus object with has_document/get_document (e.g.
            PretokCorpus from generate.py). Required for corpus_only and
            corpus_then_generate modes.
        link_retrieval_mode: How to obtain the aux doc.
            "no_op"               — inject link syntax only, no aux doc.
            "corpus_only"         — corpus lookup; no-op on miss.
            "generate"            — always generate aux doc autoregressively.
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
        link_opener_token_ids: Token IDs considered as valid link openers
            (' [' in GPT-2). Default (685,). Bare '[' (58) is excluded from
            the default — the space is part of the token, keeping the prefix
            text clean before the bracket.
        link_mid_token_id: Token ID for the '](' bigram. Default 16151.
        eos_token_id: EOS token ID. Default 50256 (GPT-2).
    """

    _VALID_MODES = frozenset({
        "no_op", "corpus_only", "generate", "corpus_then_generate",
    })

    def __init__(
        self,
        corpus=None,
        title_index: Optional[TitleIndex] = None,
        link_retrieval_mode: str = "corpus_only",
        generation_config: Optional[GenerationConfig] = None,
        layout_policy=None,
        max_display_tokens: int = 100,
        decay_factor: float = 0.95,
        max_title_tokens: int = 50,
        link_opener_token_ids: Tuple[int, ...] = (685,),
        link_mid_token_id: int = 16151,
        eos_token_id: int = 50256,
    ):
        if link_retrieval_mode not in self._VALID_MODES:
            raise ValueError(
                f"link_retrieval_mode must be one of {sorted(self._VALID_MODES)}, "
                f"got {link_retrieval_mode!r}"
            )
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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scan_prob(
        self,
        model,
        context_tokens: List[int],
        device: str = "cuda",
    ) -> float:
        """Phase-1 scan: return max P('[') across all positions.

        Cheap — one forward pass, no generation. Called for all examples
        during threshold calibration before any full annotation.
        """
        if not context_tokens:
            return 0.0
        probs = self._link_opener_probs(model, context_tokens, device)
        return float(probs.max().item())

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
        """
        if not context_tokens:
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

        # ── Step 1: forward pass 1, pick link opener position ──────────
        opener_probs = self._link_opener_probs(model, context_tokens, device)
        link_opener_prob = float(opener_probs.max().item())
        link_opener_pos = int(opener_probs.argmax().item())

        # Which opener token was preferred at that position?
        tok_tensor = torch.tensor(context_tokens, dtype=torch.long, device=device)
        full_logits = self._run_fwd(model, tok_tensor, device)   # [T, V]
        probs_at_pos = F.softmax(full_logits[link_opener_pos].float(), dim=-1)
        best_opener = max(
            self.link_opener_token_ids,
            key=lambda t: probs_at_pos[t].item(),
        )

        # ── Step 2: insert ' [', forward pass 2, pick '](' position ──────
        # best_opener is always ' [' (685) — the space is part of the token,
        # so prefix_text ends cleanly before the bracket.
        toks_with_opener = (
            context_tokens[:link_opener_pos]
            + [best_opener]
            + context_tokens[link_opener_pos:]
        )
        opener_tensor = torch.tensor(toks_with_opener, dtype=torch.long, device=device)
        fwd2_logits = self._run_fwd(model, opener_tensor, device)   # [T+1, V]

        # Search window for '](' — positions strictly after the opener
        search_start = link_opener_pos + 1
        search_end = min(len(toks_with_opener), link_opener_pos + 1 + self.max_display_tokens)
        if search_start >= search_end:
            link_mid_pos = link_opener_pos + 1
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
        target_str, title_tokens = self._generate_title(model, title_prefix, device)

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
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_fwd(self, model, tok_tensor: torch.Tensor, device: str) -> torch.Tensor:
        """Single forward pass. Returns logits [T, V] (no batch dim)."""
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

    def _link_opener_probs(
        self,
        model,
        context_tokens: List[int],
        device: str,
    ) -> torch.Tensor:
        """Run forward pass 1; return max P(opener) at each position. Shape [T]."""
        tok_tensor = torch.tensor(context_tokens, dtype=torch.long, device=device)
        logits = self._run_fwd(model, tok_tensor, device)    # [T, V]
        probs = F.softmax(logits.float(), dim=-1)            # [T, V]
        opener_ids = list(self.link_opener_token_ids)
        opener_probs = probs[:, opener_ids]                   # [T, n_openers]
        return opener_probs.max(dim=-1).values                # [T]

    def _generate_title(
        self,
        model,
        prefix_tokens: List[int],
        device: str,
    ) -> Tuple[str, List[int]]:
        """Autoregressively generate the link target title.

        Stops when any of the following are true:
          - EOS token ID is sampled (token-level: fixed single token)
          - ')' appears in the decoded accumulation (string-level: catches
            multi-character BPE tokens like ')', ').\n', 'imperial)', etc.)
          - '\n' appears in the decoded accumulation (titles never span lines)
          - max_title_tokens is exhausted

        String-level stop mirrors the approach used by MarkdownLinkDetector:
        decode the growing window each step and check for the stop character.
        This handles BPE merges that absorb ')' or '\n' into larger tokens
        that would be missed by a bare token-ID comparison.

        Returns (target_str, title_token_ids) where target_str is everything
        before the first ')' in the decoded accumulation (stripped).
        """
        tokenizer = getattr(model, "tokenizer", None)
        current_tokens = list(prefix_tokens)
        title_tokens: List[int] = []
        cfg = self.generation_config

        for _ in range(self.max_title_tokens):
            tok_tensor = torch.tensor(current_tokens, dtype=torch.long, device=device)
            logits = self._run_fwd(model, tok_tensor, device)    # [T, V]
            next_token = sample_token(
                logits[-1],
                temperature=cfg.temperature,
                top_k=cfg.top_k,
                top_p=cfg.top_p,
            )

            # Token-level EOS stop — always a single fixed token
            if next_token == self.eos_token_id:
                break

            title_tokens.append(next_token)
            current_tokens.append(next_token)

            # String-level stop: decode accumulated tokens and check for
            # a link-closing ')' (not a mid-word paren like ")iani") or '\n'.
            # Re-decode the whole accumulation each step for BPE correctness.
            # A ')' stops generation only when it is not immediately followed
            # by a word character — i.e. it closes the link rather than being
            # embedded inside a word like "Trigiani" or "U2)".
            if tokenizer is not None:
                try:
                    decoded_so_far = tokenizer.decode(title_tokens)
                    if "\n" in decoded_so_far:
                        break
                    if _re.search(r'\)(?!\w)', decoded_so_far):
                        break
                except Exception:
                    pass

        # Decode accumulated tokens and trim to clean title string.
        if title_tokens:
            try:
                raw = tokenizer.decode(title_tokens) if tokenizer is not None else (
                    "".join(chr(t % 256) for t in title_tokens)
                )
            except Exception:
                raw = ""
        else:
            raw = ""

        # Trim at the first link-closing ')' or '\n', whichever comes first.
        # Use the same regex so we trim at the same boundary we stopped on.
        m = _re.search(r'\)(?!\w)|\n', raw)
        if m:
            raw = raw[:m.start()]

        target_str = raw.strip()

        # Re-encode the clean title string so title_tokens exactly matches
        # what will appear in context_tokens (no trailing garbage from a
        # stop-trigger BPE token like ")iani").
        if target_str and tokenizer is not None:
            try:
                title_tokens = list(tokenizer.encode(target_str))
            except Exception:
                pass

        return target_str, title_tokens

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

    def _fetch_aux(
        self,
        model,
        target_str: str,
        device: str,
    ) -> Tuple[List[List[int]], List[str], bool]:
        """Fetch or generate the aux document according to link_retrieval_mode.

        Returns (aux_token_lists, aux_raw_identifiers, link_fired).
        """
        if not target_str or self.link_retrieval_mode == "no_op":
            return [], [], False

        # Try corpus — resolve generated title to a corpus raw_identifier.
        # If a TitleIndex is provided, use it (hash-norm fuzzy match) so that
        # casing/punctuation variants of valid titles still hit. Fall back to
        # verbatim has_document only when no index is available.
        if self.link_retrieval_mode in ("corpus_only", "corpus_then_generate"):
            if self.corpus is not None:
                resolved = (
                    self.title_index.lookup(target_str)
                    if self.title_index is not None
                    else (target_str if self.corpus.has_document(target_str) else None)
                )
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

        # Generate
        if self.link_retrieval_mode in ("generate", "corpus_then_generate"):
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
        """Generate an aux document for target_str using the full generation loop.

        Uses GenerationConfig with max_link_depth=0 and link_retrieval_mode="full_skip"
        so the generated doc itself never spawns sub-links.
        """
        from model.generation_loop import run_generation
        from model.identifier_utils import create_normed_identifier

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

            # Determine whether the layout policy provides a non-empty prefix
            # (e.g. IdentifierPrefixEOSLayoutPolicy prepends "# target\n\n").
            # If not (NullLayoutPolicy), fall back to encoding target_str as the
            # prompt so the model has something to condition on.
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

    # ── Panel 4: answer ───────────────────────────────────────────────
    if choices is not None:
        lines.append("")
        lines.append(_c(sep, _BOLD, use_color))
        lines.append(_c(" CHOICES", _BOLD, use_color))
        for idx, ch_toks in enumerate(choices):
            ch_text = _decode(ch_toks)
            correct_marker = (
                _c(" ✓", _GREEN, use_color)
                if label is not None and idx == label
                else ""
            )
            lines.append(f"  [{idx}] {ch_text}{correct_marker}")
    elif completion_tokens is not None:
        lines.append("")
        lines.append(_c(sep, _BOLD, use_color))
        lines.append(_c(" ANSWER", _BOLD, use_color))
        lines.append(f"  {_decode(completion_tokens)}")

    lines.append(_c(thick, _BOLD, use_color))
    return "\n".join(lines)
