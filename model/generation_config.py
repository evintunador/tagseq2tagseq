"""
Configuration dataclass for TS2TS generation.

Defines all parameters that control generation behavior, including sampling
parameters, context management, and document structure constraints.
"""
from dataclasses import dataclass, asdict
from typing import Literal, Optional

import torch


@dataclass
class GenerationConfig:
    """Configuration for generation behavior."""
    
    # Basic generation params
    max_new_tokens: int = 100
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    
    # Document structure params
    max_tokens_per_document: int = 512
    max_context_length: int = 4096
    max_auxiliary_documents: int = 6
    max_link_depth: int = 1

    # Corpus-doc truncation: when set, a fetched corpus document's body is
    # truncated (head, i.e. abstract + intro first) to this many tokens before
    # insertion. None = insert the full document. Essential for arXiv, where a
    # cited paper (~70k tokens) far exceeds the context window and would
    # otherwise be silently dropped by the can_add_document check.
    max_corpus_doc_tokens: Optional[int] = None
    
    # Corpus integration / link retrieval
    # Modes:
    #   corpus_then_generate — try corpus first, fall back to generation (default)
    #   corpus_only          — corpus lookup only; skip if not found (no generation)
    #   generate_only        — always generate aux doc; never look up corpus
    #   link_but_skip        — detect link (trace bookkeeping), don't insert any doc
    #   full_skip            — return immediately from _handle_link; no link processing
    link_retrieval_mode: str = "corpus_then_generate"
    
    # Eviction policy
    eviction_policy: Literal["drop_oldest", "stop_new"] = "drop_oldest"
    
    # Link handling
    process_prompt_links: bool = True  # Process links in initial prompt
    # Recursive link depth is controlled solely by max_link_depth (no separate allow_recursive_links)
    
    # Repetition penalty: values > 1.0 reduce probability of already-seen tokens.
    # Applied to all tokens in the current document's token list before sampling.
    # 1.0 = disabled; 1.3 is a reasonable starting value.
    repetition_penalty: float = 1.0

    # Link detection
    max_recent_link_tokens: int = 200  # How many trailing tokens of the active doc to scan per step

    # Stopping
    eos_token_id: int = 50256  # GPT-2 <|endoftext|>

    # Vocabulary bound: forbid sampling token ids >= this value. The lm_head is
    # often padded past the tokenizer's real vocab for GPU alignment (e.g. 50304
    # vs GPT-2's 50257); those padded slots decode to invalid tokens. None = no
    # bound (allow the full logit width). Defaults to GPT-2's real vocab.
    allowed_vocab_size: Optional[int] = 50257

    # Trace
    record_trace: bool = True  # Whether to populate GenerationResult.trace
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    def to_dict(self) -> dict:
        """Convert to dictionary for metadata storage."""
        return asdict(self)
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.max_new_tokens <= 0:
            raise ValueError(f"max_new_tokens must be positive, got {self.max_new_tokens}")
        
        if self.temperature < 0:
            raise ValueError(f"temperature must be non-negative, got {self.temperature}")
        
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError(f"top_k must be positive if specified, got {self.top_k}")
        
        if self.top_p is not None and not (0.0 < self.top_p <= 1.0):
            raise ValueError(f"top_p must be in (0, 1] if specified, got {self.top_p}")
        
        if self.max_tokens_per_document <= 0:
            raise ValueError(f"max_tokens_per_document must be positive, got {self.max_tokens_per_document}")
        
        if self.max_context_length <= 0:
            raise ValueError(f"max_context_length must be positive, got {self.max_context_length}")
        
        if self.max_auxiliary_documents < 0:
            raise ValueError(f"max_auxiliary_documents must be non-negative, got {self.max_auxiliary_documents}")
        
        if self.max_link_depth < 0:
            raise ValueError(f"max_link_depth must be non-negative, got {self.max_link_depth}")

        if self.max_corpus_doc_tokens is not None and self.max_corpus_doc_tokens <= 0:
            raise ValueError(
                f"max_corpus_doc_tokens must be positive if specified, got {self.max_corpus_doc_tokens}"
            )
        
        if self.eviction_policy not in ["drop_oldest", "stop_new"]:
            raise ValueError(f"eviction_policy must be 'drop_oldest' or 'stop_new', got {self.eviction_policy}")

        if self.max_tokens_per_document > self.max_context_length:
            raise ValueError(
                f"max_tokens_per_document ({self.max_tokens_per_document}) exceeds "
                f"max_context_length ({self.max_context_length}); a single document "
                "could never fit in the context window"
            )

        if self.max_new_tokens > self.max_tokens_per_document:
            raise ValueError(
                f"max_new_tokens ({self.max_new_tokens}) exceeds "
                f"max_tokens_per_document ({self.max_tokens_per_document}); "
                "max_tokens_per_document would always fire first, making max_new_tokens ineffective"
            )

        _valid_modes = {
            "corpus_only", "generate_only", "corpus_then_generate",
            "link_but_skip", "full_skip",
        }
        if self.link_retrieval_mode not in _valid_modes:
            raise ValueError(
                f"link_retrieval_mode must be one of {sorted(_valid_modes)}, "
                f"got {self.link_retrieval_mode!r}"
            )
