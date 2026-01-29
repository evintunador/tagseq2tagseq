"""
Configuration dataclass for DS2DS generation.

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
    
    # Corpus integration
    allow_corpus_fallback: bool = True  # Generate if corpus doesn't have doc
    
    # Eviction policy
    eviction_policy: Literal["drop_oldest", "stop_new"] = "drop_oldest"
    
    # Link handling
    process_prompt_links: bool = True  # Process links in initial prompt
    allow_recursive_links: bool = True  # Allow aux docs to create links (respects max_link_depth)
    
    # Stopping
    eos_token_id: int = 50256  # GPT-2 <|endoftext|>
    
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
        
        if self.eviction_policy not in ["drop_oldest", "stop_new"]:
            raise ValueError(f"eviction_policy must be 'drop_oldest' or 'stop_new', got {self.eviction_policy}")
