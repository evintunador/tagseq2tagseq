"""
Tokenizer configuration for link detection and special token handling.

This module provides a way to discover and cache token IDs for special
characters and keywords that are needed for link detection during training
and generation.
"""
from dataclasses import dataclass
from typing import List, Optional, Protocol, Dict, Any
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


class Tokenizer(Protocol):
    """Protocol defining minimal tokenizer interface we depend on."""
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        ...
    
    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs to text."""
        ...


@dataclass
class TokenizerConfig:
    """
    Configuration containing token IDs for special characters.
    
    This caches the token IDs so they don't need to be looked up repeatedly
    during training or generation. The IDs are discovered from the tokenizer
    once at initialization time.
    
    Attributes:
        # Markdown link detection
        bracket_open_ids: Token IDs for '[' (may include ' [' variants)
        bracket_close_paren_open_id: Token ID for ']('
        paren_close_id: Token ID for ')'
        
        # Python import detection
        import_keyword_id: Token ID for 'import'
        from_keyword_id: Token ID for 'from'
        
        # General structural tokens
        newline_id: Token ID for '\n'
        space_id: Token ID for ' '
        dot_id: Token ID for '.'
        
        # Tokenizer metadata
        name: Human-readable tokenizer name
        vocab_size: Total vocabulary size
    """
    
    # Markdown tokens
    bracket_open_ids: List[int]
    bracket_close_paren_open_id: int
    paren_close_id: int
    
    # Python tokens
    import_keyword_id: int
    from_keyword_id: int
    
    # Structural tokens
    newline_id: int
    space_id: int
    dot_id: int
    
    # Metadata
    name: str
    vocab_size: int
    
    @classmethod
    def from_tokenizer(cls, tokenizer: Any, name: str = "unknown") -> 'TokenizerConfig':
        """
        Automatically discover token IDs from a tokenizer.
        
        This probes the tokenizer with specific strings to find the token IDs
        for characters we need for link detection.
        
        Args:
            tokenizer: Any tokenizer with encode() method
            name: Human-readable name for logging
        
        Returns:
            TokenizerConfig with discovered token IDs
        
        Raises:
            ValueError: If required tokens can't be found or are ambiguous
        """
        logger.info(f"Discovering token IDs for tokenizer: {name}")
        
        # Discover bracket tokens
        # Try both with and without leading space
        bracket_open_ids = []
        
        # '[' alone
        tokens = tokenizer.encode('[')
        if len(tokens) == 1:
            bracket_open_ids.append(tokens[0])
            logger.debug(f"Found '[' = {tokens[0]}")
        else:
            logger.warning(f"'[' encoded to multiple tokens: {tokens}")
        
        # ' [' (space + bracket)
        tokens = tokenizer.encode(' [')
        if len(tokens) == 1:
            # Some tokenizers merge space + [
            bracket_open_ids.append(tokens[0])
            logger.debug(f"Found ' [' = {tokens[0]}")
        elif len(tokens) == 2:
            # Space and bracket are separate
            bracket_open_ids.append(tokens[1])
            logger.debug(f"Found '[' in ' [' = {tokens[1]}")
        
        if not bracket_open_ids:
            raise ValueError("Could not find token ID for '['")
        
        # Remove duplicates while preserving order
        bracket_open_ids = list(dict.fromkeys(bracket_open_ids))
        
        # Discover '](' token
        tokens = tokenizer.encode('](')
        if len(tokens) == 1:
            bracket_close_paren_open_id = tokens[0]
            logger.debug(f"Found '](' = {bracket_close_paren_open_id}")
        else:
            # Some tokenizers split this into ']' and '('
            # Try to find them separately
            close_bracket = tokenizer.encode(']')
            open_paren = tokenizer.encode('(')
            if len(close_bracket) == 1 and len(open_paren) == 1:
                logger.warning(
                    f"Tokenizer splits '](' into separate tokens: "
                    f"']' = {close_bracket[0]}, '(' = {open_paren[0]}"
                )
                # For now, we'll use the first token as a fallback
                # But this may require changing the detection algorithm
                bracket_close_paren_open_id = close_bracket[0]
            else:
                raise ValueError("Could not find token IDs for '](' or ']' + '('")
        
        # Discover ')' token
        tokens = tokenizer.encode(')')
        if len(tokens) == 1:
            paren_close_id = tokens[0]
            logger.debug(f"Found ')' = {paren_close_id}")
        else:
            raise ValueError(f"')' encoded to multiple tokens: {tokens}")
        
        # Discover Python keyword tokens
        import_tokens = tokenizer.encode('import')
        from_tokens = tokenizer.encode('from')
        
        # These might be multi-token, try with leading space
        import_with_space = tokenizer.encode(' import')
        from_with_space = tokenizer.encode(' from')
        
        # Prefer single-token version
        if len(import_with_space) == 1:
            import_keyword_id = import_with_space[0]
        elif len(import_tokens) == 1:
            import_keyword_id = import_tokens[0]
        else:
            logger.warning(f"'import' is multi-token: {import_tokens}, using first")
            import_keyword_id = import_tokens[0]
        
        if len(from_with_space) == 1:
            from_keyword_id = from_with_space[0]
        elif len(from_tokens) == 1:
            from_keyword_id = from_tokens[0]
        else:
            logger.warning(f"'from' is multi-token: {from_tokens}, using first")
            from_keyword_id = from_tokens[0]
        
        # Discover structural tokens
        newline_id = cls._get_single_token(tokenizer, '\n', 'newline')
        space_id = cls._get_single_token(tokenizer, ' ', 'space')
        dot_id = cls._get_single_token(tokenizer, '.', 'dot')
        
        # Get vocab size if available
        vocab_size = getattr(tokenizer, 'n_vocab', None)
        if vocab_size is None:
            vocab_size = getattr(tokenizer, 'vocab_size', None)
        if vocab_size is None:
            logger.warning("Could not determine vocab_size from tokenizer")
            vocab_size = -1
        
        config = cls(
            bracket_open_ids=bracket_open_ids,
            bracket_close_paren_open_id=bracket_close_paren_open_id,
            paren_close_id=paren_close_id,
            import_keyword_id=import_keyword_id,
            from_keyword_id=from_keyword_id,
            newline_id=newline_id,
            space_id=space_id,
            dot_id=dot_id,
            name=name,
            vocab_size=vocab_size,
        )
        
        logger.info(f"Discovered token IDs for {name}: {config}")
        return config
    
    @staticmethod
    def _get_single_token(tokenizer: Any, text: str, name: str) -> int:
        """Helper to get a single token ID or raise error."""
        tokens = tokenizer.encode(text)
        if len(tokens) != 1:
            raise ValueError(f"'{name}' ('{text}') encoded to {len(tokens)} tokens: {tokens}")
        return tokens[0]
    
    @classmethod
    def gpt2_defaults(cls) -> 'TokenizerConfig':
        """
        Get the hardcoded GPT-2 token IDs for backwards compatibility.
        
        Use this when you know you're using GPT-2 and want to skip discovery.
        """
        return cls(
            bracket_open_ids=[58, 685],  # '[' and ' ['
            bracket_close_paren_open_id=16151,  # ']('
            paren_close_id=8,  # ')'
            import_keyword_id=1330,  # 'import'
            from_keyword_id=6738,  # 'from'
            newline_id=198,  # '\n'
            space_id=220,  # ' '
            dot_id=13,  # '.'
            name="gpt2",
            vocab_size=50257,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for saving to config files."""
        return {
            'bracket_open_ids': self.bracket_open_ids,
            'bracket_close_paren_open_id': self.bracket_close_paren_open_id,
            'paren_close_id': self.paren_close_id,
            'import_keyword_id': self.import_keyword_id,
            'from_keyword_id': self.from_keyword_id,
            'newline_id': self.newline_id,
            'space_id': self.space_id,
            'dot_id': self.dot_id,
            'name': self.name,
            'vocab_size': self.vocab_size,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TokenizerConfig':
        """Deserialize from dictionary."""
        return cls(**data)
    
    def save(self, path: Path) -> None:
        """Save tokenizer config to JSON file."""
        output_path = Path(path)
        with open(output_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        
        logger.info(f"Saved tokenizer config to {output_path}")
    
    @classmethod
    def load(cls, path: Path) -> 'TokenizerConfig':
        """Load tokenizer config from JSON file."""
        input_path = Path(path)
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        config = cls.from_dict(data)
        logger.info(f"Loaded tokenizer config from {input_path}: {config.name}")
        return config
