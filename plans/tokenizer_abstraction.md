# Tokenizer Abstraction Plan

## Problem Statement

Token IDs for special characters (brackets, parentheses, imports) are currently hardcoded for GPT-2 throughout the codebase:

**model/graph_traversal/link_detectors.py**:
```python
link_start_token_ids: Optional[List[int]] = None,
link_mid_token_id: int = 16151,   # '](' in GPT-2
link_end_token_id: int = 8,       # ')' in GPT-2
```

**model/graph_traversal/link_detectors.py** (Python imports):
```python
import_token_id: int = 1330,     # 'import' in GPT-2
from_token_id: int = 6738,       # 'from' in GPT-2
newline_token_id: int = 198,     # '\n' in GPT-2
dot_token_id: int = 13,          # '.' in GPT-2
```

This prevents using:
- Different tiktoken encodings (cl100k_base for GPT-4, etc.)
- Custom tokenizers trained on domain-specific data
- Other tokenizer libraries (SentencePiece, HuggingFace, etc.)

## Goals

1. Abstract token ID constants into a configurable system
2. Support automatic token ID discovery from tokenizers
3. Maintain performance (no lookups on hot paths)
4. Support future tokenizers beyond GPT-2

## Proposed Solution

### Create TokenizerConfig Class

**File**: `model/tokenizer_config.py` (NEW)

```python
"""
Tokenizer configuration for link detection and special token handling.

This module provides a way to discover and cache token IDs for special
characters and keywords that are needed for link detection during training
and generation.
"""
from dataclasses import dataclass
from typing import List, Optional, Protocol, Dict, Any
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


def save_tokenizer_config(config: TokenizerConfig, path: str) -> None:
    """Save tokenizer config to JSON file."""
    import json
    from pathlib import Path
    
    output_path = Path(path)
    with open(output_path, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)
    
    logger.info(f"Saved tokenizer config to {output_path}")


def load_tokenizer_config(path: str) -> TokenizerConfig:
    """Load tokenizer config from JSON file."""
    import json
    from pathlib import Path
    
    input_path = Path(path)
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    config = TokenizerConfig.from_dict(data)
    logger.info(f"Loaded tokenizer config from {input_path}: {config.name}")
    return config
```

### Update Link Detectors

**File**: `model/graph_traversal/link_detectors.py`

**Before**:
```python
class MarkdownLinkDetector(TokenizedLinkDetector):
    def __init__(
        self,
        link_start_token_ids: Optional[List[int]] = None,
        link_mid_token_id: int = 16151,   # '](' in GPT-2
        link_end_token_id: int = 8,       # ')' in GPT-2
    ):
```

**After**:
```python
from model.tokenizer_config import TokenizerConfig

class MarkdownLinkDetector(TokenizedLinkDetector):
    def __init__(self, tokenizer_config: TokenizerConfig):
        """
        Initialize with tokenizer configuration.
        
        Args:
            tokenizer_config: Configuration containing token IDs
        """
        self.link_start_token_ids = set(tokenizer_config.bracket_open_ids)
        self.link_mid_token_id = tokenizer_config.bracket_close_paren_open_id
        self.link_end_token_id = tokenizer_config.paren_close_id
        
        logger.info(
            f"Initialized MarkdownLinkDetector with token IDs: "
            f"'[' = {tokenizer_config.bracket_open_ids}, "
            f"'](' = {self.link_mid_token_id}, "
            f"')' = {self.link_end_token_id}"
        )
```

**Similar changes for PythonImportDetector**:
```python
class PythonImportDetector(TokenizedLinkDetector):
    def __init__(self, tokenizer_config: TokenizerConfig):
        """
        Initialize with tokenizer configuration.
        
        Args:
            tokenizer_config: Configuration containing token IDs
        """
        self.import_token_id = tokenizer_config.import_keyword_id
        self.from_token_id = tokenizer_config.from_keyword_id
        self.newline_token_id = tokenizer_config.newline_id
        self.dot_token_id = tokenizer_config.dot_id
        
        logger.info(
            f"Initialized PythonImportDetector with token IDs: "
            f"'import' = {self.import_token_id}, "
            f"'from' = {self.from_token_id}"
        )
```

### Update Training Loop

**File**: `main.py` or wherever training is configured

```python
import tiktoken
from model.tokenizer_config import TokenizerConfig, save_tokenizer_config

# Load tokenizer
enc = tiktoken.get_encoding("gpt2")  # Or load custom tokenizer

# Discover token IDs
tokenizer_config = TokenizerConfig.from_tokenizer(enc, name="gpt2")

# Save config for reproducibility
save_tokenizer_config(tokenizer_config, run_dir / "tokenizer_config.json")

# Create link detector with config
from model.graph_traversal.link_detectors import MarkdownLinkDetector
link_detector = MarkdownLinkDetector(tokenizer_config)

# Create block mask creator
from model.graph_traversal.block_mask_creator import BlockMaskCreator
block_mask_creator = BlockMaskCreator(
    link_detector=link_detector,
    # ... other config
)
```

### Update Pretokenize

**File**: `data/pretokenize.py`

Currently it doesn't need tokenizer config since it works on text, not tokens.
But we should save the tokenizer name/version to metadata for reference.

```python
# In main pretokenize function
metadata = {
    "dtype_str": dtype_str,
    "total_tokens": total_tokens,
    "shard_filenames": shard_filenames,
    "tokenizer_name": enc.name if hasattr(enc, 'name') else "unknown",
    # ... other metadata
}
```

## Migration Strategy

### Phase 1: Add TokenizerConfig (No Breaking Changes)

1. Create `model/tokenizer_config.py` with full implementation
2. Add tests for token discovery with tiktoken
3. Add `gpt2_defaults()` method for backwards compatibility

### Phase 2: Update Link Detectors (Breaking Change)

1. Modify MarkdownLinkDetector to take TokenizerConfig
2. Modify PythonImportDetector to take TokenizerConfig
3. Update all call sites to pass config
4. Update tests

### Phase 3: Integration

1. Update training loop to create and save TokenizerConfig
2. Update generation code to load and use TokenizerConfig
3. Add integration tests with different tokenizers

### Phase 4: Documentation

1. Document how to use with custom tokenizers
2. Add examples for cl100k_base (GPT-4 tokenizer)
3. Document migration from old hardcoded IDs

## Testing Requirements

### Unit Tests

**test_tokenizer_config.py**:

```python
def test_gpt2_defaults():
    """Test GPT-2 hardcoded defaults."""
    config = TokenizerConfig.gpt2_defaults()
    assert config.bracket_open_ids == [58, 685]
    assert config.bracket_close_paren_open_id == 16151
    # ...

def test_discover_from_gpt2():
    """Test automatic discovery matches hardcoded defaults."""
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    
    discovered = TokenizerConfig.from_tokenizer(enc, name="gpt2")
    defaults = TokenizerConfig.gpt2_defaults()
    
    assert discovered.bracket_open_ids == defaults.bracket_open_ids
    assert discovered.paren_close_id == defaults.paren_close_id
    # ...

def test_discover_from_cl100k():
    """Test discovery with GPT-4 tokenizer."""
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    
    config = TokenizerConfig.from_tokenizer(enc, name="cl100k_base")
    
    # Verify the discovered IDs are different from GPT-2
    gpt2 = TokenizerConfig.gpt2_defaults()
    assert config.paren_close_id != gpt2.paren_close_id  # Likely different
    
    # Verify they actually encode/decode correctly
    assert enc.decode([config.paren_close_id]) == ')'
    assert enc.decode([config.bracket_close_paren_open_id]) == ']('

def test_serialization():
    """Test saving/loading config."""
    config = TokenizerConfig.gpt2_defaults()
    
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        save_tokenizer_config(config, f.name)
        loaded = load_tokenizer_config(f.name)
    
    assert loaded.bracket_open_ids == config.bracket_open_ids
    # ...
```

### Integration Tests

1. Train small model with GPT-2 tokenizer + auto-discovered config
2. Train small model with cl100k_base tokenizer + auto-discovered config
3. Verify link detection works correctly in both cases
4. Verify configs are saved and can be loaded

## Open Questions

1. **What if '](' is multi-token in some tokenizer?**
   - Current assumption: It's a single token
   - Reality: Some tokenizers might split it
   - Solution: Detection algorithm needs to handle multi-token sequences
   - For MVP: Raise error if multi-token, document limitation

2. **Should we validate discovered IDs?**
   - After discovery, decode each ID and verify it matches expected text
   - This catches bugs but adds overhead
   - Proposal: Yes, but make it optional with `validate=True` parameter

3. **How to handle tokenizers that don't have our required tokens?**
   - Example: A character-level tokenizer won't have '](' as single token
   - Proposal: Document minimum requirements, raise clear error if not met
   - Future: Support multi-token detection sequences

4. **Should TokenizerConfig be saved with model checkpoints?**
   - Yes - needed for inference
   - Already planning to save in run_dir/tokenizer_config.json
   - Should also be embedded in model metadata?

5. **Do we need separate configs for training vs generation?**
   - Training: Needs all tokens (markdown + python)
   - Generation: Might only need markdown (if not generating code)
   - Proposal: Single config has all tokens, use what you need

## Success Criteria

- [ ] TokenizerConfig class implemented and tested
- [ ] Automatic token discovery works for gpt2 and cl100k_base
- [ ] Link detectors updated to use TokenizerConfig
- [ ] No hardcoded token IDs remain in link detector code
- [ ] Training loop creates and saves tokenizer config
- [ ] Config is loaded during generation/inference
- [ ] Tests pass with multiple tokenizers
- [ ] Documentation explains how to use custom tokenizers
- [ ] Backwards compatibility maintained (gpt2_defaults())

## Future Extensions

1. **Multi-token sequence detection**
   - Support tokenizers where '](' is multiple tokens
   - Requires updating detection algorithm

2. **Tokenizer-specific optimizations**
   - Some tokenizers might have better patterns for detection
   - Could have tokenizer-specific detector implementations

3. **Language-specific token configs**
   - Rust: `::`
   - Java: `.` in imports
   - C++: `#include <>`
   - Each language might need different token patterns
