"""
Tests for TokenizerConfig class.
"""
import unittest
import tempfile
import json
from pathlib import Path

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

from model.tokenizer_config import TokenizerConfig


class TestTokenizerConfigDefaults(unittest.TestCase):
    """Test hardcoded GPT-2 defaults."""
    
    def test_gpt2_defaults(self):
        """Test GPT-2 hardcoded defaults match expected values."""
        config = TokenizerConfig.gpt2_defaults()
        
        self.assertEqual(config.bracket_open_ids, [58, 685])
        self.assertEqual(config.bracket_close_paren_open_id, 16151)
        self.assertEqual(config.paren_close_id, 8)
        self.assertEqual(config.import_keyword_id, 1330)
        self.assertEqual(config.from_keyword_id, 6738)
        self.assertEqual(config.newline_id, 198)
        self.assertEqual(config.space_id, 220)
        self.assertEqual(config.dot_id, 13)
        self.assertEqual(config.name, "gpt2")
        self.assertEqual(config.vocab_size, 50257)


@unittest.skipUnless(TIKTOKEN_AVAILABLE, "tiktoken not installed")
class TestTokenizerDiscovery(unittest.TestCase):
    """Test automatic token ID discovery from tokenizers."""
    
    def test_discover_from_gpt2(self):
        """Test automatic discovery works for GPT-2 and produces valid tokens."""
        enc = tiktoken.get_encoding("gpt2")
        
        discovered = TokenizerConfig.from_tokenizer(enc, name="gpt2")
        
        # Test that discovered IDs actually decode correctly
        self.assertIn('[', enc.decode([discovered.bracket_open_ids[0]]))
        self.assertEqual(enc.decode([discovered.bracket_close_paren_open_id]), '](')
        self.assertEqual(enc.decode([discovered.paren_close_id]), ')')
        self.assertIn('import', enc.decode([discovered.import_keyword_id]).lower())
        self.assertIn('from', enc.decode([discovered.from_keyword_id]).lower())
        self.assertEqual(enc.decode([discovered.newline_id]), '\n')
        self.assertEqual(enc.decode([discovered.space_id]), ' ')
        self.assertEqual(enc.decode([discovered.dot_id]), '.')
        self.assertEqual(discovered.vocab_size, 50257)
    
    def test_discover_from_cl100k(self):
        """Test discovery with GPT-4 tokenizer (cl100k_base)."""
        enc = tiktoken.get_encoding("cl100k_base")
        
        config = TokenizerConfig.from_tokenizer(enc, name="cl100k_base")
        
        # Verify config was created
        self.assertEqual(config.name, "cl100k_base")
        self.assertIsInstance(config.bracket_open_ids, list)
        self.assertGreater(len(config.bracket_open_ids), 0)
        
        # Verify they actually encode/decode correctly
        self.assertEqual(enc.decode([config.paren_close_id]), ')')
        self.assertIn(enc.decode([config.bracket_open_ids[0]]), ['[', ' ['])
        
        # Verify discovered IDs work for encoding
        self.assertIn(config.paren_close_id, enc.encode(')'))
    
    def test_bracket_variants_discovered(self):
        """Test that both '[' and ' [' variants are discovered."""
        enc = tiktoken.get_encoding("gpt2")
        config = TokenizerConfig.from_tokenizer(enc, name="gpt2")
        
        # Should have at least one bracket variant
        self.assertGreater(len(config.bracket_open_ids), 0)
        
        # All variants should decode to something with '['
        for bracket_id in config.bracket_open_ids:
            decoded = enc.decode([bracket_id])
            self.assertIn('[', decoded)
    
    def test_keyword_discovery(self):
        """Test Python keyword token discovery."""
        enc = tiktoken.get_encoding("gpt2")
        config = TokenizerConfig.from_tokenizer(enc, name="gpt2")
        
        # Check that keywords decode correctly
        import_decoded = enc.decode([config.import_keyword_id])
        from_decoded = enc.decode([config.from_keyword_id])
        
        self.assertIn('import', import_decoded.lower())
        self.assertIn('from', from_decoded.lower())


class TestTokenizerSerialization(unittest.TestCase):
    """Test saving and loading tokenizer configs."""
    
    def test_to_dict(self):
        """Test serialization to dictionary."""
        config = TokenizerConfig.gpt2_defaults()
        data = config.to_dict()
        
        self.assertIsInstance(data, dict)
        self.assertEqual(data['bracket_open_ids'], [58, 685])
        self.assertEqual(data['bracket_close_paren_open_id'], 16151)
        self.assertEqual(data['name'], 'gpt2')
        self.assertEqual(data['vocab_size'], 50257)
    
    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            'bracket_open_ids': [58, 685],
            'bracket_close_paren_open_id': 16151,
            'paren_close_id': 8,
            'import_keyword_id': 1330,
            'from_keyword_id': 6738,
            'newline_id': 198,
            'space_id': 220,
            'dot_id': 13,
            'name': 'gpt2',
            'vocab_size': 50257,
        }
        
        config = TokenizerConfig.from_dict(data)
        
        self.assertEqual(config.bracket_open_ids, [58, 685])
        self.assertEqual(config.name, 'gpt2')
        self.assertEqual(config.vocab_size, 50257)
    
    def test_round_trip_serialization(self):
        """Test that config survives round-trip serialization."""
        original = TokenizerConfig.gpt2_defaults()
        data = original.to_dict()
        restored = TokenizerConfig.from_dict(data)
        
        self.assertEqual(original.bracket_open_ids, restored.bracket_open_ids)
        self.assertEqual(original.bracket_close_paren_open_id, restored.bracket_close_paren_open_id)
        self.assertEqual(original.paren_close_id, restored.paren_close_id)
        self.assertEqual(original.import_keyword_id, restored.import_keyword_id)
        self.assertEqual(original.from_keyword_id, restored.from_keyword_id)
        self.assertEqual(original.newline_id, restored.newline_id)
        self.assertEqual(original.space_id, restored.space_id)
        self.assertEqual(original.dot_id, restored.dot_id)
        self.assertEqual(original.name, restored.name)
        self.assertEqual(original.vocab_size, restored.vocab_size)
    
    def test_save_and_load(self):
        """Test saving to and loading from file."""
        config = TokenizerConfig.gpt2_defaults()
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = Path(f.name)
        
        try:
            # Save
            config.save(temp_path)
            
            # Verify file exists and is valid JSON
            self.assertTrue(temp_path.exists())
            with open(temp_path, 'r') as f:
                data = json.load(f)
                self.assertIn('bracket_open_ids', data)
            
            # Load
            loaded = TokenizerConfig.load(temp_path)
            
            # Verify loaded config matches original
            self.assertEqual(loaded.bracket_open_ids, config.bracket_open_ids)
            self.assertEqual(loaded.name, config.name)
            self.assertEqual(loaded.vocab_size, config.vocab_size)
        finally:
            temp_path.unlink()


class TestTokenizerConfigEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    @unittest.skipUnless(TIKTOKEN_AVAILABLE, "tiktoken not installed")
    def test_missing_vocab_size_attribute(self):
        """Test handling of tokenizer without vocab_size attribute."""
        # Create a mock tokenizer that lacks vocab_size
        class MinimalTokenizer:
            def encode(self, text):
                # Simplified encoding for test
                mapping = {
                    '[': [58],
                    ' [': [220, 58],
                    '](': [16151],
                    ')': [8],
                    'import': [1330],
                    'from': [6738],
                    '\n': [198],
                    ' ': [220],
                    '.': [13],
                }
                return mapping.get(text, [0])
            
            def decode(self, tokens):
                return ''
        
        tokenizer = MinimalTokenizer()
        config = TokenizerConfig.from_tokenizer(tokenizer, name="minimal")
        
        # Should default to -1 when vocab_size can't be determined
        self.assertEqual(config.vocab_size, -1)
    
    def test_custom_tokenizer_config(self):
        """Test creating a config with custom values."""
        config = TokenizerConfig(
            bracket_open_ids=[100, 200],
            bracket_close_paren_open_id=300,
            paren_close_id=400,
            import_keyword_id=500,
            from_keyword_id=600,
            newline_id=700,
            space_id=800,
            dot_id=900,
            name="custom",
            vocab_size=10000,
        )
        
        self.assertEqual(config.name, "custom")
        self.assertEqual(config.vocab_size, 10000)
        self.assertEqual(config.bracket_open_ids, [100, 200])


if __name__ == '__main__':
    unittest.main()
