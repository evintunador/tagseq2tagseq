"""
Unit tests for token sampling utilities.
"""
import pytest
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from experiments.dagseq2dagseq.model.sampling import (
    greedy_sample,
    sample_token,
)


class TestGreedySample:
    """Tests for greedy_sample function."""
    
    def test_basic_greedy(self):
        """Test basic greedy sampling."""
        logits = torch.tensor([1.0, 5.0, 2.0, 0.5])
        token = greedy_sample(logits)
        assert token == 1  # Index of max value
    
    def test_greedy_with_batch_dim(self):
        """Test greedy sampling with batch dimension."""
        logits = torch.tensor([[1.0, 5.0, 2.0, 0.5]])
        token = greedy_sample(logits)
        assert token == 1
    
    def test_greedy_tie(self):
        """Test greedy when there's a tie (should pick first)."""
        logits = torch.tensor([5.0, 5.0, 1.0])
        token = greedy_sample(logits)
        assert token in [0, 1]  # Either is valid
    
    def test_greedy_negative_logits(self):
        """Test greedy with negative logits."""
        logits = torch.tensor([-1.0, -0.5, -2.0])
        token = greedy_sample(logits)
        assert token == 1  # Index of max (least negative)


class TestSampleToken:
    """Tests for sample_token function."""
    
    def test_temperature_zero_is_greedy(self):
        """Test that temperature=0 gives greedy sampling."""
        logits = torch.tensor([1.0, 5.0, 2.0, 0.5])
        token = sample_token(logits, temperature=0.0)
        assert token == 1  # Should match greedy
    
    def test_temperature_one(self):
        """Test sampling with temperature=1.0."""
        torch.manual_seed(42)
        logits = torch.tensor([1.0, 5.0, 2.0, 0.5])
        token = sample_token(logits, temperature=1.0)
        assert 0 <= token < 4
        assert isinstance(token, int)
    
    def test_high_temperature_flattens_distribution(self):
        """Test that high temperature flattens the distribution."""
        torch.manual_seed(42)
        # With very high temperature, all tokens should be more equally likely
        # Use a less extreme difference in logits
        logits = torch.tensor([1.0, 3.0, 1.0, 1.0])
        
        # Sample multiple times and check distribution
        samples = [sample_token(logits.clone(), temperature=5.0) for _ in range(100)]
        
        # With high temperature, we should see variety (not just token 1)
        unique_tokens = set(samples)
        assert len(unique_tokens) > 1
    
    def test_low_temperature_sharpens_distribution(self):
        """Test that low temperature sharpens the distribution."""
        torch.manual_seed(42)
        # With low temperature, should almost always pick the max
        logits = torch.tensor([1.0, 5.0, 2.0, 0.5])
        
        samples = [sample_token(logits.clone(), temperature=0.1) for _ in range(20)]
        
        # Should almost always pick token 1 (the max)
        assert samples.count(1) > 15
    
    def test_top_k_filtering(self):
        """Test top-k filtering."""
        torch.manual_seed(42)
        logits = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        
        # With top_k=2, should only sample from indices 3 and 4 (highest logits)
        samples = [sample_token(logits.clone(), temperature=1.0, top_k=2) for _ in range(50)]
        
        # Should only contain tokens 3 and 4
        unique_tokens = set(samples)
        assert unique_tokens.issubset({3, 4})
    
    def test_top_p_filtering(self):
        """Test nucleus (top-p) sampling."""
        torch.manual_seed(42)
        # Create logits where one token dominates
        logits = torch.tensor([1.0, 1.0, 1.0, 10.0])
        
        # With top_p=0.9, should mostly sample from the high-probability token
        samples = [sample_token(logits.clone(), temperature=1.0, top_p=0.9) for _ in range(50)]
        
        # Token 3 should appear very frequently
        assert samples.count(3) > 30
    
    def test_top_k_and_top_p_combined(self):
        """Test combining top-k and top-p filtering."""
        torch.manual_seed(42)
        logits = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        
        token = sample_token(logits, temperature=1.0, top_k=3, top_p=0.9)
        assert 0 <= token < 5
        assert isinstance(token, int)
    
    def test_batch_dimension_handling(self):
        """Test handling of batch dimension."""
        torch.manual_seed(42)
        logits_1d = torch.tensor([1.0, 5.0, 2.0])
        logits_2d = torch.tensor([[1.0, 5.0, 2.0]])
        
        token_1d = sample_token(logits_1d, temperature=1.0)
        token_2d = sample_token(logits_2d, temperature=1.0)
        
        # Both should work and return integers
        assert isinstance(token_1d, int)
        assert isinstance(token_2d, int)
    
    def test_all_negative_logits(self):
        """Test sampling with all negative logits."""
        torch.manual_seed(42)
        logits = torch.tensor([-10.0, -5.0, -8.0])
        token = sample_token(logits, temperature=1.0)
        assert 0 <= token < 3
    
    def test_uniform_logits(self):
        """Test sampling with uniform logits."""
        torch.manual_seed(42)
        logits = torch.tensor([1.0, 1.0, 1.0, 1.0])
        
        # Should sample roughly uniformly
        samples = [sample_token(logits.clone(), temperature=1.0) for _ in range(100)]
        unique_tokens = set(samples)
        
        # Should see all tokens at least once
        assert len(unique_tokens) >= 3


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_single_token_vocab(self):
        """Test with single token vocabulary."""
        logits = torch.tensor([1.0])
        token = sample_token(logits, temperature=1.0)
        assert token == 0
    
    def test_top_k_larger_than_vocab(self):
        """Test top_k larger than vocabulary size."""
        torch.manual_seed(42)
        logits = torch.tensor([1.0, 2.0, 3.0])
        token = sample_token(logits, temperature=1.0, top_k=100)
        assert 0 <= token < 3
    
    def test_top_k_zero(self):
        """Test top_k=0 (should disable filtering)."""
        torch.manual_seed(42)
        logits = torch.tensor([1.0, 2.0, 3.0])
        token = sample_token(logits, temperature=1.0, top_k=0)
        assert 0 <= token < 3
    
    def test_top_p_one(self):
        """Test top_p=1.0 (should include all tokens)."""
        torch.manual_seed(42)
        logits = torch.tensor([1.0, 2.0, 3.0])
        token = sample_token(logits, temperature=1.0, top_p=1.0)
        assert 0 <= token < 3
    
    def test_reproducibility_with_seed(self):
        """Test that setting seed makes sampling reproducible."""
        logits = torch.tensor([1.0, 2.0, 3.0, 4.0])
        
        torch.manual_seed(123)
        token1 = sample_token(logits.clone(), temperature=1.0)
        
        torch.manual_seed(123)
        token2 = sample_token(logits.clone(), temperature=1.0)
        
        assert token1 == token2
