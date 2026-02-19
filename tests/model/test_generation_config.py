"""
Unit tests for GenerationConfig dataclass.
"""
import pytest
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from experiments.dagseq2dagseq.model.generation_config import GenerationConfig


class TestGenerationConfigDefaults:
    """Test default values and basic instantiation."""
    
    def test_default_instantiation(self):
        """Test creating config with defaults."""
        config = GenerationConfig()
        
        assert config.max_new_tokens == 100
        assert config.temperature == 1.0
        assert config.top_k is None
        assert config.top_p is None
        assert config.max_tokens_per_document == 512
        assert config.max_context_length == 2048
        assert config.max_auxiliary_documents == 5
        assert config.max_link_depth == 1
        assert config.allow_corpus_fallback is True
        assert config.eviction_policy == "drop_oldest"
        assert config.process_prompt_links is True
        assert config.allow_recursive_links is True
        assert config.eos_token_id == 50256
        assert config.device in ["cuda", "cpu"]
    
    def test_custom_values(self):
        """Test creating config with custom values."""
        config = GenerationConfig(
            max_new_tokens=200,
            temperature=0.8,
            top_k=50,
            top_p=0.9,
            max_tokens_per_document=1024,
            max_context_length=4096,
            max_auxiliary_documents=10,
            max_link_depth=2,
            allow_corpus_fallback=False,
            eviction_policy="stop_new",
            process_prompt_links=False,
            allow_recursive_links=False,
            eos_token_id=0,
            device="cpu",
        )
        
        assert config.max_new_tokens == 200
        assert config.temperature == 0.8
        assert config.top_k == 50
        assert config.top_p == 0.9
        assert config.max_tokens_per_document == 1024
        assert config.max_context_length == 4096
        assert config.max_auxiliary_documents == 10
        assert config.max_link_depth == 2
        assert config.allow_corpus_fallback is False
        assert config.eviction_policy == "stop_new"
        assert config.process_prompt_links is False
        assert config.allow_recursive_links is False
        assert config.eos_token_id == 0
        assert config.device == "cpu"


class TestGenerationConfigValidation:
    """Test validation logic in __post_init__."""
    
    def test_negative_max_new_tokens(self):
        """Test that negative max_new_tokens raises error."""
        with pytest.raises(ValueError, match="max_new_tokens must be positive"):
            GenerationConfig(max_new_tokens=-1)
    
    def test_zero_max_new_tokens(self):
        """Test that zero max_new_tokens raises error."""
        with pytest.raises(ValueError, match="max_new_tokens must be positive"):
            GenerationConfig(max_new_tokens=0)
    
    def test_negative_temperature(self):
        """Test that negative temperature raises error."""
        with pytest.raises(ValueError, match="temperature must be non-negative"):
            GenerationConfig(temperature=-0.1)
    
    def test_zero_temperature_allowed(self):
        """Test that temperature=0 is allowed (greedy sampling)."""
        config = GenerationConfig(temperature=0.0)
        assert config.temperature == 0.0
    
    def test_negative_top_k(self):
        """Test that negative top_k raises error."""
        with pytest.raises(ValueError, match="top_k must be positive"):
            GenerationConfig(top_k=-1)
    
    def test_zero_top_k(self):
        """Test that zero top_k raises error."""
        with pytest.raises(ValueError, match="top_k must be positive"):
            GenerationConfig(top_k=0)
    
    def test_top_k_none_allowed(self):
        """Test that top_k=None is allowed."""
        config = GenerationConfig(top_k=None)
        assert config.top_k is None
    
    def test_top_p_too_low(self):
        """Test that top_p <= 0 raises error."""
        with pytest.raises(ValueError, match="top_p must be in"):
            GenerationConfig(top_p=0.0)
        
        with pytest.raises(ValueError, match="top_p must be in"):
            GenerationConfig(top_p=-0.1)
    
    def test_top_p_too_high(self):
        """Test that top_p > 1 raises error."""
        with pytest.raises(ValueError, match="top_p must be in"):
            GenerationConfig(top_p=1.5)
    
    def test_top_p_one_allowed(self):
        """Test that top_p=1.0 is allowed."""
        config = GenerationConfig(top_p=1.0)
        assert config.top_p == 1.0
    
    def test_top_p_none_allowed(self):
        """Test that top_p=None is allowed."""
        config = GenerationConfig(top_p=None)
        assert config.top_p is None
    
    def test_negative_max_tokens_per_document(self):
        """Test that negative max_tokens_per_document raises error."""
        with pytest.raises(ValueError, match="max_tokens_per_document must be positive"):
            GenerationConfig(max_tokens_per_document=-1)
    
    def test_negative_max_context_length(self):
        """Test that negative max_context_length raises error."""
        with pytest.raises(ValueError, match="max_context_length must be positive"):
            GenerationConfig(max_context_length=-1)
    
    def test_negative_max_auxiliary_documents(self):
        """Test that negative max_auxiliary_documents raises error."""
        with pytest.raises(ValueError, match="max_auxiliary_documents must be non-negative"):
            GenerationConfig(max_auxiliary_documents=-1)
    
    def test_zero_max_auxiliary_documents_allowed(self):
        """Test that zero max_auxiliary_documents is allowed."""
        config = GenerationConfig(max_auxiliary_documents=0)
        assert config.max_auxiliary_documents == 0
    
    def test_negative_max_link_depth(self):
        """Test that negative max_link_depth raises error."""
        with pytest.raises(ValueError, match="max_link_depth must be non-negative"):
            GenerationConfig(max_link_depth=-1)
    
    def test_zero_max_link_depth_allowed(self):
        """Test that zero max_link_depth is allowed."""
        config = GenerationConfig(max_link_depth=0)
        assert config.max_link_depth == 0
    
    def test_invalid_eviction_policy(self):
        """Test that invalid eviction_policy raises error."""
        with pytest.raises(ValueError, match="eviction_policy must be"):
            GenerationConfig(eviction_policy="invalid")


class TestGenerationConfigMethods:
    """Test methods of GenerationConfig."""
    
    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = GenerationConfig(
            max_new_tokens=200,
            temperature=0.8,
            top_k=50,
        )
        
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict["max_new_tokens"] == 200
        assert config_dict["temperature"] == 0.8
        assert config_dict["top_k"] == 50
        
        # Check that all fields are present
        expected_fields = [
            "max_new_tokens", "temperature", "top_k", "top_p",
            "max_tokens_per_document", "max_context_length",
            "max_auxiliary_documents", "max_link_depth",
            "allow_corpus_fallback", "eviction_policy",
            "process_prompt_links", "allow_recursive_links",
            "eos_token_id", "device"
        ]
        
        for field in expected_fields:
            assert field in config_dict


class TestEvictionPolicies:
    """Test different eviction policies."""
    
    def test_drop_oldest_policy(self):
        """Test drop_oldest eviction policy."""
        config = GenerationConfig(eviction_policy="drop_oldest")
        assert config.eviction_policy == "drop_oldest"
    
    def test_stop_new_policy(self):
        """Test stop_new eviction policy."""
        config = GenerationConfig(eviction_policy="stop_new")
        assert config.eviction_policy == "stop_new"


class TestDeviceHandling:
    """Test device configuration."""
    
    def test_explicit_cpu_device(self):
        """Test setting device to CPU explicitly."""
        config = GenerationConfig(device="cpu")
        assert config.device == "cpu"
    
    def test_explicit_cuda_device(self):
        """Test setting device to CUDA explicitly."""
        config = GenerationConfig(device="cuda")
        assert config.device == "cuda"
    
    def test_default_device(self):
        """Test that default device is set correctly."""
        config = GenerationConfig()
        expected = "cuda" if torch.cuda.is_available() else "cpu"
        assert config.device == expected
