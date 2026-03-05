"""
Tests for the generation loop and TS2TSModel.generate().

CPU tests use a mock model so they run without CUDA.
CUDA tests exercise the full forward_inference + generate() path.
"""
import pytest
import torch
import numpy as np
from unittest.mock import MagicMock
from torch.nn.attention.flex_attention import create_block_mask, BlockMask

from model.generation_config import GenerationConfig
from model.generation_result import GenerationResult, GeneratedDocument
from model.generation_loop import run_generation, _generate_doc
from model.document_context import DocumentContext


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

VOCAB_SIZE = 512
EOS = VOCAB_SIZE - 1  # last token acts as EOS in mock tests


def make_mock_model(next_tokens):
    """
    Returns a model whose forward_inference yields tokens from next_tokens in
    order, then EOS.  Logits are constructed so argmax == the desired token.
    """
    call_count = [0]

    def forward_inference(tokens, doc_spans=None, **kwargs):
        idx = call_count[0]
        call_count[0] += 1
        token = next_tokens[idx] if idx < len(next_tokens) else EOS
        # build logits: put a large value at the target token position
        logits = torch.full((1, tokens.shape[1], VOCAB_SIZE), -1e9)
        logits[0, -1, token] = 1e9
        return logits

    model = MagicMock()
    model.forward_inference.side_effect = forward_inference
    return model


def make_mock_tokenizer(tokens_for_prompt=None):
    tok = MagicMock()
    tok.encode.return_value = tokens_for_prompt if tokens_for_prompt is not None else [1, 2, 3]
    tok.decode.side_effect = lambda ids: f"<decoded:{ids}>"
    return tok


def base_config(**overrides):
    kwargs = dict(
        max_new_tokens=10,
        temperature=0.0,  # greedy
        max_tokens_per_document=512,
        max_context_length=1024,
        max_auxiliary_documents=6,
        eviction_policy="drop_oldest",
        device="cpu",
        eos_token_id=EOS,
    )
    kwargs.update(overrides)
    return GenerationConfig(**kwargs)


# ---------------------------------------------------------------------------
# run_generation — CPU mock tests
# ---------------------------------------------------------------------------

def test_run_generation_returns_generation_result():
    model = make_mock_model(next_tokens=[100, 101, 102, EOS])
    result = run_generation(
        model=model,
        prompt_tokens=[1, 2, 3],
        corpus=None,
        config=base_config(),
        link_detector=None,
        tokenizer_decode=None,
        layout_policy=None,
    )
    assert isinstance(result, GenerationResult)


def test_run_generation_root_is_generated():
    model = make_mock_model(next_tokens=[100, EOS])
    result = run_generation(
        model=model,
        prompt_tokens=[1, 2],
        corpus=None,
        config=base_config(),
        link_detector=None,
        tokenizer_decode=None,
        layout_policy=None,
    )
    assert result.root_document.is_root is True
    assert result.root_document.source == "generated"


def test_run_generation_no_aux_docs_in_stage_1():
    model = make_mock_model(next_tokens=[EOS])
    result = run_generation(
        model=model,
        prompt_tokens=[1],
        corpus=None,
        config=base_config(),
        link_detector=None,
        tokenizer_decode=None,
        layout_policy=None,
    )
    assert result.auxiliary_documents == []


def test_run_generation_stops_on_eos():
    model = make_mock_model(next_tokens=[10, 20, EOS, 30, 40])
    result = run_generation(
        model=model,
        prompt_tokens=[1],
        corpus=None,
        config=base_config(max_new_tokens=100),
        link_detector=None,
        tokenizer_decode=None,
        layout_policy=None,
    )
    doc = result.root_document
    # prompt(1) + 10 + 20 + EOS = 4 tokens
    assert doc.tokens.tolist() == [1, 10, 20, EOS]


def test_run_generation_stops_on_max_new_tokens():
    # Never emits EOS; should stop after max_new_tokens
    model = make_mock_model(next_tokens=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    result = run_generation(
        model=model,
        prompt_tokens=[1],
        corpus=None,
        config=base_config(max_new_tokens=3),
        link_detector=None,
        tokenizer_decode=None,
        layout_policy=None,
    )
    doc = result.root_document
    # prompt(1) + 3 generated = 4 tokens; truncated should NOT be set
    assert len(doc.tokens) == 4
    assert doc.truncated is False


def test_run_generation_stops_on_max_tokens_per_document():
    # Never emits EOS; max_tokens_per_document = 3 (includes prompt)
    model = make_mock_model(next_tokens=[10, 20, 30])
    result = run_generation(
        model=model,
        prompt_tokens=[1, 2],  # 2 prompt tokens
        corpus=None,
        config=base_config(max_new_tokens=100, max_tokens_per_document=3),
        link_detector=None,
        tokenizer_decode=None,
        layout_policy=None,
    )
    doc = result.root_document
    # prompt(2) + 1 generated = 3, then truncated
    assert len(doc.tokens) == 3
    assert doc.truncated is True


def test_run_generation_populates_text_when_decode_given():
    model = make_mock_model(next_tokens=[EOS])
    result = run_generation(
        model=model,
        prompt_tokens=[1],
        corpus=None,
        config=base_config(),
        link_detector=None,
        tokenizer_decode=lambda ids: "hello",
        layout_policy=None,
    )
    assert result.root_document.text == "hello"


def test_run_generation_text_none_when_no_decode():
    model = make_mock_model(next_tokens=[EOS])
    result = run_generation(
        model=model,
        prompt_tokens=[1],
        corpus=None,
        config=base_config(),
        link_detector=None,
        tokenizer_decode=None,
        layout_policy=None,
    )
    assert result.root_document.text is None


def test_run_generation_config_stored_in_result():
    model = make_mock_model(next_tokens=[EOS])
    cfg = base_config(max_new_tokens=7)
    result = run_generation(
        model=model,
        prompt_tokens=[1],
        corpus=None,
        config=cfg,
        link_detector=None,
        tokenizer_decode=None,
        layout_policy=None,
    )
    assert result.generation_config["max_new_tokens"] == 7


def test_run_generation_single_token_prompt():
    """Single-token prompt is the minimum — generation should work."""
    model = make_mock_model(next_tokens=[5, EOS])
    result = run_generation(
        model=model,
        prompt_tokens=[1],
        corpus=None,
        config=base_config(),
        link_detector=None,
        tokenizer_decode=None,
        layout_policy=None,
    )
    # prompt(1) + 5 + EOS = 3 tokens
    assert len(result.root_document.tokens) == 3


def test_run_generation_forward_inference_called_per_token():
    """forward_inference must be called once per token generated."""
    model = make_mock_model(next_tokens=[10, 20, EOS])
    run_generation(
        model=model,
        prompt_tokens=[1],
        corpus=None,
        config=base_config(),
        link_detector=None,
        tokenizer_decode=None,
        layout_policy=None,
    )
    # 3 tokens generated (10, 20, EOS) → 3 forward_inference calls
    assert model.forward_inference.call_count == 3


def test_run_generation_tokens_grow_each_step():
    """The token tensor passed to forward_inference grows by 1 each step.

    forward_inference is called with the current sequence, then the sampled
    token is appended.  So the lengths seen are:
      call 1: len(prompt)           = 2  → sample token A, append
      call 2: len(prompt) + 1       = 3  → sample token B, append
      call 3: len(prompt) + 2       = 4  → sample EOS, append, done
    """
    observed_lengths = []

    def forward_inference(tokens, doc_spans=None, **kwargs):
        observed_lengths.append(tokens.shape[1])
        token = EOS if len(observed_lengths) >= 3 else 10
        logits = torch.full((1, tokens.shape[1], VOCAB_SIZE), -1e9)
        logits[0, -1, token] = 1e9
        return logits

    model = MagicMock()
    model.forward_inference.side_effect = forward_inference

    run_generation(
        model=model,
        prompt_tokens=[1, 2],  # length 2
        corpus=None,
        config=base_config(),
        link_detector=None,
        tokenizer_decode=None,
        layout_policy=None,
    )
    assert observed_lengths == [2, 3, 4]


# ---------------------------------------------------------------------------
# TS2TSModel.generate() — CPU mock tests (no CUDA, no block mask)
# ---------------------------------------------------------------------------

def test_model_generate_requires_tokenizer():
    from model.model import TS2TSModel
    from model.modules.backbone import TS2TSBackbone
    from tunalab.modules.norms.rms_norm import RMSNorm
    import torch.nn as nn

    backbone = TS2TSBackbone(num_layers=1, model_dim=64, num_heads=2,
                              max_seq_len=128, dropout=0.0, drop_path_rate=0.0)
    emb = nn.Embedding(VOCAB_SIZE, 64)
    norm = RMSNorm(64)

    model = TS2TSModel(
        backbone=backbone,
        embedding_weight=emb.weight,
        lm_head_weight=emb.weight,
        norm=norm,
        block_mask_creator=lambda **kw: None,
        vocab_size=VOCAB_SIZE,
        tokenizer=None,
    )
    with pytest.raises(RuntimeError, match="tokenizer"):
        model.generate("hello")


# ---------------------------------------------------------------------------
# CUDA integration tests
# ---------------------------------------------------------------------------

def simple_causal_mask_creator(**batch):
    tokens = batch["tokens"]
    seq_len = tokens.shape[-1]
    def causal(b, h, q, kv):
        return q >= kv
    return create_block_mask(causal, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len,
                              device=tokens.device)


@pytest.fixture
def small_inference_model(device):
    """A tiny TS2TSModel with random weights, no tokenizer."""
    if device != "cuda":
        pytest.skip("FlexAttention requires CUDA")

    from model.model import TS2TSModel
    model = TS2TSModel.from_config(
        vocab_size=VOCAB_SIZE,
        num_layers=2,
        model_dim=64,
        num_heads=2,
        max_seq_len=128,
        dropout=0.0,
        drop_path_rate=0.0,
        block_mask_creator=simple_causal_mask_creator,
        weight_tying=True,
    )
    model.to(torch.device(device))
    return model


def test_forward_inference_output_shape(small_inference_model, device):
    tokens = torch.randint(0, VOCAB_SIZE, (1, 10), device=device)
    logits = small_inference_model.forward_inference(tokens)
    assert logits.shape == (1, 10, VOCAB_SIZE)


def test_forward_inference_no_nan(small_inference_model, device):
    tokens = torch.randint(0, VOCAB_SIZE, (1, 10), device=device)
    logits = small_inference_model.forward_inference(tokens)
    assert not torch.isnan(logits).any()
    assert not torch.isinf(logits).any()


def test_forward_inference_with_doc_spans(small_inference_model, device):
    from data.collate import DocSpan
    tokens = torch.randint(0, VOCAB_SIZE, (1, 20), device=device)
    doc_spans = [
        DocSpan(doc_id=0, normed_identifier="doc0", start=0, end=20,
                truncated=False, outgoing_identifiers=[], raw_identifier="doc0"),
    ]
    logits = small_inference_model.forward_inference(tokens, doc_spans)
    assert logits.shape == (1, 20, VOCAB_SIZE)


def test_generate_with_tokenizer_terminates(device):
    """Full generate() call with random weights should terminate without error."""
    if device != "cuda":
        pytest.skip("FlexAttention requires CUDA")

    from model.model import TS2TSModel

    model = TS2TSModel.from_config(
        vocab_size=VOCAB_SIZE,
        num_layers=2,
        model_dim=64,
        num_heads=2,
        max_seq_len=128,
        dropout=0.0,
        drop_path_rate=0.0,
        block_mask_creator=simple_causal_mask_creator,
        weight_tying=True,
        tokenizer=make_mock_tokenizer([10, 20]),
    )
    model.to(torch.device(device))

    config = GenerationConfig(
        max_new_tokens=5,
        temperature=1.0,
        device=device,
        eos_token_id=EOS,
        max_tokens_per_document=512,
        max_context_length=1024,
    )
    result = model.generate("hello world", config=config)

    assert isinstance(result, GenerationResult)
    assert result.root_document.is_root is True
    assert result.auxiliary_documents == []


def test_generate_result_has_text(device):
    """With a tokenizer, GeneratedDocument.text should be populated."""
    if device != "cuda":
        pytest.skip("FlexAttention requires CUDA")

    from model.model import TS2TSModel

    model = TS2TSModel.from_config(
        vocab_size=VOCAB_SIZE,
        num_layers=2,
        model_dim=64,
        num_heads=2,
        max_seq_len=128,
        dropout=0.0,
        drop_path_rate=0.0,
        block_mask_creator=simple_causal_mask_creator,
        weight_tying=True,
        tokenizer=make_mock_tokenizer([10, 20]),
    )
    model.to(torch.device(device))

    result = model.generate("test", config=GenerationConfig(
        max_new_tokens=3, device=device, eos_token_id=EOS,
        max_context_length=1024, max_tokens_per_document=512,
    ))
    assert result.root_document.text is not None
    assert isinstance(result.root_document.text, str)
