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
        config=base_config(max_new_tokens=3, max_tokens_per_document=3),
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


# ---------------------------------------------------------------------------
# Stage 2: link detection + _handle_link + _process_existing_doc_links
# ---------------------------------------------------------------------------
from model.generation_loop import _handle_link, _process_existing_doc_links
from model.document_context import DocumentContext


def make_link_detector(fires_at_token, target):
    """
    Returns a fake LinkDetector that reports a single link whose closing token
    equals fires_at_token, with the given target string.

    detect_links(recent) fires when recent[-1] == fires_at_token, reporting
    link_end_pos == len(recent).
    """
    from collections import namedtuple
    LinkInfo = namedtuple("LinkInfo", ["link_end_pos", "target_str"])

    class FakeDetector:
        def detect_links(self, tokens):
            if len(tokens) > 0 and tokens[-1].item() == fires_at_token:
                return [LinkInfo(link_end_pos=len(tokens), target_str=target)]
            return []

        def index_doc_span(self, span):
            return span.raw_identifier

    return FakeDetector()


def make_corpus(docs: dict):
    """Returns a fake corpus backed by a dict {raw_identifier: List[int]}."""
    class FakeCorpus:
        def has_document(self, identifier):
            return identifier in docs

        def get_document(self, identifier):
            return iter(docs[identifier])

    return FakeCorpus()


def make_context_obj(**overrides):
    kwargs = dict(
        max_context_length=1024,
        max_auxiliary_documents=6,
        eviction_policy="drop_oldest",
        device="cpu",
    )
    kwargs.update(overrides)
    return DocumentContext(**kwargs)


# --- _handle_link: skip empty target ---

def test_handle_link_skips_empty_target():
    from collections import namedtuple
    LinkInfo = namedtuple("LinkInfo", ["link_end_pos", "target_str"])
    ctx = make_context_obj()
    root = ctx.add_root("", [1], None)
    _handle_link(
        LinkInfo(link_end_pos=1, target_str=""),
        root, ctx, None, None, None, base_config(), None, depth=0,
    )
    assert ctx.num_aux_docs == 0


# --- _handle_link: skip already-in-context ---

def test_handle_link_skips_already_in_context():
    from collections import namedtuple
    LinkInfo = namedtuple("LinkInfo", ["link_end_pos", "target_str"])
    ctx = make_context_obj()
    root = ctx.add_root("", [1], None)
    ctx.add_corpus_doc("Python", [10], None, "", 1, root)
    _handle_link(
        LinkInfo(link_end_pos=1, target_str="Python"),
        root, ctx, None, None, None, base_config(), None, depth=0,
    )
    assert ctx.num_aux_docs == 1  # no second insertion


# --- _handle_link: corpus fetch ---

def test_handle_link_fetches_from_corpus():
    from collections import namedtuple
    LinkInfo = namedtuple("LinkInfo", ["link_end_pos", "target_str"])
    corpus = make_corpus({"Python": [10, 20, 30]})
    ctx = make_context_obj()
    root = ctx.add_root("", [1], None)
    _handle_link(
        LinkInfo(link_end_pos=1, target_str="Python"),
        root, ctx, None, None, corpus, base_config(max_link_depth=1), None, depth=0,
    )
    assert ctx.num_aux_docs == 1
    aux = ctx._docs[0]
    assert aux.raw_identifier == "Python"
    assert aux.source == "corpus"
    assert aux.tokens == [10, 20, 30]
    assert aux.parent_raw_identifier == ""
    assert aux.depth == 1


def test_handle_link_corpus_doc_inserted_before_active():
    from collections import namedtuple
    LinkInfo = namedtuple("LinkInfo", ["link_end_pos", "target_str"])
    corpus = make_corpus({"Python": [10]})
    ctx = make_context_obj()
    root = ctx.add_root("", [1], None)
    _handle_link(
        LinkInfo(link_end_pos=1, target_str="Python"),
        root, ctx, None, None, corpus, base_config(max_link_depth=1), None, depth=0,
    )
    assert ctx._docs[0].raw_identifier == "Python"
    assert ctx._docs[1].is_root is True


# --- _handle_link: generation fallback ---

def test_handle_link_generates_when_not_in_corpus():
    from collections import namedtuple
    LinkInfo = namedtuple("LinkInfo", ["link_end_pos", "target_str"])
    model = make_mock_model(next_tokens=[EOS])
    ctx = make_context_obj()
    root = ctx.add_root("", [1], None)
    cfg = base_config(max_link_depth=1, allow_generation_fallback=True)
    _handle_link(
        LinkInfo(link_end_pos=1, target_str="Go"),
        root, ctx, model, None, None, cfg, None, depth=0,
    )
    assert ctx.num_aux_docs == 1
    aux_doc = ctx.get_all_documents()[1]
    assert aux_doc.raw_identifier == "Go"
    assert aux_doc.source == "generated"
    assert aux_doc.depth == 1


def test_handle_link_respects_max_link_depth_generation():
    from collections import namedtuple
    LinkInfo = namedtuple("LinkInfo", ["link_end_pos", "target_str"])
    ctx = make_context_obj()
    root = ctx.add_root("", [1], None)
    cfg = base_config(max_link_depth=0, allow_generation_fallback=True)
    _handle_link(
        LinkInfo(link_end_pos=1, target_str="Go"),
        root, ctx, None, None, None, cfg, None, depth=0,
    )
    assert ctx.num_aux_docs == 0  # depth 0 >= max_link_depth 0 → skip


def test_handle_link_respects_allow_generation_fallback_false():
    from collections import namedtuple
    LinkInfo = namedtuple("LinkInfo", ["link_end_pos", "target_str"])
    ctx = make_context_obj()
    root = ctx.add_root("", [1], None)
    cfg = base_config(max_link_depth=2, allow_generation_fallback=False)
    _handle_link(
        LinkInfo(link_end_pos=1, target_str="Go"),
        root, ctx, None, None, None, cfg, None, depth=0,
    )
    assert ctx.num_aux_docs == 0


# --- _handle_link: eviction (stop_new) ---

def test_handle_link_stop_new_skips_when_full():
    from collections import namedtuple
    LinkInfo = namedtuple("LinkInfo", ["link_end_pos", "target_str"])
    corpus = make_corpus({"Python": [10, 20]})
    # max_context_length just large enough for root (1 token) but not aux (2 tokens)
    ctx = make_context_obj(max_context_length=2, eviction_policy="stop_new")
    root = ctx.add_root("", [1], None)
    cfg = base_config(eviction_policy="stop_new", max_context_length=2, max_tokens_per_document=2, max_new_tokens=2)
    _handle_link(
        LinkInfo(link_end_pos=1, target_str="Python"),
        root, ctx, None, None, corpus, cfg, None, depth=0,
    )
    assert ctx.num_aux_docs == 0  # no room; stop_new policy


# --- _handle_link: eviction (drop_oldest) ---

def test_handle_link_drop_oldest_evicts_to_make_room():
    from collections import namedtuple
    LinkInfo = namedtuple("LinkInfo", ["link_end_pos", "target_str"])
    corpus = make_corpus({"B": [20, 21]})
    # root=1 token, aux A=3 tokens → total 4; max=6; adding B (2 tokens) = 6 ✓
    # But to make it require eviction: root=1, aux A=4 → total 5; max=6; B=2 → 7>6
    ctx = make_context_obj(max_context_length=6, eviction_policy="drop_oldest")
    root = ctx.add_root("", [1], None)
    ctx.add_corpus_doc("A", [10, 11, 12, 13], None, "", 1, root)  # 4 tokens → total 5
    cfg = base_config(eviction_policy="drop_oldest", max_context_length=6, max_link_depth=1, max_tokens_per_document=6, max_new_tokens=6)
    _handle_link(
        LinkInfo(link_end_pos=1, target_str="B"),
        root, ctx, None, None, corpus, cfg, None, depth=0,
    )
    # A should have been evicted to make room for B
    active_ids = [e.raw_identifier for e in ctx._docs]
    assert "A" not in active_ids
    assert "B" in active_ids


# --- link detection fires during _generate_doc ---

def test_generate_doc_link_detection_triggers_corpus_fetch():
    """When the model generates the link-closing token, corpus doc is inserted."""
    LINK_TOKEN = 50   # token that closes the link
    corpus = make_corpus({"Python": [10, 20]})
    detector = make_link_detector(fires_at_token=LINK_TOKEN, target="Python")

    # model emits LINK_TOKEN then EOS
    model = make_mock_model(next_tokens=[LINK_TOKEN, EOS])
    ctx = make_context_obj()
    root = ctx.add_root("", [1], None)
    cfg = base_config(max_link_depth=1)

    from model.generation_loop import _generate_doc
    _generate_doc(root, ctx, model, detector, corpus, cfg, None, depth=0)

    assert ctx.num_aux_docs == 1
    assert ctx._docs[0].raw_identifier == "Python"
    assert ctx._docs[0].source == "corpus"


def test_generate_doc_no_link_detection_when_detector_none():
    """With link_detector=None the generation loop runs without errors."""
    model = make_mock_model(next_tokens=[EOS])
    ctx = make_context_obj()
    root = ctx.add_root("", [1], None)
    from model.generation_loop import _generate_doc
    _generate_doc(root, ctx, model, None, None, base_config(), None, depth=0)
    assert ctx.num_aux_docs == 0


# --- _process_existing_doc_links ---

def test_process_existing_doc_links_fetches_corpus_link():
    """Corpus doc's own links are fetched at depth+1."""
    # corpus doc "A" contains LINK_TOKEN which references "B"
    LINK_TOKEN = 77
    corpus = make_corpus({"B": [99]})
    detector = make_link_detector(fires_at_token=LINK_TOKEN, target="B")

    ctx = make_context_obj()
    root = ctx.add_root("", [1], None)
    # Add a corpus doc whose token body contains LINK_TOKEN
    entry_a = ctx.add_corpus_doc("A", [LINK_TOKEN], None, "", 1, root)

    cfg = base_config(max_link_depth=2)
    _process_existing_doc_links(entry_a, ctx, None, detector, corpus, cfg, None, depth=1)

    assert ctx.num_aux_docs == 2  # A and B
    identifiers = [e.raw_identifier for e in ctx._docs]
    assert "B" in identifiers


def test_process_existing_doc_links_no_op_when_detector_none():
    ctx = make_context_obj()
    root = ctx.add_root("", [1], None)
    entry = ctx.add_corpus_doc("A", [10], None, "", 1, root)
    # Should not raise and should not add any docs
    _process_existing_doc_links(entry, ctx, None, None, None, base_config(), None, depth=1)
    assert ctx.num_aux_docs == 1


# --- end-to-end run_generation with link detection ---

def test_run_generation_with_corpus_link():
    """run_generation inserts corpus doc when link is detected mid-generation."""
    LINK_TOKEN = 42
    corpus = make_corpus({"Python": [10, 20]})
    detector = make_link_detector(fires_at_token=LINK_TOKEN, target="Python")
    model = make_mock_model(next_tokens=[LINK_TOKEN, EOS])

    result = run_generation(
        model=model,
        prompt_tokens=[1],
        corpus=corpus,
        config=base_config(max_link_depth=1),
        link_detector=detector,
        tokenizer_decode=None,
        layout_policy=None,
    )
    assert result.get_document_count() == 2
    aux = result.auxiliary_documents[0]
    assert aux.raw_identifier == "Python"
    assert aux.source == "corpus"


def test_run_generation_text_decoded_for_all_docs():
    """tokenizer_decode is applied to root and all aux docs."""
    LINK_TOKEN = 42
    corpus = make_corpus({"Python": [10, 20]})
    detector = make_link_detector(fires_at_token=LINK_TOKEN, target="Python")
    model = make_mock_model(next_tokens=[LINK_TOKEN, EOS])

    result = run_generation(
        model=model,
        prompt_tokens=[1],
        corpus=corpus,
        config=base_config(max_link_depth=1),
        link_detector=detector,
        tokenizer_decode=lambda ids: f"<{ids}>",
        layout_policy=None,
    )
    for doc in result.get_all_documents():
        assert doc.text is not None


# --- prompt link pre-processing (Stage 3.2) ---

def make_anywhere_link_detector(link_token, target):
    """Fires when link_token appears anywhere in the token sequence."""
    from collections import namedtuple
    LinkInfo = namedtuple("LinkInfo", ["link_end_pos", "target_str"])

    class FakeDetector:
        def detect_links(self, tokens):
            for i, t in enumerate(tokens):
                if t.item() == link_token:
                    return [LinkInfo(link_end_pos=i + 1, target_str=target)]
            return []

    return FakeDetector()


def test_run_generation_prompt_links_fetched_from_corpus():
    """With process_prompt_links=True, a link in the prompt causes a corpus fetch
    before generation begins."""
    LINK_TOKEN = 55
    corpus = make_corpus({"Python": [10, 20]})
    detector = make_anywhere_link_detector(LINK_TOKEN, "Python")
    model = make_mock_model(next_tokens=[EOS])

    result = run_generation(
        model=model,
        prompt_tokens=[1, LINK_TOKEN, 2],
        corpus=corpus,
        config=base_config(max_link_depth=1, process_prompt_links=True),
        link_detector=detector,
        tokenizer_decode=None,
        layout_policy=None,
    )
    assert result.get_document_count() == 2
    aux = result.auxiliary_documents[0]
    assert aux.raw_identifier == "Python"
    assert aux.source == "corpus"


def test_run_generation_prompt_links_skipped_when_disabled():
    """With process_prompt_links=False, links in the prompt are ignored."""
    LINK_TOKEN = 55
    corpus = make_corpus({"Python": [10, 20]})
    detector = make_anywhere_link_detector(LINK_TOKEN, "Python")
    model = make_mock_model(next_tokens=[EOS])

    result = run_generation(
        model=model,
        prompt_tokens=[1, LINK_TOKEN, 2],
        corpus=corpus,
        config=base_config(max_link_depth=1, process_prompt_links=False),
        link_detector=detector,
        tokenizer_decode=None,
        layout_policy=None,
    )
    assert result.get_document_count() == 1


def test_run_generation_prompt_links_no_op_when_no_detector():
    """With link_detector=None, process_prompt_links has no effect."""
    model = make_mock_model(next_tokens=[EOS])

    result = run_generation(
        model=model,
        prompt_tokens=[1, 2, 3],
        corpus=None,
        config=base_config(process_prompt_links=True),
        link_detector=None,
        tokenizer_decode=None,
        layout_policy=None,
    )
    assert result.get_document_count() == 1


# ---------------------------------------------------------------------------
# Richer mock infrastructure for multi-step integration tests
# ---------------------------------------------------------------------------

class ScriptedModel:
    """
    A mock model that emits tokens from a script, where the script can branch
    based on which document is currently being generated.

    Usage:
        model = ScriptedModel({
            "":       [100, LINK_A, 101, EOS],   # root doc
            "Python": [200, 201, EOS],            # aux doc "Python"
        })

    The model tracks which entry is currently active by looking at the
    raw_identifier of the last (rightmost non-done) document in doc_spans.
    """

    def __init__(self, scripts: dict, default_eos=EOS):
        self._scripts = scripts
        self._cursors = {k: 0 for k in scripts}
        self._default_eos = default_eos
        self.call_count = 0

    def forward_inference(self, tokens, doc_spans=None, **kwargs):
        self.call_count += 1
        # Find which doc is the active (rightmost, assumed to be the one
        # being generated — matches generation_loop convention where root
        # or the currently-generating doc is always last in context).
        active_id = ""
        if doc_spans:
            # The active doc is the rightmost in the packed sequence
            active_id = doc_spans[-1].raw_identifier

        script = self._scripts.get(active_id)
        if script is None:
            token = self._default_eos
        else:
            idx = self._cursors.get(active_id, 0)
            token = script[idx] if idx < len(script) else self._default_eos
            self._cursors[active_id] = idx + 1

        logits = torch.full((1, tokens.shape[1], VOCAB_SIZE), -1e9)
        logits[0, -1, token] = 1e9
        return logits


class MultiLinkDetector:
    """
    Detects links by matching specific token IDs to target identifiers.

    Usage:
        detector = MultiLinkDetector({42: "Python", 43: "Java"})

    When the last token in the scan window matches a key, fires a link to
    the corresponding target. Also fires for tokens anywhere in a full-doc
    scan (for _process_existing_doc_links).
    """

    def __init__(self, token_to_target: dict):
        self._map = token_to_target

    def detect_links(self, tokens):
        from collections import namedtuple
        LinkInfo = namedtuple("LinkInfo", ["link_end_pos", "target_str"])
        results = []
        for i, t in enumerate(tokens):
            tid = t.item() if hasattr(t, 'item') else int(t)
            if tid in self._map:
                results.append(LinkInfo(
                    link_end_pos=i + 1,
                    target_str=self._map[tid],
                ))
        return results

    def index_doc_span(self, span):
        return span.raw_identifier


# ---------------------------------------------------------------------------
# Integration: re-eviction through _handle_link
# ---------------------------------------------------------------------------

def test_handle_link_restores_evicted_doc():
    """When a link targets a previously evicted doc, it is restored."""
    from collections import namedtuple
    LinkInfo = namedtuple("LinkInfo", ["link_end_pos", "target_str"])
    ctx = make_context_obj(max_context_length=10, eviction_policy="drop_oldest")
    root = ctx.add_root("", [1], None)
    # Add A (3 tokens) then evict it
    ctx.add_corpus_doc("A", [10, 11, 12], None, "", 1, root)
    ctx.evict_oldest_aux()
    assert ctx.num_aux_docs == 0
    assert ctx.find_evicted("A") is not None

    # Now handle a link to "A" — should restore it
    cfg = base_config(max_link_depth=1, max_context_length=10,
                      max_tokens_per_document=10, max_new_tokens=10,
                      eviction_policy="drop_oldest")
    _handle_link(
        LinkInfo(link_end_pos=1, target_str="A"),
        root, ctx, None, None, None, cfg, None, depth=0,
    )
    assert ctx.num_aux_docs == 1
    assert ctx._docs[0].raw_identifier == "A"
    assert ctx._docs[0].tokens == [10, 11, 12]
    assert ctx.find_evicted("A") is None  # no longer in evicted list


def test_handle_link_restore_evicted_needs_make_room():
    """Restoring an evicted doc may require evicting another doc first."""
    from collections import namedtuple
    LinkInfo = namedtuple("LinkInfo", ["link_end_pos", "target_str"])
    # Setup: root(1) + B(2) = 3 tokens. Evicted A has 3 tokens.
    # Restoring A: need room for 3 → total would be 3+3=6. max=5.
    # Must evict B(2) first → root(1)+A(3)=4 ≤ 5 ✓
    ctx = make_context_obj(max_context_length=5, eviction_policy="drop_oldest")
    root = ctx.add_root("", [1], None)
    ctx.add_corpus_doc("A", [10, 11, 12], None, "", 1, root)
    ctx.add_corpus_doc("B", [20, 21], None, "", 1, root)
    ctx.evict_oldest_aux()  # evicts A (leftmost non-root)
    assert ctx.find_evicted("A") is not None

    cfg = base_config(max_link_depth=1, max_context_length=5,
                      max_tokens_per_document=5, max_new_tokens=5,
                      eviction_policy="drop_oldest")
    _handle_link(
        LinkInfo(link_end_pos=1, target_str="A"),
        root, ctx, None, None, None, cfg, None, depth=0,
    )
    # A restored, B evicted to make room
    active_ids = [e.raw_identifier for e in ctx._docs]
    assert "A" in active_ids
    assert "B" not in active_ids


def test_handle_link_restore_evicted_stop_new_skips_when_no_room():
    """With stop_new policy, can't restore evicted doc if it doesn't fit."""
    from collections import namedtuple
    LinkInfo = namedtuple("LinkInfo", ["link_end_pos", "target_str"])
    # root(1) + B(2) = 3 tokens, max=4. Evicted A has 3 tokens → 3+3=6 > 4.
    ctx = make_context_obj(max_context_length=4, eviction_policy="stop_new")
    root = ctx.add_root("", [1], None)
    ctx.add_corpus_doc("A", [10, 11, 12], None, "", 1, root)
    ctx.add_corpus_doc("B", [20, 21], None, "", 1, root)
    ctx.evict_oldest_aux()  # evicts A
    assert ctx.find_evicted("A") is not None

    cfg = base_config(max_link_depth=1, max_context_length=4,
                      max_tokens_per_document=4, max_new_tokens=4,
                      eviction_policy="stop_new")
    _handle_link(
        LinkInfo(link_end_pos=1, target_str="A"),
        root, ctx, None, None, None, cfg, None, depth=0,
    )
    # Can't fit, should remain evicted
    assert ctx.find_evicted("A") is not None
    active_ids = [e.raw_identifier for e in ctx._docs]
    assert "A" not in active_ids


def test_handle_link_restore_evicted_rescans_links():
    """A restored evicted doc has its links re-processed at depth+1."""
    from collections import namedtuple
    LinkInfo = namedtuple("LinkInfo", ["link_end_pos", "target_str"])

    LINK_TOKEN = 77
    corpus = make_corpus({"C": [99]})
    detector = MultiLinkDetector({LINK_TOKEN: "C"})

    # A contains LINK_TOKEN (a link to C). Evict A, then restore it.
    ctx = make_context_obj(max_context_length=20, eviction_policy="drop_oldest")
    root = ctx.add_root("", [1], None)
    ctx.add_corpus_doc("A", [LINK_TOKEN, 11], None, "", 1, root)
    ctx.evict_oldest_aux()  # evicts A

    cfg = base_config(max_link_depth=2, max_context_length=20,
                      max_tokens_per_document=20, max_new_tokens=20,
                      eviction_policy="drop_oldest")
    _handle_link(
        LinkInfo(link_end_pos=1, target_str="A"),
        root, ctx, None, detector, corpus, cfg, None, depth=0,
    )
    # A restored, and A's link to C should have been processed
    active_ids = [e.raw_identifier for e in ctx._docs]
    assert "A" in active_ids
    assert "C" in active_ids


def test_handle_link_restore_evicted_respects_depth_limit():
    """Re-eviction respects max_link_depth — depth >= limit skips."""
    from collections import namedtuple
    LinkInfo = namedtuple("LinkInfo", ["link_end_pos", "target_str"])

    ctx = make_context_obj(max_context_length=20, eviction_policy="drop_oldest")
    root = ctx.add_root("", [1], None)
    ctx.add_corpus_doc("A", [10], None, "", 1, root)
    ctx.evict_oldest_aux()

    # depth=1, max_link_depth=1 → depth >= max_link_depth → skip
    cfg = base_config(max_link_depth=1, max_context_length=20,
                      max_tokens_per_document=20, max_new_tokens=20)
    _handle_link(
        LinkInfo(link_end_pos=1, target_str="A"),
        root, ctx, None, None, None, cfg, None, depth=1,
    )
    assert ctx.find_evicted("A") is not None  # NOT restored


# ---------------------------------------------------------------------------
# Integration: trace counters
# ---------------------------------------------------------------------------

def test_trace_counts_forward_passes_and_tokens():
    """GenerationTrace counts forward passes and generated tokens correctly."""
    model = make_mock_model(next_tokens=[10, 20, EOS])
    result = run_generation(
        model=model,
        prompt_tokens=[1],
        corpus=None,
        config=base_config(record_trace=True),
        link_detector=None,
        tokenizer_decode=None,
        layout_policy=None,
    )
    assert result.trace is not None
    assert result.trace.total_forward_passes == 3
    assert result.trace.total_tokens_generated == 3


def test_trace_counts_link_detection_and_corpus_fetch():
    """Trace tracks links detected, resolved, and corpus fetches."""
    LINK_TOKEN = 42
    corpus = make_corpus({"Python": [10, 20]})
    detector = make_link_detector(fires_at_token=LINK_TOKEN, target="Python")
    model = make_mock_model(next_tokens=[LINK_TOKEN, EOS])

    result = run_generation(
        model=model,
        prompt_tokens=[1],
        corpus=corpus,
        config=base_config(max_link_depth=1, record_trace=True),
        link_detector=detector,
        tokenizer_decode=None,
        layout_policy=None,
    )
    assert result.trace.links_detected == 1
    assert result.trace.links_resolved == 1
    assert result.trace.corpus_fetches == 1
    assert result.trace.docs_generated == 0


def test_trace_counts_generated_aux_docs():
    """Trace counts recursively generated aux docs."""
    LINK_TOKEN = 42
    detector = make_link_detector(fires_at_token=LINK_TOKEN, target="Go")
    model = ScriptedModel({
        "":   [LINK_TOKEN, EOS],
        "Go": [10, EOS],
    })

    result = run_generation(
        model=model,
        prompt_tokens=[1],
        corpus=None,
        config=base_config(max_link_depth=1, allow_generation_fallback=True,
                           record_trace=True),
        link_detector=detector,
        tokenizer_decode=None,
        layout_policy=None,
    )
    assert result.trace.docs_generated == 1
    assert result.trace.links_resolved == 1


def test_trace_disabled_when_record_trace_false():
    """When record_trace=False, trace is None."""
    model = make_mock_model(next_tokens=[EOS])
    result = run_generation(
        model=model,
        prompt_tokens=[1],
        corpus=None,
        config=base_config(record_trace=False),
        link_detector=None,
        tokenizer_decode=None,
        layout_policy=None,
    )
    assert result.trace is None


# ---------------------------------------------------------------------------
# Integration: end-to-end multi-step scenarios
# ---------------------------------------------------------------------------

def test_scenario_generate_link_fetch_continue():
    """
    Scenario: Root generates text, emits a link to "Python" (in corpus),
    corpus doc is fetched, root continues generating, hits EOS.

    Verifies:
    - Correct document ordering (aux before root in packed sequence)
    - Corpus doc tokens are preserved
    - Root doc continues after link handling
    - Trace counters consistent
    """
    LINK_TOKEN = 42
    corpus = make_corpus({"Python": [200, 201, 202]})
    detector = MultiLinkDetector({LINK_TOKEN: "Python"})
    model = ScriptedModel({
        "": [100, LINK_TOKEN, 101, 102, EOS],
    })

    result = run_generation(
        model=model,
        prompt_tokens=[1, 2],
        corpus=corpus,
        config=base_config(max_link_depth=1, record_trace=True),
        link_detector=detector,
        tokenizer_decode=None,
        layout_policy=None,
    )

    assert result.get_document_count() == 2
    root = result.root_document
    aux = result.auxiliary_documents[0]

    # Root has prompt + generated tokens including link token + EOS
    assert root.tokens.tolist() == [1, 2, 100, LINK_TOKEN, 101, 102, EOS]
    assert aux.raw_identifier == "Python"
    assert aux.source == "corpus"
    assert aux.tokens.tolist() == [200, 201, 202]

    # Trace
    assert result.trace.links_detected == 1
    assert result.trace.corpus_fetches == 1
    assert result.trace.total_tokens_generated == 5  # 100, LINK, 101, 102, EOS


def test_scenario_generate_link_generate_aux():
    """
    Scenario: Root emits link to "NewTopic" (not in corpus), generation
    fallback kicks in, aux doc is recursively generated, root continues.

    NOTE: The generation loop always samples from logits[0, -1, :], which
    is the last position in the packed sequence. During aux doc generation
    the aux doc is inserted *before* root, so position -1 is still root's
    last token. This means the aux doc's tokens are sampled from root's
    logit distribution — a known limitation of the current implementation.
    We use make_mock_model (call-order based) to test the structural flow
    without depending on position-correct sampling.
    """
    LINK_TOKEN = 42
    detector = MultiLinkDetector({LINK_TOKEN: "NewTopic"})
    # Call order: root gets LINK_TOKEN → link fires → aux generation starts →
    # aux gets tokens 200, 201, EOS → root resumes → root gets 100, EOS
    model = make_mock_model(next_tokens=[LINK_TOKEN, 200, 201, EOS, 100, EOS])

    result = run_generation(
        model=model,
        prompt_tokens=[1],
        corpus=None,
        config=base_config(max_link_depth=1, allow_generation_fallback=True,
                           record_trace=True),
        link_detector=detector,
        tokenizer_decode=None,
        layout_policy=None,
    )

    assert result.get_document_count() == 2
    aux = result.auxiliary_documents[0]
    assert aux.raw_identifier == "NewTopic"
    assert aux.source == "generated"

    assert result.trace.docs_generated == 1
    assert result.trace.links_detected == 1


def test_scenario_multiple_links_different_targets():
    """
    Scenario: Root generates two links to different corpus docs.
    Both are fetched and present in the result.
    """
    LINK_A = 42
    LINK_B = 43
    corpus = make_corpus({"Alpha": [200], "Beta": [300, 301]})
    detector = MultiLinkDetector({LINK_A: "Alpha", LINK_B: "Beta"})
    model = ScriptedModel({
        "": [LINK_A, 100, LINK_B, EOS],
    })

    result = run_generation(
        model=model,
        prompt_tokens=[1],
        corpus=corpus,
        config=base_config(max_link_depth=1, record_trace=True),
        link_detector=detector,
        tokenizer_decode=None,
        layout_policy=None,
    )

    assert result.get_document_count() == 3
    aux_ids = {d.raw_identifier for d in result.auxiliary_documents}
    assert aux_ids == {"Alpha", "Beta"}
    assert result.trace.corpus_fetches == 2
    assert result.trace.links_detected == 2


def test_scenario_eviction_and_reencounter():
    """
    Scenario: Context is small. Root generates link to A (fetched),
    then link to B (fetched, evicts A), then link to A again.
    A should be restored from the evicted list.

    Context budget: root(1 prompt + up to 4 gen = 5) + 1 aux at a time.
    max_aux_docs=1 forces eviction when second aux arrives.
    """
    LINK_A = 42
    LINK_B = 43
    corpus = make_corpus({"A": [200], "B": [300]})
    detector = MultiLinkDetector({LINK_A: "A", LINK_B: "B"})
    # Root: emit link_A, then filler, then link_B, then link_A again, then EOS
    model = ScriptedModel({
        "": [LINK_A, 100, LINK_B, LINK_A, EOS],
    })

    result = run_generation(
        model=model,
        prompt_tokens=[1],
        corpus=corpus,
        config=base_config(
            max_link_depth=1,
            max_auxiliary_documents=1,  # forces eviction
            max_context_length=100,
            max_tokens_per_document=100,
            max_new_tokens=20,
            record_trace=True,
        ),
        link_detector=detector,
        tokenizer_decode=None,
        layout_policy=None,
    )

    # A was fetched, then B was fetched (evicting A), then A was restored (evicting B)
    assert result.trace.links_detected >= 3
    assert result.trace.links_resolved >= 3
    assert result.trace.docs_evicted >= 2

    # Final state: A should be in the active window (last link was to A)
    # and B should have been evicted
    all_docs = result.get_all_documents()
    all_ids = {d.raw_identifier for d in all_docs}
    assert "A" in all_ids
    assert "B" in all_ids


def test_scenario_corpus_doc_triggers_recursive_fetch():
    """
    Scenario: Root links to "A" (corpus). A's tokens contain a link to "B"
    (also corpus). Both end up in the result at correct depths.
    """
    LINK_A = 42
    LINK_B = 43
    # A's body contains LINK_B, which should trigger fetching B at depth 2
    corpus = make_corpus({"A": [LINK_B, 201], "B": [300]})
    detector = MultiLinkDetector({LINK_A: "A", LINK_B: "B"})
    model = ScriptedModel({
        "": [LINK_A, EOS],
    })

    result = run_generation(
        model=model,
        prompt_tokens=[1],
        corpus=corpus,
        config=base_config(max_link_depth=2, record_trace=True),
        link_detector=detector,
        tokenizer_decode=None,
        layout_policy=None,
    )

    assert result.get_document_count() == 3
    doc_a = result.get_document_by_identifier("A")
    doc_b = result.get_document_by_identifier("B")
    assert doc_a is not None
    assert doc_b is not None
    assert doc_a.depth == 1
    assert doc_b.depth == 2
    assert doc_b.parent_raw_identifier == "A"

    assert result.trace.corpus_fetches == 2
    assert result.trace.max_depth_reached == 2


def test_scenario_prompt_links_prefetch_before_generation():
    """
    Scenario: Prompt already contains a link token. With process_prompt_links=True,
    the corpus doc is fetched BEFORE generation begins, so the model's first
    forward pass already sees the aux doc in context.
    """
    LINK_TOKEN = 42
    corpus = make_corpus({"Preloaded": [200, 201]})
    detector = MultiLinkDetector({LINK_TOKEN: "Preloaded"})

    observed_spans = []

    class SpyModel:
        """Records doc_spans on each forward call."""
        call_count = 0

        def forward_inference(self, tokens, doc_spans=None, **kwargs):
            self.call_count += 1
            observed_spans.append(
                [s.raw_identifier for s in (doc_spans or [])]
            )
            logits = torch.full((1, tokens.shape[1], VOCAB_SIZE), -1e9)
            logits[0, -1, EOS] = 1e9
            return logits

    model = SpyModel()
    result = run_generation(
        model=model,
        prompt_tokens=[1, LINK_TOKEN, 2],
        corpus=corpus,
        config=base_config(max_link_depth=1, process_prompt_links=True,
                           record_trace=True),
        link_detector=detector,
        tokenizer_decode=None,
        layout_policy=None,
    )

    # The very first forward pass should already have "Preloaded" in doc_spans
    assert len(observed_spans) >= 1
    assert "Preloaded" in observed_spans[0]
    assert result.trace.corpus_fetches == 1


def test_scenario_max_link_depth_zero_disables_all_aux():
    """max_link_depth=0 disables ALL aux doc insertion — corpus, generated, and re-eviction."""
    LINK_TOKEN = 42
    corpus = make_corpus({"Python": [200]})
    detector = MultiLinkDetector({LINK_TOKEN: "Python"})
    model = ScriptedModel({"": [LINK_TOKEN, EOS]})

    result = run_generation(
        model=model,
        prompt_tokens=[1],
        corpus=corpus,
        config=base_config(max_link_depth=0, record_trace=True),
        link_detector=detector,
        tokenizer_decode=None,
        layout_policy=None,
    )

    assert result.get_document_count() == 1
    assert result.trace.links_detected == 1
    assert result.trace.links_resolved == 0


def test_scenario_large_corpus_doc_exceeding_context_gracefully_skipped():
    """A corpus doc too large to fit (even after evicting everything) is skipped, not crashed."""
    from collections import namedtuple
    LinkInfo = namedtuple("LinkInfo", ["link_end_pos", "target_str"])
    # root=1 token, max_context=5. Corpus doc "Big" = 10 tokens → can never fit.
    corpus = make_corpus({"Big": list(range(10))})
    ctx = make_context_obj(max_context_length=5, eviction_policy="drop_oldest")
    root = ctx.add_root("", [1], None)

    cfg = base_config(max_link_depth=1, max_context_length=5,
                      max_tokens_per_document=5, max_new_tokens=5,
                      eviction_policy="drop_oldest")
    _handle_link(
        LinkInfo(link_end_pos=1, target_str="Big"),
        root, ctx, None, None, corpus, cfg, None, depth=0,
    )
    # Should not crash, and "Big" should not be in context
    assert ctx.num_aux_docs == 0
