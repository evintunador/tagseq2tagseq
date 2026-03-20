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
        root, ctx, None, None, corpus, base_config(max_link_depth=0), None, depth=0,
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
        root, ctx, None, None, corpus, base_config(max_link_depth=0), None, depth=0,
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
    cfg = base_config(eviction_policy="stop_new", max_context_length=2)
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
    cfg = base_config(eviction_policy="drop_oldest", max_context_length=6, max_link_depth=0)
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
