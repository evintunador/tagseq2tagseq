"""Unit tests for eval/benchmark_harness — synthetic port, no network/GPU.

Covers the frozen pieces a builder agent will be judged by: Tier 0 invariants,
Tier 1 oracle audit (precision + fire-rate parity failure modes), and Tier C
dedup policy. Tier 2 needs a trained checkpoint and is exercised by the
calibration runs, not unit tests.
"""
from __future__ import annotations

import pytest

from eval.benchmark_harness.schema import AuxDoc, CrossDocExample, PortAdapter
from eval.benchmark_harness.dedup import run_dedup, file_hash

pytest.importorskip("tree_sitter")
pytest.importorskip("tree_sitter_python")


# ─── synthetic python port ────────────────────────────────────────────────────

GOOD_CONTEXT = "import utils.helpers\n\nx = utils.helpers.f(\n"
GOOD_AUX = (AuxDoc(path="utils/helpers.py", content="def f(a):\n    return a\n"),)


def _mk_example(context=GOOD_CONTEXT, target="    1)\n", aux=GOOD_AUX,
                repo="acme/widget", file_path="main.py"):
    return CrossDocExample(repo=repo, file_path=file_path, context=context,
                           target=target, aux=aux)


def _mk_port(examples, identifier_fn=None):
    from model.graph_traversal.python_import_detector import PythonImportDetector
    return PortAdapter(
        name="synthetic_python",
        language="python",
        examples_fn=lambda n: examples[:n] if n else list(examples),
        identifier_fn=identifier_fn or (lambda repo, path, content: f"{repo}:{path}"),
        detector_factory=lambda decode_fn: PythonImportDetector(decode_fn),
    )


@pytest.fixture(scope="module")
def enc_dec():
    import tiktoken
    tok = tiktoken.get_encoding("gpt2")
    return (lambda t: tok.encode(t, disallowed_special=()), tok.decode)


# ─── Tier 0 ──────────────────────────────────────────────────────────────────

def test_tier0_passes_on_good_port(enc_dec):
    from eval.benchmark_harness.tier0 import run_tier0
    port = _mk_port([_mk_example() for _ in range(5)])
    rep = run_tier0(port, enc_dec[0])
    assert rep.passed, rep.failures


def test_tier0_fails_on_cropped_imports(enc_dec):
    from eval.benchmark_harness.tier0 import run_tier0
    port = _mk_port([_mk_example(context="x = f(\n") for _ in range(5)])
    rep = run_tier0(port, enc_dec[0])
    assert not rep.passed
    assert any("import" in f for f in rep.failures)


def test_tier0_fails_on_absolute_aux_path(enc_dec):
    from eval.benchmark_harness.tier0 import run_tier0
    bad_aux = (AuxDoc(path="/abs/utils/helpers.py", content="def f(a): ..."),)
    port = _mk_port([_mk_example(aux=bad_aux) for _ in range(3)])
    rep = run_tier0(port, enc_dec[0])
    assert not rep.passed
    assert any("repo-relative" in f for f in rep.failures)


def test_tier0_fails_on_empty_target(enc_dec):
    from eval.benchmark_harness.tier0 import run_tier0
    port = _mk_port([_mk_example(target="   ")])
    rep = run_tier0(port, enc_dec[0])
    assert not rep.passed


def test_tier0_fails_on_nondeterministic_port(enc_dec):
    from eval.benchmark_harness.tier0 import run_tier0
    calls = []

    def examples_fn(n):
        calls.append(1)
        return [_mk_example(target=f"    {len(calls)})\n")]

    from model.graph_traversal.python_import_detector import PythonImportDetector
    port = PortAdapter(
        name="flaky", language="python", examples_fn=examples_fn,
        identifier_fn=lambda r, p, c: f"{r}:{p}",
        detector_factory=lambda d: PythonImportDetector(d))
    rep = run_tier0(port, enc_dec[0])
    assert not rep.passed
    assert any("deterministic" in f for f in rep.failures)


def test_tier0_allows_same_path_different_content(enc_dec):
    from eval.benchmark_harness.tier0 import run_tier0
    aux = (AuxDoc(path="utils/helpers.py", content="def f(a): ..."),
           AuxDoc(path="utils/helpers.py", content="def g(b): ..."))
    port = _mk_port([_mk_example(aux=aux)])
    rep = run_tier0(port, enc_dec[0])
    assert rep.passed, rep.failures


# ─── Tier 1 ──────────────────────────────────────────────────────────────────

def test_tier1_passes_on_good_port(enc_dec):
    from eval.benchmark_harness.tier1 import run_tier1
    port = _mk_port([_mk_example() for _ in range(4)])
    rep = run_tier1(port, *enc_dec)
    assert rep.passed, rep.failures
    assert rep.precision == 1.0
    assert rep.n_port_fired == 4


def test_tier1_fails_on_broken_identifier_shaping(enc_dec):
    """Identifier shaping that mangles paths loses every resolvable link —
    the java-source-root failure class the fire-rate parity gate exists for."""
    from eval.benchmark_harness.tier1 import run_tier1
    port = _mk_port(
        [_mk_example() for _ in range(4)],
        identifier_fn=lambda repo, path, content: f"{repo}:BROKEN/{path}",
    )
    rep = run_tier1(port, *enc_dec)
    assert not rep.passed
    assert rep.n_port_fired == 0
    assert rep.n_oracle_reachable == 4
    assert any("fire-rate" in f for f in rep.failures)


def test_tier1_unreachable_aux_does_not_count(enc_dec):
    """Aux docs the context never imports contribute nothing to either side."""
    from eval.benchmark_harness.tier1 import run_tier1
    aux = (AuxDoc(path="unrelated/other.py", content="def h(): ..."),)
    port = _mk_port([_mk_example(aux=aux) for _ in range(3)])
    rep = run_tier1(port, *enc_dec)
    assert rep.n_oracle_reachable == 0
    assert rep.n_port_fired == 0
    assert rep.passed, rep.failures  # 0 ≥ 0.9×0 — vacuous parity, precision 1.0


# ─── Tier C dedup ────────────────────────────────────────────────────────────

def test_dedup_repo_intersection_drops_example():
    examples = [_mk_example(repo="acme/widget"), _mk_example(repo="clean/repo")]
    survivors, rep = run_dedup("p", examples, training_repos={"acme/widget"})
    assert rep.n_repo_overlap_dropped == 1
    assert [e.repo for e in survivors] == ["clean/repo"]
    assert rep.overlapping_repos == ["acme/widget"]


def test_dedup_hash_drops_copy_pasted_primary():
    ex = _mk_example(repo="clean/repo")
    hashes = {file_hash(ex.context)}
    survivors, rep = run_dedup("p", [ex], training_repos=set(),
                               training_hashes=hashes)
    assert survivors == []
    assert rep.n_hash_dropped == 1


def test_dedup_hash_drops_only_matching_aux():
    aux = (AuxDoc(path="a.py", content="def a(): ...\n"),
           AuxDoc(path="b.py", content="def b(): ...\n"))
    ex = _mk_example(repo="clean/repo", aux=aux)
    hashes = {file_hash(aux[0].content)}
    survivors, rep = run_dedup("p", [ex], training_repos=set(),
                               training_hashes=hashes)
    assert len(survivors) == 1
    assert [d.path for d in survivors[0].aux] == ["b.py"]
    assert rep.n_aux_docs_hash_dropped == 1


def test_dedup_hash_normalization_ignores_whitespace():
    assert file_hash("def f():\n    pass\n") == file_hash("def f():   \n\n    pass")


def test_scope_carving_python():
    """Use-site anchoring + the three nested target widths, on a synthetic file
    whose only aux-symbol use is mid-function."""
    from eval.benchmark_harness.scopes import scope_example

    aux = (AuxDoc(path="utils/helpers.py",
                  content="def compute(x):\n    return x*2\n"),)
    full = ("from utils.helpers import compute\n\n"
            "def main():\n    a = 1\n    b = compute(a)\n"
            "    if b > 0:\n        print(b)\n    return b\n")
    ex = CrossDocExample(repo="r", file_path="main.py",
                         context="from utils.helpers import compute\n\n"
                                 "def main():\n    a = 1",
                         target="\n    b = compute(a)", aux=aux, full_file=full)

    line = scope_example(ex, "python", "use_line")
    assert line is not None and line.matched_symbols == ("compute",)
    assert "compute(a)" in line.target and "print(b)" not in line.target

    block = scope_example(ex, "python", "use_block")
    assert "print(b)" in block.target and "return b" in block.target

    rest = scope_example(ex, "python", "rest_of_doc")
    assert rest.target.count("\n") >= block.target.count("\n")

    native = scope_example(ex, "python", "native")
    assert native.target == ex.target and native.use_site_line == -1


def test_scope_drops_when_no_use_site():
    """No line uses any aux-declared symbol → dropped for use-scopes."""
    from eval.benchmark_harness.scopes import scope_example
    aux = (AuxDoc(path="a.py", content="def unused_symbol(): ...\n"),)
    full = "x = 1\ny = 2\nz = x + y\n"
    ex = CrossDocExample(repo="r", file_path="m.py", context="x = 1",
                         target="\ny = 2", aux=aux, full_file=full)
    assert scope_example(ex, "python", "use_line") is None
    # native always survives
    assert scope_example(ex, "python", "native") is not None


def test_scope_none_without_full_file():
    from eval.benchmark_harness.scopes import scope_example
    ex = _mk_example()  # no full_file
    assert scope_example(ex, "python", "use_line") is None
    assert scope_example(ex, "python", "native") is not None


def test_tier2_oversized_pack_guard():
    """A pack longer than the model's rotary cap must be SKIPPED (counted),
    not scored — whole-file aux (Kotlin/ASE) can exceed 32k where RepoBench's
    small snippets never do. Regression for the RoPE-assertion crash."""
    from eval.benchmark_harness.tier2 import _model_max_seq_len

    class _FakeBuf:
        def __init__(self, n): self._n = n
        def size(self, _): return self._n

    class _FakeBackbone:
        def named_buffers(self):
            yield "layers.0.attn.rotary.cos", _FakeBuf(32768)

    class _FakeModel:
        backbone = _FakeBackbone()

    assert _model_max_seq_len(_FakeModel()) == 32768

    class _NoBuf:
        backbone = type("B", (), {"named_buffers": lambda self: iter(())})()
    assert _model_max_seq_len(_NoBuf(), default=4096) == 4096


def test_dedup_example_dropped_when_all_aux_hash_matched():
    aux = (AuxDoc(path="a.py", content="def a(): ...\n"),)
    ex = _mk_example(repo="clean/repo", aux=aux)
    hashes = {file_hash(aux[0].content)}
    survivors, rep = run_dedup("p", [ex], training_repos=set(),
                               training_hashes=hashes)
    assert survivors == []
    assert rep.n_hash_dropped == 1
