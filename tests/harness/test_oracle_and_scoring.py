"""
Harness self-tests.

The load-bearing test is `test_python_detector_conforms`: the EXISTING, trusted
PythonImportDetector must score ~1.0 precision & recall against the tree-sitter
oracle. If it doesn't, the harness (oracle query / key projection) is wrong — not
the detector. This is how we validate the harness before trusting it to gate new
languages.

Also verifies:
  * the scorer's precision/recall math + micro-averaging;
  * that the reward hacks the design doc calls out actually FAIL the gate
    (empty detector -> recall 0; hallucinating detector -> precision < 1).
"""
import pytest

tree_sitter = pytest.importorskip("tree_sitter")

import tiktoken
import torch

from data.graph_harness import TreeSitterOracle, score_detection, PrecisionRecall
from data.graph_harness.specs import get_spec
from model.graph_traversal.python_import_detector import PythonImportDetector


PY_SAMPLES = {
    "simple.py": "import os\nimport sys\nx = 1\n",
    "dotted.py": "import os.path\nimport a.b.c\n",
    "aliased.py": "import numpy as np\nimport a.b as ab\n",
    "from_inline.py": "from a.b import c, d\nfrom pkg import thing\n",
    "from_paren.py": "from a.b import (\n    c,\n    d,\n)\n",
    "star.py": "from a.b import *\n",
    "mixed.py": (
        "import os\n"
        "from django.db import models\n"
        "from .local import helper\n"   # relative — skipped by both sides
        "import a.b.c\n"
        "from x.y import z\n"
    ),
    "noimports.py": "def f():\n    return 42\n",
}


@pytest.fixture(scope="module")
def enc():
    return tiktoken.get_encoding("gpt2")


@pytest.fixture(scope="module")
def py_spec():
    return get_spec("python")


@pytest.fixture(scope="module")
def py_oracle(py_spec):
    return TreeSitterOracle(py_spec)


def test_python_detector_conforms(enc, py_spec, py_oracle):
    """The trusted detector scores perfectly against the independent oracle."""
    detector = PythonImportDetector(decode_fn=enc.decode)
    per_file = {}
    for name, code in PY_SAMPLES.items():
        oracle_keys = py_oracle.import_keys(code)
        ids = torch.tensor(enc.encode(code), dtype=torch.long)
        detected = [li.target_str for li in detector.detect_links(ids)]
        per_file[name] = (oracle_keys, detected)

    score = score_detection(per_file, py_spec.canonical_target)
    # The existing detector is trusted ground truth; harness must confirm it.
    assert score.recall == 1.0, (
        f"harness under-credits a correct detector: {score.summary()}; "
        f"missed={score.false_negative_examples}"
    )
    assert score.precision == 1.0, (
        f"harness over-penalizes a correct detector: {score.summary()}; "
        f"spurious={score.false_positive_examples}"
    )


def test_empty_detector_fails_recall(enc, py_spec, py_oracle):
    """The 'emit nothing' reward hack must FAIL: recall collapses."""
    per_file = {}
    for name, code in PY_SAMPLES.items():
        oracle_keys = py_oracle.import_keys(code)
        per_file[name] = (oracle_keys, [])  # detector emits nothing
    score = score_detection(per_file, py_spec.canonical_target)
    assert not score.passes(min_precision=0.95, min_recall=0.90)
    assert score.recall < 0.90


def test_hallucinating_detector_fails_precision(enc, py_spec, py_oracle):
    """The 'emit garbage' reward hack must FAIL: precision collapses."""
    per_file = {}
    for name, code in PY_SAMPLES.items():
        oracle_keys = py_oracle.import_keys(code)
        # emit the real ones PLUS fabricated targets
        fake = ["totally/made/up.py", "another/hallucination.py"]
        real = [k + ".py" for k in oracle_keys]
        per_file[name] = (oracle_keys, real + fake)
    score = score_detection(per_file, py_spec.canonical_target)
    assert not score.passes(min_precision=0.95, min_recall=0.90)
    assert score.precision < 0.95


def test_precision_recall_math():
    pr = PrecisionRecall(tp=8, fp=2, fn=0)
    assert pr.precision == pytest.approx(0.8)
    assert pr.recall == 1.0
    empty = PrecisionRecall(0, 0, 0)
    assert empty.precision == 1.0 and empty.recall == 1.0  # vacuous
    summed = pr + PrecisionRecall(2, 0, 4)
    assert (summed.tp, summed.fp, summed.fn) == (10, 2, 4)


def test_go_oracle_extracts_import_paths():
    go_spec = get_spec("go")
    oracle = TreeSitterOracle(go_spec)
    src = (
        'package main\n'
        'import "fmt"\n'
        'import (\n'
        '    "os"\n'
        '    "github.com/gin-gonic/gin"\n'
        '    m "github.com/x/y/mymod"\n'
        '    "./local"\n'
        ')\n'
    )
    keys = oracle.import_keys(src)
    assert "fmt" in keys
    assert "os" in keys
    assert "github.com/gin-gonic/gin" in keys
    assert "github.com/x/y/mymod" in keys
    assert "./local" in keys


def test_spec_requires_exactly_one_detection_path():
    from data.graph_harness.spec import LanguageSpec
    with pytest.raises(ValueError):
        LanguageSpec(
            name="broken", extensions=frozenset({"x"}),
            grammar_loader=lambda: None, canonical_target=lambda s: s,
        )  # neither path set
