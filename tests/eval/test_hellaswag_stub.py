"""
tests/eval/test_hellaswag_stub.py — verifies the HellaSwag stub behaviour.
"""
import pytest

from eval.hellaswag import run_hellaswag


def test_raises_not_implemented():
    """run_hellaswag always raises NotImplementedError until NL data is online."""
    with pytest.raises(NotImplementedError, match="HellaSwag eval deferred"):
        run_hellaswag(model=None)


def test_raises_not_implemented_with_args():
    """Verify the guard fires even when optional args are supplied."""
    with pytest.raises(NotImplementedError):
        run_hellaswag(model=None, max_examples=100, device="cpu")


def test_module_imports_without_datasets_package(monkeypatch):
    """eval.hellaswag can be imported even if the 'datasets' HuggingFace
    package is unavailable, because its import is deferred inside the stub."""
    import sys
    import importlib

    # Simulate datasets being unavailable
    original = sys.modules.get("datasets")
    sys.modules["datasets"] = None  # type: ignore[assignment]

    try:
        # Re-import to exercise the import path
        if "eval.hellaswag" in sys.modules:
            del sys.modules["eval.hellaswag"]
        import eval.hellaswag  # noqa: F401  — should not raise
    finally:
        if original is None:
            sys.modules.pop("datasets", None)
        else:
            sys.modules["datasets"] = original
