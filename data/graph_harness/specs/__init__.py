"""Per-language LanguageSpec registry for the conformance harness.

Add a language by writing a spec module here and registering it in SPECS. The
harness scoring code never changes.
"""
from ..spec import LanguageSpec
from .python_spec import PYTHON_SPEC
from .go_spec import GO_SPEC

SPECS = {
    PYTHON_SPEC.name: PYTHON_SPEC,
    GO_SPEC.name: GO_SPEC,
}


def get_spec(name: str) -> LanguageSpec:
    if name not in SPECS:
        raise KeyError(
            f"No LanguageSpec named {name!r}. Registered: {sorted(SPECS)}"
        )
    return SPECS[name]


__all__ = ["SPECS", "get_spec", "PYTHON_SPEC", "GO_SPEC"]
