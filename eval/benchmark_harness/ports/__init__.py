"""Port adapters. Each module exposes one or more PortAdapter instances.

Builder agents author a port module; the orchestrator registers it here. The
harness code above this package is frozen.

Go (colt_go) was removed 2026-07-25: the released CoLT-132K.zip ships empty
cross_file_dependency for every Go example (aux docs live in unshipped external
`godata` JSONs), so the adapter only produced zero-aux examples. The removed
adapter is preserved in git history at commit de5ba34; recover it if the
dependency JSONs are ever obtained. See
docs/crossdoc_benchmark_port_harness_DESIGN.md §per-port notes (Go).
"""
from typing import Dict

from ..schema import PortAdapter
from .repobench import REPOBENCH_PYTHON, REPOBENCH_JAVA
from .ase_kotlin import ASE_KOTLIN
from .crosscodeeval_ts import CROSSCODEEVAL_TS

PORTS: Dict[str, PortAdapter] = {
    REPOBENCH_PYTHON.name: REPOBENCH_PYTHON,
    REPOBENCH_JAVA.name: REPOBENCH_JAVA,
    ASE_KOTLIN.name: ASE_KOTLIN,
    CROSSCODEEVAL_TS.name: CROSSCODEEVAL_TS,
}


def get_port(name: str) -> PortAdapter:
    if name not in PORTS:
        raise KeyError(f"No port named {name!r}. Registered: {sorted(PORTS)}")
    return PORTS[name]
