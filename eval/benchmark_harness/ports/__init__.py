"""Port adapters. Each module exposes one or more PortAdapter instances.

Builder agents add their port module here and register it in PORTS; the
harness code above this package is frozen.
"""
from typing import Dict

from ..schema import PortAdapter
from .repobench import REPOBENCH_PYTHON, REPOBENCH_JAVA

PORTS: Dict[str, PortAdapter] = {
    REPOBENCH_PYTHON.name: REPOBENCH_PYTHON,
    REPOBENCH_JAVA.name: REPOBENCH_JAVA,
}


def get_port(name: str) -> PortAdapter:
    if name not in PORTS:
        raise KeyError(f"No port named {name!r}. Registered: {sorted(PORTS)}")
    return PORTS[name]
