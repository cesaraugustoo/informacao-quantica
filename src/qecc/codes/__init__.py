"""Quantum error-correcting codes and decoherence-free subspace implementations."""

from __future__ import annotations

from qecc.backends.base import QECCCode
from qecc.codes.bit_flip import BitFlip3QubitCode
from qecc.codes.dfs import TwoQubitDFSDemo
from qecc.codes.five_qubit import FiveQubitPerfectCode
from qecc.codes.phase_flip import PhaseFlip3QubitCode

_CODE_REGISTRY: dict[str, type[QECCCode]] = {
    "bit_flip_3": BitFlip3QubitCode,
    "phase_flip_3": PhaseFlip3QubitCode,
    "five_qubit": FiveQubitPerfectCode,
    "dfs_2": TwoQubitDFSDemo,
}


def list_available_codes() -> list[str]:
    """Return supported QECC/DFS code names."""
    return list(_CODE_REGISTRY.keys())


def get_code(name: str) -> QECCCode:
    """Instantiate a QECC or DFS code by registered name."""
    if not isinstance(name, str) or name not in _CODE_REGISTRY:
        raise ValueError(
            f"Unknown code {name!r}. Available codes: {list_available_codes()}"
        )
    return _CODE_REGISTRY[name]()


__all__ = [
    "BitFlip3QubitCode",
    "FiveQubitPerfectCode",
    "PhaseFlip3QubitCode",
    "TwoQubitDFSDemo",
    "get_code",
    "list_available_codes",
]
