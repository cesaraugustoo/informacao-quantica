"""Quantum error channels and deterministic error injectors."""

from qecc.channels.pauli import (
    CHAR_TO_PAULI,
    PAULI_CHARS,
    PAULI_I,
    PAULI_X,
    PAULI_Y,
    PAULI_Z,
    ChannelSpec,
    ChannelType,
    DeterministicErrorInjector,
    StochasticPauliChannel,
    codes_to_strings,
    coherent_collective_dephasing_ops,
)

__all__ = [
    "CHAR_TO_PAULI",
    "PAULI_CHARS",
    "PAULI_I",
    "PAULI_X",
    "PAULI_Y",
    "PAULI_Z",
    "ChannelSpec",
    "ChannelType",
    "DeterministicErrorInjector",
    "StochasticPauliChannel",
    "codes_to_strings",
    "coherent_collective_dephasing_ops",
]
