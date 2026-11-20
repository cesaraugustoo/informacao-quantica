"""Stochastic Pauli error channels and deterministic error injectors."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Literal, Sequence

import numpy as np
from pydantic import ConfigDict, field_validator
from pydantic.dataclasses import dataclass as pydantic_dataclass

if TYPE_CHECKING:
    from qecc.backends.base import GateOp

PAULI_I: int = 0
PAULI_X: int = 1
PAULI_Y: int = 2
PAULI_Z: int = 3
PAULI_CHARS: tuple[str, ...] = ("I", "X", "Y", "Z")
CHAR_TO_PAULI: dict[str, int] = {"I": 0, "X": 1, "Y": 2, "Z": 3}

ChannelType = Literal["bit_flip", "phase_flip", "depolarizing", "collective_dephasing"]


@pydantic_dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class ChannelSpec:
    """Validated specification of a stochastic Pauli noise channel."""

    channel_type: ChannelType
    p: float

    @field_validator("p", mode="before")
    @classmethod
    def _validate_p(cls, v: Any) -> float:
        if isinstance(v, (bool, np.bool_)):
            raise ValueError("p must be a float in [0.0, 1.0], not bool")
        try:
            fv = float(v)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"p must be a real number in [0.0, 1.0], got {v!r}") from exc
        if not math.isfinite(fv) or not (0.0 <= fv <= 1.0):
            raise ValueError(f"p must be in [0.0, 1.0], got {v!r}")
        return fv


def _validate_positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    ival = int(value)
    if ival < 1:
        raise ValueError(f"{name} must be >= 1, got {ival}")
    return ival


def codes_to_strings(codes: np.ndarray) -> list[str]:
    """Convert a (num_shots, n_qubits) or (n_qubits,) uint8 Pauli array to strings."""
    arr = np.asarray(codes)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2 or arr.shape[0] < 1 or arr.shape[1] < 1:
        raise ValueError(f"Expected 1D or 2D non-empty array, got shape {arr.shape}")
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"Expected integer dtype for Pauli codes, got {arr.dtype}")
    if np.any((arr < 0) | (arr > 3)):
        raise ValueError("Pauli codes must be in {0, 1, 2, 3}")
    return ["".join(PAULI_CHARS[int(c)] for c in row) for row in arr]


class StochasticPauliChannel:
    """PRNG-backed sampler for stochastic Pauli channels."""

    def __init__(self, spec: ChannelSpec) -> None:
        if not isinstance(spec, ChannelSpec):
            raise TypeError(f"spec must be a ChannelSpec, got {type(spec).__name__}")
        self.spec = spec

    @property
    def channel_type(self) -> ChannelType:
        return self.spec.channel_type

    @property
    def p(self) -> float:
        return self.spec.p

    def sample(
        self,
        n_qubits: int,
        num_shots: int,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Sample (num_shots, n_qubits) uint8 Pauli codes in {0=I, 1=X, 2=Y, 3=Z}."""
        nq = _validate_positive_int(n_qubits, "n_qubits")
        ns = _validate_positive_int(num_shots, "num_shots")

        if rng is None:
            rng = np.random.default_rng()
        elif not isinstance(rng, np.random.Generator):
            raise TypeError(f"rng must be an np.random.Generator or None, got {type(rng).__name__}")

        p = self.spec.p
        ctype = self.spec.channel_type

        if ctype == "bit_flip":
            u = rng.random((ns, nq))
            codes = np.where(u < p, np.uint8(PAULI_X), np.uint8(PAULI_I))
            return np.ascontiguousarray(codes, dtype=np.uint8)

        if ctype == "phase_flip":
            u = rng.random((ns, nq))
            codes = np.where(u < p, np.uint8(PAULI_Z), np.uint8(PAULI_I))
            return np.ascontiguousarray(codes, dtype=np.uint8)

        if ctype == "depolarizing":
            u = rng.random((ns, nq))
            codes = np.zeros((ns, nq), dtype=np.uint8)
            codes[u < p] = PAULI_Z
            codes[u < (2.0 * p / 3.0)] = PAULI_Y
            codes[u < (p / 3.0)] = PAULI_X
            return codes

        if ctype == "collective_dephasing":
            flips = rng.random(ns) < p
            col = np.where(flips[:, None], np.uint8(PAULI_Z), np.uint8(PAULI_I))
            return np.ascontiguousarray(np.broadcast_to(col, (ns, nq)), dtype=np.uint8)

        raise ValueError(f"Unsupported channel_type: {ctype!r}")


def coherent_collective_dephasing_ops(n_qubits: int, phi: float) -> list[GateOp]:
    """Return coherent collective dephasing gates Rz(phi)^otimes n for DFS verification."""
    from qecc.backends.base import GateOp

    nq = _validate_positive_int(n_qubits, "n_qubits")
    if isinstance(phi, (bool, np.bool_)):
        raise ValueError("phi must be a finite float, not bool")
    fphi = float(phi)
    if not math.isfinite(fphi):
        raise ValueError(f"phi must be finite, got {phi!r}")
    return [GateOp(name="RZ", targets=(q,), params=(fphi,)) for q in range(nq)]


class DeterministicErrorInjector:
    """Validation and conversion utilities for deterministic Pauli error injection."""

    @staticmethod
    def validate_pauli_string(pauli_str: str, n_qubits: int) -> str:
        nq = _validate_positive_int(n_qubits, "n_qubits")
        if not isinstance(pauli_str, str):
            raise ValueError(f"pauli_str must be a str, got {type(pauli_str).__name__}")
        if len(pauli_str) != nq:
            raise ValueError(
                f"pauli_str length {len(pauli_str)} does not match n_qubits={nq}"
            )
        for ch in pauli_str:
            if ch not in CHAR_TO_PAULI:
                raise ValueError(
                    f"Invalid Pauli character {ch!r} in {pauli_str!r}; expected one of 'I', 'X', 'Y', 'Z'"
                )
        return pauli_str

    @staticmethod
    def validate_single_qubit_error(qubit_idx: int, op: str, n_qubits: int) -> tuple[int, str]:
        nq = _validate_positive_int(n_qubits, "n_qubits")
        if isinstance(qubit_idx, bool) or not isinstance(qubit_idx, (int, np.integer)):
            raise ValueError(f"qubit_idx must be an integer, got {qubit_idx!r}")
        q = int(qubit_idx)
        if not (0 <= q < nq):
            raise ValueError(f"qubit_idx {q} out of range [0, {nq})")
        if not isinstance(op, str) or op not in CHAR_TO_PAULI:
            raise ValueError(f"op must be one of 'I', 'X', 'Y', 'Z', got {op!r}")
        return q, op

    @staticmethod
    def from_single_qubit(qubit_idx: int, op: str, n_qubits: int) -> str:
        q, valid_op = DeterministicErrorInjector.validate_single_qubit_error(
            qubit_idx, op, n_qubits
        )
        return "I" * q + valid_op + "I" * (n_qubits - q - 1)

    @staticmethod
    def to_codes(pauli_str: str, n_qubits: int | None = None) -> np.ndarray:
        if n_qubits is None:
            if not isinstance(pauli_str, str) or len(pauli_str) < 1:
                raise ValueError("pauli_str must be a non-empty string when n_qubits is None")
            nq = len(pauli_str)
        else:
            nq = _validate_positive_int(n_qubits, "n_qubits")
        valid = DeterministicErrorInjector.validate_pauli_string(pauli_str, nq)
        return np.array([CHAR_TO_PAULI[c] for c in valid], dtype=np.uint8)

    @staticmethod
    def batch_to_codes(pauli_strings: Sequence[str], n_qubits: int) -> np.ndarray:
        nq = _validate_positive_int(n_qubits, "n_qubits")
        if isinstance(pauli_strings, str) or len(pauli_strings) == 0:
            raise ValueError("pauli_strings must be a non-empty sequence of strings")
        rows = [DeterministicErrorInjector.to_codes(s, nq) for s in pauli_strings]
        return np.ascontiguousarray(np.vstack(rows), dtype=np.uint8)

    @staticmethod
    def to_gate_ops(
        error: str | tuple[int, str],
        n_qubits: int,
        *,
        include_identity: bool = False,
    ) -> list[GateOp]:
        from qecc.backends.base import GateOp

        if isinstance(error, str):
            pauli_str = DeterministicErrorInjector.validate_pauli_string(error, n_qubits)
        elif isinstance(error, tuple) and len(error) == 2:
            pauli_str = DeterministicErrorInjector.from_single_qubit(error[0], error[1], n_qubits)
        else:
            raise ValueError(
                f"error must be a Pauli str or (qubit_idx, op) tuple, got {error!r}"
            )
        return [
            GateOp(name=ch, targets=(q,))  # type: ignore[arg-type]
            for q, ch in enumerate(pauli_str)
            if include_identity or ch != "I"
        ]
