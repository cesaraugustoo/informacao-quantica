"""3-qubit phase-flip code (Hadamard-dual 3-data + 2-ancilla code)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from qecc.backends.base import GateOp

if TYPE_CHECKING:
    from qecc.backends.base import GateName


class PhaseFlip3QubitCode:
    """Three-qubit phase-flip error-correcting code protecting against single-qubit Z errors.

    Encoding maps ``alpha|0> + beta|1>`` (on qubit 0) to ``alpha|+++> + beta|--->``
    with stabilizers ``("XXI", "IXX")`` extracted onto ancilla qubits 3 and 4.
    """

    name: str = "phase_flip_3"
    n_data: int = 3
    n_ancilla: int = 2
    n_total: int = 5
    stabilizers: tuple[str, ...] = ("XXI", "IXX")
    _SYNDROME_TABLE: dict[str, str] = {
        "00": "III",
        "10": "ZII",
        "11": "IZI",
        "01": "IIZ",
    }

    @property
    def syndrome_table(self) -> dict[str, str]:
        """Return a fresh copy of the syndrome-to-recovery Pauli lookup table."""
        return dict(self._SYNDROME_TABLE)

    def encode_ops(self) -> list[GateOp]:
        """Return fresh gate sequence encoding qubit 0 into the Hadamard-rotated repetition space."""
        return [
            GateOp("CNOT", targets=(1,), controls=(0,)),
            GateOp("CNOT", targets=(2,), controls=(0,)),
            GateOp("H", targets=(0,)),
            GateOp("H", targets=(1,)),
            GateOp("H", targets=(2,)),
        ]

    def syndrome_ops(self) -> list[GateOp]:
        """Return fresh non-destructive X-stabilizer extraction gates onto ancillas 3 and 4."""
        return [
            GateOp("H", targets=(3,)),
            GateOp("CNOT", targets=(0,), controls=(3,)),
            GateOp("CNOT", targets=(1,), controls=(3,)),
            GateOp("H", targets=(3,)),
            GateOp("H", targets=(4,)),
            GateOp("CNOT", targets=(1,), controls=(4,)),
            GateOp("CNOT", targets=(2,), controls=(4,)),
            GateOp("H", targets=(4,)),
        ]

    def decode_ops(self) -> list[GateOp]:
        """Return fresh gate sequence unencoding data qubits back to qubit 0."""
        return [
            GateOp("H", targets=(0,)),
            GateOp("H", targets=(1,)),
            GateOp("H", targets=(2,)),
            GateOp("CNOT", targets=(2,), controls=(0,)),
            GateOp("CNOT", targets=(1,), controls=(0,)),
        ]

    def recovery_ops(self, syndrome: str) -> list[GateOp]:
        """Return conditional recovery gates for a 2-bit ancilla syndrome string."""
        if not isinstance(syndrome, str) or syndrome not in self._SYNDROME_TABLE:
            raise ValueError(
                f"Invalid syndrome {syndrome!r} for {self.name!r}; "
                f"expected one of {sorted(self._SYNDROME_TABLE)}"
            )
        pauli_str = self._SYNDROME_TABLE[syndrome]
        ops: list[GateOp] = []
        for q, ch in enumerate(pauli_str):
            if ch != "I":
                gate_name: GateName = ch  # type: ignore[assignment]
                ops.append(GateOp(gate_name, targets=(q,)))
        return ops
