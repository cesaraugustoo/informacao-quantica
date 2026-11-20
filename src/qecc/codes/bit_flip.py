"""3-qubit bit-flip repetition code ([[3, 1, 1]] / 3-data + 2-ancilla)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from qecc.backends.base import GateOp

if TYPE_CHECKING:
    from qecc.backends.base import GateName


class BitFlip3QubitCode:
    """Three-qubit bit-flip error-correcting code protecting against single-qubit X errors.

    Encoding maps ``alpha|0> + beta|1>`` (on qubit 0) to ``alpha|000> + beta|111>``
    with stabilizers ``("ZZI", "IZZ")`` extracted onto ancilla qubits 3 and 4.
    """

    name: str = "bit_flip_3"
    n_data: int = 3
    n_ancilla: int = 2
    n_total: int = 5
    stabilizers: tuple[str, ...] = ("ZZI", "IZZ")
    _SYNDROME_TABLE: dict[str, str] = {
        "00": "III",
        "10": "XII",
        "11": "IXI",
        "01": "IIX",
    }

    @property
    def syndrome_table(self) -> dict[str, str]:
        """Return a fresh copy of the syndrome-to-recovery Pauli lookup table."""
        return dict(self._SYNDROME_TABLE)

    def encode_ops(self) -> list[GateOp]:
        """Return fresh gate sequence encoding qubit 0 into data qubits 0, 1, 2."""
        return [
            GateOp("CNOT", targets=(1,), controls=(0,)),
            GateOp("CNOT", targets=(2,), controls=(0,)),
        ]

    def syndrome_ops(self) -> list[GateOp]:
        """Return fresh non-destructive stabilizer extraction gates onto ancillas 3 and 4."""
        return [
            GateOp("CNOT", targets=(3,), controls=(0,)),
            GateOp("CNOT", targets=(3,), controls=(1,)),
            GateOp("CNOT", targets=(4,), controls=(1,)),
            GateOp("CNOT", targets=(4,), controls=(2,)),
        ]

    def decode_ops(self) -> list[GateOp]:
        """Return fresh gate sequence unencoding data qubits back to qubit 0."""
        return [
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
