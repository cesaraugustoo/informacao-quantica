"""Five-qubit [[5, 1, 3]] perfect quantum error-correcting code."""

from __future__ import annotations

from typing import TYPE_CHECKING

from qecc.backends.base import GateOp

if TYPE_CHECKING:
    from qecc.backends.base import GateName


class FiveQubitPerfectCode:
    """Laflamme-Miquel-Paz-Zurek [[5, 1, 3]] perfect code (5 data + 4 ancilla qubits).

    Stabilizer generators are ``("XZZXI", "IXZZX", "XIXZZ", "ZXIXZ")``, logical operators
    are ``X_L = XXXXX`` and ``Z_L = ZZZZZ``, and all 15 single-qubit Pauli errors {X, Y, Z}
    map bijectively to the 15 non-zero 4-bit syndromes on ancilla qubits 5..8.
    """

    name: str = "five_qubit"
    n_data: int = 5
    n_ancilla: int = 4
    n_total: int = 9
    stabilizers: tuple[str, ...] = ("XZZXI", "IXZZX", "XIXZZ", "ZXIXZ")
    _SYNDROME_TABLE: dict[str, str] = {
        "0000": "IIIII",
        "0001": "XIIII",
        "1010": "ZIIII",
        "1011": "YIIII",
        "1000": "IXIII",
        "0101": "IZIII",
        "1101": "IYIII",
        "1100": "IIXII",
        "0010": "IIZII",
        "1110": "IIYII",
        "0110": "IIIXI",
        "1001": "IIIZI",
        "1111": "IIIYI",
        "0011": "IIIIX",
        "0100": "IIIIZ",
        "0111": "IIIIY",
    }

    @property
    def syndrome_table(self) -> dict[str, str]:
        """Return a fresh copy of the 16-entry syndrome-to-recovery Pauli lookup table."""
        return dict(self._SYNDROME_TABLE)

    def encode_ops(self) -> list[GateOp]:
        """Return fresh 16-gate Clifford encoder preserving exact logical relative phase."""
        return [
            GateOp("Z", targets=(0,)),
            GateOp("H", targets=(0,)),
            GateOp("CNOT", targets=(1,), controls=(0,)),
            GateOp("CNOT", targets=(2,), controls=(0,)),
            GateOp("CNOT", targets=(3,), controls=(0,)),
            GateOp("CNOT", targets=(4,), controls=(0,)),
            GateOp("H", targets=(0,)),
            GateOp("H", targets=(1,)),
            GateOp("H", targets=(2,)),
            GateOp("H", targets=(3,)),
            GateOp("H", targets=(4,)),
            GateOp("CZ", targets=(1,), controls=(0,)),
            GateOp("CZ", targets=(4,), controls=(0,)),
            GateOp("CZ", targets=(2,), controls=(1,)),
            GateOp("CZ", targets=(3,), controls=(2,)),
            GateOp("CZ", targets=(4,), controls=(3,)),
        ]

    def syndrome_ops(self) -> list[GateOp]:
        """Return fresh 24-gate non-destructive stabilizer extraction onto ancillas 5..8."""
        ops: list[GateOp] = []
        for idx, stab in enumerate(self.stabilizers):
            anc = self.n_data + idx
            ops.append(GateOp("H", targets=(anc,)))
            for q, pauli in enumerate(stab):
                if pauli == "X":
                    ops.append(GateOp("CNOT", targets=(q,), controls=(anc,)))
                elif pauli == "Z":
                    ops.append(GateOp("CZ", targets=(q,), controls=(anc,)))
            ops.append(GateOp("H", targets=(anc,)))
        return ops

    def decode_ops(self) -> list[GateOp]:
        """Return fresh 16-gate decoder (exact unitary inverse of encode_ops)."""
        return list(reversed(self.encode_ops()))

    def recovery_ops(self, syndrome: str) -> list[GateOp]:
        """Return conditional single-qubit Pauli recovery gates for a 4-bit syndrome."""
        if not isinstance(syndrome, str) or syndrome not in self._SYNDROME_TABLE:
            raise ValueError(
                f"Invalid syndrome {syndrome!r} for {self.name!r}; "
                f"expected a 4-bit binary string"
            )
        pauli_str = self._SYNDROME_TABLE[syndrome]
        ops: list[GateOp] = []
        for q, ch in enumerate(pauli_str):
            if ch != "I":
                gate_name: GateName = ch  # type: ignore[assignment]
                ops.append(GateOp(gate_name, targets=(q,)))
        return ops
