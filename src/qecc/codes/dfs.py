"""Two-qubit decoherence-free subspace (DFS) against collective dephasing."""

from __future__ import annotations

from qecc.backends.base import GateOp


class TwoQubitDFSDemo:
    """Two-qubit decoherence-free subspace ``span{|01>, |10>}`` with zero ancillas.

    Encoding maps ``alpha|0> + beta|1>`` (on qubit 0) to ``alpha|01> + beta|10>``,
    which is invariant up to global phase under collective dephasing ``Z (x) Z``
    and ``Rz(phi) (x) Rz(phi)`` without requiring syndrome extraction or ancillas.
    """

    name: str = "dfs_2"
    n_data: int = 2
    n_ancilla: int = 0
    n_total: int = 2
    stabilizers: tuple[str, ...] = ()

    @property
    def syndrome_table(self) -> dict[str, str]:
        """Return empty syndrome table (no ancilla measurements used by DFS)."""
        return {}

    def encode_ops(self) -> list[GateOp]:
        """Return fresh gate sequence encoding qubit 0 into ``span{|01>, |10>}``."""
        return [
            GateOp("X", targets=(1,)),
            GateOp("CNOT", targets=(1,), controls=(0,)),
        ]

    def syndrome_ops(self) -> list[GateOp]:
        """Return empty list since passive DFS uses no syndrome extraction."""
        return []

    def decode_ops(self) -> list[GateOp]:
        """Return fresh gate sequence unencoding ``span{|01>, |10>}`` back to qubit 0."""
        return [
            GateOp("CNOT", targets=(1,), controls=(0,)),
            GateOp("X", targets=(1,)),
        ]

    def recovery_ops(self, syndrome: str) -> list[GateOp]:
        """Return empty recovery for ``''``; raise ValueError for any invalid syndrome."""
        if not isinstance(syndrome, str) or syndrome != "":
            raise ValueError(
                f"Invalid syndrome {syndrome!r} for zero-ancilla code {self.name!r}; expected ''"
            )
        return []
