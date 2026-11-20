"""Core backend protocols, gate operations, and shared cycle validation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Protocol, Sequence, runtime_checkable

import numpy as np

from qecc.channels.pauli import ChannelSpec, DeterministicErrorInjector

GateName = Literal["I", "X", "Y", "Z", "H", "S", "Sdg", "RX", "RY", "RZ", "CNOT", "CZ"]

VALID_GATES: frozenset[str] = frozenset(
    {"I", "X", "Y", "Z", "H", "S", "Sdg", "RX", "RY", "RZ", "CNOT", "CZ"}
)
SINGLE_QUBIT_NON_PARAM: frozenset[str] = frozenset({"I", "X", "Y", "Z", "H", "S", "Sdg"})
SINGLE_QUBIT_ROTATION: frozenset[str] = frozenset({"RX", "RY", "RZ"})
TWO_QUBIT_CONTROLLED: frozenset[str] = frozenset({"CNOT", "CZ"})


def _coerce_qubit_tuple(qubits: Sequence[int], field_name: str) -> tuple[int, ...]:
    out: list[int] = []
    for q in qubits:
        if isinstance(q, bool) or not isinstance(q, (int, np.integer)):
            raise ValueError(f"{field_name} entries must be integers, got {q!r}")
        iq = int(q)
        if iq < 0:
            raise ValueError(f"{field_name} qubit index must be >= 0, got {iq}")
        out.append(iq)
    return tuple(out)


def _coerce_param_tuple(params: Sequence[float]) -> tuple[float, ...]:
    out: list[float] = []
    for val in params:
        if isinstance(val, (bool, np.bool_)):
            raise ValueError(f"Gate parameter must be a finite float, got {val!r}")
        try:
            fval = float(val)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Gate parameter must be numeric, got {val!r}") from exc
        if not math.isfinite(fval):
            raise ValueError(f"Gate parameter must be finite, got {val!r}")
        out.append(fval)
    return tuple(out)


@dataclass(frozen=True, slots=True)
class GateOp:
    """Immutable validated quantum gate operation."""

    name: GateName
    targets: tuple[int, ...]
    controls: tuple[int, ...] = ()
    params: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if self.name not in VALID_GATES:
            raise ValueError(f"Unsupported gate name {self.name!r}; expected one of {sorted(VALID_GATES)}")

        targets = _coerce_qubit_tuple(self.targets, "targets")
        controls = _coerce_qubit_tuple(self.controls, "controls")
        params = _coerce_param_tuple(self.params)

        if len(set(targets)) != len(targets):
            raise ValueError(f"Duplicate qubit indices in targets={targets}")
        if len(set(controls)) != len(controls):
            raise ValueError(f"Duplicate qubit indices in controls={controls}")
        if not set(targets).isdisjoint(controls):
            raise ValueError(f"targets {targets} and controls {controls} must be disjoint")

        if self.name in SINGLE_QUBIT_NON_PARAM:
            if len(targets) != 1 or len(controls) != 0 or len(params) != 0:
                raise ValueError(
                    f"Gate {self.name} requires 1 target, 0 controls, 0 params; "
                    f"got targets={targets}, controls={controls}, params={params}"
                )
        elif self.name in SINGLE_QUBIT_ROTATION:
            if len(targets) != 1 or len(controls) != 0 or len(params) != 1:
                raise ValueError(
                    f"Gate {self.name} requires 1 target, 0 controls, 1 param; "
                    f"got targets={targets}, controls={controls}, params={params}"
                )
        elif self.name in TWO_QUBIT_CONTROLLED:
            if len(controls) != 1 or len(targets) != 1 or len(params) != 0:
                raise ValueError(
                    f"Gate {self.name} requires 1 control, 1 target, 0 params; "
                    f"got targets={targets}, controls={controls}, params={params}"
                )

        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "controls", controls)
        object.__setattr__(self, "params", params)


def validate_ops_qubits(ops: Sequence[GateOp], n_total: int) -> None:
    """Validate that all gate qubit indices lie within [0, n_total)."""
    if isinstance(n_total, bool) or not isinstance(n_total, (int, np.integer)) or int(n_total) < 1:
        raise ValueError(f"n_total must be an integer >= 1, got {n_total!r}")
    limit = int(n_total)
    for op in ops:
        if not isinstance(op, GateOp):
            raise TypeError(f"Expected GateOp, got {type(op).__name__}")
        for q in op.targets:
            if not (0 <= q < limit):
                raise ValueError(f"Gate {op.name} target qubit {q} out of range [0, {limit})")
        for q in op.controls:
            if not (0 <= q < limit):
                raise ValueError(f"Gate {op.name} control qubit {q} out of range [0, {limit})")


@runtime_checkable
class QECCCode(Protocol):
    """Protocol implemented by QECC codes and baselines."""

    name: str
    n_data: int
    n_ancilla: int
    n_total: int
    stabilizers: tuple[str, ...]
    syndrome_table: dict[str, str]

    def encode_ops(self) -> list[GateOp]: ...
    def syndrome_ops(self) -> list[GateOp]: ...
    def decode_ops(self) -> list[GateOp]: ...
    def recovery_ops(self, syndrome: str) -> list[GateOp]: ...


@dataclass(slots=True)
class SimulationResult:
    """Output container returned by QECCBackend.simulate_cycle."""

    code_name: str
    backend_name: str
    p: float
    num_shots: int
    syndromes: list[str]
    syndrome_counts: dict[str, int]
    rho_out: np.ndarray
    fidelities: np.ndarray
    mean_fidelity: float
    std_error: float
    elapsed_seconds: float


@runtime_checkable
class QECCBackend(Protocol):
    """Protocol implemented by swappable QECC simulation backends."""

    name: str

    def simulate_cycle(
        self,
        code: QECCCode,
        initial_state: np.ndarray,
        channel: ChannelSpec | str | None,
        p: float,
        num_shots: int,
        seed: int | None = None,
        deterministic_errors: list[str] | str | None = None,
    ) -> SimulationResult: ...


@dataclass(slots=True)
class PreparedCycleInputs:
    """Validated and normalized inputs shared across all QECCBackend implementations."""

    psi: np.ndarray
    n_data: int
    n_ancilla: int
    n_total: int
    p: float
    num_shots: int
    encode_ops: list[GateOp]
    syndrome_ops: list[GateOp]
    decode_ops: list[GateOp]
    error_mode: Literal["none", "deterministic", "stochastic"]
    deterministic_codes: np.ndarray | None
    channel_spec: ChannelSpec | None
    rng: np.random.Generator


def validate_and_prepare_cycle_inputs(
    code: QECCCode,
    initial_state: np.ndarray,
    channel: ChannelSpec | str | None,
    p: float,
    num_shots: int,
    seed: int | None = None,
    deterministic_errors: list[str] | str | None = None,
) -> PreparedCycleInputs:
    """Validate and normalize inputs identically across all QECCBackend implementations."""
    if not isinstance(code, QECCCode):
        raise TypeError(f"code must implement QECCCode protocol, got {type(code).__name__}")

    for attr in ("n_data", "n_ancilla", "n_total"):
        val = getattr(code, attr)
        if isinstance(val, bool) or not isinstance(val, (int, np.integer)):
            raise ValueError(f"code.{attr} must be an integer, got {val!r}")

    n_data = int(code.n_data)
    n_ancilla = int(code.n_ancilla)
    n_total = int(code.n_total)
    if n_data < 1 or n_ancilla < 0 or n_total != n_data + n_ancilla:
        raise ValueError(
            f"Invalid code qubit counts: n_data={n_data}, n_ancilla={n_ancilla}, n_total={n_total}"
        )

    psi = np.asarray(initial_state, dtype=np.complex128)
    if psi.shape != (2,) or not np.all(np.isfinite(psi)):
        raise ValueError(f"initial_state must be a finite shape-(2,) vector, got shape {psi.shape}")
    norm_sq = float(np.vdot(psi, psi).real)
    if abs(norm_sq - 1.0) > 1e-6:
        raise ValueError(f"initial_state must be unit-normalized (||psi||^2 = {norm_sq})")
    psi = psi / math.sqrt(norm_sq)

    if isinstance(num_shots, bool) or not isinstance(num_shots, (int, np.integer)):
        raise ValueError(f"num_shots must be a positive integer, got {num_shots!r}")
    shots = int(num_shots)
    if shots < 1:
        raise ValueError(f"num_shots must be >= 1, got {shots}")

    if isinstance(p, (bool, np.bool_)):
        raise ValueError("p must be a float in [0.0, 1.0], not bool")
    try:
        fp = float(p)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"p must be a float in [0.0, 1.0], got {p!r}") from exc
    if not math.isfinite(fp) or not (0.0 <= fp <= 1.0):
        raise ValueError(f"p must be in [0.0, 1.0], got {p!r}")

    if seed is None:
        rng = np.random.default_rng()
    elif isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError(f"seed must be an integer or None, got {seed!r}")
    else:
        rng = np.random.default_rng(int(seed))

    encode_ops = list(code.encode_ops())
    syndrome_ops = list(code.syndrome_ops())
    decode_ops = list(code.decode_ops())
    validate_ops_qubits(encode_ops, n_total)
    validate_ops_qubits(syndrome_ops, n_total)
    validate_ops_qubits(decode_ops, n_total)

    error_mode: Literal["none", "deterministic", "stochastic"]
    deterministic_codes: np.ndarray | None = None
    channel_spec: ChannelSpec | None = None

    if deterministic_errors is not None:
        if channel is not None:
            raise ValueError("Cannot specify both channel and deterministic_errors")
        if isinstance(deterministic_errors, str):
            deterministic_codes = DeterministicErrorInjector.to_codes(
                deterministic_errors, n_data
            )[None, :]
        elif isinstance(deterministic_errors, (list, tuple)):
            if len(deterministic_errors) not in (1, shots):
                raise ValueError(
                    f"deterministic_errors list length ({len(deterministic_errors)}) "
                    f"must be 1 or match num_shots ({shots})"
                )
            deterministic_codes = DeterministicErrorInjector.batch_to_codes(
                deterministic_errors, n_data
            )
        else:
            raise ValueError(
                "deterministic_errors must be a str, list[str], or None"
            )
        error_mode = "deterministic"
    else:
        if channel is None:
            if fp > 0.0:
                raise ValueError("channel must be provided when p > 0 and deterministic_errors is None")
            error_mode = "none"
        else:
            if isinstance(channel, str):
                channel_spec = ChannelSpec(channel_type=channel, p=fp)  # type: ignore[arg-type]
            elif isinstance(channel, ChannelSpec):
                if not math.isclose(channel.p, fp, rel_tol=0.0, abs_tol=1e-12):
                    raise ValueError(f"channel.p ({channel.p}) must match p ({fp})")
                channel_spec = channel
            else:
                raise ValueError(
                    f"channel must be ChannelSpec, str, or None, got {type(channel).__name__}"
                )
            error_mode = "none" if fp == 0.0 else "stochastic"

    return PreparedCycleInputs(
        psi=psi,
        n_data=n_data,
        n_ancilla=n_ancilla,
        n_total=n_total,
        p=fp,
        num_shots=shots,
        encode_ops=encode_ops,
        syndrome_ops=syndrome_ops,
        decode_ops=decode_ops,
        error_mode=error_mode,
        deterministic_codes=deterministic_codes,
        channel_spec=channel_spec,
        rng=rng,
    )
