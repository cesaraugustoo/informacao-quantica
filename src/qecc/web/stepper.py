"""Stage-by-stage exact big-endian QECC & DFS statevector stepper."""

from __future__ import annotations

import cmath
import math
from functools import lru_cache
from typing import Any, Sequence

import numpy as np

from qecc.analysis.analytical import VALID_CHANNELS, _logical_basis_states
from qecc.backends.base import GateOp, QECCCode, validate_ops_qubits
from qecc.backends.numpy_backend import _apply_ops_np
from qecc.channels.pauli import (
    PAULI_CHARS,
    ChannelSpec,
    DeterministicErrorInjector,
    StochasticPauliChannel,
    coherent_collective_dephasing_ops,
)
from qecc.codes import get_code
from qecc.codes.dfs import TwoQubitDFSDemo
from qecc.metrics.fidelity import partial_trace_to_qubit0, quantum_fidelity
from qecc.metrics.states import InitialStateSampler

STAGE_KEYS: tuple[str, ...] = (
    "initial",
    "encoded",
    "noisy",
    "syndrome_extracted",
    "recovered",
    "decoded",
)

VALID_ERROR_MODES: frozenset[str] = frozenset(
    {
        "pauli_string",
        "single_qubit_pauli",
        "stochastic_sample",
        "su2_rotation",
        "coherent_collective",
    }
)


class _RawSingleQubitCode:
    """Private unprotected 1-qubit baseline implementing ``QECCCode`` protocol.

    Kept private to ``qecc.web`` so ``qecc.codes._CODE_REGISTRY`` remains untouched.
    """

    name: str = "raw"
    n_data: int = 1
    n_ancilla: int = 0
    n_total: int = 1
    stabilizers: tuple[str, ...] = ()

    @property
    def syndrome_table(self) -> dict[str, str]:
        return {}

    def encode_ops(self) -> list[GateOp]:
        return []

    def syndrome_ops(self) -> list[GateOp]:
        return []

    def decode_ops(self) -> list[GateOp]:
        return []

    def recovery_ops(self, syndrome: str) -> list[GateOp]:
        if not isinstance(syndrome, str) or syndrome != "":
            raise ValueError(
                f"Invalid syndrome {syndrome!r} for zero-ancilla code {self.name!r}; expected ''"
            )
        return []


def validate_finite_real(value: Any, name: str) -> float:
    """Validate that ``value`` is a finite real number (rejecting booleans)."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a finite real number, got {value!r}")
    fval = float(value)
    if not math.isfinite(fval):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return fval


def validate_nonbool_int(value: Any, name: str) -> int:
    """Validate that ``value`` is an integer (rejecting booleans)."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    return int(value)


def serialize_gate_op(op: GateOp) -> dict[str, Any]:
    """Serialize a ``GateOp`` into a JSON-compatible dictionary."""
    return {
        "name": op.name,
        "targets": [int(q) for q in op.targets],
        "controls": [int(q) for q in op.controls],
        "params": [float(p) for p in op.params],
    }


def serialize_complex_matrix(mat: np.ndarray) -> list[list[list[float]]]:
    """Serialize a 2D complex NumPy array as nested ``[[re, im], ...]`` lists."""
    arr = np.asarray(mat, dtype=np.complex128)
    return [
        [[float(np.real(val)), float(np.imag(val))] for val in row]
        for row in arr
    ]


def rho_to_bloch_and_coherence(rho: np.ndarray) -> tuple[list[float], float]:
    """Extract Bloch vector ``[rx, ry, rz]`` and ``|rho_01|`` from a 2x2 matrix."""
    rx = float(2.0 * np.real(rho[0, 1]))
    ry = float(-2.0 * np.imag(rho[0, 1]))
    rz = float(np.real(rho[0, 0] - rho[1, 1]))
    abs_rho01 = float(abs(rho[0, 1]))
    return [rx, ry, rz], abs_rho01


def resolve_input_state(
    state_preset: str,
    *,
    theta: float | None = None,
    phi_state: float | None = None,
    amplitudes: Sequence[Sequence[float]] | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Resolve cardinal, Bloch-angle, custom-amplitude, or Haar single-qubit state."""
    if not isinstance(state_preset, str):
        raise ValueError(f"state_preset must be a string, got {state_preset!r}")

    cardinals = InitialStateSampler.cardinal_states()
    if state_preset in cardinals:
        return cardinals[state_preset].copy()

    if state_preset == "bloch":
        if theta is None:
            raise ValueError("theta is required when state_preset == 'bloch'")
        f_theta = validate_finite_real(theta, "theta")
        f_phi = 0.0 if phi_state is None else validate_finite_real(phi_state, "phi_state")
        psi = np.array(
            [
                math.cos(0.5 * f_theta),
                cmath.exp(1.0j * f_phi) * math.sin(0.5 * f_theta),
            ],
            dtype=np.complex128,
        )
        norm = float(np.linalg.norm(psi))
        return np.asarray(psi / norm, dtype=np.complex128)

    if state_preset == "custom":
        if amplitudes is None or isinstance(amplitudes, (str, bytes)) or len(amplitudes) != 2:
            raise ValueError("amplitudes must be a 2-element list [[re0, im0], [re1, im1]]")
        parsed: list[complex] = []
        for idx, pair in enumerate(amplitudes):
            if isinstance(pair, (str, bytes)) or len(pair) != 2:
                raise ValueError(f"amplitudes[{idx}] must be a [re, im] pair")
            re_val = validate_finite_real(pair[0], f"amplitudes[{idx}][0]")
            im_val = validate_finite_real(pair[1], f"amplitudes[{idx}][1]")
            parsed.append(complex(re_val, im_val))
        psi = np.array(parsed, dtype=np.complex128)
        norm_sq = float(np.vdot(psi, psi).real)
        if abs(norm_sq - 1.0) > 1e-6:
            raise ValueError(
                f"custom initial_state must be unit-normalized (||psi||^2 = {norm_sq})"
            )
        return np.asarray(psi / math.sqrt(norm_sq), dtype=np.complex128)

    if state_preset == "haar":
        if seed is not None:
            seed = validate_nonbool_int(seed, "seed")
        return InitialStateSampler.haar_random_states(1, seed=seed)[0]

    raise ValueError(f"Unsupported state_preset {state_preset!r}")


@lru_cache(maxsize=8)
def _cached_logical_basis_matrix(code_name: str) -> np.ndarray:
    """Return ``(2**n_data, 2)`` column matrix ``[|0_L>, |1_L>]`` in big-endian order."""
    if code_name == "raw":
        return np.eye(2, dtype=np.complex128)
    code = get_code(code_name)
    ket0_l, ket1_l = _logical_basis_states(code)
    return np.column_stack((ket0_l, ket1_l))


def build_su2_rotation(
    qubit: int,
    theta: float,
    nx: float,
    ny: float,
    nz: float,
    n_data: int,
) -> tuple[list[GateOp], dict[str, dict[str, float]], str]:
    """Construct exact ``GateOp`` list and Pauli coefficients for ``U(theta, n_hat)``."""
    q = validate_nonbool_int(qubit, "su2_qubit")
    if not (0 <= q < n_data):
        raise ValueError(f"su2_qubit {q} out of range [0, {n_data})")
    f_theta = validate_finite_real(theta, "su2_theta")
    f_nx = validate_finite_real(nx, "su2_nx")
    f_ny = validate_finite_real(ny, "su2_ny")
    f_nz = validate_finite_real(nz, "su2_nz")

    axis_norm = math.hypot(f_nx, f_ny, f_nz)
    if axis_norm <= 1e-12:
        raise ValueError("SU(2) rotation axis (su2_nx, su2_ny, su2_nz) must be non-zero")

    ux = f_nx / axis_norm
    uy = f_ny / axis_norm
    uz = f_nz / axis_norm

    half = 0.5 * f_theta
    c_val = math.cos(half)
    s_val = math.sin(half)

    pauli_coeffs = {
        "I": {"re": float(c_val), "im": 0.0, "prob": float(c_val * c_val)},
        "X": {"re": 0.0, "im": float(-ux * s_val), "prob": float((ux * s_val) ** 2)},
        "Y": {"re": 0.0, "im": float(-uy * s_val), "prob": float((uy * s_val) ** 2)},
        "Z": {"re": 0.0, "im": float(-uz * s_val), "prob": float((uz * s_val) ** 2)},
    }

    if abs(abs(ux) - 1.0) < 1e-12 and abs(uy) < 1e-12 and abs(uz) < 1e-12:
        ops = [GateOp(name="RX", targets=(q,), params=(ux * f_theta,))]
    elif abs(abs(uy) - 1.0) < 1e-12 and abs(ux) < 1e-12 and abs(uz) < 1e-12:
        ops = [GateOp(name="RY", targets=(q,), params=(uy * f_theta,))]
    elif abs(abs(uz) - 1.0) < 1e-12 and abs(ux) < 1e-12 and abs(uy) < 1e-12:
        ops = [GateOp(name="RZ", targets=(q,), params=(uz * f_theta,))]
    else:
        u00 = complex(c_val, -uz * s_val)
        u10 = complex(uy * s_val, -ux * s_val)
        abs_u00 = abs(u00)
        abs_u10 = abs(u10)
        beta = 2.0 * math.atan2(abs_u10, abs_u00)

        if abs_u00 < 1e-15:
            gamma = 0.0
            alpha = 2.0 * cmath.phase(u10)
        elif abs_u10 < 1e-15:
            gamma = 0.0
            alpha = -2.0 * cmath.phase(u00)
        else:
            sum_half = -cmath.phase(u00)
            diff_half = cmath.phase(u10)
            alpha = sum_half + diff_half
            gamma = sum_half - diff_half

        ops = [
            GateOp(name="RZ", targets=(q,), params=(float(gamma),)),
            GateOp(name="RY", targets=(q,), params=(float(beta),)),
            GateOp(name="RZ", targets=(q,), params=(float(alpha),)),
        ]

    label = f"SU2(q={q}, theta={f_theta:.4g}, n=[{ux:.3g}, {uy:.3g}, {uz:.3g}])"
    return ops, pauli_coeffs, label


def _project_logical_rho(
    vec: np.ndarray,
    n_data: int,
    n_ancilla: int,
    basis_matrix: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Trace out ancillas and project onto ``{|0_L>, |1_L>}`` without renormalizing leakage."""
    da = vec.reshape(1 << n_data, 1 << n_ancilla)
    rho_data = da @ da.conj().T
    projected = basis_matrix.conj().T @ rho_data @ basis_matrix
    projected = np.asarray(0.5 * (projected + projected.conj().T), dtype=np.complex128)
    subspace_prob = float(np.real(np.trace(projected)))
    return projected, subspace_prob


def _format_nonzero_statevector(
    vec: np.ndarray,
    n_total: int,
    *,
    max_terms: int = 32,
    prob_tol: float = 1e-12,
) -> list[dict[str, Any]]:
    """Return top computational basis amplitudes sorted by descending probability."""
    probs = (vec.real * vec.real) + (vec.imag * vec.imag)
    indices = np.flatnonzero(probs > prob_tol)
    if indices.size == 0:
        return []
    order = sorted(
        (int(idx) for idx in indices),
        key=lambda idx: (-float(probs[idx]), idx),
    )[:max_terms]
    return [
        {
            "basis": format(idx, f"0{n_total}b"),
            "re": float(vec[idx].real),
            "im": float(vec[idx].imag),
            "prob": float(probs[idx]),
        }
        for idx in order
    ]


def _build_error_for_code(
    code: QECCCode,
    error_mode: str,
    *,
    pauli_string: str | None = None,
    error_qubit: int | None = None,
    error_op: str | None = None,
    channel: str | None = None,
    p: float | None = None,
    seed: int | None = None,
    su2_qubit: int | None = None,
    su2_theta: float | None = None,
    su2_nx: float | None = None,
    su2_ny: float | None = None,
    su2_nz: float | None = None,
    collective_phi: float | None = None,
) -> tuple[list[GateOp], dict[str, dict[str, float]] | None, str]:
    """Validate error parameters and construct ``GateOp`` list for ``code``."""
    if error_mode not in VALID_ERROR_MODES:
        raise ValueError(
            f"Unsupported error_mode {error_mode!r}; expected one of {sorted(VALID_ERROR_MODES)}"
        )

    n_data = int(code.n_data)

    if error_mode == "pauli_string":
        if pauli_string is None:
            raise ValueError("pauli_string is required when error_mode == 'pauli_string'")
        valid_p = DeterministicErrorInjector.validate_pauli_string(pauli_string, n_data)
        ops = DeterministicErrorInjector.to_gate_ops(valid_p, n_data)
        return ops, None, valid_p

    if error_mode == "single_qubit_pauli":
        if error_qubit is None or error_op is None:
            raise ValueError(
                "error_qubit and error_op are required when error_mode == 'single_qubit_pauli'"
            )
        q = validate_nonbool_int(error_qubit, "error_qubit")
        p_str = DeterministicErrorInjector.from_single_qubit(q, error_op, n_data)
        ops = DeterministicErrorInjector.to_gate_ops(p_str, n_data)
        return ops, None, p_str

    if error_mode == "stochastic_sample":
        if channel is None or channel not in VALID_CHANNELS:
            raise ValueError(f"channel must be one of {VALID_CHANNELS}, got {channel!r}")
        if p is None:
            raise ValueError("p is required when error_mode == 'stochastic_sample'")
        fp = validate_finite_real(p, "p")
        if not (0.0 <= fp <= 1.0):
            raise ValueError(f"p must lie in [0.0, 1.0], got {fp}")
        if seed is not None:
            seed = validate_nonbool_int(seed, "seed")
        rng = np.random.default_rng(seed)
        spec = ChannelSpec(channel_type=channel, p=fp)  # type: ignore[arg-type]
        sampled_codes = StochasticPauliChannel(spec).sample(n_data, 1, rng)[0]
        sampled_str = "".join(PAULI_CHARS[int(c)] for c in sampled_codes)
        ops = DeterministicErrorInjector.to_gate_ops(sampled_str, n_data)
        return ops, None, f"{channel} (p={fp:.4g}): {sampled_str}"

    if error_mode == "su2_rotation":
        if (
            su2_qubit is None
            or su2_theta is None
            or su2_nx is None
            or su2_ny is None
            or su2_nz is None
        ):
            raise ValueError(
                "su2_qubit, su2_theta, su2_nx, su2_ny, and su2_nz are required "
                "when error_mode == 'su2_rotation'"
            )
        return build_su2_rotation(su2_qubit, su2_theta, su2_nx, su2_ny, su2_nz, n_data)

    # coherent_collective
    if collective_phi is None:
        raise ValueError("collective_phi is required when error_mode == 'coherent_collective'")
    f_phi = validate_finite_real(collective_phi, "collective_phi")
    ops = coherent_collective_dephasing_ops(n_data, f_phi)
    half = 0.5 * f_phi
    c_val = math.cos(half)
    s_val = math.sin(half)
    pauli_coeffs = {
        "I": {"re": float(c_val), "im": 0.0, "prob": float(c_val * c_val)},
        "X": {"re": 0.0, "im": 0.0, "prob": 0.0},
        "Y": {"re": 0.0, "im": 0.0, "prob": 0.0},
        "Z": {"re": 0.0, "im": float(-s_val), "prob": float(s_val * s_val)},
    }
    return ops, pauli_coeffs, f"Rz({f_phi:.4g})^{n_data}"


def _execute_code_stepper(
    code: QECCCode,
    psi: np.ndarray,
    error_ops: list[GateOp],
    selected_syndrome: str | None = None,
) -> dict[str, Any]:
    """Execute the 6-stage big-endian statevector stepper on a single ``QECCCode``."""
    n_data = int(code.n_data)
    n_ancilla = int(code.n_ancilla)
    n_total = int(code.n_total)

    validate_ops_qubits(error_ops, n_total)
    basis_matrix = _cached_logical_basis_matrix(code.name)

    state = np.zeros((1, 1 << n_total), dtype=np.complex128)
    state[0, 0] = psi[0]
    state[0, 1 << (n_total - 1)] = psi[1]

    stages: list[dict[str, Any]] = []

    def _record_stage(
        key: str,
        applied_ops: Sequence[GateOp],
        *,
        use_logical_projection: bool,
        syndrome_probs: dict[str, float] | None = None,
        chosen_syndrome: str | None = None,
        rec_ops: Sequence[GateOp] | None = None,
    ) -> None:
        vec = state[0]
        if use_logical_projection:
            rho, subspace_prob = _project_logical_rho(vec, n_data, n_ancilla, basis_matrix)
            fidelity = float(np.clip(np.real(np.vdot(psi, rho @ psi)), 0.0, 1.0))
        else:
            rho = partial_trace_to_qubit0(vec, n_total)
            subspace_prob = float(np.real(np.trace(rho)))
            fidelity = float(quantum_fidelity(psi, rho))

        bloch, abs_rho01 = rho_to_bloch_and_coherence(rho)
        stages.append(
            {
                "key": key,
                "gate_ops": [serialize_gate_op(op) for op in applied_ops],
                "logical_rho": serialize_complex_matrix(rho),
                "logical_bloch": bloch,
                "logical_abs_rho01": abs_rho01,
                "code_subspace_prob": subspace_prob,
                "fidelity_vs_input": fidelity,
                "statevector_nonzero": _format_nonzero_statevector(vec, n_total),
                "syndrome_probabilities": syndrome_probs,
                "selected_syndrome": chosen_syndrome,
                "recovery_ops": [serialize_gate_op(op) for op in rec_ops]
                if rec_ops is not None
                else [],
            }
        )

    # 1. initial
    _record_stage("initial", [], use_logical_projection=False)

    # 2. encoded
    enc_ops = list(code.encode_ops())
    validate_ops_qubits(enc_ops, n_total)
    if enc_ops:
        _apply_ops_np(state, enc_ops, n_total)
    _record_stage("encoded", enc_ops, use_logical_projection=True)

    # 3. noisy
    if error_ops:
        _apply_ops_np(state, error_ops, n_total)
    _record_stage("noisy", error_ops, use_logical_projection=True)

    # 4. syndrome_extracted
    syn_ops = list(code.syndrome_ops())
    validate_ops_qubits(syn_ops, n_total)
    if syn_ops:
        _apply_ops_np(state, syn_ops, n_total)

    if n_ancilla == 0:
        if selected_syndrome is not None and selected_syndrome != "":
            raise ValueError(
                f"Invalid selected_syndrome {selected_syndrome!r} for zero-ancilla code "
                f"{code.name!r}; expected ''"
            )
        syndrome_probs = {"": 1.0}
        chosen_syndrome = ""
    else:
        da = state[0].reshape(1 << n_data, 1 << n_ancilla)
        probs = (da.real * da.real + da.imag * da.imag).sum(axis=0)
        probs_sum = max(float(probs.sum()), 1e-30)
        probs = probs / probs_sum
        syndrome_probs = {
            format(m, f"0{n_ancilla}b"): float(probs[m])
            for m in range(1 << n_ancilla)
        }

        if selected_syndrome is not None:
            if not isinstance(selected_syndrome, str) or selected_syndrome not in code.syndrome_table:
                raise ValueError(
                    f"Invalid selected_syndrome {selected_syndrome!r} for {code.name!r}; "
                    f"expected one of {sorted(code.syndrome_table)}"
                )
            m_chosen = int(selected_syndrome, 2)
            if float(probs[m_chosen]) <= 1e-15:
                raise ValueError(
                    f"Cannot project onto zero-probability syndrome branch {selected_syndrome!r}"
                )
            chosen_syndrome = selected_syndrome
        else:
            m_chosen = int(np.argmax(probs))
            chosen_syndrome = format(m_chosen, f"0{n_ancilla}b")

        surv = da[:, m_chosen] / math.sqrt(max(float(probs[m_chosen]), 1e-30))
        da.fill(0.0)
        da[:, m_chosen] = surv

    _record_stage(
        "syndrome_extracted",
        syn_ops,
        use_logical_projection=True,
        syndrome_probs=syndrome_probs,
        chosen_syndrome=chosen_syndrome,
    )

    # 5. recovered
    rec_ops = list(code.recovery_ops(chosen_syndrome))
    validate_ops_qubits(rec_ops, n_total)
    if rec_ops:
        _apply_ops_np(state, rec_ops, n_total)
    _record_stage(
        "recovered",
        rec_ops,
        use_logical_projection=True,
        chosen_syndrome=chosen_syndrome,
        rec_ops=rec_ops,
    )

    # 6. decoded
    dec_ops = list(code.decode_ops())
    validate_ops_qubits(dec_ops, n_total)
    if dec_ops:
        _apply_ops_np(state, dec_ops, n_total)
    _record_stage("decoded", dec_ops, use_logical_projection=False)

    final_stage = stages[-1]
    return {
        "stages": stages,
        "final_summary": {
            "rho": final_stage["logical_rho"],
            "bloch": final_stage["logical_bloch"],
            "abs_rho01": final_stage["logical_abs_rho01"],
            "fidelity": final_stage["fidelity_vs_input"],
        },
    }


def _adapted_error_ops(
    target_code: QECCCode,
    error_mode: str,
    *,
    pauli_string: str | None = None,
    error_qubit: int | None = None,
    error_op: str | None = None,
    channel: str | None = None,
    p: float | None = None,
    seed: int | None = None,
    su2_qubit: int | None = None,
    su2_theta: float | None = None,
    su2_nx: float | None = None,
    su2_ny: float | None = None,
    su2_nz: float | None = None,
    collective_phi: float | None = None,
) -> list[GateOp]:
    """Adapt the active error configuration cleanly to ``raw`` (n=1) or ``dfs_2`` (n=2)."""
    n_target = int(target_code.n_data)

    if error_mode == "pauli_string":
        assert pauli_string is not None
        if n_target == 1:
            adapted_str = pauli_string[0]
        elif len(pauli_string) >= n_target:
            adapted_str = pauli_string[:n_target]
        else:
            adapted_str = pauli_string + "I" * (n_target - len(pauli_string))
        ops, _, _ = _build_error_for_code(
            target_code,
            "pauli_string",
            pauli_string=adapted_str,
        )
        return ops

    if error_mode == "single_qubit_pauli":
        assert error_qubit is not None and error_op is not None
        q = validate_nonbool_int(error_qubit, "error_qubit")
        adapted_q = q if q < n_target else 0
        ops, _, _ = _build_error_for_code(
            target_code,
            "single_qubit_pauli",
            error_qubit=adapted_q,
            error_op=error_op,
        )
        return ops

    if error_mode == "stochastic_sample":
        ops, _, _ = _build_error_for_code(
            target_code,
            "stochastic_sample",
            channel=channel,
            p=p,
            seed=seed,
        )
        return ops

    if error_mode == "su2_rotation":
        assert su2_qubit is not None
        q = validate_nonbool_int(su2_qubit, "su2_qubit")
        adapted_q = q if q < n_target else 0
        ops, _, _ = _build_error_for_code(
            target_code,
            "su2_rotation",
            su2_qubit=adapted_q,
            su2_theta=su2_theta,
            su2_nx=su2_nx,
            su2_ny=su2_ny,
            su2_nz=su2_nz,
        )
        return ops

    ops, _, _ = _build_error_for_code(
        target_code,
        "coherent_collective",
        collective_phi=collective_phi,
    )
    return ops


def run_step_evolution(
    *,
    code_name: str,
    state_preset: str = "+i",
    theta: float | None = None,
    phi_state: float | None = None,
    amplitudes: Sequence[Sequence[float]] | None = None,
    error_mode: str = "pauli_string",
    pauli_string: str | None = None,
    error_qubit: int | None = None,
    error_op: str | None = None,
    channel: str | None = None,
    p: float | None = None,
    seed: int | None = None,
    su2_qubit: int | None = None,
    su2_theta: float | None = None,
    su2_nx: float | None = None,
    su2_ny: float | None = None,
    su2_nz: float | None = None,
    collective_phi: float | None = None,
    selected_syndrome: str | None = None,
) -> dict[str, Any]:
    """Run the 6-stage QECC/DFS stepper plus 3-way ``raw`` vs ``qecc`` vs ``dfs_2`` comparison."""
    code = get_code(code_name)
    psi = resolve_input_state(
        state_preset,
        theta=theta,
        phi_state=phi_state,
        amplitudes=amplitudes,
        seed=seed,
    )
    rho_in = np.outer(psi, psi.conj())
    bloch_in, abs_rho01_in = rho_to_bloch_and_coherence(rho_in)

    effective_seed = (
        int(np.random.default_rng().integers(0, 2**31 - 1))
        if error_mode == "stochastic_sample" and seed is None
        else seed
    )

    err_ops, pauli_coeffs, err_label = _build_error_for_code(
        code,
        error_mode,
        pauli_string=pauli_string,
        error_qubit=error_qubit,
        error_op=error_op,
        channel=channel,
        p=p,
        seed=effective_seed,
        su2_qubit=su2_qubit,
        su2_theta=su2_theta,
        su2_nx=su2_nx,
        su2_ny=su2_ny,
        su2_nz=su2_nz,
        collective_phi=collective_phi,
    )

    primary = _execute_code_stepper(code, psi, err_ops, selected_syndrome=selected_syndrome)

    raw_code = _RawSingleQubitCode()
    raw_ops = _adapted_error_ops(
        raw_code,
        error_mode,
        pauli_string=pauli_string,
        error_qubit=error_qubit,
        error_op=error_op,
        channel=channel,
        p=p,
        seed=effective_seed,
        su2_qubit=su2_qubit,
        su2_theta=su2_theta,
        su2_nx=su2_nx,
        su2_ny=su2_ny,
        su2_nz=su2_nz,
        collective_phi=collective_phi,
    )
    raw_run = _execute_code_stepper(raw_code, psi, raw_ops)

    if code_name == "dfs_2":
        dfs_summary = dict(primary["final_summary"])
    else:
        dfs_code = TwoQubitDFSDemo()
        dfs_ops = _adapted_error_ops(
            dfs_code,
            error_mode,
            pauli_string=pauli_string,
            error_qubit=error_qubit,
            error_op=error_op,
            channel=channel,
            p=p,
            seed=effective_seed,
            su2_qubit=su2_qubit,
            su2_theta=su2_theta,
            su2_nx=su2_nx,
            su2_ny=su2_ny,
            su2_nz=su2_nz,
            collective_phi=collective_phi,
        )
        dfs_summary = _execute_code_stepper(dfs_code, psi, dfs_ops)["final_summary"]

    qecc_summary = dict(primary["final_summary"])
    qecc_summary["code"] = code_name

    return {
        "code": code.name,
        "n_data": int(code.n_data),
        "n_ancilla": int(code.n_ancilla),
        "n_total": int(code.n_total),
        "big_endian": True,
        "input_state": {
            "amplitudes": [
                [float(psi[0].real), float(psi[0].imag)],
                [float(psi[1].real), float(psi[1].imag)],
            ],
            "rho": serialize_complex_matrix(rho_in),
            "bloch": bloch_in,
            "abs_rho01": abs_rho01_in,
        },
        "error_applied": {
            "mode": error_mode,
            "label": err_label,
            "pauli_coefficients": pauli_coeffs,
            "ops": [serialize_gate_op(op) for op in err_ops],
        },
        "stages": primary["stages"],
        "comparison": {
            "raw": raw_run["final_summary"],
            "qecc": qecc_summary,
            "dfs_2": dfs_summary,
        },
    }
