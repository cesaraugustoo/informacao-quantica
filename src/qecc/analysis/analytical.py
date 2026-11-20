"""Analytical Pauli-channel fidelities, weight enumerators, Knill-Laflamme checks, and thresholds."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Sequence

import numpy as np
from scipy.optimize import brentq

from qecc.backends.base import GateOp, QECCCode
from qecc.codes import FiveQubitPerfectCode, get_code

VALID_CODES: tuple[str, ...] = (
    "raw",
    "bit_flip_3",
    "phase_flip_3",
    "five_qubit",
    "dfs_2",
)
VALID_CHANNELS: tuple[str, ...] = (
    "bit_flip",
    "phase_flip",
    "depolarizing",
    "collective_dephasing",
)
VALID_METRICS: tuple[str, ...] = ("entanglement", "state")

_PAULIS: tuple[str, ...] = ("I", "X", "Y", "Z")
_PAULI_MATS: dict[str, np.ndarray] = {
    "I": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.complex128),
    "X": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128),
    "Y": np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128),
    "Z": np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128),
}


@dataclass(frozen=True, slots=True)
class KnillLaflammeResult:
    """Result of checking Knill-Laflamme conditions <i_L|E_a^dag E_b|j_L> = C_ab delta_ij."""

    code_name: str
    passed: bool
    error_set: tuple[str, ...]
    c_matrix: np.ndarray
    max_diagonal_diff: float
    max_off_diagonal: float
    max_violation: float


def _validate_code_and_channel(code_name: str, channel_type: str) -> None:
    if not isinstance(code_name, str) or code_name not in VALID_CODES:
        raise ValueError(
            f"Unknown code_name {code_name!r}; expected one of {VALID_CODES}"
        )
    if not isinstance(channel_type, str) or channel_type not in VALID_CHANNELS:
        raise ValueError(
            f"Unknown channel_type {channel_type!r}; expected one of {VALID_CHANNELS}"
        )


def _validate_metric(metric: str) -> None:
    if not isinstance(metric, str) or metric not in VALID_METRICS:
        raise ValueError(f"Unknown metric {metric!r}; expected one of {VALID_METRICS}")


def _contains_bool(obj: Any) -> bool:
    if isinstance(obj, (bool, np.bool_)):
        return True
    if isinstance(obj, (list, tuple)):
        return any(_contains_bool(item) for item in obj)
    return False


def _normalize_probability_input(
    p: float | int | np.number | np.ndarray | Sequence[Any],
) -> tuple[float | np.ndarray, bool]:
    """Validate scalar or array-like probability p in [0.0, 1.0].

    Returns ``(normalized_value, is_scalar)``.
    """
    if isinstance(p, (bool, np.bool_)) or _contains_bool(p):
        raise ValueError("p must be a real number or array in [0.0, 1.0], not bool")
    if isinstance(p, (str, bytes)) or p is None:
        raise ValueError(f"p must be a real number or array in [0.0, 1.0], got {p!r}")

    if isinstance(p, np.ndarray):
        if np.issubdtype(p.dtype, np.bool_) or np.issubdtype(p.dtype, np.complexfloating):
            raise ValueError(f"p array cannot have bool or complex dtype, got {p.dtype}")
        if p.dtype.kind in {"U", "S", "O", "V"}:
            raise ValueError(f"p array must be numeric, got dtype {p.dtype}")
        try:
            arr = p.astype(np.float64, copy=False)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"p array must be convertible to float64: {exc}") from exc
        if not np.all(np.isfinite(arr)):
            raise ValueError("p array values must all be finite (no NaN or inf)")
        if np.any((arr < 0.0) | (arr > 1.0)):
            raise ValueError("p array values must lie in [0.0, 1.0]")
        return arr, False

    if isinstance(p, Sequence):
        arr_raw = np.asarray(p)
        if np.issubdtype(arr_raw.dtype, np.bool_) or np.issubdtype(
            arr_raw.dtype, np.complexfloating
        ):
            raise ValueError(f"p sequence cannot have bool or complex elements, got {arr_raw.dtype}")
        if arr_raw.dtype.kind in {"U", "S", "O", "V"}:
            raise ValueError(f"p sequence must be numeric, got dtype {arr_raw.dtype}")
        try:
            arr = arr_raw.astype(np.float64, copy=False)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"p sequence must be convertible to float64: {exc}") from exc
        if not np.all(np.isfinite(arr)):
            raise ValueError("p values must all be finite (no NaN or inf)")
        if np.any((arr < 0.0) | (arr > 1.0)):
            raise ValueError("p values must lie in [0.0, 1.0]")
        return arr, False

    if isinstance(p, complex):
        raise ValueError(f"p must be a real number in [0.0, 1.0], got {p!r}")

    try:
        fp = float(p)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"p must be a real number in [0.0, 1.0], got {p!r}") from exc
    if not math.isfinite(fp) or not (0.0 <= fp <= 1.0):
        raise ValueError(f"p must lie in [0.0, 1.0], got {p!r}")
    return fp, True


def _pauli_commutes(a: str, b: str) -> bool:
    """Return True iff multi-qubit Pauli strings ``a`` and ``b`` commute."""
    anticommutes = 0
    for ca, cb in zip(a, b, strict=True):
        if ca != "I" and cb != "I" and ca != cb:
            anticommutes ^= 1
    return anticommutes == 0


def _pauli_multiply_unsigned(a: str, b: str) -> str:
    """Multiply two Pauli strings up to global phase."""
    out: list[str] = []
    for ca, cb in zip(a, b, strict=True):
        if ca == "I":
            out.append(cb)
        elif cb == "I":
            out.append(ca)
        elif ca == cb:
            out.append("I")
        else:
            # Remaining third non-identity Pauli
            for cand in ("X", "Y", "Z"):
                if cand != ca and cand != cb:
                    out.append(cand)
                    break
    return "".join(out)


def _pauli_weight(pauli: str) -> int:
    return sum(1 for ch in pauli if ch != "I")


@lru_cache(maxsize=1)
def _cached_five_qubit_enumerator() -> dict[str, dict[int, int]]:
    """Enumerate all 4^5 = 1024 5-qubit Pauli operators into logical cosets."""
    code = FiveQubitPerfectCode()
    stabilizers = code.stabilizers
    syndrome_table = code.syndrome_table
    logical_x = "XXXXX"
    logical_z = "ZZZZZ"

    cosets: dict[str, dict[int, int]] = {
        op: {w: 0 for w in range(code.n_data + 1)} for op in _PAULIS
    }

    for pauli_tuple in itertools.product(_PAULIS, repeat=code.n_data):
        error = "".join(pauli_tuple)
        syndrome = "".join(
            "0" if _pauli_commutes(error, stab) else "1" for stab in stabilizers
        )
        recovery = syndrome_table[syndrome]
        residual = _pauli_multiply_unsigned(recovery, error)

        # Residual must commute with all stabilizers (in normalizer N(S))
        for stab in stabilizers:
            if not _pauli_commutes(residual, stab):
                raise RuntimeError(
                    f"Syndrome table inconsistency for error {error!r}, recovery {recovery!r}"
                )

        comm_x = _pauli_commutes(residual, logical_x)
        comm_z = _pauli_commutes(residual, logical_z)
        if comm_x and comm_z:
            logical_op = "I"
        elif comm_x and not comm_z:
            logical_op = "X"
        elif not comm_x and not comm_z:
            logical_op = "Y"
        else:
            logical_op = "Z"

        cosets[logical_op][_pauli_weight(error)] += 1

    return cosets


def five_qubit_weight_enumerator() -> dict[str, dict[int, int]]:
    """Return the exact 1024-term ``[[5,1,3]]`` Pauli weight enumerator per logical coset.

    Each of the 4^5 = 1024 Pauli strings over ``{I, X, Y, Z}^5`` is mapped to its
    syndrome from ``FiveQubitPerfectCode.stabilizers``, corrected via
    ``FiveQubitPerfectCode.syndrome_table``, and classified by residual action on the
    logical subspace (``"I"``, ``"X"``, ``"Y"``, ``"Z"``).
    """
    cached = _cached_five_qubit_enumerator()
    return {k: dict(v) for k, v in cached.items()}


@lru_cache(maxsize=2)
def _cached_five_qubit_single_axis_enumerator(axis: str) -> dict[int, int]:
    """Enumerate all 2^5 = 32 pure-X or pure-Z errors for logical-I preservation."""
    if axis not in ("X", "Z"):
        raise ValueError(f"axis must be 'X' or 'Z', got {axis!r}")
    code = FiveQubitPerfectCode()
    stabilizers = code.stabilizers
    syndrome_table = code.syndrome_table
    logical_x = "XXXXX"
    logical_z = "ZZZZZ"

    counts = {w: 0 for w in range(code.n_data + 1)}
    for bits in itertools.product(("I", axis), repeat=code.n_data):
        error = "".join(bits)
        syndrome = "".join(
            "0" if _pauli_commutes(error, stab) else "1" for stab in stabilizers
        )
        recovery = syndrome_table[syndrome]
        residual = _pauli_multiply_unsigned(recovery, error)
        if _pauli_commutes(residual, logical_x) and _pauli_commutes(residual, logical_z):
            counts[_pauli_weight(error)] += 1
    return counts


def five_qubit_single_axis_weight_enumerator(axis: str) -> dict[int, int]:
    """Return logical-I weight counts for pure-``X`` (bit-flip) or pure-``Z`` (phase-flip)."""
    return dict(_cached_five_qubit_single_axis_enumerator(axis))


def _gate_matrix_on_n_qubits(op: GateOp, n_qubits: int) -> np.ndarray:
    """Construct the exact 2^n x 2^n big-endian unitary matrix for a GateOp."""
    dim = 1 << n_qubits
    if op.name in ("I", "X", "Y", "Z", "H", "S", "Sdg", "RX", "RY", "RZ"):
        if op.name == "H":
            u = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128) / math.sqrt(2.0)
        elif op.name == "S":
            u = np.array([[1.0, 0.0], [0.0, 1.0j]], dtype=np.complex128)
        elif op.name == "Sdg":
            u = np.array([[1.0, 0.0], [0.0, -1.0j]], dtype=np.complex128)
        elif op.name == "RX":
            theta = op.params[0] / 2.0
            u = np.array(
                [
                    [math.cos(theta), -1.0j * math.sin(theta)],
                    [-1.0j * math.sin(theta), math.cos(theta)],
                ],
                dtype=np.complex128,
            )
        elif op.name == "RY":
            theta = op.params[0] / 2.0
            u = np.array(
                [
                    [math.cos(theta), -math.sin(theta)],
                    [math.sin(theta), math.cos(theta)],
                ],
                dtype=np.complex128,
            )
        elif op.name == "RZ":
            theta = op.params[0] / 2.0
            u = np.array(
                [
                    [np.exp(-1.0j * theta), 0.0],
                    [0.0, np.exp(1.0j * theta)],
                ],
                dtype=np.complex128,
            )
        else:
            u = _PAULI_MATS[op.name]
        target = op.targets[0]
        mat = np.array([[1.0 + 0.0j]], dtype=np.complex128)
        for q in range(n_qubits):
            mat = np.kron(mat, u if q == target else _PAULI_MATS["I"])
        return mat

    if op.name in ("CNOT", "CZ"):
        ctrl = op.controls[0]
        tgt = op.targets[0]
        mat = np.zeros((dim, dim), dtype=np.complex128)
        ctrl_shift = n_qubits - 1 - ctrl
        tgt_shift = n_qubits - 1 - tgt
        for basis in range(dim):
            c_bit = (basis >> ctrl_shift) & 1
            if op.name == "CNOT":
                out_basis = basis ^ (1 << tgt_shift) if c_bit else basis
                mat[out_basis, basis] = 1.0
            else:
                t_bit = (basis >> tgt_shift) & 1
                mat[basis, basis] = -1.0 if (c_bit and t_bit) else 1.0
        return mat

    raise ValueError(f"Unsupported gate {op.name!r}")


def _pauli_string_matrix(pauli_str: str) -> np.ndarray:
    mat = np.array([[1.0 + 0.0j]], dtype=np.complex128)
    for ch in pauli_str:
        mat = np.kron(mat, _PAULI_MATS[ch])
    return mat


def _logical_basis_states(code: QECCCode) -> tuple[np.ndarray, np.ndarray]:
    """Construct exact |0_L> and |1_L> on n_data qubits from ``code.encode_ops()``."""
    n_data = code.n_data
    dim = 1 << n_data
    ket0_l = np.zeros(dim, dtype=np.complex128)
    ket1_l = np.zeros(dim, dtype=np.complex128)
    ket0_l[0] = 1.0
    # Big-endian qubit 0 MSB => |1>|0...0> has index 1 << (n_data - 1)
    ket1_l[1 << (n_data - 1)] = 1.0

    for op in code.encode_ops():
        for q in (*op.controls, *op.targets):
            if q >= n_data:
                raise ValueError(
                    f"encode_ops() for {code.name!r} acts on ancilla qubit {q}"
                )
        unitary = _gate_matrix_on_n_qubits(op, n_data)
        ket0_l = unitary @ ket0_l
        ket1_l = unitary @ ket1_l

    return ket0_l, ket1_l


def _default_correctable_error_set(code_name: str) -> tuple[str, ...]:
    if code_name == "bit_flip_3":
        return ("III", "XII", "IXI", "IIX")
    if code_name == "phase_flip_3":
        return ("III", "ZII", "IZI", "IIZ")
    if code_name == "five_qubit":
        code = FiveQubitPerfectCode()
        return tuple(code.syndrome_table.values())
    if code_name == "dfs_2":
        return ("II", "ZZ")
    raise ValueError(
        f"Knill-Laflamme verification unsupported for code_name={code_name!r}"
    )


def verify_knill_laflamme(
    code_name: str,
    *,
    tol: float = 1e-12,
) -> KnillLaflammeResult:
    """Verify Knill-Laflamme conditions ``<i_L|E_a^dag E_b|j_L> = C_ab delta_ij``.

    Constructs ``|0_L>`` and ``|1_L>`` directly from ``code.encode_ops()`` and checks
    diagonal equality and logical orthogonality across all pairs ``(E_a, E_b)`` in the
    code's target correctable error set.
    """
    if code_name not in {"bit_flip_3", "phase_flip_3", "five_qubit", "dfs_2"}:
        raise ValueError(
            f"Unsupported code_name {code_name!r} for Knill-Laflamme verification"
        )
    code = get_code(code_name)
    error_set = _default_correctable_error_set(code_name)
    ket0_l, ket1_l = _logical_basis_states(code)

    err_mats = [_pauli_string_matrix(e) for e in error_set]
    e_ket0 = [m @ ket0_l for m in err_mats]
    e_ket1 = [m @ ket1_l for m in err_mats]

    m_len = len(error_set)
    c_mat = np.zeros((m_len, m_len), dtype=np.complex128)
    max_diag_diff = 0.0
    max_off_diag = 0.0

    for a in range(m_len):
        for b in range(m_len):
            val00 = complex(np.vdot(e_ket0[a], e_ket0[b]))
            val11 = complex(np.vdot(e_ket1[a], e_ket1[b]))
            val01 = complex(np.vdot(e_ket0[a], e_ket1[b]))
            val10 = complex(np.vdot(e_ket1[a], e_ket0[b]))

            c_mat[a, b] = 0.5 * (val00 + val11)
            max_diag_diff = max(max_diag_diff, abs(val00 - val11))
            max_off_diag = max(max_off_diag, abs(val01), abs(val10))

    max_herm_err = float(np.max(np.abs(c_mat - c_mat.conj().T)))
    max_violation = max(max_diag_diff, max_off_diag, max_herm_err)
    return KnillLaflammeResult(
        code_name=code_name,
        passed=bool(max_violation <= tol),
        error_set=error_set,
        c_matrix=c_mat,
        max_diagonal_diff=float(max_diag_diff),
        max_off_diagonal=float(max_off_diag),
        max_violation=float(max_violation),
    )


def _entanglement_fidelity_eval(
    code_name: str, channel_type: str, p: float | np.ndarray
) -> float | np.ndarray:
    """Exact entanglement fidelity P_I(p) for all 5 codes x 4 channels."""
    q = 1.0 - p

    if code_name == "raw":
        return q

    if code_name == "bit_flip_3":
        if channel_type == "bit_flip":
            return 1.0 - 3.0 * (p**2) + 2.0 * (p**3)
        if channel_type == "phase_flip":
            return (q**3) + 3.0 * (p**2) * q
        if channel_type == "depolarizing":
            return (q**3) + p * (q**2) + (p**2) * q + (1.0 / 9.0) * (p**3)
        if channel_type == "collective_dephasing":
            return q

    if code_name == "phase_flip_3":
        if channel_type == "phase_flip":
            return 1.0 - 3.0 * (p**2) + 2.0 * (p**3)
        if channel_type == "bit_flip":
            return (q**3) + 3.0 * (p**2) * q
        if channel_type == "depolarizing":
            return (q**3) + p * (q**2) + (p**2) * q + (1.0 / 9.0) * (p**3)
        if channel_type == "collective_dephasing":
            return q

    if code_name == "five_qubit":
        if channel_type == "depolarizing":
            enum_i = _cached_five_qubit_enumerator()["I"]
            total = 0.0 * p
            for weight, count in enum_i.items():
                if count > 0:
                    total = total + count * (q ** (5 - weight)) * ((p / 3.0) ** weight)
            return total
        if channel_type in ("bit_flip", "phase_flip"):
            axis = "X" if channel_type == "bit_flip" else "Z"
            counts = _cached_five_qubit_single_axis_enumerator(axis)
            total = 0.0 * p
            for weight, count in counts.items():
                if count > 0:
                    total = total + count * (q ** (5 - weight)) * (p**weight)
            return total
        if channel_type == "collective_dephasing":
            return q

    if code_name == "dfs_2":
        if channel_type == "collective_dephasing":
            return np.ones_like(p, dtype=np.float64) if isinstance(p, np.ndarray) else 1.0
        if channel_type == "phase_flip":
            return (q**2) + (p**2)
        if channel_type == "bit_flip":
            return q
        if channel_type == "depolarizing":
            return 1.0 - (5.0 / 3.0) * p + (8.0 / 9.0) * (p**2)

    raise ValueError(
        f"Unsupported combination code_name={code_name!r}, channel_type={channel_type!r}"
    )


def analytical_fidelity(
    code_name: str,
    channel_type: str,
    p: float | int | np.number | np.ndarray | Sequence[Any],
    *,
    metric: str = "entanglement",
) -> float | np.ndarray:
    """Compute exact analytical fidelity for ``(code_name, channel_type, p)``.

    Parameters
    ----------
    code_name:
        One of ``{"raw", "bit_flip_3", "phase_flip_3", "five_qubit", "dfs_2"}``.
    channel_type:
        One of ``{"bit_flip", "phase_flip", "depolarizing", "collective_dephasing"}``.
    p:
        Error rate scalar or array-like with values in ``[0.0, 1.0]``.
    metric:
        ``"entanglement"`` (exact Pauli preservation probability ``P_I(p)``) or
        ``"state"`` (Haar-averaged state fidelity ``1 - (2/3)*(1 - P_I(p))``).
    """
    _validate_code_and_channel(code_name, channel_type)
    _validate_metric(metric)
    p_norm, is_scalar = _normalize_probability_input(p)

    f_ent = _entanglement_fidelity_eval(code_name, channel_type, p_norm)
    if metric == "state":
        result = 1.0 - (2.0 / 3.0) * (1.0 - f_ent)
    else:
        result = f_ent

    if is_scalar:
        return float(result)
    return np.asarray(result, dtype=np.float64)


@lru_cache(maxsize=32)
def _cached_gain_threshold(code_name: str, channel_type: str) -> float:
    if code_name == "raw":
        return 0.0

    if (code_name, channel_type) in {
        ("bit_flip_3", "bit_flip"),
        ("phase_flip_3", "phase_flip"),
    }:
        return 0.5

    if (code_name, channel_type) == ("dfs_2", "collective_dephasing"):
        return 1.0

    if code_name == "five_qubit" and channel_type in {
        "depolarizing",
        "bit_flip",
        "phase_flip",
    }:
        def diff(prob: float) -> float:
            return float(_entanglement_fidelity_eval(code_name, channel_type, prob)) - (
                1.0 - prob
            )

        root = brentq(diff, 1e-4, 0.45, xtol=1e-15, rtol=1e-15)
        return float(root)

    return 0.0


def analytical_gain_threshold(code_name: str, channel_type: str) -> float:
    """Return upper error-rate threshold ``p*`` where ``F_code(p) > F_raw(p)`` on ``(0, p*)``.

    Returns ``0.0`` when the code never beats the unprotected physical qubit in the
    physical forward-noise regime ``p in (0, 0.5]``.
    """
    _validate_code_and_channel(code_name, channel_type)
    return _cached_gain_threshold(code_name, channel_type)
