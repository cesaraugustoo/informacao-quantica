"""Analytics, simulation, sweep, and algebraic theory service for ``qecc.web``."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from qecc.analysis.analytical import (
    VALID_CHANNELS,
    VALID_CODES,
    VALID_METRICS,
    _pauli_commutes,
    _pauli_multiply_unsigned,
    analytical_fidelity,
    analytical_gain_threshold,
    five_qubit_single_axis_weight_enumerator,
    five_qubit_weight_enumerator,
    verify_knill_laflamme,
)
from qecc.backends import get_backend, list_available_backends
from qecc.backends.base import QECCBackend, QECCCode
from qecc.codes import get_code, list_available_codes
from qecc.metrics.states import InitialStateSampler
from qecc.web.stepper import (
    _RawSingleQubitCode,
    resolve_input_state,
    rho_to_bloch_and_coherence,
    serialize_complex_matrix,
    serialize_gate_op,
    validate_finite_real,
    validate_nonbool_int,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SWEEPS_JSON = REPO_ROOT / "results" / "data" / "fidelity_sweeps.json"
_UNSET: Any = object()

_LOGICAL_OPS: dict[str, dict[str, str]] = {
    "raw": {"X_L": "X", "Z_L": "Z"},
    "bit_flip_3": {"X_L": "XXX", "Z_L": "ZII"},
    "phase_flip_3": {"X_L": "ZZZ", "Z_L": "XII"},
    "five_qubit": {"X_L": "XXXXX", "Z_L": "ZZZZZ"},
    "dfs_2": {"X_L": "XX", "Z_L": "ZI"},
}


def _resolve_backend_name(requested: str) -> str:
    if not isinstance(requested, str):
        raise ValueError(f"backend must be a string, got {requested!r}")
    avail = list_available_backends()
    if requested == "auto":
        return "torch_cuda" if "torch_cuda" in avail else "numpy"
    if requested not in {"numpy", "torch_cuda", "eigenkit_unitary"}:
        raise ValueError(f"Unknown backend {requested!r}; available={avail}")
    return requested


def _resolve_code_obj(code_name: str) -> QECCCode:
    if code_name == "raw":
        return _RawSingleQubitCode()
    return get_code(code_name)


def get_metadata() -> dict[str, Any]:
    """Return static code, channel, backend, cardinal-state, and threshold metadata."""
    codes_meta: dict[str, Any] = {}
    for code_name in (*list_available_codes(), "raw"):
        code_obj = _resolve_code_obj(code_name)
        codes_meta[code_name] = {
            "n_data": int(code_obj.n_data),
            "n_ancilla": int(code_obj.n_ancilla),
            "n_total": int(code_obj.n_total),
            "stabilizers": list(code_obj.stabilizers),
            "logical_ops": dict(_LOGICAL_OPS[code_name]),
            "syndrome_table": dict(code_obj.syndrome_table),
            "encode_ops": [serialize_gate_op(op) for op in code_obj.encode_ops()],
            "syndrome_ops": [serialize_gate_op(op) for op in code_obj.syndrome_ops()],
            "decode_ops": [serialize_gate_op(op) for op in code_obj.decode_ops()],
        }

    avail_backends = set(list_available_backends())
    backends_meta = [
        {
            "name": "numpy",
            "available": "numpy" in avail_backends,
            "stochastic_supported": True,
        },
        {
            "name": "torch_cuda",
            "available": "torch_cuda" in avail_backends,
            "stochastic_supported": True,
        },
        {
            "name": "eigenkit_unitary",
            "available": "eigenkit_unitary" in avail_backends,
            "stochastic_supported": False,
        },
    ]

    cardinals_meta: dict[str, Any] = {}
    for key, vec in InitialStateSampler.cardinal_states().items():
        bloch, _ = rho_to_bloch_and_coherence(np.outer(vec, vec.conj()))
        cardinals_meta[key] = {
            "amplitudes": [
                [float(vec[0].real), float(vec[0].imag)],
                [float(vec[1].real), float(vec[1].imag)],
            ],
            "bloch": bloch,
        }

    gain_thresholds = {
        f"{code_name}/{channel_name}": float(analytical_gain_threshold(code_name, channel_name))
        for code_name in list_available_codes()
        for channel_name in VALID_CHANNELS
    }

    return {
        "codes": codes_meta,
        "channels": list(VALID_CHANNELS),
        "backends": backends_meta,
        "cardinal_states": cardinals_meta,
        "gain_thresholds": gain_thresholds,
    }


def _run_backend_cycle(
    backend_inst: QECCBackend,
    code_obj: QECCCode,
    *,
    channel: str | None,
    p: float,
    num_shots: int,
    state_mode: str,
    theta: float | None = None,
    phi_state: float | None = None,
    amplitudes: Sequence[Sequence[float]] | None = None,
    seed: int | None = None,
    deterministic_errors: str | list[str] | None = None,
) -> dict[str, Any]:
    """Run single-state or 6-cardinal averaged cycle on ``backend_inst``."""
    if state_mode == "cardinal_avg":
        cardinals = InitialStateSampler.cardinal_states()
        state_keys = ("0", "1", "+", "-", "+i", "-i")
        base_shots = num_shots // len(state_keys)
        rem = num_shots % len(state_keys)

        all_fids: list[np.ndarray] = []
        rho_weighted = np.zeros((2, 2), dtype=np.complex128)
        syndrome_counts: dict[str, int] = {}
        elapsed_total = 0.0

        for idx, skey in enumerate(state_keys):
            s_shots = base_shots + (1 if idx < rem else 0)
            if s_shots <= 0:
                continue
            sub_seed = None if seed is None else seed + idx * 1009
            det_arg: str | list[str] | None
            if isinstance(deterministic_errors, list):
                if len(deterministic_errors) == 1:
                    det_arg = deterministic_errors
                else:
                    raise ValueError(
                        "deterministic_errors list with length > 1 is unsupported with state_mode='cardinal_avg'"
                    )
            else:
                det_arg = deterministic_errors

            res = backend_inst.simulate_cycle(
                code=code_obj,
                initial_state=cardinals[skey],
                channel=channel,
                p=p,
                num_shots=s_shots,
                seed=sub_seed,
                deterministic_errors=det_arg,
            )
            all_fids.append(res.fidelities)
            rho_weighted += (float(s_shots) / float(num_shots)) * res.rho_out
            elapsed_total += float(res.elapsed_seconds)
            for syn, cnt in res.syndrome_counts.items():
                syndrome_counts[syn] = syndrome_counts.get(syn, 0) + int(cnt)

        combined = np.concatenate(all_fids)
        mean_fid = float(np.mean(combined))
        std_err = (
            float(np.std(combined, ddof=1) / math.sqrt(combined.size))
            if combined.size > 1
            else 0.0
        )
        rho_out = np.asarray(0.5 * (rho_weighted + rho_weighted.conj().T), dtype=np.complex128)
        return {
            "mean_fidelity": mean_fid,
            "std_error": std_err,
            "syndrome_counts": syndrome_counts,
            "rho_out": rho_out,
            "elapsed_seconds": elapsed_total,
        }

    psi = resolve_input_state(
        state_mode,
        theta=theta,
        phi_state=phi_state,
        amplitudes=amplitudes,
        seed=seed,
    )
    res = backend_inst.simulate_cycle(
        code=code_obj,
        initial_state=psi,
        channel=channel,
        p=p,
        num_shots=num_shots,
        seed=seed,
        deterministic_errors=deterministic_errors,
    )
    return {
        "mean_fidelity": float(res.mean_fidelity),
        "std_error": float(res.std_error),
        "syndrome_counts": {str(k): int(v) for k, v in res.syndrome_counts.items()},
        "rho_out": np.asarray(res.rho_out, dtype=np.complex128),
        "elapsed_seconds": float(res.elapsed_seconds),
    }


def run_simulation(
    *,
    code: str,
    channel: str | None = _UNSET,
    p: float = 0.05,
    num_shots: int = 1000,
    state_mode: str = "+i",
    theta: float | None = None,
    phi_state: float | None = None,
    amplitudes: Sequence[Sequence[float]] | None = None,
    seed: int | None = 42,
    backend: str = "auto",
    deterministic_errors: str | list[str] | None = None,
) -> dict[str, Any]:
    """Execute a Monte Carlo or deterministic QECC simulation cycle via ``get_backend``."""
    if code not in VALID_CODES:
        raise ValueError(f"Unknown code {code!r}; expected one of {VALID_CODES}")

    fp = validate_finite_real(p, "p")
    if not (0.0 <= fp <= 1.0):
        raise ValueError(f"p must lie in [0.0, 1.0], got {fp}")
    shots = validate_nonbool_int(num_shots, "num_shots")
    if shots < 1 or shots > 200_000:
        raise ValueError(f"num_shots must be in [1, 200000], got {shots}")
    if seed is not None:
        seed = validate_nonbool_int(seed, "seed")

    if deterministic_errors is not None:
        if channel is _UNSET:
            channel_resolved: str | None = None
        elif channel is not None:
            raise ValueError("Cannot specify both channel and deterministic_errors")
        else:
            channel_resolved = None
        if fp != 0.0:
            raise ValueError(
                f"Non-zero p ({fp}) cannot be combined with deterministic_errors; require p == 0.0"
            )
    else:
        if channel is _UNSET:
            channel_resolved = "depolarizing"
        elif channel is None:
            if fp > 0.0:
                raise ValueError("channel must be provided when p > 0 and deterministic_errors is None")
            channel_resolved = None
        elif channel not in VALID_CHANNELS:
            raise ValueError(f"Unknown channel {channel!r}; expected one of {VALID_CHANNELS}")
        else:
            channel_resolved = channel

    code_obj = _resolve_code_obj(code)
    backend_used = _resolve_backend_name(backend)
    backend_inst = get_backend(backend_used)

    primary = _run_backend_cycle(
        backend_inst,
        code_obj,
        channel=channel_resolved,
        p=fp,
        num_shots=shots,
        state_mode=state_mode,
        theta=theta,
        phi_state=phi_state,
        amplitudes=amplitudes,
        seed=seed,
        deterministic_errors=deterministic_errors,
    )

    rho_out = primary["rho_out"]
    bloch_out, abs_rho01 = rho_to_bloch_and_coherence(rho_out)

    if code == "raw":
        raw_comparison = {
            "fidelity": primary["mean_fidelity"],
            "abs_rho01": abs_rho01,
        }
    else:
        raw_det: str | list[str] | None = None
        if isinstance(deterministic_errors, str):
            raw_det = deterministic_errors[0]
        elif isinstance(deterministic_errors, list):
            raw_det = [err[0] for err in deterministic_errors]

        raw_res = _run_backend_cycle(
            backend_inst,
            _RawSingleQubitCode(),
            channel=channel_resolved,
            p=fp,
            num_shots=shots,
            state_mode=state_mode,
            theta=theta,
            phi_state=phi_state,
            amplitudes=amplitudes,
            seed=seed,
            deterministic_errors=raw_det,
        )
        _, raw_abs_rho01 = rho_to_bloch_and_coherence(raw_res["rho_out"])
        raw_comparison = {
            "fidelity": raw_res["mean_fidelity"],
            "abs_rho01": raw_abs_rho01,
        }

    ana_ent: float | None = None
    ana_state: float | None = None
    if deterministic_errors is None and channel_resolved is not None:
        ana_ent = float(analytical_fidelity(code, channel_resolved, fp, metric="entanglement"))
        ana_state = float(analytical_fidelity(code, channel_resolved, fp, metric="state"))

    return {
        "code": code,
        "backend_used": backend_used,
        "p": fp,
        "num_shots": shots,
        "state_mode": state_mode,
        "mean_fidelity": primary["mean_fidelity"],
        "std_error": primary["std_error"],
        "syndrome_counts": primary["syndrome_counts"],
        "rho_out": serialize_complex_matrix(rho_out),
        "bloch_out": bloch_out,
        "abs_rho01": abs_rho01,
        "raw_comparison": raw_comparison,
        "analytical_entanglement_fidelity": ana_ent,
        "analytical_state_fidelity": ana_state,
        "elapsed_seconds": primary["elapsed_seconds"],
    }


def _analytical_coherence_plus_i(
    code_name: str,
    channel_type: str,
    p_grid: np.ndarray,
) -> np.ndarray:
    """Compute exact analytical ``|rho_01|`` on ``|+i>`` across ``p_grid``."""
    p = np.asarray(p_grid, dtype=np.float64)
    f_ent = np.asarray(
        analytical_fidelity(code_name, channel_type, p, metric="entanglement"),
        dtype=np.float64,
    )

    if code_name == "raw":
        if channel_type == "depolarizing":
            return 0.5 * np.abs(1.0 - (4.0 / 3.0) * p)
        return 0.5 * np.abs(1.0 - 2.0 * p)

    if code_name in ("bit_flip_3", "phase_flip_3"):
        matched_channel = "bit_flip" if code_name == "bit_flip_3" else "phase_flip"
        if channel_type == matched_channel:
            return 0.5 * np.abs(2.0 * f_ent - 1.0)
        if channel_type in ("bit_flip", "phase_flip"):
            return 0.5 * np.abs(1.0 - 2.0 * p) ** 3
        if channel_type == "depolarizing":
            return 0.5 * np.abs(1.0 - (4.0 / 3.0) * p) ** 3
        if channel_type == "collective_dephasing":
            return 0.5 * np.abs(1.0 - 2.0 * p)

    if code_name == "five_qubit":
        if channel_type == "depolarizing":
            return 0.5 * np.abs(1.0 - (4.0 / 3.0) * (1.0 - f_ent))
        if channel_type in ("bit_flip", "phase_flip"):
            q = 1.0 - p
            poly = (q**5) + 5.0 * p * (q**4) - 5.0 * (p**4) * q - (p**5)
            return 0.5 * np.abs(poly)
        if channel_type == "collective_dephasing":
            return 0.5 * np.abs(1.0 - 2.0 * p)

    if code_name == "dfs_2":
        if channel_type == "collective_dephasing":
            return np.full_like(p, 0.5, dtype=np.float64)
        if channel_type == "phase_flip":
            return 0.5 * ((1.0 - 2.0 * p) ** 2)
        if channel_type == "bit_flip":
            return 0.5 * np.abs(1.0 - 2.0 * p)
        if channel_type == "depolarizing":
            return 0.5 * ((1.0 - (4.0 / 3.0) * p) ** 2)

    raise ValueError(f"Unsupported combination ({code_name!r}, {channel_type!r})")


def _load_cached_mc_sweeps(
    codes: Sequence[str],
    channels: Sequence[str],
    requested_metric: str,
) -> dict[str, list[dict[str, Any]]]:
    """Load and group cached MC rows from ``results/data/fidelity_sweeps.json``."""
    if not DEFAULT_SWEEPS_JSON.is_file():
        return {}
    payload = json.loads(DEFAULT_SWEEPS_JSON.read_text(encoding="utf-8"))
    records = payload.get("records", [])
    code_set = set(codes)
    channel_set = set(channels)
    grouped: dict[str, list[dict[str, Any]]] = {}

    for row in records:
        c_name = row.get("code")
        ch_name = row.get("channel")
        if c_name not in code_set or ch_name not in channel_set:
            continue
        state_mode = str(row.get("state_mode", ""))
        row_metric = "state" if state_mode in {"cardinal_avg", "state"} else "entanglement"
        mean_f = float(row["mean_fidelity"])
        std_err = float(row.get("std_error", 0.0))
        if row_metric == requested_metric:
            converted_f = mean_f
            converted_se = std_err
        elif requested_metric == "state":
            converted_f = 1.0 - (2.0 / 3.0) * (1.0 - mean_f)
            converted_se = (2.0 / 3.0) * std_err
        else:
            converted_f = 1.0 - 1.5 * (1.0 - mean_f)
            converted_se = 1.5 * std_err

        entry = dict(row)
        entry["metric"] = row_metric
        entry["converted_fidelity"] = float(converted_f)
        entry["std_error"] = float(converted_se)
        key = f"{c_name}/{ch_name}"
        grouped.setdefault(key, []).append(entry)

    return grouped


def run_sweeps(
    *,
    codes: Sequence[str] = ("bit_flip_3", "phase_flip_3", "five_qubit", "dfs_2"),
    channels: Sequence[str] = ("bit_flip", "phase_flip", "depolarizing", "collective_dephasing"),
    metric: str = "entanglement",
    p_min: float = 0.0,
    p_max: float = 0.5,
    num_p: int = 21,
    run_live_mc: bool = False,
    num_shots: int = 2000,
    seed: int | None = 20261118,
    backend: str = "auto",
) -> dict[str, Any]:
    """Compute analytical curves, ``p*`` thresholds, coherence curves, and cached/live MC sweeps."""
    if not codes:
        raise ValueError("codes must be a non-empty list")
    if not channels:
        raise ValueError("channels must be a non-empty list")
    for c in codes:
        if c not in VALID_CODES:
            raise ValueError(f"Unknown code {c!r}; expected one of {VALID_CODES}")
    for ch in channels:
        if ch not in VALID_CHANNELS:
            raise ValueError(f"Unknown channel {ch!r}; expected one of {VALID_CHANNELS}")
    if metric not in VALID_METRICS:
        raise ValueError(f"Unknown metric {metric!r}; expected one of {VALID_METRICS}")

    fp_min = validate_finite_real(p_min, "p_min")
    fp_max = validate_finite_real(p_max, "p_max")
    if not (0.0 <= fp_min < fp_max <= 1.0):
        raise ValueError(f"Require 0.0 <= p_min < p_max <= 1.0, got ({fp_min}, {fp_max})")
    n_p = validate_nonbool_int(num_p, "num_p")
    if n_p < 2 or n_p > 501:
        raise ValueError(f"num_p must be in [2, 501], got {n_p}")
    n_shots = validate_nonbool_int(num_shots, "num_shots")
    if n_shots < 1 or n_shots > 100_000:
        raise ValueError(f"num_shots must be in [1, 100000], got {n_shots}")
    if seed is not None:
        seed = validate_nonbool_int(seed, "seed")
    if isinstance(run_live_mc, (int, str)) and not isinstance(run_live_mc, bool):
        raise ValueError("run_live_mc must be a bool")

    p_grid = np.linspace(fp_min, fp_max, n_p, dtype=np.float64)
    raw_arr = np.asarray(
        analytical_fidelity("raw", "bit_flip", p_grid, metric=metric),
        dtype=np.float64,
    )

    analytical_curves: dict[str, Any] = {}
    gain_thresholds: dict[str, float] = {}
    coherence_curves: dict[str, Any] = {}

    for code_name in codes:
        for ch_name in channels:
            key = f"{code_name}/{ch_name}"
            f_arr = np.asarray(
                analytical_fidelity(code_name, ch_name, p_grid, metric=metric),
                dtype=np.float64,
            )
            delta_arr = f_arr - raw_arr
            p_star = float(analytical_gain_threshold(code_name, ch_name))

            analytical_curves[key] = {
                "fidelities": [float(v) for v in f_arr],
                "delta_f": [float(v) for v in delta_arr],
                "p_star": p_star,
            }
            gain_thresholds[key] = p_star

            coh_arr = _analytical_coherence_plus_i(code_name, ch_name, p_grid)
            coh_raw = _analytical_coherence_plus_i("raw", ch_name, p_grid)
            coherence_curves[key] = {
                "p_grid": [float(v) for v in p_grid],
                "abs_rho01_analytical": [float(v) for v in coh_arr],
                "abs_rho01_raw": [float(v) for v in coh_raw],
            }

    if run_live_mc:
        resolved_backend = _resolve_backend_name(backend)
        backend_inst = get_backend(resolved_backend)
        exact_zero_pairs = {
            ("bit_flip_3", "bit_flip"),
            ("phase_flip_3", "phase_flip"),
            ("raw", "bit_flip"),
        }
        exact_plus_i_pairs = {
            ("dfs_2", "collective_dephasing"),
            ("phase_flip_3", "collective_dephasing"),
            ("raw", "collective_dephasing"),
            ("raw", "phase_flip"),
        }
        mc_sweeps: dict[str, list[dict[str, Any]]] = {}
        pair_idx = 0
        for code_name in codes:
            code_obj = _resolve_code_obj(code_name)
            for ch_name in channels:
                key = f"{code_name}/{ch_name}"
                if metric == "state":
                    s_mode = "cardinal_avg"
                    row_metric = "state"
                elif (code_name, ch_name) in exact_zero_pairs:
                    s_mode = "0"
                    row_metric = "entanglement"
                elif (code_name, ch_name) in exact_plus_i_pairs:
                    s_mode = "+i"
                    row_metric = "entanglement"
                else:
                    s_mode = "cardinal_avg"
                    row_metric = "state"

                rows: list[dict[str, Any]] = []
                for p_idx, p_val in enumerate(p_grid):
                    p_f = float(p_val)
                    pt_seed = None if seed is None else seed + pair_idx * 1000 + p_idx
                    res = _run_backend_cycle(
                        backend_inst,
                        code_obj,
                        channel=ch_name,
                        p=p_f,
                        num_shots=n_shots,
                        state_mode=s_mode,
                        seed=pt_seed,
                    )
                    mean_f = float(res["mean_fidelity"])
                    std_err = float(res["std_error"])
                    if metric == "entanglement" and row_metric == "state":
                        converted_f = 1.0 - 1.5 * (1.0 - mean_f)
                        std_err = 1.5 * std_err
                    else:
                        converted_f = mean_f

                    rows.append(
                        {
                            "code": code_name,
                            "channel": ch_name,
                            "p": p_f,
                            "mean_fidelity": mean_f,
                            "std_error": std_err,
                            "num_shots": n_shots,
                            "backend": resolved_backend,
                            "state_mode": s_mode,
                            "metric": row_metric,
                            "converted_fidelity": float(converted_f),
                        }
                    )
                mc_sweeps[key] = rows
                pair_idx += 1
    else:
        mc_sweeps = _load_cached_mc_sweeps(codes, channels, metric)

    return {
        "metric": metric,
        "p_grid": [float(v) for v in p_grid],
        "raw_curve": [float(v) for v in raw_arr],
        "analytical_curves": analytical_curves,
        "gain_thresholds": gain_thresholds,
        "mc_sweeps": mc_sweeps,
        "coherence_curves": coherence_curves,
    }


def _build_aliasing_examples(code_name: str, code_obj: QECCCode) -> list[dict[str, Any]]:
    """Construct verified syndrome-aliasing and uncorrectable-error examples."""
    if code_name == "dfs_2":
        return [
            {
                "error": "ZI",
                "syndrome": "",
                "recovery": "",
                "residual": "ZI",
                "logical_effect": "Z_L",
                "note": "Symmetry-breaking single-qubit phase error induces logical Z_L phase flip (undetected).",
            },
            {
                "error": "XI",
                "syndrome": "",
                "recovery": "",
                "residual": "XI",
                "logical_effect": "leakage",
                "note": "Bit-flip leaks state out of decoherence-free subspace span{|01>, |10>}.",
            },
        ]

    candidates_by_code: dict[str, list[tuple[str, str]]] = {
        "bit_flip_3": [("IXX", "X_L"), ("XIX", "X_L"), ("XXI", "X_L")],
        "phase_flip_3": [("IZZ", "X_L"), ("ZIZ", "X_L"), ("ZZI", "X_L")],
        "five_qubit": [("XXIII", "Z_L"), ("ZZIII", "Y_L")],
    }
    logical_ops = _LOGICAL_OPS[code_name]
    x_l = logical_ops["X_L"]
    z_l = logical_ops["Z_L"]

    examples: list[dict[str, Any]] = []
    for err_str, expected_effect in candidates_by_code[code_name]:
        syndrome = "".join(
            "0" if _pauli_commutes(err_str, stab) else "1"
            for stab in code_obj.stabilizers
        )
        recovery = code_obj.syndrome_table[syndrome]
        residual = _pauli_multiply_unsigned(recovery, err_str)
        comm_x = _pauli_commutes(residual, x_l)
        comm_z = _pauli_commutes(residual, z_l)
        if comm_x and comm_z:
            effect = "I_L"
        elif comm_x and not comm_z:
            effect = "X_L"
        elif not comm_x and not comm_z:
            effect = "Y_L"
        else:
            effect = "Z_L"
        if effect != expected_effect:
            raise RuntimeError(
                f"Aliasing logical effect mismatch for {code_name} error {err_str!r}: "
                f"got {effect!r}, expected {expected_effect!r}"
            )
        examples.append(
            {
                "error": err_str,
                "syndrome": syndrome,
                "recovery": recovery,
                "residual": residual,
                "logical_effect": effect,
                "note": (
                    f"Weight-2 error {err_str} aliases syndrome {syndrome}; "
                    f"recovery {recovery} leaves logical operator {effect} ({residual})."
                ),
            }
        )
    return examples


def get_theory(code_name: str) -> dict[str, Any]:
    """Return stabilizers, syndrome table, Knill-Laflamme C-matrix, enumerator, and aliasing."""
    if code_name not in set(list_available_codes()):
        raise ValueError(
            f"Unsupported code_name {code_name!r} for theory endpoint; "
            f"expected one of {list_available_codes()}"
        )

    code_obj = get_code(code_name)
    kl = verify_knill_laflamme(code_name, tol=1e-12)

    return {
        "code": code_name,
        "stabilizers": list(code_obj.stabilizers),
        "logical_ops": dict(_LOGICAL_OPS[code_name]),
        "syndrome_table": dict(code_obj.syndrome_table),
        "knill_laflamme": {
            "satisfies": bool(kl.passed),
            "max_violation": float(kl.max_violation),
            "max_diagonal_diff": float(kl.max_diagonal_diff),
            "max_off_diagonal": float(kl.max_off_diagonal),
            "error_basis": list(kl.error_set),
            "c_matrix": serialize_complex_matrix(kl.c_matrix),
        },
        "weight_enumerator": five_qubit_weight_enumerator(),
        "single_axis_weight_enumerator": {
            "X": five_qubit_single_axis_weight_enumerator("X"),
            "Z": five_qubit_single_axis_weight_enumerator("Z"),
        },
        "aliasing_examples": _build_aliasing_examples(code_name, code_obj),
    }
