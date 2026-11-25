"""Parity, validation, HTTP server, and CLI tests for ``qecc.web`` (Milestone M1)."""

from __future__ import annotations

import http.client
import itertools
import json
import math
import os
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import numpy as np
import pytest
from pydantic import ValidationError

import qecc
from qecc.analysis.analytical import (
    analytical_fidelity,
    analytical_gain_threshold,
    five_qubit_weight_enumerator,
    verify_knill_laflamme,
)
from qecc.backends import NumPyCPUBackend, list_available_backends
from qecc.backends.base import GateOp
from qecc.backends.numpy_backend import _apply_ops_np, _single_qubit_matrix
from qecc.codes import get_code, list_available_codes
from qecc.metrics import InitialStateSampler, quantum_fidelity
from qecc.web import app as web_app
from qecc.web.analytics_service import (
    get_metadata,
    get_theory,
    run_simulation,
    run_sweeps,
)
from qecc.web.app import (
    MAX_BODY_BYTES,
    SimulateRequest,
    StepRequest,
    SweepsRequest,
    create_server,
)
from qecc.web.stepper import (
    STAGE_KEYS,
    _RawSingleQubitCode,
    build_su2_rotation,
    resolve_input_state,
    run_step_evolution,
)

PARITY_TOL = 1e-12


def _deser_rho(raw_rho: list[list[list[float]]]) -> np.ndarray:
    arr = np.zeros((2, 2), dtype=np.complex128)
    for i in range(2):
        for j in range(2):
            arr[i, j] = complex(raw_rho[i][j][0], raw_rho[i][j][1])
    return arr


def _http_request(
    url: str,
    *,
    method: str = "GET",
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, str], bytes]:
    req = urllib.request.Request(url, data=body, method=method)
    if headers:
        for k, v in headers.items():
            req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            return int(resp.status), dict(resp.headers.items()), resp.read()
    except urllib.error.HTTPError as err:
        return int(err.code), dict(err.headers.items()), err.read()


def _http_json(
    base_url: str,
    path: str,
    payload: dict[str, Any] | None = None,
) -> tuple[int, Any]:
    if payload is None:
        status, _, raw = _http_request(f"{base_url}{path}", method="GET")
    else:
        raw_body = json.dumps(payload).encode("utf-8")
        status, _, raw = _http_request(
            f"{base_url}{path}",
            method="POST",
            body=raw_body,
            headers={"Content-Type": "application/json"},
        )
    return status, json.loads(raw.decode("utf-8"))


# ---------------------------------------------------------------------------
# 1. Stepper Parity (<= 1e-12) Across All 4 Codes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "code_name",
    ["bit_flip_3", "phase_flip_3", "five_qubit", "dfs_2"],
)
@pytest.mark.parametrize("state_preset", ["0", "1", "+", "-", "+i", "-i"])
def test_stepper_parity_all_codes_and_cardinals(
    code_name: str,
    state_preset: str,
) -> None:
    backend = NumPyCPUBackend()
    code = get_code(code_name)
    psi = InitialStateSampler.cardinal_states()[state_preset]

    if code_name == "dfs_2":
        errors = ["II", "ZZ"]
    else:
        errors = list(code.syndrome_table.values())

    for err_str in errors:
        step_res = run_step_evolution(
            code_name=code_name,
            state_preset=state_preset,
            error_mode="pauli_string",
            pauli_string=err_str,
        )
        stages = step_res["stages"]
        assert [s["key"] for s in stages] == list(STAGE_KEYS)

        sim_res = backend.simulate_cycle(
            code=code,
            initial_state=psi,
            channel=None,
            p=0.0,
            num_shots=1,
            seed=123,
            deterministic_errors=err_str,
        )
        decoded = stages[-1]
        rho_decoded = _deser_rho(decoded["logical_rho"])
        assert np.max(np.abs(rho_decoded - sim_res.rho_out)) <= PARITY_TOL
        assert abs(decoded["fidelity_vs_input"] - quantum_fidelity(psi, sim_res.rho_out)) <= PARITY_TOL
        assert abs(decoded["logical_abs_rho01"] - abs(sim_res.rho_out[0, 1])) <= PARITY_TOL
        assert abs(decoded["fidelity_vs_input"] - 1.0) <= PARITY_TOL

        syn_stage = stages[3]
        assert syn_stage["selected_syndrome"] == sim_res.syndromes[0]
        assert abs(stages[1]["code_subspace_prob"] - 1.0) <= PARITY_TOL
        assert abs(stages[4]["code_subspace_prob"] - 1.0) <= PARITY_TOL

        # Verify 3-way comparison parity
        assert np.max(np.abs(_deser_rho(step_res["comparison"]["qecc"]["rho"]) - sim_res.rho_out)) <= PARITY_TOL
        raw_sim = backend.simulate_cycle(
            code=_RawSingleQubitCode(),
            initial_state=psi,
            channel=None,
            p=0.0,
            num_shots=1,
            deterministic_errors=err_str[0],
        )
        assert np.max(np.abs(_deser_rho(step_res["comparison"]["raw"]["rho"]) - raw_sim.rho_out)) <= PARITY_TOL
        dfs_sim = backend.simulate_cycle(
            code=get_code("dfs_2"),
            initial_state=psi,
            channel=None,
            p=0.0,
            num_shots=1,
            deterministic_errors=err_str[:2],
        )
        assert np.max(np.abs(_deser_rho(step_res["comparison"]["dfs_2"]["rho"]) - dfs_sim.rho_out)) <= PARITY_TOL


def test_stepper_bloch_and_custom_state_parity() -> None:
    backend = NumPyCPUBackend()
    code = get_code("five_qubit")

    # Bloch state with omitted phi_state (defaults to 0.0) and explicit phi_state
    step_bloch = run_step_evolution(
        code_name="five_qubit",
        state_preset="bloch",
        theta=1.1,
        error_mode="single_qubit_pauli",
        error_qubit=3,
        error_op="Y",
    )
    psi_bloch = resolve_input_state("bloch", theta=1.1, phi_state=None)
    sim_bloch = backend.simulate_cycle(
        code=code,
        initial_state=psi_bloch,
        channel=None,
        p=0.0,
        num_shots=1,
        deterministic_errors="IIIYI",
    )
    assert np.max(np.abs(_deser_rho(step_bloch["stages"][-1]["logical_rho"]) - sim_bloch.rho_out)) <= PARITY_TOL
    assert abs(step_bloch["stages"][-1]["fidelity_vs_input"] - 1.0) <= PARITY_TOL

    # Custom state
    amps = [[0.6, 0.0], [0.0, 0.8]]
    step_custom = run_step_evolution(
        code_name="bit_flip_3",
        state_preset="custom",
        amplitudes=amps,
        error_mode="single_qubit_pauli",
        error_qubit=2,
        error_op="X",
    )
    psi_custom = np.array([0.6 + 0.0j, 0.0 + 0.8j], dtype=np.complex128)
    sim_custom = backend.simulate_cycle(
        code=get_code("bit_flip_3"),
        initial_state=psi_custom,
        channel=None,
        p=0.0,
        num_shots=1,
        deterministic_errors="IIX",
    )
    assert np.max(np.abs(_deser_rho(step_custom["stages"][-1]["logical_rho"]) - sim_custom.rho_out)) <= PARITY_TOL


def test_stepper_weight2_aliasing_failures() -> None:
    # bit_flip_3 weight-2 error IXX aliases syndrome '10' -> recovery 'XII' -> residual 'XXX' = X_L
    res_bf = run_step_evolution(
        code_name="bit_flip_3",
        state_preset="0",
        error_mode="pauli_string",
        pauli_string="IXX",
    )
    assert res_bf["stages"][3]["selected_syndrome"] == "10"
    assert res_bf["stages"][-1]["fidelity_vs_input"] <= PARITY_TOL

    # five_qubit weight-2 error XXIII aliases syndrome '1001' -> recovery 'IIIZI' -> residual 'XXIZI' = Z_L
    res_5q = run_step_evolution(
        code_name="five_qubit",
        state_preset="+",
        error_mode="pauli_string",
        pauli_string="XXIII",
    )
    assert res_5q["stages"][3]["selected_syndrome"] == "1001"
    assert res_5q["stages"][-1]["fidelity_vs_input"] <= PARITY_TOL


# ---------------------------------------------------------------------------
# 2. Continuous SU(2) Discretization & Coherent Collective Dephasing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "theta,nx,ny,nz",
    [
        (0.83, 1.0, -2.0, 0.5),
        (math.pi, 0.3, 0.4, -0.8),
        (0.0, 0.0, 0.0, 1.0),
        (1.25, -1.0, 0.0, 0.0),
        (2.1, 0.0, 1.0, 0.0),
    ],
)
def test_su2_zyz_decomposition_exact_unitary(
    theta: float,
    nx: float,
    ny: float,
    nz: float,
) -> None:
    ops, coeffs, _ = build_su2_rotation(0, theta, nx, ny, nz, n_data=3)
    prob_sum = sum(entry["prob"] for entry in coeffs.values())
    assert abs(prob_sum - 1.0) <= PARITY_TOL

    norm = math.sqrt(nx * nx + ny * ny + nz * nz)
    ux, uy, uz = nx / norm, ny / norm, nz / norm
    half = 0.5 * theta
    c, s = math.cos(half), math.sin(half)
    u_exact = np.array(
        [
            [complex(c, -uz * s), complex(-uy * s, -ux * s)],
            [complex(uy * s, -ux * s), complex(c, uz * s)],
        ],
        dtype=np.complex128,
    )

    u_ops = np.eye(2, dtype=np.complex128)
    for op in ops:
        u_ops = _single_qubit_matrix(op) @ u_ops

    # Check exact equality including SU(2) global sign!
    assert np.max(np.abs(u_ops - u_exact)) <= PARITY_TOL


def test_su2_large_axis_normalization_no_overflow() -> None:
    theta = 1.2
    ops, coeffs, _ = build_su2_rotation(0, theta, 1e200, 1e200, 1e200, n_data=3)
    inv_sqrt3 = 1.0 / math.sqrt(3.0)
    s_val = math.sin(0.5 * theta)
    c_val = math.cos(0.5 * theta)

    assert abs(coeffs["I"]["re"] - c_val) <= PARITY_TOL
    for axis in ("X", "Y", "Z"):
        assert abs(coeffs[axis]["im"] + inv_sqrt3 * s_val) <= PARITY_TOL
        assert abs(coeffs[axis]["prob"] - (s_val * s_val) / 3.0) <= PARITY_TOL

    u_ops = np.eye(2, dtype=np.complex128)
    for op in ops:
        u_ops = _single_qubit_matrix(op) @ u_ops
    u_exact = np.array(
        [
            [complex(c_val, -inv_sqrt3 * s_val), complex(-inv_sqrt3 * s_val, -inv_sqrt3 * s_val)],
            [complex(inv_sqrt3 * s_val, -inv_sqrt3 * s_val), complex(c_val, inv_sqrt3 * s_val)],
        ],
        dtype=np.complex128,
    )
    assert np.max(np.abs(u_ops - u_exact)) <= PARITY_TOL

    req = StepRequest.model_validate(
        {
            "code": "bit_flip_3",
            "error_mode": "su2_rotation",
            "su2_qubit": 0,
            "su2_theta": theta,
            "su2_nx": 1e200,
            "su2_ny": 1e200,
            "su2_nz": 1e200,
        }
    )
    assert req.su2_nx == 1e200

    step_res = run_step_evolution(
        code_name="bit_flip_3",
        state_preset="+i",
        error_mode="su2_rotation",
        su2_qubit=0,
        su2_theta=theta,
        su2_nx=1e200,
        su2_ny=1e200,
        su2_nz=1e200,
    )
    step_coeffs = step_res["error_applied"]["pauli_coefficients"]
    for axis in ("X", "Y", "Z"):
        assert abs(step_coeffs[axis]["prob"] - (s_val * s_val) / 3.0) <= PARITY_TOL


def test_unseeded_stochastic_sample_synchronizes_comparisons() -> None:
    backend = NumPyCPUBackend()
    psi = InitialStateSampler.cardinal_states()["+i"]
    dfs_code = get_code("dfs_2")
    raw_code = _RawSingleQubitCode()

    for _ in range(12):
        res = run_step_evolution(
            code_name="bit_flip_3",
            state_preset="+i",
            error_mode="stochastic_sample",
            channel="phase_flip",
            p=0.5,
            seed=None,
        )
        sampled_str = res["error_applied"]["label"].split(": ", 1)[1]
        raw_sim = backend.simulate_cycle(
            code=raw_code,
            initial_state=psi,
            channel=None,
            p=0.0,
            num_shots=1,
            deterministic_errors=sampled_str[0],
        )
        dfs_sim = backend.simulate_cycle(
            code=dfs_code,
            initial_state=psi,
            channel=None,
            p=0.0,
            num_shots=1,
            deterministic_errors=sampled_str[:2],
        )
        assert abs(res["comparison"]["raw"]["fidelity"] - raw_sim.mean_fidelity) <= PARITY_TOL
        assert abs(res["comparison"]["dfs_2"]["fidelity"] - dfs_sim.mean_fidelity) <= PARITY_TOL


def test_su2_discretization_syndrome_branches_parity() -> None:
    code = get_code("five_qubit")
    psi = InitialStateSampler.cardinal_states()["+i"]
    ops, coeffs, _ = build_su2_rotation(2, 0.9, 1.0, 1.5, -0.7, n_data=5)

    # Check direct _apply_ops_np up to syndrome extraction
    state = np.zeros((1, 1 << code.n_total), dtype=np.complex128)
    state[0, 0] = psi[0]
    state[0, 1 << (code.n_total - 1)] = psi[1]
    _apply_ops_np(state, code.encode_ops(), code.n_total)
    _apply_ops_np(state, ops, code.n_total)
    _apply_ops_np(state, code.syndrome_ops(), code.n_total)
    da = state[0].reshape(1 << code.n_data, 1 << code.n_ancilla)
    probs = (da.real**2 + da.imag**2).sum(axis=0)

    # Non-zero branches in five_qubit for single-qubit SU(2) on qubit 2:
    # '0000' (I), '1100' (IIXII), '0010' (IIZII), '1110' (IIYII)
    expected_nonzero = {
        "0000": coeffs["I"]["prob"],
        "1100": coeffs["X"]["prob"],
        "0010": coeffs["Z"]["prob"],
        "1110": coeffs["Y"]["prob"],
    }

    res_default = run_step_evolution(
        code_name="five_qubit",
        state_preset="+i",
        error_mode="su2_rotation",
        su2_qubit=2,
        su2_theta=0.9,
        su2_nx=1.0,
        su2_ny=1.5,
        su2_nz=-0.7,
    )
    syn_probs = res_default["stages"][3]["syndrome_probabilities"]
    assert abs(sum(syn_probs.values()) - 1.0) <= PARITY_TOL
    for syn_key, exp_p in expected_nonzero.items():
        assert abs(syn_probs[syn_key] - exp_p) <= PARITY_TOL
        assert abs(probs[int(syn_key, 2)] - exp_p) <= PARITY_TOL

        # Selecting each non-zero branch collapses and recovers exact |+i> (since SU(2) is weight <= 1!)
        branch_res = run_step_evolution(
            code_name="five_qubit",
            state_preset="+i",
            error_mode="su2_rotation",
            su2_qubit=2,
            su2_theta=0.9,
            su2_nx=1.0,
            su2_ny=1.5,
            su2_nz=-0.7,
            selected_syndrome=syn_key,
        )
        assert branch_res["stages"][3]["selected_syndrome"] == syn_key
        assert abs(branch_res["stages"][-1]["fidelity_vs_input"] - 1.0) <= PARITY_TOL

    # Selecting a zero-probability syndrome branch must raise ValueError
    with pytest.raises(ValueError, match="zero-probability"):
        run_step_evolution(
            code_name="five_qubit",
            state_preset="+i",
            error_mode="su2_rotation",
            su2_qubit=2,
            su2_theta=0.9,
            su2_nx=1.0,
            su2_ny=1.5,
            su2_nz=-0.7,
            selected_syndrome="0001",
        )


@pytest.mark.parametrize("phi", [0.0, 0.73, math.pi, -1.41])
def test_coherent_collective_dephasing_dfs_invariance(phi: float) -> None:
    res = run_step_evolution(
        code_name="bit_flip_3",
        state_preset="+i",
        error_mode="coherent_collective",
        collective_phi=phi,
    )
    # dfs_2 is completely invariant under Rz(phi) x Rz(phi)
    dfs_cmp = res["comparison"]["dfs_2"]
    assert abs(dfs_cmp["fidelity"] - 1.0) <= PARITY_TOL
    assert abs(dfs_cmp["abs_rho01"] - 0.5) <= PARITY_TOL

    # raw undergoes Rz(phi) rotation on |+i>: fidelity = cos^2(phi/2)
    raw_cmp = res["comparison"]["raw"]
    assert abs(raw_cmp["fidelity"] - (math.cos(0.5 * phi) ** 2)) <= PARITY_TOL

    # bit_flip_3 undergoes Rz(phi)^{\otimes 3} -> logical Z rotation by 3*phi: fidelity = cos^2(3*phi/2)
    qecc_cmp = res["comparison"]["qecc"]
    assert abs(qecc_cmp["fidelity"] - (math.cos(1.5 * phi) ** 2)) <= PARITY_TOL


# ---------------------------------------------------------------------------
# 3. Analytics & Theory Service Parity (<= 1e-12)
# ---------------------------------------------------------------------------


def test_analytics_simulate_parity_with_numpy_backend() -> None:
    backend = NumPyCPUBackend()
    psi = InitialStateSampler.cardinal_states()["+i"]

    for code_name in ("raw", "bit_flip_3", "phase_flip_3", "five_qubit", "dfs_2"):
        sim_out = run_simulation(
            code=code_name,
            channel="depolarizing",
            p=0.08,
            num_shots=500,
            state_mode="+i",
            seed=777,
            backend="numpy",
        )
        code_obj = _RawSingleQubitCode() if code_name == "raw" else get_code(code_name)
        direct = backend.simulate_cycle(
            code=code_obj,
            initial_state=psi,
            channel="depolarizing",
            p=0.08,
            num_shots=500,
            seed=777,
        )
        assert abs(sim_out["mean_fidelity"] - direct.mean_fidelity) <= PARITY_TOL
        assert abs(sim_out["std_error"] - direct.std_error) <= PARITY_TOL
        assert sim_out["syndrome_counts"] == direct.syndrome_counts
        assert np.max(np.abs(_deser_rho(sim_out["rho_out"]) - direct.rho_out)) <= PARITY_TOL
        assert abs(sim_out["abs_rho01"] - abs(direct.rho_out[0, 1])) <= PARITY_TOL

        expected_ent = float(analytical_fidelity(code_name, "depolarizing", 0.08, metric="entanglement"))
        expected_st = float(analytical_fidelity(code_name, "depolarizing", 0.08, metric="state"))
        assert abs(sim_out["analytical_entanglement_fidelity"] - expected_ent) <= PARITY_TOL
        assert abs(sim_out["analytical_state_fidelity"] - expected_st) <= PARITY_TOL


def test_analytics_simulate_cardinal_avg_and_eigenkit() -> None:
    res_avg = run_simulation(
        code="five_qubit",
        channel="depolarizing",
        p=0.05,
        num_shots=600,
        state_mode="cardinal_avg",
        seed=101,
        backend="numpy",
    )
    assert sum(res_avg["syndrome_counts"].values()) == 600
    assert abs(res_avg["analytical_state_fidelity"] - float(analytical_fidelity("five_qubit", "depolarizing", 0.05, metric="state"))) <= PARITY_TOL

    if "eigenkit_unitary" in list_available_backends():
        det_ek = run_simulation(
            code="bit_flip_3",
            deterministic_errors="IXI",
            p=0.0,
            num_shots=1,
            state_mode="+",
            backend="eigenkit_unitary",
        )
        assert abs(det_ek["mean_fidelity"] - 1.0) <= PARITY_TOL

        with pytest.raises(NotImplementedError):
            run_simulation(
                code="bit_flip_3",
                channel="bit_flip",
                p=0.05,
                num_shots=10,
                backend="eigenkit_unitary",
            )


def test_analytics_sweeps_metric_separation_thresholds_and_coherence() -> None:
    p_grid = np.linspace(0.0, 0.5, 21)

    for metric in ("entanglement", "state"):
        sw = run_sweeps(
            codes=["bit_flip_3", "five_qubit", "dfs_2"],
            channels=["bit_flip", "depolarizing", "collective_dephasing"],
            metric=metric,
            p_min=0.0,
            p_max=0.5,
            num_p=21,
            run_live_mc=False,
        )
        expected_raw = analytical_fidelity("raw", "bit_flip", p_grid, metric=metric)
        assert np.max(np.abs(np.array(sw["raw_curve"]) - expected_raw)) <= PARITY_TOL

        # Verify delta_f sign flip across p* for five_qubit/depolarizing
        curve_5q = sw["analytical_curves"]["five_qubit/depolarizing"]
        p_star_5q = analytical_gain_threshold("five_qubit", "depolarizing")
        assert abs(curve_5q["p_star"] - p_star_5q) <= PARITY_TOL
        assert abs(sw["gain_thresholds"]["five_qubit/depolarizing"] - p_star_5q) <= PARITY_TOL

        for p_val, df_val in zip(sw["p_grid"], curve_5q["delta_f"]):
            if 0.0 < p_val < p_star_5q:
                assert df_val > 0.0
            elif p_val > p_star_5q:
                assert df_val < 0.0

        # Cached MC sweeps preserve raw fidelity and tag state_mode + metric cleanly
        cached_5q = sw["mc_sweeps"]["five_qubit/depolarizing"]
        assert len(cached_5q) == 21
        assert all(row["state_mode"] == "cardinal_avg" for row in cached_5q)
        assert all(row["metric"] == "state" for row in cached_5q)

    # Verify coherence_curves against exact Kraus enumeration for representative pairs
    sw_coh = run_sweeps(
        codes=["bit_flip_3", "dfs_2"],
        channels=["bit_flip", "collective_dephasing", "depolarizing"],
        metric="entanglement",
        p_min=0.0,
        p_max=0.5,
        num_p=6,
        run_live_mc=False,
    )
    p_check = sw_coh["p_grid"][1]  # p = 0.1
    psi_plus_i = InitialStateSampler.cardinal_states()["+i"]
    backend = NumPyCPUBackend()
    dfs_code = get_code("dfs_2")
    rho_dfs = np.zeros((2, 2), dtype=np.complex128)
    for tup in itertools.product(["I", "X", "Y", "Z"], repeat=2):
        prob = 1.0
        for ch in tup:
            prob *= (1.0 - p_check) if ch == "I" else (p_check / 3.0)
        rho_dfs += prob * backend.simulate_cycle(
            dfs_code, psi_plus_i, None, 0.0, 1, deterministic_errors="".join(tup)
        ).rho_out
    assert abs(sw_coh["coherence_curves"]["dfs_2/depolarizing"]["abs_rho01_analytical"][1] - abs(rho_dfs[0, 1])) <= PARITY_TOL

    # Live MC sweep test
    sw_live = run_sweeps(
        codes=["dfs_2"],
        channels=["collective_dephasing"],
        metric="entanglement",
        p_min=0.0,
        p_max=0.2,
        num_p=3,
        run_live_mc=True,
        num_shots=100,
        seed=5,
        backend="numpy",
    )
    assert len(sw_live["mc_sweeps"]["dfs_2/collective_dephasing"]) == 3
    assert all(abs(row["mean_fidelity"] - 1.0) <= PARITY_TOL for row in sw_live["mc_sweeps"]["dfs_2/collective_dephasing"])


def test_analytics_sweeps_live_mc_entanglement_all_20_pairs_parity() -> None:
    sw = run_sweeps(
        codes=["raw", "bit_flip_3", "phase_flip_3", "five_qubit", "dfs_2"],
        channels=["bit_flip", "phase_flip", "depolarizing", "collective_dephasing"],
        metric="entanglement",
        p_min=0.0,
        p_max=0.5,
        num_p=3,
        run_live_mc=True,
        num_shots=4000,
        seed=12345,
        backend="numpy",
    )
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
    assert len(sw["mc_sweeps"]) == 20
    for key, rows in sw["mc_sweeps"].items():
        code_name, ch_name = key.split("/", 1)
        ana_vals = sw["analytical_curves"][key]["fidelities"]
        assert len(rows) == 3
        for idx, row in enumerate(rows):
            if (code_name, ch_name) in exact_zero_pairs:
                assert row["state_mode"] == "0"
                assert row["metric"] == "entanglement"
            elif (code_name, ch_name) in exact_plus_i_pairs:
                assert row["state_mode"] == "+i"
                assert row["metric"] == "entanglement"
            else:
                assert row["state_mode"] == "cardinal_avg"
                assert row["metric"] == "state"
                expected_conv = 1.0 - 1.5 * (1.0 - row["mean_fidelity"])
                assert abs(row["converted_fidelity"] - expected_conv) <= PARITY_TOL
            diff = abs(row["converted_fidelity"] - ana_vals[idx])
            assert diff < 0.03, f"{key} at p={row['p']}: diff={diff:.6f}"


def test_theory_service_kl_enumerator_and_aliasing() -> None:
    expected_enum = five_qubit_weight_enumerator()
    assert sum(sum(w.values()) for w in expected_enum.values()) == 1024

    for code_name in ("bit_flip_3", "phase_flip_3", "five_qubit", "dfs_2"):
        th = get_theory(code_name)
        kl_ref = verify_knill_laflamme(code_name, tol=1e-12)
        assert th["knill_laflamme"]["satisfies"] is True
        assert th["knill_laflamme"]["max_violation"] <= PARITY_TOL
        assert abs(th["knill_laflamme"]["max_violation"] - kl_ref.max_violation) <= PARITY_TOL
        assert th["knill_laflamme"]["error_basis"] == list(kl_ref.error_set)
        c_mat = np.array(
            [
                [complex(cell[0], cell[1]) for cell in row]
                for row in th["knill_laflamme"]["c_matrix"]
            ],
            dtype=np.complex128,
        )
        assert np.max(np.abs(c_mat - kl_ref.c_matrix)) <= PARITY_TOL
        assert th["weight_enumerator"] == expected_enum
        assert len(th["aliasing_examples"]) >= 1

    with pytest.raises(ValueError):
        get_theory("raw")


# ---------------------------------------------------------------------------
# 4. Strict Negative Validation Tests (Pydantic + Stepper + Analytics)
# ---------------------------------------------------------------------------


def test_strict_pydantic_rejects_bool_nan_inf_extra_and_invalid_states() -> None:
    # Reject bool in numeric fields
    with pytest.raises(ValidationError):
        SimulateRequest.model_validate({"code": "bit_flip_3", "p": True})
    with pytest.raises(ValidationError):
        SimulateRequest.model_validate({"code": "bit_flip_3", "num_shots": False})
    with pytest.raises(ValidationError):
        SimulateRequest.model_validate({"code": "bit_flip_3", "seed": True})
    with pytest.raises(ValidationError):
        StepRequest.model_validate(
            {
                "code": "bit_flip_3",
                "error_mode": "single_qubit_pauli",
                "error_qubit": True,
                "error_op": "X",
            }
        )
    with pytest.raises(ValidationError):
        StepRequest.model_validate(
            {
                "code": "bit_flip_3",
                "state_preset": "bloch",
                "theta": True,
                "error_mode": "pauli_string",
                "pauli_string": "III",
            }
        )
    with pytest.raises(ValidationError):
        SweepsRequest.model_validate({"num_p": True})

    # Reject NaN / Inf
    with pytest.raises(ValidationError):
        SimulateRequest.model_validate({"code": "bit_flip_3", "p": float("nan")})
    with pytest.raises(ValidationError):
        SimulateRequest.model_validate({"code": "bit_flip_3", "p": float("inf")})

    # Reject out-of-range p, num_shots, extra fields, unnormalized custom, zero SU(2) axis
    with pytest.raises(ValidationError):
        SimulateRequest.model_validate({"code": "bit_flip_3", "p": -0.01})
    with pytest.raises(ValidationError):
        SimulateRequest.model_validate({"code": "bit_flip_3", "p": 1.01})
    with pytest.raises(ValidationError):
        SimulateRequest.model_validate({"code": "bit_flip_3", "num_shots": 0})
    with pytest.raises(ValidationError):
        SimulateRequest.model_validate({"code": "bit_flip_3", "unexpected": 1})
    with pytest.raises(ValidationError):
        StepRequest.model_validate(
            {
                "code": "bit_flip_3",
                "state_preset": "custom",
                "amplitudes": [[0.0, 0.0], [0.0, 0.0]],
                "error_mode": "pauli_string",
                "pauli_string": "III",
            }
        )
    with pytest.raises(ValidationError):
        StepRequest.model_validate(
            {
                "code": "bit_flip_3",
                "state_preset": "custom",
                "amplitudes": [[2.0, 0.0], [0.0, 0.0]],
                "error_mode": "pauli_string",
                "pauli_string": "III",
            }
        )
    with pytest.raises(ValidationError):
        StepRequest.model_validate(
            {
                "code": "bit_flip_3",
                "error_mode": "su2_rotation",
                "su2_qubit": 0,
                "su2_theta": 0.5,
                "su2_nx": 0.0,
                "su2_ny": 0.0,
                "su2_nz": 0.0,
            }
        )
    with pytest.raises(ValidationError):
        StepRequest.model_validate(
            {
                "code": "dfs_2",
                "error_mode": "pauli_string",
                "pauli_string": "ZZ",
                "selected_syndrome": "00",
            }
        )
    with pytest.raises(ValidationError):
        StepRequest.model_validate(
            {
                "code": "bit_flip_3",
                "error_mode": "pauli_string",
                "pauli_string": "II",
            }
        )


# ---------------------------------------------------------------------------
# 5. Live Ephemeral-Port HTTP Server, Static Safety & CLI Tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def live_server() -> str:
    server = create_server("127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[0], server.server_address[1]
    try:
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def test_http_endpoints_roundtrip(live_server: str) -> None:
    status, health = _http_json(live_server, "/api/health")
    assert status == 200
    assert health["status"] == "ok"
    assert health["codes"] == ["bit_flip_3", "phase_flip_3", "five_qubit", "dfs_2"]

    status, meta = _http_json(live_server, "/api/metadata")
    assert status == 200
    assert set(meta["codes"]) == {"bit_flip_3", "phase_flip_3", "five_qubit", "dfs_2", "raw"}
    assert meta["channels"] == ["bit_flip", "phase_flip", "depolarizing", "collective_dephasing"]
    assert abs(meta["gain_thresholds"]["five_qubit/depolarizing"] - analytical_gain_threshold("five_qubit", "depolarizing")) <= PARITY_TOL

    status, th = _http_json(live_server, "/api/theory?code=five_qubit")
    assert status == 200
    assert th["code"] == "five_qubit"
    assert th["knill_laflamme"]["satisfies"] is True

    status, bad_th = _http_json(live_server, "/api/theory?code=raw")
    assert status == 400
    assert "error" in bad_th

    status, missing_th = _http_json(live_server, "/api/theory")
    assert status == 400
    assert "error" in missing_th

    # Wrong HTTP method -> 405
    status, method_err = _http_json(live_server, "/api/step")
    assert status == 405
    assert "error" in method_err
    status, method_err_post = _http_json(live_server, "/api/health", {})
    assert status == 405
    assert "error" in method_err_post

    # Custom amplitudes via HTTP JSON list[list[float]]
    status, custom_step = _http_json(
        live_server,
        "/api/step",
        {
            "code": "dfs_2",
            "state_preset": "custom",
            "amplitudes": [[0.6, 0.0], [0.0, 0.8]],
            "error_mode": "pauli_string",
            "pauli_string": "ZZ",
        },
    )
    assert status == 200
    assert abs(custom_step["stages"][-1]["fidelity_vs_input"] - 1.0) <= PARITY_TOL

    status, step_out = _http_json(
        live_server,
        "/api/step",
        {
            "code": "phase_flip_3",
            "state_preset": "+i",
            "error_mode": "single_qubit_pauli",
            "error_qubit": 1,
            "error_op": "Z",
        },
    )
    assert status == 200
    assert [s["key"] for s in step_out["stages"]] == list(STAGE_KEYS)
    assert abs(step_out["stages"][-1]["fidelity_vs_input"] - 1.0) <= PARITY_TOL

    status, sim_out = _http_json(
        live_server,
        "/api/simulate",
        {
            "code": "bit_flip_3",
            "channel": "bit_flip",
            "p": 0.05,
            "num_shots": 200,
            "state_mode": "+",
            "seed": 42,
            "backend": "numpy",
        },
    )
    assert status == 200
    assert sim_out["backend_used"] == "numpy"

    status, sw_out = _http_json(
        live_server,
        "/api/sweeps",
        {
            "codes": ["bit_flip_3", "five_qubit"],
            "channels": ["bit_flip", "depolarizing"],
            "metric": "state",
            "p_min": 0.0,
            "p_max": 0.5,
            "num_p": 11,
            "run_live_mc": False,
        },
    )
    assert status == 200
    assert sw_out["metric"] == "state"
    assert abs(sw_out["raw_curve"][-1] - (1.0 - (2.0 / 3.0) * 0.5)) <= PARITY_TOL


def test_http_negative_validation_and_static_safety(
    live_server: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Invalid JSON / NaN / non-dict JSON -> 400
    for bad_body in (b"{bad json", b'{"p": NaN}', b"[1, 2, 3]"):
        status, _, raw = _http_request(
            f"{live_server}/api/simulate",
            method="POST",
            body=bad_body,
            headers={"Content-Type": "application/json"},
        )
        assert status == 400
        assert "error" in json.loads(raw.decode("utf-8"))

    # Wrong Content-Type -> 415
    status, _, raw = _http_request(
        f"{live_server}/api/simulate",
        method="POST",
        body=b'{"code":"bit_flip_3"}',
        headers={"Content-Type": "text/plain"},
    )
    assert status == 415

    # Oversized Content-Length -> 413
    parsed_srv = urlsplit(live_server)
    conn = http.client.HTTPConnection(parsed_srv.hostname, parsed_srv.port, timeout=5.0)
    try:
        conn.putrequest("POST", "/api/simulate")
        conn.putheader("Content-Type", "application/json")
        conn.putheader("Content-Length", str(MAX_BODY_BYTES + 16))
        conn.endheaders()
        resp = conn.getresponse()
        assert resp.status == 413
        assert "error" in json.loads(resp.read().decode("utf-8"))
    finally:
        conn.close()

    # Pydantic validation error -> 422
    status, err = _http_json(
        live_server,
        "/api/simulate",
        {"code": "bit_flip_3", "p": True},
    )
    assert status == 422
    assert "error" in err

    # Unknown route -> 404
    status, err = _http_json(live_server, "/api/does_not_exist")
    assert status == 404
    assert "error" in err

    # eigenkit_unitary stochastic rejection -> 501
    if "eigenkit_unitary" in list_available_backends():
        status, err = _http_json(
            live_server,
            "/api/simulate",
            {
                "code": "bit_flip_3",
                "channel": "bit_flip",
                "p": 0.05,
                "num_shots": 10,
                "backend": "eigenkit_unitary",
            },
        )
        assert status == 501
        assert "error" in err

    # Path traversal protection
    for bad_path in (
        "/static/../app.py",
        "/static/%2e%2e/app.py",
        "/static//etc/passwd",
        "/static/sub/../../app.py",
    ):
        status, _, raw = _http_request(f"{live_server}{bad_path}")
        assert status == 403
        assert "error" in json.loads(raw.decode("utf-8"))

    # Missing static asset returns clean 404 JSON
    status, _, raw = _http_request(f"{live_server}/static/nonexistent_asset_xyz.js")
    assert status == 404
    assert "error" in json.loads(raw.decode("utf-8"))

    # Monkeypatched static directory verifies MIME types and root index.html serving
    static_dir = tmp_path / "static"
    static_dir.mkdir()
    (static_dir / "index.html").write_text("<html><body>QECC</body></html>", encoding="utf-8")
    (static_dir / "app.js").write_text("console.log('ok');", encoding="utf-8")
    (static_dir / "styles.css").write_text("body { margin: 0; }", encoding="utf-8")
    monkeypatch.setattr(web_app, "STATIC_DIR", static_dir)

    status, headers, raw = _http_request(f"{live_server}/")
    assert status == 200
    assert "text/html" in headers["Content-Type"]
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert b"QECC" in raw

    status, headers, _ = _http_request(f"{live_server}/static/app.js")
    assert status == 200
    assert "application/javascript" in headers["Content-Type"]

    status, headers, _ = _http_request(f"{live_server}/static/styles.css")
    assert status == 200
    assert "text/css" in headers["Content-Type"]


def test_cli_and_module_entrypoint_and_invariants() -> None:
    # Preserved package invariants
    assert list_available_codes() == ["bit_flip_3", "phase_flip_3", "five_qubit", "dfs_2"]
    assert len(qecc.__all__) == 26
    assert "raw" not in list_available_codes()

    repo_root = Path(__file__).resolve().parents[2]
    venv_python = repo_root / ".venv" / "bin" / "python"
    py_exe = str(venv_python) if venv_python.exists() else sys.executable
    sub_env = dict(os.environ)
    sub_env["PYTHONPATH"] = str(repo_root / "src")

    # CLI serve --help and python -m qecc.web --help
    for cmd in (
        [py_exe, "-m", "qecc.cli", "serve", "--help"],
        [py_exe, "-m", "qecc.web", "--help"],
    ):
        proc = subprocess.run(
            cmd,
            cwd=str(repo_root),
            env=sub_env,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        assert proc.returncode == 0
        assert "--host" in proc.stdout
        assert "--port" in proc.stdout
        assert "--no-browser" in proc.stdout

    # Subprocess ephemeral-port startup and live /api/health probe
    proc = subprocess.Popen(
        [py_exe, "-m", "qecc.web", "--port", "0", "--no-browser"],
        cwd=str(repo_root),
        env=sub_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert proc.stdout is not None
        line = proc.stdout.readline().strip()
        assert "http://127.0.0.1:" in line
        url = line.split("http://", 1)[1]
        base_url = f"http://{url}"
        status, payload = _http_json(base_url, "/api/health")
        assert status == 200
        assert payload["status"] == "ok"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


def test_get_metadata_and_root_404_when_static_absent(
    live_server: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    meta = get_metadata()
    assert len(meta["cardinal_states"]) == 6
    assert set(meta["gain_thresholds"]) == {
        f"{c}/{ch}"
        for c in ("bit_flip_3", "phase_flip_3", "five_qubit", "dfs_2")
        for ch in ("bit_flip", "phase_flip", "depolarizing", "collective_dephasing")
    }

    missing_dir = tmp_path / "missing_static"
    monkeypatch.setattr(web_app, "STATIC_DIR", missing_dir)
    status, _, raw = _http_request(f"{live_server}/")
    assert status == 404
    assert "error" in json.loads(raw.decode("utf-8"))


# ---------------------------------------------------------------------------
# 6. Tier 5 White-Box Adversarial Backend, Schema, HTTP, Math & CLI Tests
# ---------------------------------------------------------------------------


def test_theory_single_axis_weight_enumerator_five_qubit() -> None:
    from qecc.analysis.analytical import five_qubit_single_axis_weight_enumerator

    th = get_theory("five_qubit")
    sa = th["single_axis_weight_enumerator"]
    expected = {0: 1, 1: 5, 2: 0, 3: 0, 4: 0, 5: 0}
    assert sa["X"] == expected
    assert sa["Z"] == expected
    assert sa["X"] == five_qubit_single_axis_weight_enumerator("X")
    assert sa["Z"] == five_qubit_single_axis_weight_enumerator("Z")


def test_cached_mc_sweeps_std_error_metric_conversion_parity() -> None:
    sweeps_path = Path(__file__).resolve().parents[2] / "results" / "data" / "fidelity_sweeps.json"
    raw_records = json.loads(sweeps_path.read_text(encoding="utf-8"))["records"]
    raw_5q_dep = {
        round(float(r["p"]), 8): r
        for r in raw_records
        if r["code"] == "five_qubit" and r["channel"] == "depolarizing"
    }
    raw_bf_bf = {
        round(float(r["p"]), 8): r
        for r in raw_records
        if r["code"] == "bit_flip_3" and r["channel"] == "bit_flip"
    }

    sw_ent = run_sweeps(
        codes=["five_qubit", "bit_flip_3"],
        channels=["depolarizing", "bit_flip"],
        metric="entanglement",
        p_min=0.0,
        p_max=0.5,
        num_p=21,
        run_live_mc=False,
    )
    for row in sw_ent["mc_sweeps"]["five_qubit/depolarizing"]:
        ref = raw_5q_dep[round(float(row["p"]), 8)]
        assert row["metric"] == "state"
        assert abs(row["converted_fidelity"] - (1.0 - 1.5 * (1.0 - float(ref["mean_fidelity"])))) <= PARITY_TOL
        assert abs(row["std_error"] - 1.5 * float(ref["std_error"])) <= PARITY_TOL
    for row in sw_ent["mc_sweeps"]["bit_flip_3/bit_flip"]:
        ref = raw_bf_bf[round(float(row["p"]), 8)]
        assert row["metric"] == "entanglement"
        assert abs(row["converted_fidelity"] - float(ref["mean_fidelity"])) <= PARITY_TOL
        assert abs(row["std_error"] - float(ref["std_error"])) <= PARITY_TOL

    sw_st = run_sweeps(
        codes=["five_qubit", "bit_flip_3"],
        channels=["depolarizing", "bit_flip"],
        metric="state",
        p_min=0.0,
        p_max=0.5,
        num_p=21,
        run_live_mc=False,
    )
    for row in sw_st["mc_sweeps"]["bit_flip_3/bit_flip"]:
        ref = raw_bf_bf[round(float(row["p"]), 8)]
        assert row["metric"] == "entanglement"
        assert abs(row["converted_fidelity"] - (1.0 - (2.0 / 3.0) * (1.0 - float(ref["mean_fidelity"])))) <= PARITY_TOL
        assert abs(row["std_error"] - (2.0 / 3.0) * float(ref["std_error"])) <= PARITY_TOL
    for row in sw_st["mc_sweeps"]["five_qubit/depolarizing"]:
        ref = raw_5q_dep[round(float(row["p"]), 8)]
        assert row["metric"] == "state"
        assert abs(row["converted_fidelity"] - float(ref["mean_fidelity"])) <= PARITY_TOL
        assert abs(row["std_error"] - float(ref["std_error"])) <= PARITY_TOL


def test_negative_seed_schema_and_http_422() -> None:
    with pytest.raises(ValidationError):
        StepRequest.model_validate(
            {
                "code": "bit_flip_3",
                "error_mode": "pauli_string",
                "pauli_string": "III",
                "seed": -1,
            }
        )
    with pytest.raises(ValidationError):
        SimulateRequest.model_validate({"code": "bit_flip_3", "seed": -1})
    with pytest.raises(ValidationError):
        SweepsRequest.model_validate({"seed": -1})


def test_http_adversarial_methods_headers_traversal_and_eigenkit(live_server: str) -> None:
    # 1. Non-finite JSON constants across all POST endpoints -> 400
    for endpoint, raw_payload in (
        ("/api/step", b'{"code":"bit_flip_3","error_mode":"coherent_collective","collective_phi":Infinity}'),
        ("/api/step", b'{"code":"bit_flip_3","error_mode":"coherent_collective","collective_phi":-Infinity}'),
        ("/api/simulate", b'{"code":"bit_flip_3","p":NaN}'),
        ("/api/sweeps", b'{"p_max":Infinity}'),
    ):
        status, _, raw = _http_request(
            f"{live_server}{endpoint}",
            method="POST",
            body=raw_payload,
            headers={"Content-Type": "application/json"},
        )
        assert status == 400
        assert "error" in json.loads(raw.decode("utf-8"))

    # 2. Negative, non-integer, zero Content-Length -> 400 via raw HTTP connection
    parsed_srv = urlsplit(live_server)
    for bad_cl in ("-5", "abc", "0"):
        conn = http.client.HTTPConnection(parsed_srv.hostname, parsed_srv.port, timeout=5.0)
        try:
            conn.putrequest("POST", "/api/simulate")
            conn.putheader("Content-Type", "application/json")
            conn.putheader("Content-Length", bad_cl)
            conn.endheaders()
            resp = conn.getresponse()
            assert resp.status == 400
            assert "error" in json.loads(resp.read().decode("utf-8"))
        finally:
            conn.close()

    # 3. Negative seed -> 422 on all 3 POST endpoints (regardless of error_mode)
    for endpoint, payload in (
        ("/api/step", {"code": "bit_flip_3", "error_mode": "pauli_string", "pauli_string": "III", "seed": -1}),
        ("/api/step", {"code": "bit_flip_3", "error_mode": "stochastic_sample", "channel": "bit_flip", "p": 0.1, "seed": -1}),
        ("/api/simulate", {"code": "bit_flip_3", "seed": -1}),
        ("/api/sweeps", {"seed": -1}),
    ):
        status, body = _http_json(live_server, endpoint, payload)
        assert status == 422
        assert "error" in body

    # 4. Static traversal & encoding edge cases
    for forbidden in (
        "/static/",
        "/static/.",
        "/static/%5c..%5capp.py",
        "/static/foo%00bar.js",
    ):
        status, _, raw = _http_request(f"{live_server}{forbidden}")
        assert status == 403
        assert "error" in json.loads(raw.decode("utf-8"))

    status, _, raw = _http_request(f"{live_server}/static/%252e%252e/app.py")
    assert status == 404
    assert "error" in json.loads(raw.decode("utf-8"))

    # 5. Unsupported HTTP methods -> structured JSON 405 (never HTML 501)
    for method in ("PUT", "DELETE", "PATCH", "OPTIONS"):
        for path in ("/api/health", "/api/metadata", "/api/theory?code=five_qubit", "/api/step", "/api/simulate", "/api/sweeps"):
            status, headers, raw = _http_request(f"{live_server}{path}", method=method)
            assert status == 405, f"{method} {path} returned {status}"
            assert "application/json" in headers.get("Content-Type", "")
            assert "error" in json.loads(raw.decode("utf-8"))

    # HEAD returns status & headers with empty body
    status, headers, raw = _http_request(f"{live_server}/api/health", method="HEAD")
    assert status == 200
    assert "application/json" in headers.get("Content-Type", "")
    assert int(headers.get("Content-Length", "0")) > 0
    assert raw == b""

    status, _, raw = _http_request(f"{live_server}/api/step", method="HEAD")
    assert status == 405
    assert raw == b""

    # 6. eigenkit_unitary live MC sweep -> 501; deterministic p=0.0 cardinal_avg -> 200 parity
    if "eigenkit_unitary" in list_available_backends():
        status, body = _http_json(
            live_server,
            "/api/sweeps",
            {
                "codes": ["bit_flip_3"],
                "channels": ["bit_flip"],
                "p_min": 0.0,
                "p_max": 0.2,
                "num_p": 2,
                "run_live_mc": True,
                "num_shots": 10,
                "backend": "eigenkit_unitary",
            },
        )
        assert status == 501
        assert "error" in body

        st_ek, res_ek = _http_json(
            live_server,
            "/api/simulate",
            {
                "code": "five_qubit",
                "channel": "depolarizing",
                "p": 0.0,
                "num_shots": 6,
                "state_mode": "cardinal_avg",
                "backend": "eigenkit_unitary",
            },
        )
        st_np, res_np = _http_json(
            live_server,
            "/api/simulate",
            {
                "code": "five_qubit",
                "channel": "depolarizing",
                "p": 0.0,
                "num_shots": 6,
                "state_mode": "cardinal_avg",
                "backend": "numpy",
            },
        )
        assert st_ek == 200 and st_np == 200
        assert abs(res_ek["mean_fidelity"] - res_np["mean_fidelity"]) <= PARITY_TOL
        assert abs(res_ek["mean_fidelity"] - 1.0) <= PARITY_TOL


def test_math_edge_cases_haar_su2_and_cli_invalid_ports() -> None:
    # Haar state determinism
    s1 = resolve_input_state("haar", seed=12345)
    s2 = resolve_input_state("haar", seed=12345)
    s3 = resolve_input_state("haar", seed=54321)
    assert np.allclose(s1, s2, atol=PARITY_TOL, rtol=0.0)
    assert not np.allclose(s1, s3, atol=1e-6, rtol=0.0)
    assert abs(float(np.linalg.norm(s1)) - 1.0) <= PARITY_TOL

    # Extreme SU(2) angles & near-pole axes
    for theta, (nx, ny, nz) in (
        (-1.23, (1e-13, 1e-13, 1.0)),
        (5.5, (1.0, 1e-13, -1e-13)),
        (1e-9, (-1e-13, 1.0, 1e-13)),
        (-math.pi, (0.3, -0.4, 0.85)),
    ):
        ops, coeffs, _ = build_su2_rotation(0, theta, nx, ny, nz, n_data=5)
        assert abs(sum(c["prob"] for c in coeffs.values()) - 1.0) <= PARITY_TOL
        norm = math.hypot(nx, ny, nz)
        ux, uy, uz = nx / norm, ny / norm, nz / norm
        half = 0.5 * theta
        c_v, s_v = math.cos(half), math.sin(half)
        u_exact = np.array(
            [
                [complex(c_v, -uz * s_v), complex(-uy * s_v, -ux * s_v)],
                [complex(uy * s_v, -ux * s_v), complex(c_v, uz * s_v)],
            ],
            dtype=np.complex128,
        )
        u_ops = np.eye(2, dtype=np.complex128)
        for op in ops:
            u_ops = _single_qubit_matrix(op) @ u_ops
        assert np.max(np.abs(u_ops - u_exact)) <= PARITY_TOL

    # CLI invalid port -> exit code 1
    repo_root = Path(__file__).resolve().parents[2]
    venv_python = repo_root / ".venv" / "bin" / "python"
    py_exe = str(venv_python) if venv_python.exists() else sys.executable
    sub_env = dict(os.environ)
    sub_env["PYTHONPATH"] = str(repo_root / "src")

    for bad_port in ("-1", "70000"):
        proc = subprocess.run(
            [py_exe, "-m", "qecc.cli", "serve", "--port", bad_port, "--no-browser"],
            cwd=str(repo_root),
            env=sub_env,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        assert proc.returncode == 1
        with pytest.raises(ValueError):
            create_server("127.0.0.1", int(bad_port))


def test_http_keepalive_post_errors_and_allow_header(live_server: str) -> None:
    parsed_srv = urlsplit(live_server)
    conn = http.client.HTTPConnection(parsed_srv.hostname, parsed_srv.port, timeout=5.0)

    def _assert_followup_health(*, expect_keepalive: bool = False) -> None:
        conn.request("GET", "/api/health")
        followup = conn.getresponse()
        raw_fu = followup.read()
        assert followup.status == 200, f"Follow-up GET failed: {followup.status} {raw_fu!r}"
        if expect_keepalive:
            assert followup.will_close is False
        fu_data = json.loads(raw_fu.decode("utf-8"))
        assert fu_data["status"] == "ok"

    try:
        # 1. POST /api/health with JSON body -> 405 JSON + Allow: GET, HEAD + clean follow-up
        body_bad = b'{"bad":1}'
        conn.request(
            "POST",
            "/api/health",
            body=body_bad,
            headers={"Content-Type": "application/json", "Content-Length": str(len(body_bad))},
        )
        resp = conn.getresponse()
        raw = resp.read()
        assert resp.status == 405
        allow_hdr = resp.getheader("Allow") or ""
        assert "GET" in allow_hdr and "HEAD" in allow_hdr
        assert resp.getheader("Connection") == "close"
        assert "application/json" in (resp.getheader("Content-Type") or "")
        assert "error" in json.loads(raw.decode("utf-8"))
        _assert_followup_health()

        # 2. POST /api/unknown with JSON body -> 404 JSON + clean follow-up
        conn.request(
            "POST",
            "/api/unknown",
            body=body_bad,
            headers={"Content-Type": "application/json", "Content-Length": str(len(body_bad))},
        )
        resp = conn.getresponse()
        raw = resp.read()
        assert resp.status == 404
        assert resp.getheader("Connection") == "close"
        assert "error" in json.loads(raw.decode("utf-8"))
        _assert_followup_health()

        # 3. POST /api/simulate with Transfer-Encoding: chunked -> 400 JSON + clean follow-up
        conn.putrequest("POST", "/api/simulate")
        conn.putheader("Content-Type", "application/json")
        conn.putheader("Transfer-Encoding", "chunked")
        conn.endheaders()
        conn.send(b'16\r\n{"code":"bit_flip_3"}\r\n0\r\n\r\n')
        resp = conn.getresponse()
        raw = resp.read()
        assert resp.status == 400
        assert resp.getheader("Connection") == "close"
        assert "error" in json.loads(raw.decode("utf-8"))
        _assert_followup_health()

        # 4. GET /api/step with unexpected body -> 405 JSON + Allow: POST + clean follow-up
        conn.putrequest("GET", "/api/step")
        conn.putheader("Content-Length", str(len(body_bad)))
        conn.endheaders()
        conn.send(body_bad)
        resp = conn.getresponse()
        raw = resp.read()
        assert resp.status == 405
        assert "POST" in (resp.getheader("Allow") or "")
        assert resp.getheader("Connection") == "close"
        assert "error" in json.loads(raw.decode("utf-8"))
        _assert_followup_health()

        # 5. Valid POST with Content-Type charset parameter -> 200 + persistent keep-alive follow-up
        valid_body = b'{"code":"bit_flip_3","p":0.0,"num_shots":5}'
        conn.request(
            "POST",
            "/api/simulate",
            body=valid_body,
            headers={
                "Content-Type": "application/json; charset=utf-8",
                "Content-Length": str(len(valid_body)),
            },
        )
        resp = conn.getresponse()
        raw = resp.read()
        assert resp.status == 200
        assert resp.will_close is False
        sim = json.loads(raw.decode("utf-8"))
        assert sim["code"] == "bit_flip_3"
        _assert_followup_health(expect_keepalive=True)
    finally:
        conn.close()
