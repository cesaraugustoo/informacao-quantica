"""Tier 2 opaque-box E2E tests for Features F1-F11 (BVA, invalid inputs, edge cases, & error handling).

All ``qecc.*`` imports are deferred inside functions/fixtures so ``pytest --collect-only tests/e2e``
succeeds cleanly with zero top-level ``qecc`` dependencies.
"""

from __future__ import annotations

import math
from typing import Iterator

import numpy as np
import pytest

from .web_harness import (
    BrowserSession,
    PageSession,
    ServerHandle,
    analytical_coherence_oracle,
    assert_json_error,
    decode_complex_mat,
    decode_complex_vec,
    find_leak_tokens,
    http_request_json,
    http_request_raw,
    list_backends_oracle,
)

pytestmark = pytest.mark.tier2


@pytest.fixture(scope="module")
def live_server() -> Iterator[ServerHandle]:
    with ServerHandle(use_cli=False, host="127.0.0.1", port=0) as srv:
        yield srv


@pytest.fixture(scope="module")
def browser_session() -> Iterator[BrowserSession]:
    with BrowserSession() as browser:
        yield browser


@pytest.fixture()
def page(browser_session: BrowserSession, live_server: ServerHandle) -> Iterator[PageSession]:
    pg = browser_session.new_page()
    try:
        pg.goto(live_server.base_url)
        pg.wait_for_testid("code-select")
        yield pg
    finally:
        pg.close()


# ---------------------------------------------------------------------------
# Feature F1 Boundaries: HTTP Routing, Path Traversal, Malformed Payloads
# ---------------------------------------------------------------------------


def test_f1_bva_static_path_traversal_blocked(live_server: ServerHandle) -> None:
    for traversal_path in (
        "/static/../../etc/passwd",
        "/static/%2e%2e/app.py",
        "/static/..%2f__init__.py",
        "/static/../pyproject.toml",
    ):
        status, _, raw = http_request_raw(live_server.base_url, traversal_path)
        assert status in (400, 403, 404), f"Path traversal {traversal_path!r} returned status={status}"
        assert b"root:x:0:0" not in raw


def test_f1_bva_unknown_api_route_returns_404_json(live_server: ServerHandle) -> None:
    status, _, body = http_request_json(live_server.base_url, "/api/nonexistent_route_xyz")
    assert status == 404
    assert isinstance(body, dict) and ("error" in body or "detail" in body)


def test_f1_bva_malformed_json_body_returns_400_or_422(live_server: ServerHandle) -> None:
    status, _, body = http_request_json(
        live_server.base_url,
        "/api/step",
        method="POST",
        raw_body=b"{not valid json",
    )
    assert_json_error(status, body, allowed_statuses=(400, 422))


def test_f1_bva_wrong_http_methods_rejected(live_server: ServerHandle) -> None:
    st_get, _, b_get = http_request_json(live_server.base_url, "/api/step", method="GET")
    assert_json_error(st_get, b_get, allowed_statuses=(400, 404, 405, 422))

    st_post, _, b_post = http_request_json(
        live_server.base_url, "/api/health", method="POST", payload={}
    )
    assert_json_error(st_post, b_post, allowed_statuses=(400, 404, 405, 422))


def test_f1_bva_empty_post_body_rejected_and_server_recovers(live_server: ServerHandle) -> None:
    st_err, _, body_err = http_request_json(
        live_server.base_url,
        "/api/simulate",
        method="POST",
        raw_body=b"",
        headers={"Content-Length": "0"},
    )
    assert_json_error(st_err, body_err, allowed_statuses=(400, 422))

    st_ok, _, body_ok = http_request_json(live_server.base_url, "/api/health")
    assert st_ok == 200
    assert body_ok["status"] == "ok"


# ---------------------------------------------------------------------------
# Feature F2 Boundaries: Invalid Codes & Raw Constraints
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_code", ["shor_9", "", 123, None])
def test_f2_bva_unknown_code_rejected_on_step_and_simulate(
    live_server: ServerHandle, bad_code: object
) -> None:
    st_step, _, b_step = http_request_json(
        live_server.base_url,
        "/api/step",
        method="POST",
        payload={"code": bad_code, "state_preset": "+i", "error_mode": "pauli_string", "pauli_string": "III"},
    )
    assert_json_error(st_step, b_step)

    st_sim, _, b_sim = http_request_json(
        live_server.base_url,
        "/api/simulate",
        method="POST",
        payload={"code": bad_code, "channel": "bit_flip", "p": 0.05, "num_shots": 10, "state_mode": "0"},
    )
    assert_json_error(st_sim, b_sim)


def test_f2_bva_raw_rejected_on_theory(live_server: ServerHandle) -> None:
    st, _, body = http_request_json(live_server.base_url, "/api/theory?code=raw")
    assert_json_error(st, body)


def test_f2_bva_raw_rejected_on_step(live_server: ServerHandle) -> None:
    st, _, body = http_request_json(
        live_server.base_url,
        "/api/step",
        method="POST",
        payload={"code": "raw", "state_preset": "+i", "error_mode": "pauli_string", "pauli_string": "I"},
    )
    assert_json_error(st, body)


def test_f2_bva_code_case_sensitivity_rejected(live_server: ServerHandle) -> None:
    for bad in ("Bit_Flip_3", "FIVE_QUBIT", "DFS_2"):
        st, _, body = http_request_json(
            live_server.base_url,
            "/api/step",
            method="POST",
            payload={"code": bad, "state_preset": "0", "error_mode": "pauli_string", "pauli_string": "III"},
        )
        assert_json_error(st, body)


def test_f2_bva_metadata_uncorrupted_after_invalid_requests(live_server: ServerHandle) -> None:
    http_request_json(live_server.base_url, "/api/step", method="POST", payload={"code": "invalid"})
    st, _, meta = http_request_json(live_server.base_url, "/api/metadata")
    assert st == 200
    assert set(["bit_flip_3", "phase_flip_3", "five_qubit", "dfs_2"]).issubset(set(meta["codes"].keys()))


# ---------------------------------------------------------------------------
# Feature F3 Boundaries: Syndromes, Zero-Probability Branches, & Weight-2 Aliasing
# ---------------------------------------------------------------------------


def test_f3_bva_invalid_selected_syndrome_format_rejected(live_server: ServerHandle) -> None:
    for cname, bad_synd in (("bit_flip_3", "99"), ("bit_flip_3", "000"), ("five_qubit", "00")):
        st, _, body = http_request_json(
            live_server.base_url,
            "/api/step",
            method="POST",
            payload={
                "code": cname,
                "state_preset": "+i",
                "error_mode": "pauli_string",
                "pauli_string": "III" if cname == "bit_flip_3" else "IIIII",
                "selected_syndrome": bad_synd,
            },
        )
        assert_json_error(st, body)


def test_f3_bva_zero_probability_syndrome_branch_rejected_cleanly(live_server: ServerHandle) -> None:
    st, _, body = http_request_json(
        live_server.base_url,
        "/api/step",
        method="POST",
        payload={
            "code": "bit_flip_3",
            "state_preset": "+i",
            "error_mode": "pauli_string",
            "pauli_string": "XII",
            "selected_syndrome": "11",
        },
    )
    assert_json_error(st, body)


def test_f3_bva_nonempty_selected_syndrome_for_dfs2_rejected(live_server: ServerHandle) -> None:
    st, _, body = http_request_json(
        live_server.base_url,
        "/api/step",
        method="POST",
        payload={
            "code": "dfs_2",
            "state_preset": "+i",
            "error_mode": "pauli_string",
            "pauli_string": "II",
            "selected_syndrome": "0",
        },
    )
    assert_json_error(st, body)


def test_f3_bva_bit_flip_3_weight2_aliasing_ixx_logical_flip(live_server: ServerHandle) -> None:
    st, _, body = http_request_json(
        live_server.base_url,
        "/api/step",
        method="POST",
        payload={
            "code": "bit_flip_3",
            "state_preset": "0",
            "error_mode": "pauli_string",
            "pauli_string": "IXX",
        },
    )
    assert st == 200
    assert body["stages"][3]["selected_syndrome"] == "10"
    assert abs(float(body["stages"][5]["fidelity_vs_input"]) - 0.0) <= 1e-12


def test_f3_bva_phase_flip_3_undetected_single_x_logical_flip(live_server: ServerHandle) -> None:
    st, _, body = http_request_json(
        live_server.base_url,
        "/api/step",
        method="POST",
        payload={
            "code": "phase_flip_3",
            "state_preset": "+",
            "error_mode": "pauli_string",
            "pauli_string": "XII",
        },
    )
    assert st == 200
    assert body["stages"][3]["selected_syndrome"] == "00"
    assert abs(float(body["stages"][5]["fidelity_vs_input"]) - 0.0) <= 1e-12


# ---------------------------------------------------------------------------
# Feature F4 Boundaries: SU(2) Axis/Qubit/Angle Validation & Exact Boundaries
# ---------------------------------------------------------------------------


def test_f4_bva_zero_norm_su2_axis_rejected(live_server: ServerHandle) -> None:
    st, _, body = http_request_json(
        live_server.base_url,
        "/api/step",
        method="POST",
        payload={
            "code": "bit_flip_3",
            "state_preset": "+i",
            "error_mode": "su2_rotation",
            "su2_qubit": 0,
            "su2_theta": 1.0,
            "su2_nx": 0.0,
            "su2_ny": 0.0,
            "su2_nz": 0.0,
        },
    )
    assert_json_error(st, body)


def test_f4_bva_su2_qubit_out_of_range_and_bool_rejected(live_server: ServerHandle) -> None:
    for bad_q in (-1, 3, True):
        st, _, body = http_request_json(
            live_server.base_url,
            "/api/step",
            method="POST",
            payload={
                "code": "bit_flip_3",
                "state_preset": "+i",
                "error_mode": "su2_rotation",
                "su2_qubit": bad_q,
                "su2_theta": 0.5,
                "su2_nx": 1.0,
                "su2_ny": 0.0,
                "su2_nz": 0.0,
            },
        )
        assert_json_error(st, body)


def test_f4_bva_nonfinite_su2_theta_and_collective_phi_rejected(live_server: ServerHandle) -> None:
    st1, _, b1 = http_request_json(
        live_server.base_url,
        "/api/step",
        method="POST",
        payload={
            "code": "bit_flip_3",
            "state_preset": "+i",
            "error_mode": "su2_rotation",
            "su2_qubit": 0,
            "su2_theta": "NaN",
            "su2_nx": 1.0,
            "su2_ny": 0.0,
            "su2_nz": 0.0,
        },
    )
    assert_json_error(st1, b1)

    st2, _, b2 = http_request_json(
        live_server.base_url,
        "/api/step",
        method="POST",
        payload={
            "code": "dfs_2",
            "state_preset": "+i",
            "error_mode": "coherent_collective",
            "collective_phi": None,
        },
    )
    assert_json_error(st2, b2)


def test_f4_bva_su2_theta_zero_and_two_pi_identity_boundaries(live_server: ServerHandle) -> None:
    for theta in (0.0, 2.0 * math.pi):
        st, _, body = http_request_json(
            live_server.base_url,
            "/api/step",
            method="POST",
            payload={
                "code": "bit_flip_3",
                "state_preset": "+i",
                "error_mode": "su2_rotation",
                "su2_qubit": 0,
                "su2_theta": theta,
                "su2_nx": 1.0,
                "su2_ny": 0.0,
                "su2_nz": 0.0,
            },
        )
        assert st == 200
        assert abs(float(body["stages"][3]["syndrome_probabilities"]["00"]) - 1.0) <= 1e-12
        assert abs(float(body["stages"][5]["fidelity_vs_input"]) - 1.0) <= 1e-12


def test_f4_bva_su2_theta_pi_exact_x_flip_boundary(live_server: ServerHandle) -> None:
    st, _, body = http_request_json(
        live_server.base_url,
        "/api/step",
        method="POST",
        payload={
            "code": "bit_flip_3",
            "state_preset": "+i",
            "error_mode": "su2_rotation",
            "su2_qubit": 0,
            "su2_theta": math.pi,
            "su2_nx": 1.0,
            "su2_ny": 0.0,
            "su2_nz": 0.0,
        },
    )
    assert st == 200
    assert abs(float(body["stages"][3]["syndrome_probabilities"]["10"]) - 1.0) <= 1e-12
    assert body["stages"][3]["selected_syndrome"] == "10"
    assert abs(float(body["stages"][5]["fidelity_vs_input"]) - 1.0) <= 1e-12


# ---------------------------------------------------------------------------
# Feature F5 Boundaries: Bloch Sphere & |rho_01| Limits
# ---------------------------------------------------------------------------


def test_f5_bva_basis_states_zero_coherence_and_polar_bloch(live_server: ServerHandle) -> None:
    for label, expected_rz in (("0", 1.0), ("1", -1.0)):
        st, _, body = http_request_json(
            live_server.base_url,
            "/api/step",
            method="POST",
            payload={
                "code": "bit_flip_3",
                "state_preset": label,
                "error_mode": "pauli_string",
                "pauli_string": "III",
            },
        )
        assert st == 200
        assert abs(float(body["input_state"]["abs_rho01"])) <= 1e-12
        assert np.max(np.abs(np.asarray(body["input_state"]["bloch"], dtype=float) - [0.0, 0.0, expected_rz])) <= 1e-12


def test_f5_bva_bit_flip_3_coherence_zero_crossing_at_p_half(live_server: ServerHandle) -> None:
    st, _, body = http_request_json(
        live_server.base_url,
        "/api/sweeps",
        method="POST",
        payload={
            "codes": ["bit_flip_3"],
            "channels": ["bit_flip"],
            "metric": "entanglement",
            "p_min": 0.0,
            "p_max": 0.5,
            "num_p": 3,
            "run_live_mc": False,
        },
    )
    assert st == 200
    val_half = float(body["coherence_curves"]["bit_flip_3/bit_flip"]["abs_rho01_analytical"][-1])
    assert abs(val_half - 0.0) <= 1e-12


def test_f5_bva_five_qubit_depolarizing_coherence_below_raw_above_threshold(
    live_server: ServerHandle,
) -> None:
    st, _, body = http_request_json(
        live_server.base_url,
        "/api/sweeps",
        method="POST",
        payload={
            "codes": ["five_qubit"],
            "channels": ["depolarizing"],
            "metric": "entanglement",
            "p_min": 0.3,
            "p_max": 0.3,
            "num_p": 2,
            "run_live_mc": False,
        },
    )
    # Note: if num_p=2 requires p_min < p_max, test over [0.25, 0.35]
    if st in (400, 422):
        st, _, body = http_request_json(
            live_server.base_url,
            "/api/sweeps",
            method="POST",
            payload={
                "codes": ["five_qubit"],
                "channels": ["depolarizing"],
                "metric": "entanglement",
                "p_min": 0.25,
                "p_max": 0.35,
                "num_p": 3,
                "run_live_mc": False,
            },
        )
    assert st == 200
    coh = body["coherence_curves"]["five_qubit/depolarizing"]
    assert float(coh["abs_rho01_analytical"][1]) < float(coh["abs_rho01_raw"][1])


def test_f5_bva_dfs2_p1_and_phi_pi_vs_bit_flip_failure(live_server: ServerHandle) -> None:
    st1, _, b1 = http_request_json(
        live_server.base_url,
        "/api/step",
        method="POST",
        payload={
            "code": "dfs_2",
            "state_preset": "+i",
            "error_mode": "coherent_collective",
            "collective_phi": math.pi,
        },
    )
    assert st1 == 200
    assert abs(float(b1["stages"][5]["logical_abs_rho01"]) - 0.5) <= 1e-12

    assert abs(analytical_coherence_oracle("dfs_2", "collective_dephasing", 1.0) - 0.5) <= 1e-12
    assert abs(analytical_coherence_oracle("dfs_2", "bit_flip", 0.5) - 0.0) <= 1e-12


def test_f5_bva_bloch_norm_never_exceeds_unit_sphere(live_server: ServerHandle) -> None:
    for p in (0.0, 0.1, 0.3, 0.5):
        st, _, body = http_request_json(
            live_server.base_url,
            "/api/simulate",
            method="POST",
            payload={
                "code": "five_qubit",
                "channel": "depolarizing",
                "p": p,
                "num_shots": 200,
                "state_mode": "+i",
                "seed": 42,
                "backend": "numpy",
            },
        )
        assert st == 200
        norm = float(np.linalg.norm(np.asarray(body["bloch_out"], dtype=float)))
        assert norm <= 1.0 + 1e-12


# ---------------------------------------------------------------------------
# Feature F6 Boundaries: Input State Validation & Eigenstate Invariance
# ---------------------------------------------------------------------------


def test_f6_bva_custom_zero_amplitudes_rejected(live_server: ServerHandle) -> None:
    st, _, body = http_request_json(
        live_server.base_url,
        "/api/step",
        method="POST",
        payload={
            "code": "bit_flip_3",
            "state_preset": "custom",
            "amplitudes": [[0.0, 0.0], [0.0, 0.0]],
            "error_mode": "pauli_string",
            "pauli_string": "III",
        },
    )
    assert_json_error(st, body)


def test_f6_bva_custom_wrong_length_amplitudes_rejected(live_server: ServerHandle) -> None:
    for amps in ([[1.0, 0.0]], [[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]]):
        st, _, body = http_request_json(
            live_server.base_url,
            "/api/step",
            method="POST",
            payload={
                "code": "bit_flip_3",
                "state_preset": "custom",
                "amplitudes": amps,
                "error_mode": "pauli_string",
                "pauli_string": "III",
            },
        )
        assert_json_error(st, body)


def test_f6_bva_nonfinite_bloch_angles_and_amplitudes_rejected(live_server: ServerHandle) -> None:
    st1, _, b1 = http_request_json(
        live_server.base_url,
        "/api/step",
        method="POST",
        payload={
            "code": "bit_flip_3",
            "state_preset": "bloch",
            "theta": "NaN",
            "phi_state": 0.0,
            "error_mode": "pauli_string",
            "pauli_string": "III",
        },
    )
    assert_json_error(st1, b1)

    st2, _, b2 = http_request_json(
        live_server.base_url,
        "/api/step",
        method="POST",
        payload={
            "code": "bit_flip_3",
            "state_preset": "custom",
            "amplitudes": [[None, 0.0], [1.0, 0.0]],
            "error_mode": "pauli_string",
            "pauli_string": "III",
        },
    )
    assert_json_error(st2, b2)


def test_f6_bva_unnormalized_custom_amplitudes_consistent_handling(live_server: ServerHandle) -> None:
    st, _, body = http_request_json(
        live_server.base_url,
        "/api/step",
        method="POST",
        payload={
            "code": "bit_flip_3",
            "state_preset": "custom",
            "amplitudes": [[3.0, 0.0], [0.0, 4.0]],
            "error_mode": "pauli_string",
            "pauli_string": "III",
        },
    )
    if st == 200:
        amps = decode_complex_vec(body["input_state"]["amplitudes"])
        assert np.max(np.abs(amps - np.array([0.6 + 0.0j, 0.0 + 0.8j]))) <= 1e-12
    else:
        assert_json_error(st, body)


def test_f6_bva_eigenstate_invariance_boundaries(live_server: ServerHandle) -> None:
    st1, _, b1 = http_request_json(
        live_server.base_url,
        "/api/step",
        method="POST",
        payload={
            "code": "bit_flip_3",
            "state_preset": "+",
            "error_mode": "pauli_string",
            "pauli_string": "XXX",
        },
    )
    assert st1 == 200
    assert abs(float(b1["stages"][5]["fidelity_vs_input"]) - 1.0) <= 1e-12

    # For phase_flip_3, Z_L = XXX (leaves |0> invariant) and X_L = ZZZ (leaves |+> invariant)
    st2, _, b2 = http_request_json(
        live_server.base_url,
        "/api/step",
        method="POST",
        payload={
            "code": "phase_flip_3",
            "state_preset": "0",
            "error_mode": "pauli_string",
            "pauli_string": "XXX",
        },
    )
    assert st2 == 200
    assert abs(float(b2["stages"][5]["fidelity_vs_input"]) - 1.0) <= 1e-12

    st3, _, b3 = http_request_json(
        live_server.base_url,
        "/api/step",
        method="POST",
        payload={
            "code": "phase_flip_3",
            "state_preset": "+",
            "error_mode": "pauli_string",
            "pauli_string": "ZZZ",
        },
    )
    assert st3 == 200
    assert abs(float(b3["stages"][5]["fidelity_vs_input"]) - 1.0) <= 1e-12


# ---------------------------------------------------------------------------
# Feature F7 Boundaries: Noise, Parameters & Backend Error Handling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_p", [-0.01, 1.01, True, "abc"])
def test_f7_bva_invalid_p_out_of_range_and_bool_rejected(
    live_server: ServerHandle, bad_p: object
) -> None:
    st, _, body = http_request_json(
        live_server.base_url,
        "/api/simulate",
        method="POST",
        payload={
            "code": "bit_flip_3",
            "channel": "bit_flip",
            "p": bad_p,
            "num_shots": 10,
            "state_mode": "0",
        },
    )
    assert_json_error(st, body)


@pytest.mark.parametrize("bad_shots", [0, -5, True])
def test_f7_bva_invalid_num_shots_rejected(live_server: ServerHandle, bad_shots: object) -> None:
    st, _, body = http_request_json(
        live_server.base_url,
        "/api/simulate",
        method="POST",
        payload={
            "code": "bit_flip_3",
            "channel": "bit_flip",
            "p": 0.05,
            "num_shots": bad_shots,
            "state_mode": "0",
        },
    )
    assert_json_error(st, body)


@pytest.mark.parametrize("bad_channel", ["amplitude_damping", "", "bitflip"])
def test_f7_bva_invalid_channel_rejected(live_server: ServerHandle, bad_channel: str) -> None:
    st, _, body = http_request_json(
        live_server.base_url,
        "/api/simulate",
        method="POST",
        payload={
            "code": "bit_flip_3",
            "channel": bad_channel,
            "p": 0.05,
            "num_shots": 10,
            "state_mode": "0",
        },
    )
    assert_json_error(st, body)


def test_f7_bva_invalid_pauli_string_and_error_qubit_rejected(live_server: ServerHandle) -> None:
    for bad_ps in ("XIA", "XXXX", ""):
        st, _, body = http_request_json(
            live_server.base_url,
            "/api/step",
            method="POST",
            payload={
                "code": "bit_flip_3",
                "state_preset": "+i",
                "error_mode": "pauli_string",
                "pauli_string": bad_ps,
            },
        )
        assert_json_error(st, body)

    for bad_q in (-1, 3):
        st, _, body = http_request_json(
            live_server.base_url,
            "/api/step",
            method="POST",
            payload={
                "code": "bit_flip_3",
                "state_preset": "+i",
                "error_mode": "single_qubit_pauli",
                "error_qubit": bad_q,
                "error_op": "X",
            },
        )
        assert_json_error(st, body)


def test_f7_bva_eigenkit_unitary_stochastic_p_gt_0_rejected_cleanly(live_server: ServerHandle) -> None:
    if "eigenkit_unitary" not in list_backends_oracle():
        pytest.skip("eigenkit_unitary backend not installed in environment")

    st, _, body = http_request_json(
        live_server.base_url,
        "/api/simulate",
        method="POST",
        payload={
            "code": "five_qubit",
            "channel": "depolarizing",
            "p": 0.1,
            "num_shots": 10,
            "state_mode": "+i",
            "backend": "eigenkit_unitary",
        },
    )
    assert_json_error(st, body, allowed_statuses=(400, 501))


# ---------------------------------------------------------------------------
# Feature F8 Boundaries: Sweep Grid Bounds, Endpoints & Metrics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_grid",
    [
        {"p_min": 0.6, "p_max": 0.2, "num_p": 5},
        {"p_min": -0.1, "p_max": 0.5, "num_p": 5},
        {"p_min": 0.0, "p_max": 1.1, "num_p": 5},
        {"p_min": 0.0, "p_max": 0.5, "num_p": 1},
    ],
)
def test_f8_bva_invalid_sweep_grid_params_rejected(
    live_server: ServerHandle, bad_grid: dict[str, object]
) -> None:
    st, _, body = http_request_json(
        live_server.base_url,
        "/api/sweeps",
        method="POST",
        payload={
            "codes": ["bit_flip_3"],
            "channels": ["bit_flip"],
            "metric": "entanglement",
            "run_live_mc": False,
            **bad_grid,
        },
    )
    assert_json_error(st, body)


def test_f8_bva_invalid_sweep_metric_rejected(live_server: ServerHandle) -> None:
    st, _, body = http_request_json(
        live_server.base_url,
        "/api/sweeps",
        method="POST",
        payload={
            "codes": ["bit_flip_3"],
            "channels": ["bit_flip"],
            "metric": "process",
            "p_min": 0.0,
            "p_max": 0.5,
            "num_p": 5,
            "run_live_mc": False,
        },
    )
    assert_json_error(st, body)


def test_f8_bva_sweep_endpoints_p0_and_p05(live_server: ServerHandle) -> None:
    st, _, body = http_request_json(
        live_server.base_url,
        "/api/sweeps",
        method="POST",
        payload={
            "codes": ["bit_flip_3", "five_qubit"],
            "channels": ["bit_flip", "depolarizing"],
            "metric": "entanglement",
            "p_min": 0.0,
            "p_max": 0.5,
            "num_p": 3,
            "run_live_mc": False,
        },
    )
    assert st == 200
    bf = body["analytical_curves"]["bit_flip_3/bit_flip"]["fidelities"]
    fq = body["analytical_curves"]["five_qubit/depolarizing"]["fidelities"]
    assert abs(float(bf[0]) - 1.0) <= 1e-12
    assert abs(float(fq[0]) - 1.0) <= 1e-12
    assert abs(float(bf[-1]) - 0.5) <= 1e-12
    assert float(fq[-1]) < float(body["raw_curve"][-1])


def test_f8_bva_sweep_p1_boundary_dfs2_vs_bit_flip_3(live_server: ServerHandle) -> None:
    st, _, body = http_request_json(
        live_server.base_url,
        "/api/sweeps",
        method="POST",
        payload={
            "codes": ["dfs_2", "bit_flip_3"],
            "channels": ["collective_dephasing", "bit_flip"],
            "metric": "entanglement",
            "p_min": 0.0,
            "p_max": 1.0,
            "num_p": 3,
            "run_live_mc": False,
        },
    )
    assert st == 200
    assert abs(float(body["analytical_curves"]["dfs_2/collective_dephasing"]["fidelities"][-1]) - 1.0) <= 1e-12
    assert abs(float(body["analytical_curves"]["bit_flip_3/bit_flip"]["fidelities"][-1]) - 0.0) <= 1e-12


def test_f8_bva_minimal_two_point_grid_succeeds(live_server: ServerHandle) -> None:
    st, _, body = http_request_json(
        live_server.base_url,
        "/api/sweeps",
        method="POST",
        payload={
            "codes": ["bit_flip_3"],
            "channels": ["bit_flip"],
            "metric": "entanglement",
            "p_min": 0.1,
            "p_max": 0.3,
            "num_p": 2,
            "run_live_mc": False,
        },
    )
    assert st == 200
    assert len(body["p_grid"]) == 2
    assert abs(float(body["p_grid"][0]) - 0.1) <= 1e-12
    assert abs(float(body["p_grid"][1]) - 0.3) <= 1e-12


# ---------------------------------------------------------------------------
# Feature F9 Boundaries: Theory Validation, Weight-2 Aliasing & Hermiticity
# ---------------------------------------------------------------------------


def test_f9_bva_theory_missing_or_invalid_code_rejected(live_server: ServerHandle) -> None:
    for path in ("/api/theory?code=", "/api/theory?code=unknown_code"):
        st, _, body = http_request_json(live_server.base_url, path)
        assert_json_error(st, body)


def test_f9_bva_theory_raw_rejected(live_server: ServerHandle) -> None:
    st, _, body = http_request_json(live_server.base_url, "/api/theory?code=raw")
    assert_json_error(st, body)


def test_f9_bva_five_qubit_weight2_aliasing_xxiii_logical_error(live_server: ServerHandle) -> None:
    # XXIII -> syndrome "1001" -> recovery "IIIZI" -> residual XXIZI = Z_L (flips |+> to |->)
    st, _, body = http_request_json(
        live_server.base_url,
        "/api/step",
        method="POST",
        payload={
            "code": "five_qubit",
            "state_preset": "+",
            "error_mode": "pauli_string",
            "pauli_string": "XXIII",
        },
    )
    assert st == 200
    assert body["stages"][3]["selected_syndrome"] == "1001"
    assert abs(float(body["stages"][5]["fidelity_vs_input"]) - 0.0) <= 1e-12


def test_f9_bva_dfs2_theory_empty_stabilizers_and_syndrome_table(live_server: ServerHandle) -> None:
    st, _, body = http_request_json(live_server.base_url, "/api/theory?code=dfs_2")
    assert st == 200
    assert body["stabilizers"] == []
    assert body["syndrome_table"] == {}
    assert body["knill_laflamme"]["satisfies"] is True


def test_f9_bva_knill_laflamme_c_matrix_hermiticity_all_codes(live_server: ServerHandle) -> None:
    for cname in ("bit_flip_3", "phase_flip_3", "five_qubit", "dfs_2"):
        st, _, body = http_request_json(live_server.base_url, f"/api/theory?code={cname}")
        assert st == 200
        c_mat = decode_complex_mat(body["knill_laflamme"]["c_matrix"])
        assert np.max(np.abs(c_mat - c_mat.conj().T)) <= 1e-12


# ---------------------------------------------------------------------------
# Feature F10 Boundaries: Seminar Navigation Limits & Responsive Layout
# ---------------------------------------------------------------------------


def test_f10_bva_seminar_slide_boundary_first_and_last(page: PageSession) -> None:
    page.ensure_visible("seminar-panel")
    # Exercise Prev/Next buttons, slide <select>, or topic tabs at boundaries cleanly
    page.eval_js(
        """(() => {
            const p = document.querySelector('[data-testid="seminar-panel"]');
            if (!p) return;
            const prev = p.querySelector('[data-testid*="prev"]') || Array.from(p.querySelectorAll('button')).find(b => /prev|anter/i.test(b.innerText));
            if (prev) prev.click();
            const next = p.querySelector('[data-testid*="next"]') || Array.from(p.querySelectorAll('button')).find(b => /next|próxim|avanç/i.test(b.innerText));
            if (next) {
                for (let i = 0; i < 25; i++) next.click();
            }
            const sel = p.querySelector('select');
            if (sel && sel.options && sel.options.length > 0) {
                sel.value = sel.options[sel.options.length - 1].value;
                sel.dispatchEvent(new Event('change', {bubbles: true}));
                sel.value = sel.options[0].value;
                sel.dispatchEvent(new Event('change', {bubbles: true}));
            }
            const tabs = Array.from(p.querySelectorAll('[role="tab"], [data-slide], [data-topic], [data-testid*="topic"], [data-testid*="slide"]'))
                .filter(el => el.tagName !== 'SELECT');
            if (tabs.length > 0) {
                tabs[tabs.length - 1].click();
                tabs[0].click();
            }
        })()"""
    )
    txt = page.eval_js('document.querySelector(\'[data-testid="seminar-panel"]\').innerText')
    assert len(txt.strip()) > 20
    assert not find_leak_tokens(txt)
    page.assert_no_errors()


def test_f10_bva_seminar_invalid_hash_or_state_fallback_graceful(
    page: PageSession, live_server: ServerHandle
) -> None:
    page.goto(f"{live_server.base_url}/#slide-999-invalid")
    page.wait_for_testid("seminar-panel")
    page.ensure_visible("seminar-panel")
    page.assert_no_errors()


def test_f10_bva_seminar_rapid_repeated_stepping_no_errors(page: PageSession) -> None:
    page.ensure_visible("seminar-panel")
    page.eval_js(
        """(() => {
            const p = document.querySelector('[data-testid="seminar-panel"]');
            if (!p) return;
            const next = p.querySelector('[data-testid*="next"]') || Array.from(p.querySelectorAll('button')).find(b => /next|próxim|avanç/i.test(b.innerText));
            const prev = p.querySelector('[data-testid*="prev"]') || Array.from(p.querySelectorAll('button')).find(b => /prev|anter/i.test(b.innerText));
            for (let i = 0; i < 10; i++) {
                if (next) next.click();
                if (prev && i % 2 === 1) prev.click();
            }
            const tabs = Array.from(p.querySelectorAll('[role="tab"], [data-slide], [data-topic], [data-testid*="topic"], [data-testid*="slide"]'))
                .filter(el => el.tagName !== 'SELECT');
            if (tabs.length > 1) {
                for (let i = 0; i < 10; i++) tabs[i % tabs.length].click();
            }
            const sel = p.querySelector('select');
            if (sel && sel.options && sel.options.length > 1) {
                for (let i = 0; i < 10; i++) {
                    sel.value = sel.options[i % sel.options.length].value;
                    sel.dispatchEvent(new Event('change', {bubbles: true}));
                }
            }
        })()"""
    )
    page.assert_no_errors()


def test_f10_bva_seminar_mid_simulation_controls_remain_responsive(page: PageSession) -> None:
    page.select_option("code-select", "five_qubit")
    page.ensure_visible("seminar-panel")
    page.ensure_visible("code-select")
    page.select_option("code-select", "dfs_2")
    assert page.eval_js('document.querySelector(\'[data-testid="code-select"]\').value') == "dfs_2"
    page.assert_no_errors()


def test_f10_bva_seminar_narrow_viewport_no_overflow(page: PageSession) -> None:
    page.ensure_visible("seminar-panel")
    for w, h in ((768, 1024), (390, 844)):
        page.set_viewport(w, h, mobile=True)
        assert page.check_no_horizontal_overflow(), (
            f"Seminar overflow at {w}x{h}: {page.viewport_dimensions()}"
        )
    page.assert_no_errors()


# ---------------------------------------------------------------------------
# Feature F11 Boundaries: Rapid Toggle, State Preservation & UTF-8 Diacritics
# ---------------------------------------------------------------------------


def test_f11_bva_rapid_language_toggle_no_duplicates_or_leaks(page: PageSession) -> None:
    for lang in ("pt-br", "en", "pt-br", "en", "pt-br"):
        page.switch_language(lang, settle_s=0.05)
    txt = page.body_text()
    assert not find_leak_tokens(txt)
    page.assert_no_errors()


def test_f11_bva_language_switch_preserves_simulator_selections(page: PageSession) -> None:
    page.select_option("code-select", "five_qubit")
    page.select_option("channel-select", "depolarizing")
    page.select_option("state-select", "+i")
    page.fill_input("p-input", "0.12")

    page.switch_language("pt-br")
    assert page.eval_js('document.querySelector(\'[data-testid="code-select"]\').value') == "five_qubit"
    assert page.eval_js('document.querySelector(\'[data-testid="channel-select"]\').value') == "depolarizing"
    assert page.eval_js('document.querySelector(\'[data-testid="state-select"]\').value') == "+i"
    assert float(page.eval_js('document.querySelector(\'[data-testid="p-input"]\').value')) == pytest.approx(0.12)

    page.switch_language("en")
    assert page.eval_js('document.querySelector(\'[data-testid="code-select"]\').value') == "five_qubit"
    page.assert_no_errors()


def test_f11_bva_invalid_p_input_ui_no_crash_both_langs(page: PageSession) -> None:
    for lang in ("en", "pt-br"):
        page.switch_language(lang)
        page.fill_input("p-input", "1.5")
        txt = page.body_text()
        assert not find_leak_tokens(txt)
        # Note: if UI triggers an API validation call returning 422, browser logs network error;
        # uncaught JS exceptions must ALWAYS be zero!
        assert not page.exceptions, f"Uncaught JS exception on invalid p input in {lang}: {page.exceptions}"
        page.fill_input("p-input", "0.05")
        page.clear_errors()


def test_f11_bva_pt_br_no_untranslated_en_ui_headers(page: PageSession) -> None:
    page.switch_language("pt-br")
    txt = page.body_text()
    assert not find_leak_tokens(txt)
    assert "i18n." not in txt
    pt_lower = txt.lower()
    assert any(w in pt_lower for w in ("síndrome", "estabilizador", "fidelidade", "coerência"))
    page.assert_no_errors()


def test_f11_bva_pt_br_utf8_diacritics_without_mojibake(page: PageSession) -> None:
    page.switch_language("pt-br")
    txt = page.body_text()
    for mojibake in ("Ã£", "Ã§", "Ã©", "Ãª", "Ã³", "Ãº", "ï¿½"):
        assert mojibake not in txt, f"Detected UTF-8 mojibake {mojibake!r} in pt-br UI"
    assert any(ch in txt for ch in ("ã", "ç", "é", "ê", "ó", "ú", "í")), (
        "Expected valid Portuguese UTF-8 diacritics in pt-br UI"
    )
    page.assert_no_errors()
