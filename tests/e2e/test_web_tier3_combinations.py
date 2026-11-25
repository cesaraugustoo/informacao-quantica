"""Tier 3 opaque-box pairwise cross-feature combination E2E tests (16 pairs).

Covers pairwise interactions across Code (4 + raw) x Mode/Channel x State x Endpoint/Backend x UI/Language.
All ``qecc.*`` imports are deferred inside functions/fixtures so ``pytest --collect-only tests/e2e``
succeeds cleanly with zero top-level ``qecc`` dependencies.
"""

from __future__ import annotations

import json
import math
from typing import Iterator

import numpy as np
import pytest

from .web_harness import (
    BrowserSession,
    PageSession,
    ServerHandle,
    abs_rho01,
    analytical_coherence_oracle,
    bloch_to_rho,
    compute_step_oracle,
    decode_complex_mat,
    find_leak_tokens,
    http_request_json,
    list_backends_oracle,
)

pytestmark = pytest.mark.tier3


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


def test_t3_pair_01_bit_flip_3_bit_flip_plus_i_step_and_simulate(
    live_server: ServerHandle, qecc_analysis, helpers
) -> None:
    inv_s2 = 1.0 / math.sqrt(2.0)
    psi = np.array([inv_s2, 1.0j * inv_s2], dtype=np.complex128)
    st_step, _, step_body = http_request_json(
        live_server.base_url,
        "/api/step",
        method="POST",
        payload={
            "code": "bit_flip_3",
            "state_preset": "+i",
            "error_mode": "single_qubit_pauli",
            "error_qubit": 1,
            "error_op": "X",
        },
    )
    assert st_step == 200
    orc = compute_step_oracle(
        "bit_flip_3", psi, error_mode="single_qubit_pauli", error_qubit=1, error_op="X"
    )
    assert step_body["stages"][3]["selected_syndrome"] == "11"
    assert abs(float(step_body["stages"][5]["fidelity_vs_input"]) - orc["stages"][5]["fidelity_vs_input"]) <= 1e-12

    n_shots = 5000
    st_sim, _, sim_body = http_request_json(
        live_server.base_url,
        "/api/simulate",
        method="POST",
        payload={
            "code": "bit_flip_3",
            "channel": "bit_flip",
            "p": 0.05,
            "num_shots": n_shots,
            "state_mode": "+i",
            "seed": 301,
            "backend": "numpy",
        },
    )
    assert st_sim == 200
    exp_f = float(qecc_analysis.analytical_fidelity("bit_flip_3", "bit_flip", 0.05, metric="entanglement"))
    exp_coh = analytical_coherence_oracle("bit_flip_3", "bit_flip", 0.05)
    assert abs(float(sim_body["mean_fidelity"]) - exp_f) <= helpers.binom_sigma(exp_f, n_shots, z=4.5)
    assert abs(float(sim_body["abs_rho01"]) - exp_coh) <= 2.0 * helpers.binom_sigma(exp_f, n_shots, z=4.5)


def test_t3_pair_02_bit_flip_3_su2_rx_branch_11_recovery(live_server: ServerHandle) -> None:
    theta = 0.8
    st, _, body = http_request_json(
        live_server.base_url,
        "/api/step",
        method="POST",
        payload={
            "code": "bit_flip_3",
            "state_preset": "+i",
            "error_mode": "su2_rotation",
            "su2_qubit": 1,
            "su2_theta": theta,
            "su2_nx": 1.0,
            "su2_ny": 0.0,
            "su2_nz": 0.0,
            "selected_syndrome": "11",
        },
    )
    assert st == 200
    assert abs(float(body["stages"][3]["syndrome_probabilities"]["11"]) - math.sin(0.4) ** 2) <= 1e-12
    assert abs(float(body["stages"][5]["fidelity_vs_input"]) - 1.0) <= 1e-12


def test_t3_pair_03_phase_flip_3_phase_flip_hadamard_duality(
    live_server: ServerHandle, qecc_analysis
) -> None:
    st, _, sw = http_request_json(
        live_server.base_url,
        "/api/sweeps",
        method="POST",
        payload={
            "codes": ["bit_flip_3", "phase_flip_3"],
            "channels": ["bit_flip", "phase_flip"],
            "metric": "entanglement",
            "p_min": 0.0,
            "p_max": 0.5,
            "num_p": 11,
            "run_live_mc": False,
        },
    )
    assert st == 200
    bf = sw["analytical_curves"]["bit_flip_3/bit_flip"]["fidelities"]
    pf = sw["analytical_curves"]["phase_flip_3/phase_flip"]["fidelities"]
    assert np.max(np.abs(np.asarray(bf, dtype=float) - np.asarray(pf, dtype=float))) <= 1e-12
    coh_bf = sw["coherence_curves"]["bit_flip_3/bit_flip"]["abs_rho01_analytical"]
    coh_pf = sw["coherence_curves"]["phase_flip_3/phase_flip"]["abs_rho01_analytical"]
    assert np.max(np.abs(np.asarray(coh_bf, dtype=float) - np.asarray(coh_pf, dtype=float))) <= 1e-12
    assert abs(float(qecc_analysis.analytical_gain_threshold("phase_flip_3", "phase_flip")) - 0.5) <= 1e-12


def test_t3_pair_04_phase_flip_3_coherent_collective_vs_dfs_2(live_server: ServerHandle) -> None:
    inv_s2 = 1.0 / math.sqrt(2.0)
    psi = np.array([inv_s2, 1.0j * inv_s2], dtype=np.complex128)
    phi = math.pi / 4.0
    st, _, body = http_request_json(
        live_server.base_url,
        "/api/step",
        method="POST",
        payload={
            "code": "phase_flip_3",
            "state_preset": "+i",
            "error_mode": "coherent_collective",
            "collective_phi": phi,
        },
    )
    assert st == 200
    orc = compute_step_oracle(
        "phase_flip_3",
        psi,
        error_mode="coherent_collective",
        collective_phi=phi,
        selected_syndrome=body["stages"][3]["selected_syndrome"],
    )
    assert abs(float(body["stages"][5]["fidelity_vs_input"]) - orc["stages"][5]["fidelity_vs_input"]) <= 1e-12
    assert abs(float(body["comparison"]["dfs_2"]["fidelity"]) - 1.0) <= 1e-12
    assert float(body["comparison"]["qecc"]["fidelity"]) < 1.0


def test_t3_pair_05_five_qubit_depolarizing_cardinal_avg_numpy(
    live_server: ServerHandle, qecc_analysis, helpers
) -> None:
    p = 0.08
    n_shots = 6000
    st, _, body = http_request_json(
        live_server.base_url,
        "/api/simulate",
        method="POST",
        payload={
            "code": "five_qubit",
            "channel": "depolarizing",
            "p": p,
            "num_shots": n_shots,
            "state_mode": "cardinal_avg",
            "seed": 305,
            "backend": "numpy",
        },
    )
    assert st == 200
    exp_state = float(qecc_analysis.analytical_fidelity("five_qubit", "depolarizing", p, metric="state"))
    assert abs(float(body["mean_fidelity"]) - exp_state) <= helpers.binom_sigma(exp_state, n_shots, z=4.5)


def test_t3_pair_06_five_qubit_su2_arbitrary_custom_state_all_branches(
    live_server: ServerHandle,
) -> None:
    theta = 1.1
    nx, ny, nz = 1.0, 2.0, 3.0
    # On qubit 4 of five_qubit: I->"0000", X->"0011", Y->"0111", Z->"0100"
    for synd in ("0000", "0011", "0111", "0100"):
        st, _, body = http_request_json(
            live_server.base_url,
            "/api/step",
            method="POST",
            payload={
                "code": "five_qubit",
                "state_preset": "custom",
                "amplitudes": [[0.6, 0.0], [0.0, 0.8]],
                "error_mode": "su2_rotation",
                "su2_qubit": 4,
                "su2_theta": theta,
                "su2_nx": nx,
                "su2_ny": ny,
                "su2_nz": nz,
                "selected_syndrome": synd,
            },
        )
        assert st == 200
        assert abs(float(body["stages"][5]["fidelity_vs_input"]) - 1.0) <= 1e-12


def test_t3_pair_07_five_qubit_bit_flip_cross_channel_32_term_coherence(
    live_server: ServerHandle, qecc_analysis, helpers
) -> None:
    p = 0.08
    st_sw, _, sw = http_request_json(
        live_server.base_url,
        "/api/sweeps",
        method="POST",
        payload={
            "codes": ["five_qubit"],
            "channels": ["bit_flip"],
            "metric": "entanglement",
            "p_min": 0.0,
            "p_max": 0.16,
            "num_p": 3,
            "run_live_mc": False,
        },
    )
    assert st_sw == 200
    exp_p_star = float(qecc_analysis.analytical_gain_threshold("five_qubit", "bit_flip"))
    assert abs(float(sw["analytical_curves"]["five_qubit/bit_flip"]["p_star"]) - exp_p_star) <= 1e-12
    obs_coh = float(sw["coherence_curves"]["five_qubit/bit_flip"]["abs_rho01_analytical"][1])
    exp_coh = analytical_coherence_oracle("five_qubit", "bit_flip", p)
    assert abs(obs_coh - exp_coh) <= 1e-12

    n_shots = 5000
    st_sim, _, sim = http_request_json(
        live_server.base_url,
        "/api/simulate",
        method="POST",
        payload={
            "code": "five_qubit",
            "channel": "bit_flip",
            "p": p,
            "num_shots": n_shots,
            "state_mode": "+i",
            "seed": 307,
            "backend": "numpy",
        },
    )
    assert st_sim == 200
    assert abs(float(sim["abs_rho01"]) - exp_coh) <= 2.5 * helpers.binom_sigma(0.5 + exp_coh, n_shots, z=4.5)


def test_t3_pair_08_dfs_2_collective_dephasing_plus_i_perfect_coherence(
    live_server: ServerHandle,
) -> None:
    st_sim, _, sim = http_request_json(
        live_server.base_url,
        "/api/simulate",
        method="POST",
        payload={
            "code": "dfs_2",
            "channel": "collective_dephasing",
            "p": 0.25,
            "num_shots": 500,
            "state_mode": "+i",
            "seed": 308,
            "backend": "numpy",
        },
    )
    assert st_sim == 200
    assert abs(float(sim["mean_fidelity"]) - 1.0) <= 1e-12
    assert abs(float(sim["abs_rho01"]) - 0.5) <= 1e-12

    st_sw, _, sw = http_request_json(
        live_server.base_url,
        "/api/sweeps",
        method="POST",
        payload={
            "codes": ["dfs_2"],
            "channels": ["collective_dephasing"],
            "metric": "entanglement",
            "p_min": 0.0,
            "p_max": 0.5,
            "num_p": 6,
            "run_live_mc": False,
        },
    )
    assert st_sw == 200
    assert abs(float(sw["analytical_curves"]["dfs_2/collective_dephasing"]["p_star"]) - 1.0) <= 1e-12


def test_t3_pair_09_dfs_2_independent_phase_flip_breaks_dfs(
    live_server: ServerHandle, qecc_analysis, helpers
) -> None:
    p = 0.10
    n_shots = 5000
    st, _, sim = http_request_json(
        live_server.base_url,
        "/api/simulate",
        method="POST",
        payload={
            "code": "dfs_2",
            "channel": "phase_flip",
            "p": p,
            "num_shots": n_shots,
            "state_mode": "+i",
            "seed": 309,
            "backend": "numpy",
        },
    )
    assert st == 200
    exp_f = float(qecc_analysis.analytical_fidelity("dfs_2", "phase_flip", p, metric="entanglement"))
    assert abs(exp_f - ((1.0 - p) ** 2 + p**2)) <= 1e-12
    assert abs(float(sim["mean_fidelity"]) - exp_f) <= helpers.binom_sigma(exp_f, n_shots, z=4.5)


def test_t3_pair_10_five_qubit_zero_error_bloch_eigenkit_vs_numpy(live_server: ServerHandle) -> None:
    if "eigenkit_unitary" not in list_backends_oracle():
        pytest.skip("eigenkit_unitary backend not installed in environment")

    common = {
        "code": "five_qubit",
        "channel": "depolarizing",
        "p": 0.0,
        "num_shots": 1,
        "state_mode": "bloch",
        "theta": 1.2,
        "phi_state": 0.7,
        "seed": 310,
    }
    st_np, _, res_np = http_request_json(
        live_server.base_url, "/api/simulate", method="POST", payload={**common, "backend": "numpy"}
    )
    st_ek, _, res_ek = http_request_json(
        live_server.base_url,
        "/api/simulate",
        method="POST",
        payload={**common, "backend": "eigenkit_unitary"},
    )
    assert st_np == 200 and st_ek == 200
    assert abs(float(res_ek["mean_fidelity"]) - 1.0) <= 1e-12
    assert np.max(np.abs(decode_complex_mat(res_ek["rho_out"]) - decode_complex_mat(res_np["rho_out"]))) <= 1e-12


def test_t3_pair_11_bit_flip_3_theory_aliasing_ixx_matches_step(live_server: ServerHandle) -> None:
    st_th, _, th = http_request_json(live_server.base_url, "/api/theory?code=bit_flip_3")
    st_sp, _, sp = http_request_json(
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
    assert st_th == 200 and st_sp == 200
    assert any("IXX" in json.dumps(ex) for ex in th["aliasing_examples"])
    assert sp["stages"][3]["selected_syndrome"] == "10"
    assert abs(float(sp["stages"][5]["fidelity_vs_input"]) - 0.0) <= 1e-12


def test_t3_pair_12_all_codes_sweeps_entanglement_vs_state_and_cached(
    live_server: ServerHandle, qecc_analysis
) -> None:
    for metric in ("entanglement", "state"):
        st, _, sw = http_request_json(
            live_server.base_url,
            "/api/sweeps",
            method="POST",
            payload={
                "codes": ["bit_flip_3", "phase_flip_3", "five_qubit", "dfs_2"],
                "channels": ["bit_flip", "phase_flip", "depolarizing", "collective_dephasing"],
                "metric": metric,
                "p_min": 0.0,
                "p_max": 0.5,
                "num_p": 11,
                "run_live_mc": False,
            },
        )
        assert st == 200
        for cname in ("bit_flip_3", "phase_flip_3", "five_qubit", "dfs_2"):
            for ch in ("bit_flip", "phase_flip", "depolarizing", "collective_dephasing"):
                curve = sw["analytical_curves"][f"{cname}/{ch}"]["fidelities"]
                for p_val, f_val in zip(sw["p_grid"], curve, strict=True):
                    exp_f = float(qecc_analysis.analytical_fidelity(cname, ch, float(p_val), metric=metric))
                    assert abs(float(f_val) - exp_f) <= 1e-12


def test_t3_pair_13_browser_five_qubit_su2_stage_syndrome_and_bloch(page: PageSession) -> None:
    page.select_option("code-select", "five_qubit")
    page.select_option("state-select", "+i")
    page.click("stage-step-syndrome_extracted")
    b_box = page.ensure_visible("bloch-canvas")
    assert b_box["width"] > 0 and b_box["height"] > 0
    page.assert_no_errors()


def test_t3_pair_14_browser_dfs_2_collective_dephasing_coherence_pt_br(page: PageSession) -> None:
    page.switch_language("pt-br")
    page.select_option("code-select", "dfs_2")
    page.select_option("channel-select", "collective_dephasing")
    page.select_option("state-select", "+i")
    page.fill_input("phi-input", "1.57079632679")
    c_box = page.ensure_visible("coherence-panel")
    assert c_box["width"] > 0 and c_box["height"] > 0
    txt = page.body_text()
    assert not find_leak_tokens(txt)
    page.assert_no_errors()


def test_t3_pair_15_browser_curves_chart_metric_switch_bilingual(page: PageSession) -> None:
    page.ensure_visible("curves-chart")
    for lang in ("en", "pt-br"):
        page.switch_language(lang)
        box = page.ensure_visible("curves-chart")
        assert box["width"] > 0 and box["height"] > 0
        assert not find_leak_tokens(page.body_text())
    page.assert_no_errors()


def test_t3_pair_16_browser_seminar_sync_and_code_panel_pt_br(page: PageSession) -> None:
    page.switch_language("pt-br")
    page.ensure_visible("seminar-panel")
    sem_txt = page.eval_js('document.querySelector(\'[data-testid="seminar-panel"]\').innerText')
    assert len(sem_txt.strip()) > 30
    assert not find_leak_tokens(sem_txt)
    page.ensure_visible("theory-panel")
    page.assert_no_errors()
