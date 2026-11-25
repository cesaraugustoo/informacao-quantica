"""Tier 4 opaque-box realistic end-to-end graduate seminar, self-study & concurrency workflows.

All ``qecc.*`` imports are deferred inside functions/fixtures so ``pytest --collect-only tests/e2e``
succeeds cleanly with zero top-level ``qecc`` dependencies.
"""

from __future__ import annotations

import concurrent.futures
import json
import math
from typing import Any, Iterator

import numpy as np
import pytest

from .web_harness import (
    REQUIRED_TESTIDS,
    STAGE_KEYS,
    BrowserSession,
    ServerHandle,
    analytical_coherence_oracle,
    collect_seminar_panel_texts,
    compute_step_oracle,
    decode_complex_mat,
    find_leak_tokens,
    http_request_json,
)

pytestmark = pytest.mark.tier4


@pytest.fixture(scope="module")
def live_server() -> Iterator[ServerHandle]:
    with ServerHandle(use_cli=False, host="127.0.0.1", port=0) as srv:
        yield srv


@pytest.fixture(scope="module")
def browser_session() -> Iterator[BrowserSession]:
    with BrowserSession() as browser:
        yield browser


def test_scenario_1_live_graduate_seminar_presentation_en_and_pt_br(
    live_server: ServerHandle, browser_session: BrowserSession
) -> None:
    """Walk through live seminar presentation in EN and PT-BR, stepping topics/slides,
    inspecting five_qubit SU(2) discretization and p* threshold curves with 0 browser errors.
    """
    pg = browser_session.new_page()
    try:
        pg.set_viewport(1440, 900, mobile=False)
        pg.goto(live_server.base_url)
        pg.wait_for_testid("code-select")

        # 1. Start in EN, walk seminar-panel topics/slides
        pg.switch_language("en")
        sem_box = pg.ensure_visible("seminar-panel")
        assert sem_box["width"] > 0 and sem_box["height"] > 0

        en_texts = collect_seminar_panel_texts(pg, max_steps=25)

        joined_en = "\n".join(en_texts).lower()
        assert "no-cloning" in joined_en or "cloning" in joined_en
        assert "knill" in joined_en and "laflamme" in joined_en

        # 2. Switch to PT-BR mid-presentation and verify terminology & layout
        pg.switch_language("pt-br")
        pt_text = pg.body_text()
        assert not find_leak_tokens(pt_text)
        assert pg.check_no_horizontal_overflow()

        # 3. Live five_qubit demonstration & threshold curve view
        pg.ensure_visible("code-select")
        pg.select_option("code-select", "five_qubit")
        pg.select_option("channel-select", "depolarizing")
        pg.select_option("state-select", "+i")
        pg.fill_input("p-input", "0.05")
        curves_box = pg.ensure_visible("curves-chart")
        assert curves_box["width"] > 0 and curves_box["height"] > 0

        pg.assert_no_errors()
    finally:
        pg.close()


def test_scenario_2_bilingual_graduate_self_study_all_codes_and_kl_theory(
    live_server: ServerHandle, browser_session: BrowserSession, qecc_analysis
) -> None:
    """Graduate self-study across bit_flip_3 -> phase_flip_3 -> five_qubit -> dfs_2,
    verifying KL C_ab matrices, custom Bloch states, and mid-session en <-> pt-br switching.
    """
    pg = browser_session.new_page()
    try:
        pg.set_viewport(1280, 800, mobile=False)
        pg.goto(live_server.base_url)
        pg.wait_for_testid("code-select")

        for cname in ("bit_flip_3", "phase_flip_3", "five_qubit", "dfs_2"):
            pg.select_option("code-select", cname)
            st_th, _, th = http_request_json(live_server.base_url, f"/api/theory?code={cname}")
            assert st_th == 200
            kl_ref = qecc_analysis.verify_knill_laflamme(cname)
            assert th["knill_laflamme"]["satisfies"] is True
            assert np.max(np.abs(decode_complex_mat(th["knill_laflamme"]["c_matrix"]) - kl_ref.c_matrix)) <= 1e-12

            th_box = pg.ensure_visible("theory-panel")
            assert th_box["width"] > 0 and th_box["height"] > 0

        # Switch to pt-br mid-session and verify selections + UI invariants preserved
        pg.switch_language("pt-br")
        assert pg.eval_js('document.querySelector(\'[data-testid="code-select"]\').value') == "dfs_2"
        for tid in REQUIRED_TESTIDS:
            assert pg.has_element(tid)
        assert not find_leak_tokens(pg.body_text())
        assert pg.check_no_horizontal_overflow()
        pg.assert_no_errors()
    finally:
        pg.close()


def test_scenario_3_continuous_su2_discretization_branch_collapse_browser(
    live_server: ServerHandle, browser_session: BrowserSession
) -> None:
    """Verify continuous SU(2) discretization across all 4 syndrome branches on five_qubit
    and interactive 6-stage walkthrough in headless Chromium.
    """
    theta = 1.05
    nx, ny, nz = 1.0, -1.0, 2.0
    inv_s2 = 1.0 / math.sqrt(2.0)
    psi = np.array([inv_s2, 1.0j * inv_s2], dtype=np.complex128)

    # API branch-collapse audit on five_qubit qubit 0:
    # I -> 0000, X -> 0001, Y -> 1011, Z -> 1010
    total_prob = 0.0
    for synd in ("0000", "0001", "1011", "1010"):
        st, _, body = http_request_json(
            live_server.base_url,
            "/api/step",
            method="POST",
            payload={
                "code": "five_qubit",
                "state_preset": "+i",
                "error_mode": "su2_rotation",
                "su2_qubit": 0,
                "su2_theta": theta,
                "su2_nx": nx,
                "su2_ny": ny,
                "su2_nz": nz,
                "selected_syndrome": synd,
            },
        )
        assert st == 200
        orc = compute_step_oracle(
            "five_qubit",
            psi,
            error_mode="su2_rotation",
            su2_qubit=0,
            su2_theta=theta,
            su2_nx=nx,
            su2_ny=ny,
            su2_nz=nz,
            selected_syndrome=synd,
        )
        total_prob += float(body["stages"][3]["syndrome_probabilities"][synd])
        assert abs(float(body["stages"][5]["fidelity_vs_input"]) - 1.0) <= 1e-12
        assert np.max(np.abs(decode_complex_mat(body["stages"][5]["logical_rho"]) - orc["stages"][5]["logical_rho"])) <= 1e-12
    assert abs(total_prob - 1.0) <= 1e-12

    # Browser 6-stage stepping
    pg = browser_session.new_page()
    try:
        pg.set_viewport(1440, 900, mobile=False)
        pg.goto(live_server.base_url)
        pg.wait_for_testid("code-select")
        pg.select_option("code-select", "five_qubit")
        pg.select_option("state-select", "+i")
        for key in STAGE_KEYS:
            pg.click(f"stage-step-{key}")
        assert pg.ensure_visible("bloch-canvas")["width"] > 0
        pg.assert_no_errors()
    finally:
        pg.close()


def test_scenario_4_active_qecc_vs_passive_dfs_coherence_and_symmetry_breaking(
    live_server: ServerHandle, browser_session: BrowserSession
) -> None:
    """Compare active phase_flip_3 vs passive dfs_2 under coherent Rz(phi) and stochastic
    collective_dephasing vs independent phase_flip symmetry breaking.
    """
    for phi in (0.0, math.pi / 4.0, math.pi / 2.0, math.pi):
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
        assert abs(float(body["comparison"]["dfs_2"]["fidelity"]) - 1.0) <= 1e-12
        assert abs(float(body["comparison"]["dfs_2"]["abs_rho01"]) - 0.5) <= 1e-12
        assert abs(float(body["comparison"]["raw"]["fidelity"]) - math.cos(phi / 2.0) ** 2) <= 1e-12

    # Symmetry breaking: independent phase_flip degrades dfs_2 while collective_dephasing keeps 1.0
    st_sw, _, sw = http_request_json(
        live_server.base_url,
        "/api/sweeps",
        method="POST",
        payload={
            "codes": ["dfs_2", "phase_flip_3"],
            "channels": ["collective_dephasing", "phase_flip"],
            "metric": "entanglement",
            "p_min": 0.0,
            "p_max": 0.5,
            "num_p": 6,
            "run_live_mc": False,
        },
    )
    assert st_sw == 200
    assert abs(float(sw["analytical_curves"]["dfs_2/collective_dephasing"]["fidelities"][-1]) - 1.0) <= 1e-12
    assert float(sw["analytical_curves"]["dfs_2/phase_flip"]["fidelities"][2]) < 1.0
    assert abs(float(sw["coherence_curves"]["dfs_2/collective_dephasing"]["abs_rho01_analytical"][-1]) - 0.5) <= 1e-12

    pg = browser_session.new_page()
    try:
        pg.goto(live_server.base_url)
        pg.wait_for_testid("code-select")
        pg.select_option("code-select", "dfs_2")
        pg.select_option("channel-select", "collective_dephasing")
        pg.select_option("state-select", "+i")
        pg.fill_input("phi-input", str(math.pi / 2.0))
        assert pg.ensure_visible("coherence-panel")["width"] > 0
        pg.assert_no_errors()
    finally:
        pg.close()


def test_scenario_5_full_sweep_dataset_parity_audit_and_metric_separation(
    live_server: ServerHandle, results_dir, qecc_analysis
) -> None:
    """Audit all 11 sweep curves and coherence_at_p005 in results/data/fidelity_sweeps.json
    against /api/sweeps and verify strict entanglement vs Haar state fidelity separation (<= 1e-12).
    """
    with (results_dir / "data" / "fidelity_sweeps.json").open("r", encoding="utf-8") as fh:
        ref = json.load(fh)

    st_e, _, sw_e = http_request_json(
        live_server.base_url,
        "/api/sweeps",
        method="POST",
        payload={
            "codes": ["bit_flip_3", "phase_flip_3", "five_qubit", "dfs_2"],
            "channels": ["bit_flip", "phase_flip", "depolarizing", "collective_dephasing"],
            "metric": "entanglement",
            "p_min": 0.0,
            "p_max": 0.5,
            "num_p": 21,
            "run_live_mc": False,
        },
    )
    st_s, _, sw_s = http_request_json(
        live_server.base_url,
        "/api/sweeps",
        method="POST",
        payload={
            "codes": ["bit_flip_3", "phase_flip_3", "five_qubit", "dfs_2"],
            "channels": ["bit_flip", "phase_flip", "depolarizing", "collective_dephasing"],
            "metric": "state",
            "p_min": 0.0,
            "p_max": 0.5,
            "num_p": 21,
            "run_live_mc": False,
        },
    )
    assert st_e == 200 and st_s == 200

    # Verify coherence_at_p005 (index 2 on 21-point [0, 0.5] grid => p = 0.05)
    assert abs(float(sw_e["p_grid"][2]) - 0.05) <= 1e-12
    for coh_ref in ref["coherence_at_p005"]:
        key = f"{coh_ref['code']}/{coh_ref['channel']}"
        api_coh = sw_e["coherence_curves"][key]
        assert abs(float(api_coh["abs_rho01_analytical"][2]) - float(coh_ref["abs_rho01_analytical"])) <= 1e-12
        assert abs(float(api_coh["abs_rho01_raw"][2]) - float(coh_ref["abs_rho01_raw"])) <= 1e-12
        assert abs(float(api_coh["abs_rho01_analytical"][2]) - analytical_coherence_oracle(coh_ref["code"], coh_ref["channel"], 0.05)) <= 1e-12

    # Verify all records' analytical_entanglement_fidelity & analytical_state_fidelity (<= 1e-12)
    p_index = {round(float(p), 6): idx for idx, p in enumerate(sw_e["p_grid"])}
    for rec in ref["records"]:
        idx = p_index[round(float(rec["p"]), 6)]
        cname, ch = rec["code"], rec["channel"]
        if cname == "raw":
            assert abs(float(sw_e["raw_curve"][idx]) - float(rec["analytical_entanglement_fidelity"])) <= 1e-12
            assert abs(float(sw_s["raw_curve"][idx]) - float(rec["analytical_state_fidelity"])) <= 1e-12
        else:
            key = f"{cname}/{ch}"
            f_e = float(sw_e["analytical_curves"][key]["fidelities"][idx])
            f_s = float(sw_s["analytical_curves"][key]["fidelities"][idx])
            assert abs(f_e - float(rec["analytical_entanglement_fidelity"])) <= 1e-12
            assert abs(f_s - float(rec["analytical_state_fidelity"])) <= 1e-12
            assert abs(f_s - (1.0 - (2.0 / 3.0) * (1.0 - f_e))) <= 1e-12
            if float(rec["p"]) > 0.0 and f_e < 1.0 - 1e-12:
                assert f_s > f_e


def test_scenario_6_concurrent_multi_client_and_responsive_desktop_mobile_audit(
    live_server: ServerHandle, browser_session: BrowserSession
) -> None:
    """Exercise concurrent ThreadingHTTPServer API requests across all endpoints plus
    desktop (1440x900) and mobile (390x844) responsive layout audit in both EN and PT-BR.
    """
    def _worker(idx: int) -> tuple[str, int, Any]:
        mode = idx % 4
        if mode == 0:
            st, _, body = http_request_json(
                live_server.base_url,
                "/api/step",
                method="POST",
                payload={
                    "code": "five_qubit",
                    "state_preset": "+i",
                    "error_mode": "pauli_string",
                    "pauli_string": "IYIII",
                },
            )
            return ("step", st, body["stages"][5]["fidelity_vs_input"])
        if mode == 1:
            st, _, body = http_request_json(
                live_server.base_url,
                "/api/simulate",
                method="POST",
                payload={
                    "code": "dfs_2",
                    "channel": "collective_dephasing",
                    "p": 0.2,
                    "num_shots": 100,
                    "state_mode": "+i",
                    "seed": 400 + idx,
                    "backend": "numpy",
                },
            )
            return ("simulate", st, body["mean_fidelity"])
        if mode == 2:
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
                    "num_p": 5,
                    "run_live_mc": False,
                },
            )
            return ("sweeps", st, body["gain_thresholds"])
        st, _, body = http_request_json(live_server.base_url, "/api/theory?code=five_qubit")
        return ("theory", st, body["knill_laflamme"]["satisfies"])

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(_worker, range(16)))

    for kind, st, val in results:
        assert st == 200, f"Concurrent {kind} failed with status={st}"
        if kind in ("step", "simulate"):
            assert abs(float(val) - 1.0) <= 1e-12
        elif kind == "theory":
            assert val is True

    # Responsive desktop (1440x900) and mobile (390x844) audit in both languages
    pg = browser_session.new_page()
    try:
        pg.goto(live_server.base_url)
        pg.wait_for_testid("code-select")
        for lang in ("en", "pt-br"):
            pg.switch_language(lang)
            for w, h, mobile in ((1440, 900, False), (390, 844, True)):
                pg.set_viewport(w, h, mobile=mobile)
                assert pg.check_no_horizontal_overflow(), (
                    f"Overflow at {w}x{h} ({lang}): {pg.viewport_dimensions()}"
                )
            assert not find_leak_tokens(pg.body_text())
        pg.assert_no_errors()
    finally:
        pg.close()
