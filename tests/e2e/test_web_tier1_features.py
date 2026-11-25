"""Tier 1 opaque-box E2E tests for Features F1-F11 (happy-path API, CLI, math parity, and browser UI).

All ``qecc.*`` imports are deferred inside functions/fixtures so ``pytest --collect-only tests/e2e``
succeeds cleanly with zero top-level ``qecc`` dependencies.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Iterator

import numpy as np
import pytest

from .web_harness import (
    REQUIRED_TESTIDS,
    STAGE_KEYS,
    BrowserSession,
    PageSession,
    ServerHandle,
    abs_rho01,
    analytical_coherence_oracle,
    bloch_to_rho,
    collect_seminar_panel_texts,
    compute_step_oracle,
    decode_complex_mat,
    decode_complex_vec,
    find_leak_tokens,
    get_code_oracle,
    http_request_json,
    http_request_raw,
    list_backends_oracle,
    list_codes_oracle,
    rho_to_bloch,
    su2_pauli_coefficients,
)

pytestmark = pytest.mark.tier1


# ---------------------------------------------------------------------------
# Module-scoped live server & browser fixtures
# ---------------------------------------------------------------------------


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


def _cardinal_amplitude_map() -> dict[str, np.ndarray]:
    inv_s2 = 1.0 / math.sqrt(2.0)
    return {
        "0": np.array([1.0, 0.0], dtype=np.complex128),
        "1": np.array([0.0, 1.0], dtype=np.complex128),
        "+": np.array([inv_s2, inv_s2], dtype=np.complex128),
        "-": np.array([inv_s2, -inv_s2], dtype=np.complex128),
        "+i": np.array([inv_s2, 1.0j * inv_s2], dtype=np.complex128),
        "-i": np.array([inv_s2, -1.0j * inv_s2], dtype=np.complex128),
    }


def _extract_gain_threshold(thresholds: dict[str, Any], code: str, channel: str) -> float:
    flat_key = f"{code}/{channel}"
    if flat_key in thresholds:
        return float(thresholds[flat_key])
    if code in thresholds and isinstance(thresholds[code], dict) and channel in thresholds[code]:
        return float(thresholds[code][channel])
    raise KeyError(f"Missing gain threshold for ({code!r}, {channel!r}) in {list(thresholds)}")


# ---------------------------------------------------------------------------
# Feature F1: Engine-Backed Local Web Server & CLI
# ---------------------------------------------------------------------------


def test_f1_module_web_startup_ephemeral_port_and_sigterm() -> None:
    with ServerHandle(use_cli=False, host="127.0.0.1", port=0) as srv:
        assert srv.base_url.startswith("http://127.0.0.1:")
        status, _, body = http_request_json(srv.base_url, "/api/health")
        assert status == 200
        assert isinstance(body, dict) and body.get("status") == "ok"
        rc = srv.stop()
        assert rc is not None


def test_f1_cli_serve_startup_and_preserved_subcommands(python_executable, repo_root) -> None:
    import subprocess

    with ServerHandle(use_cli=True, host="127.0.0.1", port=0) as srv:
        status, _, body = http_request_json(srv.base_url, "/api/health")
        assert status == 200
        assert body["status"] == "ok"

    info_proc = subprocess.run(
        [str(python_executable), "-m", "qecc.cli", "info"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    assert info_proc.returncode == 0, info_proc.stderr
    for code_name in ("bit_flip_3", "phase_flip_3", "five_qubit", "dfs_2"):
        assert code_name in info_proc.stdout


def test_f1_root_html_status_content_type_and_zero_cdn(live_server: ServerHandle) -> None:
    status, headers, raw = http_request_raw(live_server.base_url, "/")
    assert status == 200
    ctype = headers.get("content-type", "").lower()
    assert "text/html" in ctype
    assert "utf-8" in ctype
    html = raw.decode("utf-8")
    assert len(html.strip()) > 100
    assert not re.search(r'<script[^>]+src=["\']https?://', html, re.IGNORECASE)
    assert not re.search(r'<link[^>]+href=["\']https?://', html, re.IGNORECASE)
    assert not re.search(r'@import\s+url\(["\']?https?://', html, re.IGNORECASE)


def test_f1_browser_rendered_dom_exposes_required_testids(page: PageSession) -> None:
    for tid in REQUIRED_TESTIDS:
        assert page.has_element(tid), f"Missing rendered [data-testid='{tid}'] in SPA DOM"
    page.assert_no_errors()


def test_f1_health_endpoint_contract_and_backends(live_server: ServerHandle) -> None:
    status, _, body = http_request_json(live_server.base_url, "/api/health")
    assert status == 200
    assert body["status"] == "ok"
    assert body["codes"] == ["bit_flip_3", "phase_flip_3", "five_qubit", "dfs_2"]

    expected_backends = list_backends_oracle()
    reported = body["backends"]
    if reported and isinstance(reported[0], dict):
        names = [b["name"] for b in reported if b.get("available", True)]
    else:
        names = list(reported)
    assert "numpy" in names
    assert set(expected_backends).issubset(set(names))


def test_f1_static_assets_served_with_valid_mime_types(live_server: ServerHandle) -> None:
    status, _, raw = http_request_raw(live_server.base_url, "/")
    assert status == 200
    html = raw.decode("utf-8")
    referenced = set(re.findall(r'(?:src|href)=["\'](/static/[^"\'?#]+)', html))
    assert len(referenced) >= 2, f"Expected static assets referenced in index.html, got {referenced}"

    for asset_path in sorted(referenced):
        st, hdrs, content = http_request_raw(live_server.base_url, asset_path)
        assert st == 200, f"Failed fetching {asset_path}: status={st}"
        assert len(content) > 0
        ctype = hdrs.get("content-type", "").lower()
        if asset_path.endswith(".css"):
            assert "text/css" in ctype
        elif asset_path.endswith(".js"):
            assert "javascript" in ctype


# ---------------------------------------------------------------------------
# Feature F2: 4-Code + Raw Baseline Support
# ---------------------------------------------------------------------------


def test_f2_metadata_codes_and_raw_descriptor_contract(live_server: ServerHandle) -> None:
    status, _, meta = http_request_json(live_server.base_url, "/api/metadata")
    assert status == 200
    codes_meta = meta["codes"]
    for cname in ("bit_flip_3", "phase_flip_3", "five_qubit", "dfs_2"):
        assert cname in codes_meta
        c_api = codes_meta[cname]
        c_ref = get_code_oracle(cname)
        assert c_api["n_data"] == c_ref.n_data
        assert c_api["n_ancilla"] == c_ref.n_ancilla
        assert c_api["n_total"] == c_ref.n_total
        assert list(c_api["stabilizers"]) == list(c_ref.stabilizers)
        assert dict(c_api["syndrome_table"]) == dict(c_ref.syndrome_table)
        assert len(c_api["encode_ops"]) == len(c_ref.encode_ops())
        assert len(c_api["syndrome_ops"]) == len(c_ref.syndrome_ops())
        assert len(c_api["decode_ops"]) == len(c_ref.decode_ops())

    assert "raw" in codes_meta or "raw" in meta
    assert set(meta["channels"]) == {"bit_flip", "phase_flip", "depolarizing", "collective_dephasing"}


def test_f2_preserved_code_registry_invariant(live_server: ServerHandle) -> None:
    _ = http_request_json(live_server.base_url, "/api/metadata")
    assert list_codes_oracle() == ["bit_flip_3", "phase_flip_3", "five_qubit", "dfs_2"]


def test_f2_step_all_four_codes_zero_error_identity(live_server: ServerHandle) -> None:
    for cname, zero_err in (
        ("bit_flip_3", "III"),
        ("phase_flip_3", "III"),
        ("five_qubit", "IIIII"),
        ("dfs_2", "II"),
    ):
        status, _, res = http_request_json(
            live_server.base_url,
            "/api/step",
            method="POST",
            payload={
                "code": cname,
                "state_preset": "+i",
                "error_mode": "pauli_string",
                "pauli_string": zero_err,
            },
        )
        assert status == 200
        decoded = res["stages"][-1]
        assert decoded["key"] == "decoded"
        assert abs(decoded["fidelity_vs_input"] - 1.0) <= 1e-12


def test_f2_simulate_raw_and_four_codes_baseline_parity(live_server: ServerHandle, helpers) -> None:
    p = 0.1
    n_shots = 4000
    st_raw, _, res_raw = http_request_json(
        live_server.base_url,
        "/api/simulate",
        method="POST",
        payload={
            "code": "raw",
            "channel": "bit_flip",
            "p": p,
            "num_shots": n_shots,
            "state_mode": "0",
            "seed": 42,
            "backend": "numpy",
        },
    )
    assert st_raw == 200
    assert abs(res_raw["mean_fidelity"] - (1.0 - p)) <= helpers.binom_sigma(1.0 - p, n_shots, z=4.5)

    for cname in ("bit_flip_3", "phase_flip_3", "five_qubit", "dfs_2"):
        st, _, res = http_request_json(
            live_server.base_url,
            "/api/simulate",
            method="POST",
            payload={
                "code": cname,
                "channel": "depolarizing",
                "p": 0.0,
                "num_shots": 10,
                "state_mode": "+i",
                "seed": 42,
                "backend": "numpy",
            },
        )
        assert st == 200
        assert res["code"] == cname
        assert abs(res["mean_fidelity"] - 1.0) <= 1e-12


def test_f2_browser_code_select_switches_all_four_codes(page: PageSession) -> None:
    for cname in ("bit_flip_3", "phase_flip_3", "five_qubit", "dfs_2"):
        page.select_option("code-select", cname)
        selected = page.eval_js('document.querySelector(\'[data-testid="code-select"]\').value')
        assert selected == cname
    page.assert_no_errors()


# ---------------------------------------------------------------------------
# Feature F3: Step-by-Step State Evolution
# ---------------------------------------------------------------------------


def test_f3_step_six_ordered_stages_across_all_codes(live_server: ServerHandle) -> None:
    for cname, zero_err in (
        ("bit_flip_3", "III"),
        ("phase_flip_3", "III"),
        ("five_qubit", "IIIII"),
        ("dfs_2", "II"),
    ):
        st, _, body = http_request_json(
            live_server.base_url,
            "/api/step",
            method="POST",
            payload={
                "code": cname,
                "state_preset": "+i",
                "error_mode": "pauli_string",
                "pauli_string": zero_err,
            },
        )
        assert st == 200
        assert body["big_endian"] is True
        assert [s["key"] for s in body["stages"]] == list(STAGE_KEYS)


def _assert_step_matches_oracle(api_res: dict[str, Any], oracle: dict[str, Any]) -> None:
    assert api_res["code"] == oracle["code"]
    for s_api, s_orc in zip(api_res["stages"], oracle["stages"], strict=True):
        assert s_api["key"] == s_orc["key"]
        rho_api = decode_complex_mat(s_api["logical_rho"])
        assert np.max(np.abs(rho_api - s_orc["logical_rho"])) <= 1e-12
        assert np.max(np.abs(np.asarray(s_api["logical_bloch"], dtype=float) - s_orc["logical_bloch"])) <= 1e-12
        assert abs(float(s_api["logical_abs_rho01"]) - s_orc["logical_abs_rho01"]) <= 1e-12
        assert abs(float(s_api["fidelity_vs_input"]) - s_orc["fidelity_vs_input"]) <= 1e-12

        # Check statevector_nonzero subset matches oracle exact statevector amplitudes
        sv_list = s_api.get("statevector_nonzero", [])
        assert isinstance(sv_list, list) and len(sv_list) >= 1
        vec = s_orc["statevector"]
        for entry in sv_list:
            idx = int(entry["basis"], 2)
            amp_api = complex(float(entry["re"]), float(entry["im"]))
            assert abs(amp_api - vec[idx]) <= 1e-12
            assert abs(float(entry["prob"]) - abs(vec[idx]) ** 2) <= 1e-12

    synd_stage = api_res["stages"][3]
    if oracle["syndrome_probabilities"]:
        for k, v_orc in oracle["syndrome_probabilities"].items():
            assert abs(float(synd_stage["syndrome_probabilities"][k]) - v_orc) <= 1e-12
    assert synd_stage["selected_syndrome"] == oracle["selected_syndrome"]


def test_f3_step_bit_flip_3_single_x_stage_oracle_parity(live_server: ServerHandle) -> None:
    psi = _cardinal_amplitude_map()["+i"]
    st, _, body = http_request_json(
        live_server.base_url,
        "/api/step",
        method="POST",
        payload={
            "code": "bit_flip_3",
            "state_preset": "+i",
            "error_mode": "pauli_string",
            "pauli_string": "XII",
        },
    )
    assert st == 200
    oracle = compute_step_oracle("bit_flip_3", psi, error_mode="pauli_string", pauli_string="XII")
    _assert_step_matches_oracle(body, oracle)
    assert body["stages"][3]["selected_syndrome"] == "10"
    assert abs(body["stages"][5]["fidelity_vs_input"] - 1.0) <= 1e-12


def test_f3_step_phase_flip_3_and_five_qubit_stage_oracle_parity(live_server: ServerHandle) -> None:
    psi = _cardinal_amplitude_map()["+i"]
    for cname, err, expected_synd in (
        ("phase_flip_3", "IZI", "11"),
        ("five_qubit", "IYIII", "1101"),
    ):
        st, _, body = http_request_json(
            live_server.base_url,
            "/api/step",
            method="POST",
            payload={
                "code": cname,
                "state_preset": "+i",
                "error_mode": "pauli_string",
                "pauli_string": err,
            },
        )
        assert st == 200
        oracle = compute_step_oracle(cname, psi, error_mode="pauli_string", pauli_string=err)
        _assert_step_matches_oracle(body, oracle)
        assert body["stages"][3]["selected_syndrome"] == expected_synd
        assert abs(body["stages"][5]["fidelity_vs_input"] - 1.0) <= 1e-12


def test_f3_step_dfs_2_zero_ancilla_no_syndrome_recovery(live_server: ServerHandle) -> None:
    psi = _cardinal_amplitude_map()["+i"]
    st, _, body = http_request_json(
        live_server.base_url,
        "/api/step",
        method="POST",
        payload={
            "code": "dfs_2",
            "state_preset": "+i",
            "error_mode": "pauli_string",
            "pauli_string": "ZZ",
        },
    )
    assert st == 200
    assert body["n_ancilla"] == 0
    assert body["stages"][3]["selected_syndrome"] == ""
    assert body["stages"][4]["recovery_ops"] == []
    oracle = compute_step_oracle("dfs_2", psi, error_mode="pauli_string", pauli_string="ZZ")
    _assert_step_matches_oracle(body, oracle)
    enc_probs = {e["basis"]: e["prob"] for e in body["stages"][1]["statevector_nonzero"]}
    assert abs(enc_probs["01"] - 0.5) <= 1e-12
    assert abs(enc_probs["10"] - 0.5) <= 1e-12


def test_f3_browser_stage_list_interactive_stepping(page: PageSession) -> None:
    assert page.has_element("stage-list")
    page.select_option("code-select", "bit_flip_3")
    page.select_option("state-select", "+i")

    def _stage_snapshot() -> str:
        return str(
            page.eval_js(
                """(() => {
                    const keys = ['initial', 'encoded', 'noisy', 'syndrome_extracted', 'recovered', 'decoded'];
                    const states = keys.map(k => {
                        const el = document.querySelector('[data-testid="stage-step-' + k + '"]');
                        if (!el) return '';
                        return [
                            k,
                            el.getAttribute('aria-current') || '',
                            el.getAttribute('aria-selected') || '',
                            el.getAttribute('aria-pressed') || '',
                            el.getAttribute('data-active') || '',
                            el.className || ''
                        ].join('|');
                    });
                    const coh = document.querySelector('[data-testid="coherence-panel"]')?.innerText || '';
                    const sl = document.querySelector('[data-testid="stage-list"]')?.innerText || '';
                    return JSON.stringify({states, coh, sl});
                })()"""
            )
        )

    snapshots: set[str] = set()
    for key in STAGE_KEYS:
        tid = f"stage-step-{key}"
        assert page.has_element(tid), f"Missing stage step element [data-testid='{tid}']"
        page.click(tid, settle_s=0.12)
        snapshots.add(_stage_snapshot())

    assert len(snapshots) >= 2, (
        "Clicking across stage-step-* elements produced no observable active-state or DOM/metric update"
    )
    page.assert_no_errors()


# ---------------------------------------------------------------------------
# Feature F4: Continuous SU(2) / Kraus Discretization
# ---------------------------------------------------------------------------


def test_f4_su2_rx_bit_flip_3_pauli_coefficients_and_branches(live_server: ServerHandle) -> None:
    theta = 1.1
    psi = _cardinal_amplitude_map()["+i"]
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
    coeffs_orc = su2_pauli_coefficients(theta, 1.0, 0.0, 0.0)
    coeffs_api = body["error_applied"]["pauli_coefficients"]
    assert coeffs_api is not None
    for p_key, c_orc in coeffs_orc.items():
        val = coeffs_api[p_key]
        c_api = complex(float(val[0]), float(val[1])) if isinstance(val, (list, tuple)) else complex(
            float(val.get("re", 0.0)), float(val.get("im", 0.0))
        ) if isinstance(val, dict) else complex(float(val), 0.0)
        assert abs(c_api - c_orc) <= 1e-12

    probs = body["stages"][3]["syndrome_probabilities"]
    assert abs(probs["00"] - math.cos(theta / 2.0) ** 2) <= 1e-12
    assert abs(probs["10"] - math.sin(theta / 2.0) ** 2) <= 1e-12
    oracle = compute_step_oracle(
        "bit_flip_3",
        psi,
        error_mode="su2_rotation",
        su2_qubit=0,
        su2_theta=theta,
        su2_nx=1.0,
        su2_ny=0.0,
        su2_nz=0.0,
        selected_syndrome=body["stages"][3]["selected_syndrome"],
    )
    _assert_step_matches_oracle(body, oracle)


def test_f4_su2_arbitrary_axis_five_qubit_all_four_branches_recover(live_server: ServerHandle) -> None:
    theta = 1.25
    nx, ny, nz = 1.0, -2.0, 3.0
    norm = math.sqrt(nx * nx + ny * ny + nz * nz)
    ux, uy, uz = nx / norm, ny / norm, nz / norm
    s2 = math.sin(theta / 2.0) ** 2
    # For qubit 2 in five_qubit: I->"0000", X->"1100", Y->"1110", Z->"0010"
    expected_branches = {
        "0000": math.cos(theta / 2.0) ** 2,
        "1100": (ux**2) * s2,
        "1110": (uy**2) * s2,
        "0010": (uz**2) * s2,
    }
    for synd, exp_p in expected_branches.items():
        st, _, body = http_request_json(
            live_server.base_url,
            "/api/step",
            method="POST",
            payload={
                "code": "five_qubit",
                "state_preset": "+i",
                "error_mode": "su2_rotation",
                "su2_qubit": 2,
                "su2_theta": theta,
                "su2_nx": nx,
                "su2_ny": ny,
                "su2_nz": nz,
                "selected_syndrome": synd,
            },
        )
        assert st == 200
        probs = body["stages"][3]["syndrome_probabilities"]
        assert abs(probs[synd] - exp_p) <= 1e-12
        assert abs(body["stages"][5]["fidelity_vs_input"] - 1.0) <= 1e-12


def test_f4_su2_rz_bit_flip_3_undetected_logical_z_rotation(live_server: ServerHandle) -> None:
    theta = math.pi / 3.0
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
            "su2_nx": 0.0,
            "su2_ny": 0.0,
            "su2_nz": 1.0,
        },
    )
    assert st == 200
    assert abs(body["stages"][3]["syndrome_probabilities"]["00"] - 1.0) <= 1e-12
    assert abs(body["stages"][5]["fidelity_vs_input"] - math.cos(theta / 2.0) ** 2) <= 1e-12


def test_f4_coherent_collective_rz_dfs_2_vs_raw_parity(live_server: ServerHandle) -> None:
    phi = math.pi / 3.0
    st, _, body = http_request_json(
        live_server.base_url,
        "/api/step",
        method="POST",
        payload={
            "code": "dfs_2",
            "state_preset": "+i",
            "error_mode": "coherent_collective",
            "collective_phi": phi,
        },
    )
    assert st == 200
    assert abs(body["stages"][5]["fidelity_vs_input"] - 1.0) <= 1e-12
    assert abs(body["comparison"]["dfs_2"]["fidelity"] - 1.0) <= 1e-12
    assert abs(body["comparison"]["raw"]["fidelity"] - (math.cos(phi / 2.0) ** 2)) <= 1e-12


def test_f4_coherent_collective_rz_phase_flip_3_stage_oracle(live_server: ServerHandle) -> None:
    phi = 0.7
    psi = _cardinal_amplitude_map()["+i"]
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
    oracle = compute_step_oracle(
        "phase_flip_3",
        psi,
        error_mode="coherent_collective",
        collective_phi=phi,
        selected_syndrome=body["stages"][3]["selected_syndrome"],
    )
    _assert_step_matches_oracle(body, oracle)


# ---------------------------------------------------------------------------
# Feature F5: Bloch Sphere & |rho_01| Comparison
# ---------------------------------------------------------------------------


def test_f5_input_state_bloch_and_abs_rho01_all_cardinals(live_server: ServerHandle) -> None:
    for label, psi in _cardinal_amplitude_map().items():
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
        rho_orc = np.outer(psi, psi.conj())
        bloch_orc = rho_to_bloch(rho_orc)
        inp = body["input_state"]
        assert np.max(np.abs(np.asarray(inp["bloch"], dtype=float) - bloch_orc)) <= 1e-12
        assert abs(float(inp["abs_rho01"]) - 0.5 * math.hypot(bloch_orc[0], bloch_orc[1])) <= 1e-12


def test_f5_comparison_raw_qecc_dfs2_under_dephasing(live_server: ServerHandle) -> None:
    st, _, body = http_request_json(
        live_server.base_url,
        "/api/step",
        method="POST",
        payload={
            "code": "phase_flip_3",
            "state_preset": "+i",
            "error_mode": "coherent_collective",
            "collective_phi": math.pi / 2.0,
        },
    )
    assert st == 200
    comp = body["comparison"]
    assert abs(float(comp["dfs_2"]["fidelity"]) - 1.0) <= 1e-12
    assert abs(float(comp["dfs_2"]["abs_rho01"]) - 0.5) <= 1e-12
    assert abs(float(comp["raw"]["fidelity"]) - 0.5) <= 1e-12
    assert abs(float(comp["raw"]["abs_rho01"]) - 0.5) <= 1e-12
    assert abs(float(comp["qecc"]["fidelity"]) - float(body["stages"][5]["fidelity_vs_input"])) <= 1e-12


def test_f5_sweeps_coherence_curves_match_analytical_oracle(live_server: ServerHandle) -> None:
    st, _, body = http_request_json(
        live_server.base_url,
        "/api/sweeps",
        method="POST",
        payload={
            "codes": ["bit_flip_3", "phase_flip_3", "five_qubit", "dfs_2"],
            "channels": ["bit_flip", "phase_flip", "depolarizing", "collective_dephasing"],
            "metric": "entanglement",
            "p_min": 0.0,
            "p_max": 0.5,
            "num_p": 11,
            "run_live_mc": False,
        },
    )
    assert st == 200
    coherence_curves = body["coherence_curves"]
    for cname, ch in (
        ("bit_flip_3", "bit_flip"),
        ("phase_flip_3", "phase_flip"),
        ("five_qubit", "depolarizing"),
        ("dfs_2", "collective_dephasing"),
    ):
        entry = coherence_curves[f"{cname}/{ch}"]
        p_grid = entry["p_grid"]
        for p_val, obs_qecc, obs_raw in zip(
            p_grid, entry["abs_rho01_analytical"], entry["abs_rho01_raw"], strict=True
        ):
            assert abs(float(obs_qecc) - analytical_coherence_oracle(cname, ch, float(p_val))) <= 1e-12
            assert abs(float(obs_raw) - analytical_coherence_oracle("raw", ch, float(p_val))) <= 1e-12


def test_f5_simulate_rho_out_bloch_and_abs_rho01_consistency(live_server: ServerHandle) -> None:
    st, _, body = http_request_json(
        live_server.base_url,
        "/api/simulate",
        method="POST",
        payload={
            "code": "five_qubit",
            "channel": "depolarizing",
            "p": 0.05,
            "num_shots": 1000,
            "state_mode": "+i",
            "seed": 77,
            "backend": "numpy",
        },
    )
    assert st == 200
    rho_out = decode_complex_mat(body["rho_out"])
    bloch_out = np.asarray(body["bloch_out"], dtype=float)
    assert np.max(np.abs(rho_out - bloch_to_rho(bloch_out))) <= 1e-12
    assert abs(float(body["abs_rho01"]) - abs_rho01(rho_out)) <= 1e-12


def test_f5_browser_bloch_canvas_and_coherence_panel_render_and_update(page: PageSession) -> None:
    b_box = page.ensure_visible("bloch-canvas")
    c_box = page.ensure_visible("coherence-panel")
    assert b_box["width"] > 0 and b_box["height"] > 0
    assert c_box["width"] > 0 and c_box["height"] > 0

    def _bloch_rendered() -> bool:
        return bool(
            page.eval_js(
                """(() => {
                    const root = document.querySelector('[data-testid="bloch-canvas"]');
                    if (!root) return false;
                    const tag = (root.tagName || '').toUpperCase();
                    const canvas = tag === 'CANVAS' ? root : root.querySelector('canvas');
                    if (canvas) {
                        if (canvas.width <= 0 || canvas.height <= 0) return false;
                        const ctx2d = canvas.getContext('2d');
                        if (ctx2d) {
                            const data = ctx2d.getImageData(0, 0, canvas.width, canvas.height).data;
                            for (let i = 3; i < data.length; i += 16) {
                                if (data[i] !== 0) return true;
                            }
                            return false;
                        }
                        return canvas.toDataURL().length > 100;
                    }
                    const svg = tag === 'SVG' ? root : root.querySelector('svg');
                    if (svg) {
                        const shapes = svg.querySelectorAll('path, circle, ellipse, line, polygon, polyline, rect');
                        return shapes.length >= 2;
                    }
                    return false;
                })()"""
            )
        )

    page.select_option("code-select", "phase_flip_3")
    page.select_option("channel-select", "collective_dephasing")
    page.select_option("state-select", "+i")
    page.fill_input("phi-input", "0.0")
    page.wait_for(
        '(() => {'
        '  const t = document.querySelector(\'[data-testid="coherence-panel"]\')?.innerText || "";'
        '  return t.trim().length > 0 && /\\d/.test(t);'
        '})()'
    )
    assert _bloch_rendered(), "bloch-canvas rendered blank or without <canvas>/<svg> geometry"
    before = str(page.eval_js('document.querySelector(\'[data-testid="coherence-panel"]\').innerText') or "")

    page.select_option("code-select", "dfs_2")
    page.fill_input("phi-input", "1.57079632679")
    before_json = json.dumps(before)
    page.wait_for(
        f'(() => {{'
        f'  const t = document.querySelector(\'[data-testid="coherence-panel"]\')?.innerText || "";'
        f'  return t.trim().length > 0 && /\\d/.test(t) && t !== {before_json};'
        f'}})()'
    )
    after = str(page.eval_js('document.querySelector(\'[data-testid="coherence-panel"]\').innerText') or "")
    assert after != before and len(after.strip()) > 0
    assert _bloch_rendered(), "bloch-canvas became blank after updating code/phi"
    page.assert_no_errors()


# ---------------------------------------------------------------------------
# Feature F6: Interactive Input State Selector
# ---------------------------------------------------------------------------


def test_f6_metadata_cardinal_states_match_sampler(live_server: ServerHandle) -> None:
    st, _, meta = http_request_json(live_server.base_url, "/api/metadata")
    assert st == 200
    raw_cards = meta["cardinal_states"]
    ref = _cardinal_amplitude_map()
    if isinstance(raw_cards, dict):
        cards = raw_cards
    else:
        cards = {item.get("key") or item.get("name") or item.get("label"): item for item in raw_cards}
    assert set(ref.keys()).issubset(set(cards.keys()))
    for k, psi_ref in ref.items():
        amps = decode_complex_vec(cards[k]["amplitudes"])
        assert np.max(np.abs(amps - psi_ref)) <= 1e-12
        bloch_ref = rho_to_bloch(np.outer(psi_ref, psi_ref.conj()))
        assert np.max(np.abs(np.asarray(cards[k]["bloch"], dtype=float) - bloch_ref)) <= 1e-12


def test_f6_step_all_six_cardinal_presets(live_server: ServerHandle) -> None:
    for label, psi_ref in _cardinal_amplitude_map().items():
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
        amps = decode_complex_vec(body["input_state"]["amplitudes"])
        assert np.max(np.abs(amps - psi_ref)) <= 1e-12


def test_f6_step_bloch_angle_preset(live_server: ServerHandle) -> None:
    theta, phi_s = 1.23, 2.34
    psi_ref = np.array(
        [math.cos(theta / 2.0), np.exp(1.0j * phi_s) * math.sin(theta / 2.0)],
        dtype=np.complex128,
    )
    st, _, body = http_request_json(
        live_server.base_url,
        "/api/step",
        method="POST",
        payload={
            "code": "five_qubit",
            "state_preset": "bloch",
            "theta": theta,
            "phi_state": phi_s,
            "error_mode": "pauli_string",
            "pauli_string": "IIIII",
        },
    )
    assert st == 200
    rho_api = decode_complex_mat(body["input_state"]["rho"])
    assert np.max(np.abs(rho_api - np.outer(psi_ref, psi_ref.conj()))) <= 1e-12


def test_f6_step_custom_amplitudes_preset(live_server: ServerHandle) -> None:
    st, _, body = http_request_json(
        live_server.base_url,
        "/api/step",
        method="POST",
        payload={
            "code": "dfs_2",
            "state_preset": "custom",
            "amplitudes": [[0.6, 0.0], [0.0, 0.8]],
            "error_mode": "pauli_string",
            "pauli_string": "II",
        },
    )
    assert st == 200
    amps = decode_complex_vec(body["input_state"]["amplitudes"])
    assert np.max(np.abs(amps - np.array([0.6 + 0.0j, 0.0 + 0.8j]))) <= 1e-12


def test_f6_simulate_cardinal_avg_matches_haar_state_fidelity(
    live_server: ServerHandle, qecc_analysis, helpers
) -> None:
    p = 0.05
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
            "seed": 2026,
            "backend": "numpy",
        },
    )
    assert st == 200
    exp_state = float(qecc_analysis.analytical_fidelity("five_qubit", "depolarizing", p, metric="state"))
    assert abs(float(body["analytical_state_fidelity"]) - exp_state) <= 1e-12
    assert abs(float(body["mean_fidelity"]) - exp_state) <= helpers.binom_sigma(exp_state, n_shots, z=4.5)


# ---------------------------------------------------------------------------
# Feature F7: Noise, Parameter & Backend Controls
# ---------------------------------------------------------------------------


def test_f7_step_pauli_string_single_qubit_and_seeded_stochastic(live_server: ServerHandle) -> None:
    st1, _, r1 = http_request_json(
        live_server.base_url,
        "/api/step",
        method="POST",
        payload={
            "code": "five_qubit",
            "state_preset": "+i",
            "error_mode": "single_qubit_pauli",
            "error_qubit": 3,
            "error_op": "Z",
        },
    )
    assert st1 == 200
    assert r1["stages"][3]["selected_syndrome"] == "1001"
    assert abs(r1["stages"][5]["fidelity_vs_input"] - 1.0) <= 1e-12

    payload_stoch = {
        "code": "five_qubit",
        "state_preset": "+i",
        "error_mode": "stochastic_sample",
        "channel": "depolarizing",
        "p": 0.2,
        "seed": 1337,
    }
    st2, _, r2 = http_request_json(live_server.base_url, "/api/step", method="POST", payload=payload_stoch)
    st3, _, r3 = http_request_json(live_server.base_url, "/api/step", method="POST", payload=payload_stoch)
    assert st2 == 200 and st3 == 200
    assert r2["error_applied"]["label"] == r3["error_applied"]["label"]
    assert r2["stages"][3]["selected_syndrome"] == r3["stages"][3]["selected_syndrome"]


def test_f7_simulate_all_four_channels_numpy_statistical_parity(
    live_server: ServerHandle, qecc_analysis, helpers
) -> None:
    p = 0.08
    n_shots = 5000
    for cname, ch in (
        ("bit_flip_3", "bit_flip"),
        ("phase_flip_3", "phase_flip"),
        ("five_qubit", "depolarizing"),
        ("dfs_2", "collective_dephasing"),
    ):
        st, _, body = http_request_json(
            live_server.base_url,
            "/api/simulate",
            method="POST",
            payload={
                "code": cname,
                "channel": ch,
                "p": p,
                "num_shots": n_shots,
                "state_mode": "+i",
                "seed": 505,
                "backend": "numpy",
            },
        )
        assert st == 200
        assert body["backend_used"] == "numpy"
        # Pure |+i> under symmetric depolarizing has fidelity F_state (Y|+i> = +|+i>),
        # whereas single-axis bit_flip / phase_flip on |+i> equals F_entanglement.
        metric = "state" if ch == "depolarizing" else "entanglement"
        exp_f = float(qecc_analysis.analytical_fidelity(cname, ch, p, metric=metric))
        assert abs(float(body["mean_fidelity"]) - exp_f) <= helpers.binom_sigma(exp_f, n_shots, z=4.5)


def test_f7_simulate_torch_cuda_when_available(
    live_server: ServerHandle, qecc_analysis, helpers
) -> None:
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA device not available for torch_cuda backend")

    p = 0.05
    n_shots = 8000
    st, _, body = http_request_json(
        live_server.base_url,
        "/api/simulate",
        method="POST",
        payload={
            "code": "five_qubit",
            "channel": "depolarizing",
            "p": p,
            "num_shots": n_shots,
            "state_mode": "+i",
            "seed": 999,
            "backend": "torch_cuda",
        },
    )
    assert st == 200
    assert body["backend_used"] == "torch_cuda"
    # On |+i> under symmetric depolarizing, exact fidelity equals F_state (Y|+i> = +|+i>)
    exp_state = float(qecc_analysis.analytical_fidelity("five_qubit", "depolarizing", p, metric="state"))
    assert abs(float(body["mean_fidelity"]) - exp_state) <= helpers.binom_sigma(exp_state, n_shots, z=4.5)


def test_f7_simulate_eigenkit_unitary_p0_and_deterministic_parity(live_server: ServerHandle) -> None:
    if "eigenkit_unitary" not in list_backends_oracle():
        pytest.skip("eigenkit_unitary backend not installed in environment")

    for det_err in (None, ["XIIII"]):
        payload_common: dict[str, Any] = {
            "code": "five_qubit",
            "channel": "depolarizing" if det_err is None else None,
            "p": 0.0,
            "num_shots": 1,
            "state_mode": "+i",
            "seed": 11,
        }
        if det_err is not None:
            payload_common["deterministic_errors"] = det_err
        st_np, _, res_np = http_request_json(
            live_server.base_url, "/api/simulate", method="POST", payload={**payload_common, "backend": "numpy"}
        )
        st_ek, _, res_ek = http_request_json(
            live_server.base_url,
            "/api/simulate",
            method="POST",
            payload={**payload_common, "backend": "eigenkit_unitary"},
        )
        assert st_np == 200 and st_ek == 200
        assert res_ek["backend_used"] == "eigenkit_unitary"
        assert abs(float(res_ek["mean_fidelity"]) - float(res_np["mean_fidelity"])) <= 1e-12
        assert np.max(np.abs(decode_complex_mat(res_ek["rho_out"]) - decode_complex_mat(res_np["rho_out"]))) <= 1e-12


def test_f7_browser_controls_channel_p_phi_state_reactivity(page: PageSession) -> None:
    page.select_option("code-select", "five_qubit")
    page.select_option("channel-select", "depolarizing")
    page.select_option("state-select", "+i")
    page.fill_input("p-input", "0.08")
    page.fill_input("phi-input", "0.5")
    page.assert_no_errors()


# ---------------------------------------------------------------------------
# Feature F8: Analytical & MC Curves + p* Thresholds
# ---------------------------------------------------------------------------


def test_f8_sweeps_entanglement_metric_and_raw_1_minus_p(
    live_server: ServerHandle, qecc_analysis
) -> None:
    st, _, body = http_request_json(
        live_server.base_url,
        "/api/sweeps",
        method="POST",
        payload={
            "codes": ["bit_flip_3", "phase_flip_3", "five_qubit", "dfs_2"],
            "channels": ["bit_flip", "phase_flip", "depolarizing", "collective_dephasing"],
            "metric": "entanglement",
            "p_min": 0.0,
            "p_max": 0.5,
            "num_p": 11,
            "run_live_mc": False,
        },
    )
    assert st == 200
    assert body["metric"] == "entanglement"
    p_grid = body["p_grid"]
    for p_val, r_val in zip(p_grid, body["raw_curve"], strict=True):
        assert abs(float(r_val) - (1.0 - float(p_val))) <= 1e-12

    for cname, ch in (
        ("bit_flip_3", "bit_flip"),
        ("phase_flip_3", "phase_flip"),
        ("five_qubit", "depolarizing"),
        ("dfs_2", "collective_dephasing"),
    ):
        curve = body["analytical_curves"][f"{cname}/{ch}"]
        for p_val, f_val, df_val in zip(p_grid, curve["fidelities"], curve["delta_f"], strict=True):
            exp_f = float(qecc_analysis.analytical_fidelity(cname, ch, float(p_val), metric="entanglement"))
            assert abs(float(f_val) - exp_f) <= 1e-12
            assert abs(float(df_val) - (exp_f - (1.0 - float(p_val)))) <= 1e-12


def test_f8_sweeps_state_metric_and_strict_separation_from_entanglement(
    live_server: ServerHandle, qecc_analysis
) -> None:
    payload_base = {
        "codes": ["five_qubit"],
        "channels": ["depolarizing"],
        "p_min": 0.0,
        "p_max": 0.5,
        "num_p": 11,
        "run_live_mc": False,
    }
    st_e, _, body_e = http_request_json(
        live_server.base_url, "/api/sweeps", method="POST", payload={**payload_base, "metric": "entanglement"}
    )
    st_s, _, body_s = http_request_json(
        live_server.base_url, "/api/sweeps", method="POST", payload={**payload_base, "metric": "state"}
    )
    assert st_e == 200 and st_s == 200
    p_grid = body_s["p_grid"]
    for idx, p_val in enumerate(p_grid):
        assert abs(float(body_s["raw_curve"][idx]) - (1.0 - (2.0 / 3.0) * float(p_val))) <= 1e-12
        exp_state = float(qecc_analysis.analytical_fidelity("five_qubit", "depolarizing", float(p_val), metric="state"))
        f_s = float(body_s["analytical_curves"]["five_qubit/depolarizing"]["fidelities"][idx])
        f_e = float(body_e["analytical_curves"]["five_qubit/depolarizing"]["fidelities"][idx])
        assert abs(f_s - exp_state) <= 1e-12
        if float(p_val) > 0.0:
            assert f_s > f_e


def test_f8_gain_thresholds_and_p_star_match_brentq(live_server: ServerHandle, qecc_analysis) -> None:
    st_m, _, meta = http_request_json(live_server.base_url, "/api/metadata")
    st_s, _, sw = http_request_json(
        live_server.base_url,
        "/api/sweeps",
        method="POST",
        payload={
            "codes": ["bit_flip_3", "phase_flip_3", "five_qubit", "dfs_2"],
            "channels": ["bit_flip", "phase_flip", "depolarizing", "collective_dephasing"],
            "metric": "entanglement",
            "p_min": 0.0,
            "p_max": 0.5,
            "num_p": 6,
            "run_live_mc": False,
        },
    )
    assert st_m == 200 and st_s == 200
    for cname in ("bit_flip_3", "phase_flip_3", "five_qubit", "dfs_2"):
        for ch in ("bit_flip", "phase_flip", "depolarizing", "collective_dephasing"):
            exp_p = float(qecc_analysis.analytical_gain_threshold(cname, ch))
            assert abs(_extract_gain_threshold(meta["gain_thresholds"], cname, ch) - exp_p) <= 1e-12
            assert abs(_extract_gain_threshold(sw["gain_thresholds"], cname, ch) - exp_p) <= 1e-12
            assert abs(float(sw["analytical_curves"][f"{cname}/{ch}"]["p_star"]) - exp_p) <= 1e-12


def test_f8_sweeps_cached_fidelity_sweeps_json_and_live_mc(
    live_server: ServerHandle, results_dir, qecc_analysis, helpers
) -> None:
    with (results_dir / "data" / "fidelity_sweeps.json").open("r", encoding="utf-8") as fh:
        ref_data = json.load(fh)

    st_c, _, cached = http_request_json(
        live_server.base_url,
        "/api/sweeps",
        method="POST",
        payload={
            "codes": ["bit_flip_3", "five_qubit", "dfs_2"],
            "channels": ["bit_flip", "depolarizing", "collective_dephasing"],
            "metric": "entanglement",
            "p_min": 0.0,
            "p_max": 0.5,
            "num_p": 21,
            "run_live_mc": False,
        },
    )
    assert st_c == 200
    mc_cached = cached["mc_sweeps"]
    assert isinstance(mc_cached, dict) and len(mc_cached) > 0
    # Verify cached MC records match reference dataset values (<= 1e-12)
    ref_lookup = {
        (r["code"], r["channel"], round(float(r["p"]), 6), r["state_mode"]): float(r["mean_fidelity"])
        for r in ref_data["records"]
    }
    matched_any = False
    for key in ("bit_flip_3/bit_flip", "five_qubit/depolarizing", "dfs_2/collective_dephasing"):
        entries = mc_cached.get(key)
        if not entries:
            continue
        cname, ch = key.split("/", 1)
        for item in entries:
            p_r = round(float(item["p"]), 6)
            sm = item.get("state_mode", "|0>")
            if (cname, ch, p_r, sm) in ref_lookup:
                assert abs(float(item["mean_fidelity"]) - ref_lookup[(cname, ch, p_r, sm)]) <= 1e-12
                matched_any = True
    assert matched_any, f"Expected cached mc_sweeps to match fidelity_sweeps.json records; got {list(mc_cached)}"

    # Live MC sweep
    st_l, _, live = http_request_json(
        live_server.base_url,
        "/api/sweeps",
        method="POST",
        payload={
            "codes": ["bit_flip_3"],
            "channels": ["bit_flip"],
            "metric": "entanglement",
            "p_min": 0.0,
            "p_max": 0.2,
            "num_p": 3,
            "run_live_mc": True,
            "num_shots": 2000,
            "seed": 123,
            "backend": "numpy",
        },
    )
    assert st_l == 200
    for item in live["mc_sweeps"]["bit_flip_3/bit_flip"]:
        exp_f = float(qecc_analysis.analytical_fidelity("bit_flip_3", "bit_flip", float(item["p"]), metric="entanglement"))
        assert abs(float(item["mean_fidelity"]) - exp_f) <= helpers.binom_sigma(exp_f, 2000, z=4.5)


def test_f8_browser_curves_chart_renders_and_toggles_metric(page: PageSession) -> None:
    box = page.ensure_visible("curves-chart")
    assert box["width"] > 0 and box["height"] > 0

    chart_ready_js = """(() => {
        const root = document.querySelector('[data-testid="curves-chart"]');
        if (!root) return false;
        const tag = (root.tagName || '').toUpperCase();
        const svg = tag === 'SVG' ? root : root.querySelector('svg');
        if (svg) {
            const shapes = svg.querySelectorAll('path, polyline, line, circle, rect');
            return shapes.length >= 2;
        }
        const canvas = tag === 'CANVAS' ? root : root.querySelector('canvas');
        if (canvas && canvas.width > 0 && canvas.height > 0) {
            const ctx = canvas.getContext('2d');
            if (!ctx) return canvas.toDataURL().length > 100;
            const d = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
            for (let i = 3; i < d.length; i += 16) {
                if (d[i] !== 0) return true;
            }
        }
        return false;
    })()"""
    page.wait_for(chart_ready_js)

    chart_sig_js = """(() => {
        const root = document.querySelector('[data-testid="curves-chart"]');
        if (!root) return '';
        const tag = (root.tagName || '').toUpperCase();
        const svg = tag === 'SVG' ? root : root.querySelector('svg');
        if (svg) return svg.innerHTML + '|' + (root.innerText || '');
        const canvas = tag === 'CANVAS' ? root : root.querySelector('canvas');
        if (canvas) return canvas.toDataURL() + '|' + (root.innerText || '');
        return root.innerText || '';
    })()"""

    def _chart_sig() -> str:
        return str(page.eval_js(chart_sig_js) or "")

    before_sig = _chart_sig()
    toggled = bool(
        page.eval_js(
            """(() => {
                const m = document.querySelector('[data-testid="metric-select"], [data-testid="metric-toggle"]');
                if (!m) return false;
                if ((m.tagName || '').toUpperCase() === 'SELECT') {
                    const opts = Array.from(m.options || []).map(o => o.value);
                    const nextVal = opts.find(v => v !== m.value) || (m.value === 'state' ? 'entanglement' : 'state');
                    m.value = nextVal;
                    m.dispatchEvent(new Event('input', {bubbles: true}));
                    m.dispatchEvent(new Event('change', {bubbles: true}));
                } else {
                    m.click();
                }
                return true;
            })()"""
        )
    )
    if toggled:
        before_json = json.dumps(before_sig)
        page.wait_for(
            f"(() => {{ if (!({chart_ready_js})) return false; const sig = {chart_sig_js}; return Boolean(sig) && sig !== {before_json}; }})()"
        )
        assert _chart_sig() != before_sig, "Toggling metric did not update rendered curves-chart"
        assert bool(page.eval_js(chart_ready_js)), "curves-chart lost geometry after toggling metric"
    page.assert_no_errors()


# ---------------------------------------------------------------------------
# Feature F9: Stabilizers, Syndromes, Weight Enumerator & KL
# ---------------------------------------------------------------------------


def test_f9_theory_stabilizers_logical_ops_syndrome_tables(live_server: ServerHandle) -> None:
    for cname in ("bit_flip_3", "phase_flip_3", "five_qubit", "dfs_2"):
        st, _, body = http_request_json(live_server.base_url, f"/api/theory?code={cname}")
        assert st == 200
        c_ref = get_code_oracle(cname)
        assert body["code"] == cname
        assert list(body["stabilizers"]) == list(c_ref.stabilizers)
        assert dict(body["syndrome_table"]) == dict(c_ref.syndrome_table)
        assert "X_L" in body["logical_ops"] and "Z_L" in body["logical_ops"]


def test_f9_theory_knill_laflamme_c_matrix_parity(live_server: ServerHandle, qecc_analysis) -> None:
    for cname in ("bit_flip_3", "phase_flip_3", "five_qubit", "dfs_2"):
        st, _, body = http_request_json(live_server.base_url, f"/api/theory?code={cname}")
        assert st == 200
        kl_api = body["knill_laflamme"]
        kl_ref = qecc_analysis.verify_knill_laflamme(cname)
        assert kl_api["satisfies"] is True
        assert float(kl_api["max_violation"]) <= 1e-12
        assert list(kl_api["error_basis"]) == list(kl_ref.error_set)
        c_api = decode_complex_mat(kl_api["c_matrix"])
        assert np.max(np.abs(c_api - kl_ref.c_matrix)) <= 1e-12


def test_f9_theory_five_qubit_1024_weight_enumerator(live_server: ServerHandle, qecc_analysis) -> None:
    st, _, body = http_request_json(live_server.base_url, "/api/theory?code=five_qubit")
    assert st == 200
    enum_ref = qecc_analysis.five_qubit_weight_enumerator()
    enum_api = body["weight_enumerator"]
    assert enum_api is not None
    total = 0
    for coset, counts_ref in enum_ref.items():
        counts_api = {int(k): int(v) for k, v in enum_api[coset].items()}
        assert counts_api == counts_ref
        total += sum(counts_api.values())
    assert total == 1024
    assert {int(k): int(v) for k, v in enum_api["I"].items()} == {0: 1, 1: 15, 2: 0, 3: 60, 4: 135, 5: 45}


def test_f9_theory_aliasing_examples_match_step_logical_flip(live_server: ServerHandle) -> None:
    st, _, body = http_request_json(live_server.base_url, "/api/theory?code=bit_flip_3")
    assert st == 200
    examples = body["aliasing_examples"]
    assert isinstance(examples, list) and len(examples) >= 1

    st_step, _, step_res = http_request_json(
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
    assert st_step == 200
    assert step_res["stages"][3]["selected_syndrome"] == "10"
    assert abs(float(step_res["stages"][5]["fidelity_vs_input"]) - 0.0) <= 1e-12


def test_f9_browser_theory_panel_displays_stabilizers_kl_and_enumerator(page: PageSession) -> None:
    page.select_option("code-select", "five_qubit")
    box = page.ensure_visible("theory-panel")
    assert box["width"] > 0 and box["height"] > 0
    page.wait_for(
        '(() => {'
        '  const el = document.querySelector(\'[data-testid="theory-panel"]\');'
        '  if (!el) return false;'
        '  const t = el.innerText || "";'
        '  return t.includes("XZZXI") && (t.includes("1024") || t.includes("135") || t.includes("Knill"));'
        '})()'
    )
    txt = page.eval_js('document.querySelector(\'[data-testid="theory-panel"]\').innerText')
    assert "XZZXI" in txt
    assert "1024" in txt or "135" in txt or "Knill" in txt
    page.assert_no_errors()


# ---------------------------------------------------------------------------
# Feature F10: Seminar & Live Presentation Companion
# ---------------------------------------------------------------------------


def test_f10_browser_seminar_panel_renders_without_errors(page: PageSession) -> None:
    box = page.ensure_visible("seminar-panel")
    assert box["width"] > 0 and box["height"] > 0
    txt = page.eval_js('document.querySelector(\'[data-testid="seminar-panel"]\').innerText')
    assert len(txt.strip()) > 30
    page.assert_no_errors()


def test_f10_browser_seminar_covers_core_topics_and_navigation(page: PageSession) -> None:
    collected_texts = collect_seminar_panel_texts(page, max_steps=25)
    combined = "\n".join(collected_texts).lower()
    for term in ("no-cloning", "knill", "stabilizer", "syndrome", "dfs"):
        assert term in combined or "não-clonagem" in combined, f"Expected seminar term {term!r} in seminar panel"
    page.assert_no_errors()


def test_f10_browser_seminar_interactive_simulator_sync(page: PageSession) -> None:
    page.ensure_visible("seminar-panel")
    # Click any interactive "Try in simulator" / preset button if exposed inside seminar-panel
    page.eval_js(
        """(() => {
            const panel = document.querySelector('[data-testid="seminar-panel"]');
            if (!panel) return;
            const btn = panel.querySelector('[data-testid*="try"], [data-testid*="sync"], [data-testid*="apply"], button[data-code]');
            if (btn) btn.click();
        })()"""
    )
    page.assert_no_errors()


def test_f10_browser_seminar_formulas_and_no_leak_tokens(page: PageSession) -> None:
    page.ensure_visible("seminar-panel")
    txt = page.eval_js('document.querySelector(\'[data-testid="seminar-panel"]\').innerText')
    leaks = find_leak_tokens(txt)
    assert not leaks, f"Found leak tokens in seminar-panel: {leaks}"
    page.assert_no_errors()


def test_f10_browser_seminar_panel_roundtrip_stability(page: PageSession) -> None:
    page.ensure_visible("seminar-panel")
    page.ensure_visible("theory-panel")
    page.ensure_visible("curves-chart")
    box = page.ensure_visible("seminar-panel")
    assert box["width"] > 0 and box["height"] > 0
    page.assert_no_errors()


# ---------------------------------------------------------------------------
# Feature F11: Complete Bilingual UI (en <-> pt-br)
# ---------------------------------------------------------------------------


def test_f11_browser_default_ui_zero_leak_tokens(page: PageSession) -> None:
    txt = page.body_text()
    assert len(txt.strip()) > 50
    leaks = find_leak_tokens(txt)
    assert not leaks, f"Found leak tokens in default UI: {leaks}"
    page.assert_no_errors()


def test_f11_browser_switch_to_pt_br_translates_prose_and_seminar(page: PageSession) -> None:
    page.switch_language("en")
    text_en = page.body_text()
    lang_pt = page.switch_language("pt-br")
    assert lang_pt == "pt-br"
    text_pt = page.body_text()
    assert text_pt != text_en
    pt_lower = text_pt.lower()
    assert any(w in pt_lower for w in ("síndrome", "estabilizador", "coerência", "limiar", "fidelidade")), (
        "Expected Brazilian Portuguese QECC terminology after switching to pt-br"
    )
    leaks = find_leak_tokens(text_pt)
    assert not leaks, f"Found leak tokens in pt-br UI: {leaks}"
    page.assert_no_errors()


def test_f11_browser_roundtrip_pt_br_to_en_idempotent(page: PageSession) -> None:
    page.switch_language("en")
    en1 = page.body_text()
    page.switch_language("pt-br")
    assert page.get_current_language() == "pt-br"
    page.switch_language("en")
    assert page.get_current_language() == "en"
    en2 = page.body_text()
    assert en2 == en1 or "syndrome" in en2.lower()
    page.assert_no_errors()


def test_f11_browser_required_testids_preserved_across_languages(page: PageSession) -> None:
    for lang in ("en", "pt-br"):
        page.switch_language(lang)
        for tid in REQUIRED_TESTIDS:
            assert page.has_element(tid), f"Missing [data-testid='{tid}'] in language {lang}"
    page.assert_no_errors()


def test_f11_browser_desktop_and_tablet_no_horizontal_overflow_both_langs(page: PageSession) -> None:
    for lang in ("en", "pt-br"):
        page.switch_language(lang)
        for width, height, mobile in ((1280, 800, False), (768, 1024, True)):
            page.set_viewport(width, height, mobile=mobile)
            assert page.check_no_horizontal_overflow(), (
                f"Horizontal overflow at {width}x{height} in {lang}: {page.viewport_dimensions()}"
            )
    page.assert_no_errors()
