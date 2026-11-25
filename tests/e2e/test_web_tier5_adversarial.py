"""Tier 5 white-box adversarial & regression E2E tests for the QECC web system.

Covers:
1. Tier 4 regression: [[5,1,3]] Pure-X / Pure-Z (I coset) API + DOM table cells (en & pt-br).
2. Tier 4 regression: #delta-chart p_max = 1.0 unclipped yMax scaling (entanglement & state).
3. Custom normalize precision: (1,1,2,2) and (1,1,1,9) pass 1e-6 norm validation in browser.
4. Bloch angle validation: err.invalid_bloch_angles in en & pt-br (never Collective phase phi).
5. Cached MC std_error metric conversion parity (1.5x and (2/3)x within 1e-12).
6. Seminar 20-slide demo preset sweep + toast bilingual switch + Bloch drag/reset +
   Circuit M counts + SU(2) branch click sync + runSweeps chart recovery + 390x844 mobile audit.
7. Live HTTP/schema/CLI adversarial cases (Infinity/NaN, Content-Length, /static/ traversal,
   seed=-1 -> 422, unsupported HTTP methods -> JSON 405, HEAD, eigenkit_unitary, CLI ports).

All ``qecc.*`` imports are deferred inside test functions/fixtures so
``pytest --collect-only tests/e2e`` succeeds cleanly.
"""

from __future__ import annotations

import http.client
import json
import math
import subprocess
import sys
import time
from typing import Iterator
from urllib.parse import urlsplit

import pytest

from .web_harness import (
    BrowserSession,
    PageSession,
    ServerHandle,
    find_leak_tokens,
    http_request_json,
    http_request_raw,
)

pytestmark = pytest.mark.tier5

PARITY_TOL = 1e-12


@pytest.fixture(scope="module")
def live_server() -> Iterator[ServerHandle]:
    with ServerHandle(use_cli=False, host="127.0.0.1", port=0) as srv:
        yield srv


@pytest.fixture(scope="module")
def browser_session() -> Iterator[BrowserSession]:
    with BrowserSession() as browser:
        yield browser


def _wait_engine_ready(pg: PageSession, timeout: float = 10.0) -> None:
    pg.wait_for(
        '["Ready", "Pronto"].includes('
        '(document.getElementById("engine-status-badge")?.textContent || "").trim()'
        ")",
        timeout=timeout,
    )


def test_t5_1_five_qubit_single_axis_enumerator_api_and_dom(
    live_server: ServerHandle, browser_session: BrowserSession
) -> None:
    """Regression for [[5,1,3]] Pure-X (I coset) and Pure-Z (I coset) API + DOM table."""
    st, _, th = http_request_json(live_server.base_url, "/api/theory?code=five_qubit")
    assert st == 200
    expected_sa = {"0": 1, "1": 5, "2": 0, "3": 0, "4": 0, "5": 0}
    assert th["single_axis_weight_enumerator"]["X"] == expected_sa
    assert th["single_axis_weight_enumerator"]["Z"] == expected_sa

    pg = browser_session.new_page()
    try:
        pg.set_viewport(1440, 900, mobile=False)
        pg.goto(live_server.base_url)
        pg.wait_for_testid("code-select")
        _wait_engine_ready(pg)

        pg.switch_language("en")
        pg.select_option("code-select", "five_qubit")
        pg.ensure_visible("theory-panel")
        _wait_engine_ready(pg)

        pg.wait_for(
            "document.querySelectorAll('#theory-content .theory-card:last-child tbody tr').length === 7"
        )
        rows_en = pg.eval_js(
            "Array.from(document.querySelectorAll('#theory-content .theory-card:last-child tbody tr'))"
            ".map(tr => Array.from(tr.children).map(td => td.textContent.trim()))"
        )
        assert len(rows_en) == 7
        assert rows_en[6][0] == "Sum"
        assert [r[6] for r in rows_en] == ["1", "5", "0", "0", "0", "0", "6"]
        assert [r[7] for r in rows_en] == ["1", "5", "0", "0", "0", "0", "6"]

        pg.switch_language("pt-br")
        rows_pt = pg.eval_js(
            "Array.from(document.querySelectorAll('#theory-content .theory-card:last-child tbody tr'))"
            ".map(tr => Array.from(tr.children).map(td => td.textContent.trim()))"
        )
        assert rows_pt[6][0] == "Soma"
        assert [r[6] for r in rows_pt] == ["1", "5", "0", "0", "0", "0", "6"]
        assert [r[7] for r in rows_pt] == ["1", "5", "0", "0", "0", "0", "6"]
        pg.assert_no_errors()
    finally:
        pg.close()


def test_t5_2_delta_chart_pmax_1_unclipped_scaling(
    live_server: ServerHandle, browser_session: BrowserSession
) -> None:
    """Regression for #delta-chart p_max=1.0 unclipped yMax for dfs_2/collective_dephasing."""
    pg = browser_session.new_page()
    try:
        pg.set_viewport(1440, 900, mobile=False)
        pg.goto(live_server.base_url)
        pg.wait_for_testid("code-select")
        _wait_engine_ready(pg)

        pg.switch_language("en")
        pg.select_option("code-select", "dfs_2")
        pg.select_option("channel-select", "collective_dephasing")
        pg.ensure_visible("curves-chart")
        pg.fill_input("sweep-pmax", "1.0")
        pg.click("#refresh-sweeps-btn")
        _wait_engine_ready(pg)

        pg.wait_for(
            "Array.from(document.querySelectorAll('#delta-chart svg text'))"
            ".some(t => t.textContent.trim() === '1.02')"
        )
        delta_ent = pg.eval_js(
            "(() => {"
            "  const svg = document.querySelector('#delta-chart svg');"
            "  const texts = Array.from(svg.querySelectorAll('text')).map(t => t.textContent.trim());"
            "  const poly = svg.querySelector('polyline');"
            "  const pts = (poly?.getAttribute('points') || '').trim().split(/\\s+/);"
            "  const last = pts[pts.length - 1] || '';"
            "  return { texts, last };"
            "})()"
        )
        assert "1.02" in delta_ent["texts"]
        assert "1.00" in delta_ent["texts"]
        _, y_str = delta_ent["last"].split(",")
        assert float(y_str) < 40.0

        # Switch to state metric -> max delta_f = +2/3 => top tick 0.72
        pg.select_option("metric-select", "state")
        _wait_engine_ready(pg)
        pg.wait_for(
            "Array.from(document.querySelectorAll('#delta-chart svg text'))"
            ".some(t => t.textContent.trim() === '0.72')"
        )
        delta_st_texts = pg.eval_js(
            "Array.from(document.querySelectorAll('#delta-chart svg text')).map(t => t.textContent.trim())"
        )
        assert "0.72" in delta_st_texts
        pg.assert_no_errors()
    finally:
        pg.close()


def test_t5_3_custom_normalize_precision_in_browser(
    live_server: ServerHandle, browser_session: BrowserSession
) -> None:
    """normalize-custom-btn must produce high enough precision to pass 1e-6 norm validation."""
    pg = browser_session.new_page()
    try:
        pg.set_viewport(1440, 900, mobile=False)
        pg.goto(live_server.base_url)
        pg.wait_for_testid("code-select")
        _wait_engine_ready(pg)

        pg.select_option("code-select", "five_qubit")
        pg.select_option("state-select", "custom")
        pg.select_option("error-mode-select", "pauli_string")
        pg.fill_input("pauli-string-input", "IIIII")
        _wait_engine_ready(pg)

        for vec in ((1, 1, 2, 2), (1, 1, 1, 9)):
            pg.fill_input("alpha-re-input", str(vec[0]))
            pg.fill_input("alpha-im-input", str(vec[1]))
            pg.fill_input("beta-re-input", str(vec[2]))
            pg.fill_input("beta-im-input", str(vec[3]))
            pg.click("normalize-custom-btn")
            _wait_engine_ready(pg)

            state_info = pg.eval_js(
                "(() => {"
                "  const err = document.getElementById('validation-error');"
                "  const ar = Number(document.getElementById('alpha-re-input').value);"
                "  const ai = Number(document.getElementById('alpha-im-input').value);"
                "  const br = Number(document.getElementById('beta-re-input').value);"
                "  const bi = Number(document.getElementById('beta-im-input').value);"
                "  const normSq = ar*ar + ai*ai + br*br + bi*bi;"
                "  const stageText = document.getElementById('stage-detail-card')?.innerText || '';"
                "  return { hidden: err.classList.contains('hidden'), errText: err.textContent, normSq, stageText };"
                "})()"
            )
            assert state_info["hidden"] is True, f"Unexpected validation error for {vec}: {state_info['errText']}"
            assert abs(float(state_info["normSq"]) - 1.0) <= 1e-6
            assert "1.000000" in state_info["stageText"]
        pg.assert_no_errors()
    finally:
        pg.close()


def test_t5_4_bloch_angle_validation_and_bilingual_translation(
    live_server: ServerHandle, browser_session: BrowserSession
) -> None:
    """Bloch angle validation must show err.invalid_bloch_angles (not Collective phase phi) and translate."""
    pg = browser_session.new_page()
    try:
        pg.set_viewport(1440, 900, mobile=False)
        pg.goto(live_server.base_url)
        pg.wait_for_testid("code-select")
        _wait_engine_ready(pg)

        pg.switch_language("en")
        pg.select_option("state-select", "bloch")
        pg.fill_input("theta-input", "")
        pg.click("run-step-btn")

        pg.wait_for("!document.getElementById('validation-error').classList.contains('hidden')")
        en_msg = pg.eval_js("document.getElementById('validation-error').textContent.trim()")
        assert "Bloch angles" in en_msg
        assert "Collective phase" not in en_msg

        pg.switch_language("pt-br")
        pt_msg = pg.eval_js("document.getElementById('validation-error').textContent.trim()")
        assert "ângulos de Bloch" in pt_msg
        assert "fase coletiva" not in pt_msg

        pg.switch_language("en")
        pg.fill_input("theta-input", "1.5708")
        pg.click("run-step-btn")
        _wait_engine_ready(pg)
        pg.wait_for("document.getElementById('validation-error').classList.contains('hidden')")
        pg.assert_no_errors()
    finally:
        pg.close()


def test_t5_5_cached_mc_std_error_metric_conversion_parity(
    live_server: ServerHandle, results_dir
) -> None:
    """Cached MC sweeps must scale std_error consistently with converted_fidelity."""
    ref_records = json.loads((results_dir / "data" / "fidelity_sweeps.json").read_text(encoding="utf-8"))["records"]
    ref_5q = next(r for r in ref_records if r["code"] == "five_qubit" and r["channel"] == "depolarizing" and abs(float(r["p"]) - 0.05) < 1e-9)
    ref_bf = next(r for r in ref_records if r["code"] == "bit_flip_3" and r["channel"] == "bit_flip" and abs(float(r["p"]) - 0.05) < 1e-9)

    st_e, _, sw_e = http_request_json(
        live_server.base_url,
        "/api/sweeps",
        method="POST",
        payload={
            "codes": ["five_qubit"],
            "channels": ["depolarizing"],
            "metric": "entanglement",
            "p_min": 0.0,
            "p_max": 0.5,
            "num_p": 21,
            "run_live_mc": False,
        },
    )
    assert st_e == 200
    row_5q = sw_e["mc_sweeps"]["five_qubit/depolarizing"][2]
    assert abs(float(row_5q["p"]) - 0.05) <= PARITY_TOL
    assert abs(float(row_5q["converted_fidelity"]) - (1.0 - 1.5 * (1.0 - float(ref_5q["mean_fidelity"])))) <= PARITY_TOL
    assert abs(float(row_5q["std_error"]) - 1.5 * float(ref_5q["std_error"])) <= PARITY_TOL

    st_s, _, sw_s = http_request_json(
        live_server.base_url,
        "/api/sweeps",
        method="POST",
        payload={
            "codes": ["bit_flip_3"],
            "channels": ["bit_flip"],
            "metric": "state",
            "p_min": 0.0,
            "p_max": 0.5,
            "num_p": 21,
            "run_live_mc": False,
        },
    )
    assert st_s == 200
    row_bf = sw_s["mc_sweeps"]["bit_flip_3/bit_flip"][2]
    assert abs(float(row_bf["p"]) - 0.05) <= PARITY_TOL
    assert abs(float(row_bf["converted_fidelity"]) - (1.0 - (2.0 / 3.0) * (1.0 - float(ref_bf["mean_fidelity"])))) <= PARITY_TOL
    assert abs(float(row_bf["std_error"]) - (2.0 / 3.0) * float(ref_bf["std_error"])) <= PARITY_TOL


def test_t5_6_seminar_20_slides_toast_bloch_circuit_su2_sweeps_recovery_and_mobile(
    live_server: ServerHandle, browser_session: BrowserSession
) -> None:
    """Comprehensive browser adversarial checks:
    - All 20 seminar slide demo presets + #demo-toast en<->pt-br translation
    - Bloch canvas drag & reset view
    - Circuit SVG M box counts across all 4 codes
    - SU(2) .syn-bar-row click syncs #syndrome-branch-select & recovers F=1.000000
    - runSweeps catch block preserves charts on failed API response + eigenkit_unitary UI error
    - 390x844 mobile zero horizontal overflow across en/pt-br x five_qubit/dfs_2 x 6 panels
    """
    pg = browser_session.new_page()
    try:
        pg.set_viewport(1440, 900, mobile=False)
        pg.goto(live_server.base_url)
        pg.wait_for_testid("code-select")
        _wait_engine_ready(pg)
        pg.switch_language("en")

        # 1. All 20 seminar slides demo preset sweep + toast bilingual switch
        for idx in range(20):
            pg.ensure_visible("seminar-panel")
            pg.select_option("seminar-slide-select", str(idx))
            pg.wait_for_testid("seminar-apply-demo")
            pg.click("seminar-apply-demo")
            _wait_engine_ready(pg)
            err_hidden = pg.eval_js("document.getElementById('validation-error').classList.contains('hidden')")
            assert err_hidden is True, f"Validation error on slide {idx}"

        toast_en = pg.eval_js(
            "({ hidden: document.getElementById('demo-toast').classList.contains('hidden'),"
            "   text: document.getElementById('demo-toast').textContent.trim() })"
        )
        assert toast_en["hidden"] is False
        assert "Loaded slide preset into simulator:" in toast_en["text"]

        pg.switch_language("pt-br")
        toast_pt = pg.eval_js("document.getElementById('demo-toast').textContent.trim()")
        assert "Preset do slide carregado no simulador:" in toast_pt
        pg.switch_language("en")

        # 2. Bloch canvas drag & reset
        pg.ensure_visible("bloch-canvas")
        initial_url = pg.eval_js("document.getElementById('bloch-canvas').toDataURL()")
        box = pg.measure_box("bloch-canvas")
        assert box is not None and box["width"] > 0
        cx = float(box["left"]) + float(box["width"]) * 0.5
        cy = float(box["top"]) + float(box["height"]) * 0.5
        pg.cdp("Input.dispatchMouseEvent", {"type": "mouseMoved", "x": cx, "y": cy})
        pg.cdp("Input.dispatchMouseEvent", {"type": "mousePressed", "x": cx, "y": cy, "button": "left", "clickCount": 1})
        pg.cdp("Input.dispatchMouseEvent", {"type": "mouseMoved", "x": cx + 65.0, "y": cy - 35.0, "button": "left"})
        pg.cdp("Input.dispatchMouseEvent", {"type": "mouseReleased", "x": cx + 65.0, "y": cy - 35.0, "button": "left", "clickCount": 1})
        time.sleep(0.05)
        dragged_url = pg.eval_js("document.getElementById('bloch-canvas').toDataURL()")
        assert dragged_url != initial_url
        pg.click("#bloch-reset-btn")
        reset_url = pg.eval_js("document.getElementById('bloch-canvas').toDataURL()")
        assert reset_url == initial_url

        # 3. Circuit SVG M box counts across all 4 codes
        expected_m = {"bit_flip_3": 2, "phase_flip_3": 2, "five_qubit": 4, "dfs_2": 0}
        for cname, exp_cnt in expected_m.items():
            pg.select_option("code-select", cname)
            _wait_engine_ready(pg)
            counts = pg.eval_js(
                "(() => {"
                "  const svg = document.querySelector('#circuit-container svg');"
                "  const mCount = Array.from(svg.querySelectorAll('text'))"
                "    .filter(t => t.textContent.trim() === 'M').length;"
                "  const rectCount = svg.querySelectorAll('rect').length;"
                "  return { mCount, rectCount };"
                "})()"
            )
            assert counts["mCount"] == exp_cnt
            if cname == "five_qubit":
                assert counts["rectCount"] >= 45

        # 4. SU(2) .syn-bar-row click syncs #syndrome-branch-select
        pg.select_option("code-select", "five_qubit")
        pg.select_option("state-select", "+i")
        pg.select_option("error-mode-select", "su2_rotation")
        pg.fill_input("su2-qubit-input", "0")
        _wait_engine_ready(pg)
        pg.wait_for(
            "Array.from(document.querySelectorAll('.syn-bar-row'))"
            ".some(r => (r.firstElementChild?.textContent || '').trim() === '1010')"
        )
        pg.eval_js(
            "(() => {"
            "  const row = Array.from(document.querySelectorAll('.syn-bar-row'))"
            "    .find(r => (r.firstElementChild?.textContent || '').trim() === '1010');"
            "  if (row) row.click();"
            "})()"
        )
        _wait_engine_ready(pg)
        pg.wait_for("document.getElementById('syndrome-branch-select').value === '1010'")
        stage_txt = pg.eval_js("document.getElementById('stage-detail-card').innerText")
        assert "1.000000" in stage_txt

        # 5. runSweeps catch block preserves charts on failed API response + eigenkit_unitary UI error
        pg.ensure_visible("curves-chart")
        pg.click("#refresh-sweeps-btn")
        _wait_engine_ready(pg)
        assert pg.eval_js("Boolean(document.querySelector('#curves-chart svg'))") is True
        assert pg.eval_js("Boolean(document.querySelector('#delta-chart svg'))") is True
        assert pg.eval_js("Boolean(document.querySelector('#coherence-chart svg'))") is True

        # Temporarily intercept /api/sweeps in page fetch to simulate a failed server response
        pg.eval_js(
            "(() => {"
            "  window.__origFetch = window.fetch;"
            "  window.fetch = function (url, opts) {"
            "    if (String(url).includes('/api/sweeps')) {"
            "      return Promise.resolve(new Response(JSON.stringify({error: 'Injected sweep failure'}), {"
            "        status: 500, headers: {'Content-Type': 'application/json'}"
            "      }));"
            "    }"
            "    return window.__origFetch.apply(this, arguments);"
            "  };"
            "})()"
        )
        pg.click("#refresh-sweeps-btn")
        _wait_engine_ready(pg)
        pg.wait_for("!document.getElementById('validation-error').classList.contains('hidden')")
        recovered = pg.eval_js(
            "({"
            "  err: document.getElementById('validation-error').textContent,"
            "  hasCurves: Boolean(document.querySelector('#curves-chart svg')),"
            "  hasDelta: Boolean(document.querySelector('#delta-chart svg')),"
            "  hasCoh: Boolean(document.querySelector('#coherence-chart svg'))"
            "})"
        )
        assert "Injected sweep failure" in recovered["err"]
        assert recovered["hasCurves"] is True
        assert recovered["hasDelta"] is True
        assert recovered["hasCoh"] is True

        pg.eval_js("(() => { if (window.__origFetch) { window.fetch = window.__origFetch; delete window.__origFetch; } })()")
        pg.click("#refresh-sweeps-btn")
        _wait_engine_ready(pg)

        # eigenkit_unitary UI error check (if option enabled on this host)
        ek_enabled = pg.eval_js(
            "!document.querySelector('#backend-select option[value=\"eigenkit_unitary\"]')?.disabled"
        )
        if ek_enabled:
            pg.fill_input("p-input", "0.05")
            pg.select_option("backend-select", "eigenkit_unitary")
            pg.wait_for("!document.getElementById('validation-error').classList.contains('hidden')")
            ek_err = pg.eval_js("document.getElementById('validation-error').textContent")
            assert "eigenkit_unitary" in ek_err
            pg.select_option("backend-select", "numpy")
            _wait_engine_ready(pg)

        # 6. 390x844 mobile zero horizontal overflow across en/pt-br x five_qubit/dfs_2 x all 6 panels
        pg.set_viewport(390, 844, mobile=True)
        for lang in ("en", "pt-br"):
            pg.switch_language(lang)
            for cname in ("five_qubit", "dfs_2"):
                pg.select_option("code-select", cname)
                _wait_engine_ready(pg)
                for tid in ("stage-list", "bloch-canvas", "coherence-panel", "curves-chart", "theory-panel", "seminar-panel"):
                    pg.ensure_visible(tid)
                    assert pg.check_no_horizontal_overflow(), (
                        f"Mobile overflow for lang={lang} code={cname} panel={tid}: {pg.viewport_dimensions()}"
                    )
                assert not find_leak_tokens(pg.body_text())

        pg.assert_no_errors()
    finally:
        pg.close()


def test_t5_7_http_schema_math_and_cli_adversarial(
    live_server: ServerHandle, python_executable
) -> None:
    """Live server HTTP/schema/static/eigenkit + CLI invalid-port adversarial checks."""
    # 1. Infinity / -Infinity / NaN JSON constants -> 400
    for endpoint, raw_body in (
        ("/api/step", b'{"code":"bit_flip_3","error_mode":"coherent_collective","collective_phi":Infinity}'),
        ("/api/step", b'{"code":"bit_flip_3","error_mode":"coherent_collective","collective_phi":-Infinity}'),
        ("/api/simulate", b'{"code":"bit_flip_3","p":NaN}'),
        ("/api/sweeps", b'{"p_max":Infinity}'),
    ):
        st, _, body = http_request_json(live_server.base_url, endpoint, method="POST", raw_body=raw_body)
        assert st == 400
        assert isinstance(body, dict) and "error" in body

    # 2. Negative / non-int / oversized Content-Length -> 400 / 413
    parsed = urlsplit(live_server.base_url)
    for cl, expected_status in (("-10", 400), ("notanint", 400), (str(2_000_000), 413)):
        conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=5.0)
        try:
            conn.putrequest("POST", "/api/simulate")
            conn.putheader("Content-Type", "application/json")
            conn.putheader("Content-Length", cl)
            conn.endheaders()
            resp = conn.getresponse()
            assert resp.status == expected_status
            assert "error" in json.loads(resp.read().decode("utf-8"))
        finally:
            conn.close()

    # 3. /static/ path traversal & encoding edge cases
    for bad_static in ("/static/", "/static/.", "/static/%5c..%5capp.py", "/static/foo%00bar.js"):
        st, _, raw = http_request_raw(live_server.base_url, bad_static)
        assert st == 403
        assert "error" in json.loads(raw.decode("utf-8"))

    st_dbl, _, raw_dbl = http_request_raw(live_server.base_url, "/static/%252e%252e/app.py")
    assert st_dbl == 404
    assert "error" in json.loads(raw_dbl.decode("utf-8"))

    # 4. Negative seed -> 422 on /api/step, /api/simulate, /api/sweeps
    for endpoint, payload in (
        ("/api/step", {"code": "bit_flip_3", "error_mode": "pauli_string", "pauli_string": "III", "seed": -1}),
        ("/api/simulate", {"code": "bit_flip_3", "seed": -1}),
        ("/api/sweeps", {"seed": -1}),
    ):
        st, _, body = http_request_json(live_server.base_url, endpoint, method="POST", payload=payload)
        assert st == 422
        assert "error" in body

    # 5. Unsupported HTTP methods -> structured JSON 405 + HEAD empty body
    for method in ("PUT", "DELETE", "PATCH", "OPTIONS"):
        for route in ("/api/health", "/api/step"):
            st, hdrs, body = http_request_json(live_server.base_url, route, method=method)
            assert st == 405
            assert "application/json" in hdrs.get("content-type", "")
            assert isinstance(body, dict) and "error" in body

    st_h, hdrs_h, raw_h = http_request_raw(live_server.base_url, "/api/health", method="HEAD")
    assert st_h == 200
    assert raw_h == b""
    assert int(hdrs_h.get("content-length", "0")) > 0

    # 6. eigenkit_unitary live MC -> 501; p=0.0 cardinal_avg -> 200 parity
    _, _, health = http_request_json(live_server.base_url, "/api/health")
    if "eigenkit_unitary" in health.get("backends", []):
        st_501, _, body_501 = http_request_json(
            live_server.base_url,
            "/api/sweeps",
            method="POST",
            payload={
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
        assert st_501 == 501
        assert "error" in body_501

        st_ek, _, body_ek = http_request_json(
            live_server.base_url,
            "/api/simulate",
            method="POST",
            payload={
                "code": "five_qubit",
                "channel": "depolarizing",
                "p": 0.0,
                "num_shots": 6,
                "state_mode": "cardinal_avg",
                "backend": "eigenkit_unitary",
            },
        )
        st_np, _, body_np = http_request_json(
            live_server.base_url,
            "/api/simulate",
            method="POST",
            payload={
                "code": "five_qubit",
                "channel": "depolarizing",
                "p": 0.0,
                "num_shots": 6,
                "state_mode": "cardinal_avg",
                "backend": "numpy",
            },
        )
        assert st_ek == 200 and st_np == 200
        assert abs(float(body_ek["mean_fidelity"]) - float(body_np["mean_fidelity"])) <= PARITY_TOL

    # 7. CLI invalid ports -> exit code 1
    for bad_port in ("-1", "70000"):
        proc = subprocess.run(
            [str(python_executable), "-m", "qecc.cli", "serve", "--port", bad_port, "--no-browser"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        assert proc.returncode == 1


def test_t5_8_concurrent_sweeps_livemc_race_and_http_keepalive(
    live_server: ServerHandle, browser_session: BrowserSession
) -> None:
    """Verify concurrent runSweeps does not corrupt #mc-source-label and HTTP/1.1 keep-alive stays synced."""
    pg = browser_session.new_page()
    try:
        pg.set_viewport(1440, 900, mobile=False)
        pg.goto(live_server.base_url)
        pg.wait_for_testid("code-select")
        _wait_engine_ready(pg)

        pg.switch_language("en")
        pg.ensure_visible("curves-chart")
        _wait_engine_ready(pg)

        initial_label = pg.eval_js("document.getElementById('mc-source-label').textContent.trim()")
        assert "Cached 18k-shot" in initial_label

        # Intercept /api/sweeps: call #1 (live MC) is delayed; call #2 (cached refresh) fails with 500.
        pg.eval_js(
            "(() => {"
            "  window.__origFetch = window.fetch;"
            "  window.__sweepCallCount = 0;"
            "  window.fetch = function (url, opts) {"
            "    if (String(url).includes('/api/sweeps')) {"
            "      window.__sweepCallCount += 1;"
            "      const callNum = window.__sweepCallCount;"
            "      if (callNum === 1) {"
            "        const origArgs = arguments;"
            "        return new Promise((resolve, reject) => {"
            "          setTimeout(() => {"
            "            window.__origFetch.apply(window, origArgs).then(resolve, reject);"
            "          }, 500);"
            "        });"
            "      }"
            "      return Promise.resolve(new Response(JSON.stringify({error: 'Injected concurrent sweep failure'}), {"
            "        status: 500, headers: {'Content-Type': 'application/json'}"
            "      }));"
            "    }"
            "    return window.__origFetch.apply(this, arguments);"
            "  };"
            "})()"
        )

        pg.click("#run-live-mc-btn")
        pg.wait_for("window.__sweepCallCount >= 1")
        pg.click("#refresh-sweeps-btn")
        _wait_engine_ready(pg)

        pg.wait_for("!document.getElementById('validation-error').classList.contains('hidden')")
        race_state = pg.eval_js(
            "({"
            "  err: document.getElementById('validation-error').textContent.trim(),"
            "  mcLabel: document.getElementById('mc-source-label').textContent.trim(),"
            "  hasCurves: Boolean(document.querySelector('#curves-chart svg')),"
            "  hasDelta: Boolean(document.querySelector('#delta-chart svg')),"
            "  hasCoh: Boolean(document.querySelector('#coherence-chart svg'))"
            "})"
        )
        assert "Injected concurrent sweep failure" in race_state["err"]
        assert "Cached 18k-shot" in race_state["mcLabel"]
        assert "Live Monte Carlo Sweep" not in race_state["mcLabel"]
        assert race_state["hasCurves"] is True
        assert race_state["hasDelta"] is True
        assert race_state["hasCoh"] is True

        pg.switch_language("pt-br")
        pt_label = pg.eval_js("document.getElementById('mc-source-label').textContent.trim()")
        assert "Conjunto em cache de 18 mil shots" in pt_label
        assert "Varredura Monte Carlo ao Vivo" not in pt_label

        pg.eval_js(
            "(() => {"
            "  if (window.__origFetch) { window.fetch = window.__origFetch; delete window.__origFetch; }"
            "  delete window.__sweepCallCount;"
            "})()"
        )
        pg.switch_language("en")
        pg.click("#refresh-sweeps-btn")
        _wait_engine_ready(pg)
        pg.assert_no_errors()
    finally:
        pg.close()

    # Live-server persistent HTTP/1.1 verification for POST /api/health 405 + Allow: GET, HEAD
    parsed = urlsplit(live_server.base_url)
    conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=5.0)
    try:
        bad_body = b'{"bad":1}'
        conn.request(
            "POST",
            "/api/health",
            body=bad_body,
            headers={"Content-Type": "application/json", "Content-Length": str(len(bad_body))},
        )
        resp = conn.getresponse()
        raw = resp.read()
        assert resp.status == 405
        assert resp.getheader("Allow") == "GET, HEAD"
        assert resp.getheader("Connection") == "close"
        assert "error" in json.loads(raw.decode("utf-8"))

        conn.request("GET", "/api/health")
        fu = conn.getresponse()
        fu_raw = fu.read()
        assert fu.status == 200
        assert json.loads(fu_raw.decode("utf-8"))["status"] == "ok"
    finally:
        conn.close()
