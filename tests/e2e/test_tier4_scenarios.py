"""Tier 4 E2E tests: 6 realistic end-to-end application scenarios."""

from __future__ import annotations

import csv
import json
import math
import re
import subprocess

import numpy as np
import pytest
from scipy import stats

pytestmark = pytest.mark.tier4


def test_scenario_1_coherence_protection_rho01_preservation(
    qecc_backends, qecc_codes, qecc_channels, helpers
) -> None:
    backend = qecc_backends.get_backend("numpy")
    psi = helpers.canonical_states()["|+i>"]

    configs = [
        ("bit_flip_3", "bit_flip", 0.49275, 0.48, 0.45),
        ("phase_flip_3", "phase_flip", 0.49275, 0.48, 0.45),
        (
            "five_qubit",
            "depolarizing",
            0.5 * (1.0 - (4.0 / 3.0) * (1.0 - helpers.five_qubit_exact_pi(0.05))),
            0.475,
            0.46667,
        ),
    ]

    rho_in = helpers.ket_to_rho(psi)
    for code_name, ctype, expected_coh, lower_qecc, raw_coh in configs:
        code = qecc_codes.get_code(code_name)
        res0 = backend.simulate_cycle(code, psi, channel=None, p=0.0, num_shots=10, seed=4101)
        assert abs(res0.rho_out[0, 1] - rho_in[0, 1]) < 1e-10

        res = backend.simulate_cycle(
            code,
            psi,
            channel=qecc_channels.ChannelSpec(ctype, 0.05),
            p=0.05,
            num_shots=20_000,
            seed=4102,
        )
        coh = abs(res.rho_out[0, 1])
        assert abs(coh - expected_coh) < 0.015
        assert coh > lower_qecc > raw_coh


def test_scenario_2_state_tomography_reconstruction_of_decoded_qubits(
    qecc_backends, qecc_codes, qecc_channels, qecc_metrics, helpers
) -> None:
    backend = qecc_backends.get_backend("numpy")
    ketpi = helpers.canonical_states()["|+i>"]
    haar_state = qecc_metrics.InitialStateSampler().haar_random_states(1, seed=4201)[0]
    rng = np.random.default_rng(4202)

    cases = [
        ("bit_flip_3", "bit_flip", 0.08, ketpi),
        ("phase_flip_3", "phase_flip", 0.08, ketpi),
        ("five_qubit", "depolarizing", 0.06, haar_state),
    ]
    for code_name, ctype, p, psi in cases:
        code = qecc_codes.get_code(code_name)
        res = backend.simulate_cycle(
            code, psi, channel=qecc_channels.ChannelSpec(ctype, p), p=p, num_shots=15_000, seed=4203
        )
        rho_tomo = qecc_metrics.SingleQubitTomography.reconstruct(
            res.rho_out, shots_per_axis=20_000, rng=rng
        )
        assert rho_tomo.shape == (2, 2)
        assert np.allclose(rho_tomo, rho_tomo.conj().T, atol=1e-10)
        assert abs(float(np.real(np.trace(rho_tomo))) - 1.0) < 1e-10
        assert qecc_metrics.quantum_fidelity(res.rho_out, rho_tomo) > 0.995
        assert float(np.linalg.norm(res.rho_out - rho_tomo, ord="fro")) < 0.04
        f_tomo_vs_in = qecc_metrics.quantum_fidelity(psi, rho_tomo)
        assert abs(f_tomo_vs_in - res.mean_fidelity) < 0.025


def test_scenario_3_monte_carlo_gain_threshold_sweep_within_3sigma(
    qecc_backends, qecc_codes, qecc_channels, qecc_analysis, helpers
) -> None:
    backend = qecc_backends.get_backend("numpy")
    p_grid = [0.0, 0.02, 0.05, 0.08, 0.12, 0.14, 0.20, 0.30, 0.40, 0.50]
    n_shots = 10_000
    canonical = helpers.canonical_states()

    for code_name, ctype, psi in (
        ("bit_flip_3", "bit_flip", canonical["|0>"]),
        ("phase_flip_3", "phase_flip", canonical["|0>"]),
    ):
        code = qecc_codes.get_code(code_name)
        for p in p_grid:
            res = backend.simulate_cycle(
                code, psi, channel=qecc_channels.ChannelSpec(ctype, p), p=p, num_shots=n_shots, seed=4301
            )
            f_exact = float(qecc_analysis.analytical_fidelity(code_name, ctype, p, metric="entanglement"))
            tol = helpers.binom_sigma(f_exact, n_shots, z=4.5)
            assert abs(res.mean_fidelity - f_exact) <= tol
            if 0.0 < p < 0.5:
                assert res.mean_fidelity > 1.0 - p

    fq = qecc_codes.get_code("five_qubit")
    ketp = canonical["|+>"]
    for p in p_grid:
        res = backend.simulate_cycle(
            fq, ketp, channel=qecc_channels.ChannelSpec("depolarizing", p), p=p, num_shots=n_shots, seed=4302
        )
        f_exact = 1.0 - (2.0 / 3.0) * (1.0 - helpers.five_qubit_exact_pi(p))
        tol = helpers.binom_sigma(f_exact, n_shots, z=4.5)
        assert abs(res.mean_fidelity - f_exact) <= tol
        raw_state = 1.0 - (2.0 / 3.0) * p
        if p in (0.02, 0.05, 0.08, 0.12):
            assert res.mean_fidelity > raw_state
        if p in (0.20, 0.30, 0.40, 0.50):
            assert res.mean_fidelity < raw_state


def test_scenario_4_passive_dfs_vs_active_qecc_noise_regimes(
    qecc_backends, qecc_codes, qecc_channels, helpers
) -> None:
    backend = qecc_backends.get_backend("numpy")
    dfs = qecc_codes.get_code("dfs_2")
    pf = qecc_codes.get_code("phase_flip_3")
    ketpi = helpers.canonical_states()["|+i>"]
    n_shots = 10_000

    res_dfs_coll = backend.simulate_cycle(
        dfs, ketpi, channel=qecc_channels.ChannelSpec("collective_dephasing", 0.20), p=0.20, num_shots=n_shots, seed=404
    )
    res_pf_coll = backend.simulate_cycle(
        pf, ketpi, channel=qecc_channels.ChannelSpec("collective_dephasing", 0.20), p=0.20, num_shots=n_shots, seed=404
    )
    assert res_dfs_coll.mean_fidelity > 1.0 - 1e-10
    assert res_pf_coll.mean_fidelity < 0.84
    assert abs(res_pf_coll.mean_fidelity - 0.80) <= helpers.binom_sigma(0.80, n_shots, z=4.0)

    res_dfs_ind = backend.simulate_cycle(
        dfs, ketpi, channel=qecc_channels.ChannelSpec("phase_flip", 0.10), p=0.10, num_shots=n_shots, seed=404
    )
    res_pf_ind = backend.simulate_cycle(
        pf, ketpi, channel=qecc_channels.ChannelSpec("phase_flip", 0.10), p=0.10, num_shots=n_shots, seed=404
    )
    assert res_dfs_ind.mean_fidelity < 0.90
    assert abs(res_dfs_ind.mean_fidelity - 0.82) <= helpers.binom_sigma(0.82, n_shots, z=4.0)
    assert res_pf_ind.mean_fidelity > 0.95 > 0.90
    assert abs(res_pf_ind.mean_fidelity - 0.972) <= helpers.binom_sigma(0.972, n_shots, z=4.0)


def test_scenario_5_cpu_vs_rtx5080_cuda_parity_and_throughput(
    qecc_backends, qecc_codes, qecc_channels, helpers
) -> None:
    np_backend = qecc_backends.get_backend("numpy")
    cu_backend = qecc_backends.get_backend("torch_cuda")
    fq = qecc_codes.get_code("five_qubit")
    ketpi = helpers.canonical_states()["|+i>"]
    ch = qecc_channels.ChannelSpec("depolarizing", 0.10)

    # Warm-up CUDA JIT/allocator before timing
    cu_backend.simulate_cycle(fq, ketpi, channel=ch, p=0.10, num_shots=100, seed=4500)

    n_shots = 50_000
    res_np = np_backend.simulate_cycle(fq, ketpi, channel=ch, p=0.10, num_shots=n_shots, seed=4501)
    res_cu = cu_backend.simulate_cycle(fq, ketpi, channel=ch, p=0.10, num_shots=n_shots, seed=4502)

    expected = 1.0 - (2.0 / 3.0) * (1.0 - helpers.five_qubit_exact_pi(0.10))
    tol = helpers.binom_sigma(expected, n_shots, z=4.0)
    assert abs(res_np.mean_fidelity - expected) <= tol
    assert abs(res_cu.mean_fidelity - expected) <= tol

    comb_tol = 4.0 * math.sqrt(float(res_np.std_error) ** 2 + float(res_cu.std_error) ** 2 + 1e-12)
    assert abs(res_np.mean_fidelity - res_cu.mean_fidelity) <= comb_tol

    keys = [f"{i:04b}" for i in range(16)]
    c_np = [res_np.syndrome_counts.get(k, 0) for k in keys]
    c_cu = [res_cu.syndrome_counts.get(k, 0) for k in keys]
    _, p_val, _, _ = stats.chi2_contingency(np.array([c_np, c_cu]))
    assert p_val > 0.0002

    assert res_np.elapsed_seconds > 0.0
    assert res_cu.elapsed_seconds > 0.0
    assert res_cu.elapsed_seconds < res_np.elapsed_seconds


def test_scenario_6_full_seminar_and_verification_pipeline(
    repo_root, results_dir, docs_dir, scripts_dir, python_executable, qecc_analysis, helpers
) -> None:
    # 1. Analytical verification script gate
    verifier = scripts_dir / "verify_analytical_thresholds.py"
    assert verifier.is_file()
    proc = subprocess.run(
        [str(python_executable), str(verifier)],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    for tok in ("0.5", "0.1376", "0.1311"):
        assert tok in proc.stdout

    # 2. Sweep data & analytical agreement gate
    sweep_json = results_dir / "data" / "fidelity_sweeps.json"
    sweep_csv = results_dir / "data" / "fidelity_sweeps.csv"
    assert sweep_json.is_file() and sweep_csv.is_file()
    sweep_data = json.loads(sweep_json.read_text(encoding="utf-8"))
    assert "five_qubit" in helpers.flatten_json_text(sweep_data).lower()

    with sweep_csv.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) >= 15

    # Validate analytical consistency on available CSV columns if structured per-code
    cols = {k.lower(): k for k in rows[0].keys()}
    p_key = next((cols[c] for c in ("p", "error_rate", "p_error") if c in cols), None)
    assert p_key is not None
    if "code" in cols or "code_name" in cols:
        code_key = cols.get("code") or cols["code_name"]
        fid_key = next(
            (
                orig
                for low, orig in cols.items()
                if helpers.is_fidelity_metric_name(orig) and "raw" not in low and "analytical" not in low
            ),
            None,
        )
        ch_key = cols.get("channel") or cols.get("channel_type")
        if fid_key is not None and ch_key is not None:
            for r in rows:
                c_name = r[code_key].strip()
                ch_name = r[ch_key].strip()
                if c_name in {"bit_flip_3", "phase_flip_3", "five_qubit"} and ch_name in {
                    "bit_flip",
                    "phase_flip",
                    "depolarizing",
                }:
                    p_val = float(r[p_key])
                    f_sim = float(r[fid_key])
                    f_ana = float(qecc_analysis.analytical_fidelity(c_name, ch_name, p_val))
                    if abs(f_sim - f_ana) > 0.03:
                        # Allow state-fidelity conversion 1 - (2/3)*(1 - F_e) for depolarizing
                        f_ana_state = 1.0 - (2.0 / 3.0) * (1.0 - f_ana)
                        assert min(abs(f_sim - f_ana), abs(f_sim - f_ana_state)) <= 0.03
    else:
        def _wide_cols(code_tok: str) -> list[str]:
            matched: list[str] = []
            for low, orig in cols.items():
                if code_tok not in low:
                    continue
                if any(
                    bad in low
                    for bad in ("raw", "gain", "diff", "delta", "advantage", "sigma", "stderr", "variance", "shots", "seed", "time", "elapsed")
                ):
                    continue
                toks = set(low.replace("-", "_").split("_"))
                if toks & {"se", "ci", "err", "error", "var", "std", "stdev", "low", "high"}:
                    continue
                if code_tok == "five_qubit" and ("bit_flip" in low or "phase_flip" in low):
                    continue
                if code_tok == "bit_flip_3" and ("depolar" in low or "phase_flip" in low):
                    continue
                if code_tok == "phase_flip_3" and ("depolar" in low or "bit_flip" in low):
                    continue
                matched.append(orig)
            return matched

        wide_specs = [
            ("bit_flip_3", "bit_flip", _wide_cols("bit_flip_3")),
            ("phase_flip_3", "phase_flip", _wide_cols("phase_flip_3")),
            ("five_qubit", "depolarizing", _wide_cols("five_qubit")),
        ]
        assert sum(len(w[2]) for w in wide_specs) >= 1
        for r in rows:
            p_val = float(r[p_key])
            for c_name, ch_name, wcols in wide_specs:
                if not wcols:
                    continue
                f_ana = float(qecc_analysis.analytical_fidelity(c_name, ch_name, p_val))
                f_ana_state = 1.0 - (2.0 / 3.0) * (1.0 - f_ana) if ch_name == "depolarizing" else f_ana
                for col in wcols:
                    raw_cell = r.get(col)
                    if raw_cell is None or raw_cell.strip() == "":
                        continue
                    f_sim = float(raw_cell)
                    assert min(abs(f_sim - f_ana), abs(f_sim - f_ana_state)) <= 0.03

    # 3. Benchmark artifact gate
    bench_json = results_dir / "benchmarks" / "backend_benchmark.json"
    bench_md = results_dir / "benchmarks" / "backend_benchmark.md"
    assert bench_json.is_file() and bench_md.is_file()
    bench_text = helpers.flatten_json_text(json.loads(bench_json.read_text(encoding="utf-8"))).lower()
    assert "numpy" in bench_text and "torch_cuda" in bench_text

    # 4. Figure & manuscript link / formatting gate
    manuscript_path = docs_dir / "seminar_qecc.md"
    slides_path = docs_dir / "seminar_slides.md"
    manuscript_text = manuscript_path.read_text(encoding="utf-8")
    slides_text = slides_path.read_text(encoding="utf-8")
    for md_path, txt in ((manuscript_path, manuscript_text), (slides_path, slides_text)):
        assert helpers.has_balanced_fenced_code_blocks(txt)
        no_code = helpers.strip_escaped_dollars(
            helpers.strip_inline_code_spans(helpers.strip_fenced_code_blocks(txt))
        )
        assert no_code.count("$$") % 2 == 0
        for tgt in helpers.extract_markdown_image_targets(txt):
            resolved = (md_path.parent / tgt).resolve()
            if not resolved.exists():
                resolved = (repo_root / tgt).resolve()
            assert resolved.is_file() and resolved.stat().st_size > 1024

    # 5. 18-check rubric & evidence cross-check gate
    rubric_text = (docs_dir / "review_rubric.md").read_text(encoding="utf-8")
    for marker in ("TBD", "TODO", "FIXME", "XXX", "- [ ]"):
        assert marker not in rubric_text
    assert not helpers.has_rubric_fail_verdict(rubric_text)
    rubric_lines = rubric_text.splitlines()
    for rid in (
        "R-1.1", "R-1.2", "R-1.3",
        "R-2.1", "R-2.2", "R-2.3",
        "R-3.1", "R-3.2", "R-3.3", "R-3.4",
        "R-4.1", "R-4.2",
        "R-5.1",
        "R-6.1", "R-6.2", "R-6.3", "R-6.4",
        "R-7.1",
    ):
        assert any(rid in ln and "PASS" in ln for ln in rubric_lines)

    manuscript_low = manuscript_text.lower()
    assert "no-cloning" in manuscript_low or "no cloning" in manuscript_low
    assert "knill" in manuscript_low and "laflamme" in manuscript_low
    assert "willow" in manuscript_low and "288" in manuscript_text and "48" in manuscript_text
    for idx in range(16):
        assert f"{idx:04b}" in manuscript_text
