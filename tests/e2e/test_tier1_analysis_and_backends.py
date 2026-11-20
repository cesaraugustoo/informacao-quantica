"""Tier 1 E2E tests for Features F7-F8: analytical thresholds, sweeps, plots, backends & CUDA."""

from __future__ import annotations

import csv
import json
import math
import subprocess
from typing import Any

import matplotlib.image as mpimg
import numpy as np
import pytest

pytestmark = pytest.mark.tier1


def _collect_floats_by_key(
    obj: Any, key_substrings: tuple[str, ...], *, fidelity_only: bool = False, helpers: Any = None
) -> list[float]:
    vals: list[float] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            kl = str(k).lower()
            matched = (
                helpers.is_fidelity_metric_name(kl)
                if fidelity_only and helpers is not None
                else any(sub in kl for sub in key_substrings)
            )
            if matched:
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    vals.append(float(v))
                elif isinstance(v, list):
                    for item in v:
                        if isinstance(item, (int, float)) and not isinstance(item, bool):
                            vals.append(float(item))
            vals.extend(_collect_floats_by_key(v, key_substrings, fidelity_only=fidelity_only, helpers=helpers))
    elif isinstance(obj, list):
        for item in obj:
            vals.extend(_collect_floats_by_key(item, key_substrings, fidelity_only=fidelity_only, helpers=helpers))
    return vals


# ---------------------------------------------------------------------------
# Feature F7: AnalyticalThresholdsSweepsAndPlots
# ---------------------------------------------------------------------------


def test_t1_f7_analytical_gain_thresholds_exact(qecc_analysis) -> None:
    assert qecc_analysis.analytical_gain_threshold("bit_flip_3", "bit_flip") == pytest.approx(
        0.5, abs=1e-9
    )
    assert qecc_analysis.analytical_gain_threshold("phase_flip_3", "phase_flip") == pytest.approx(
        0.5, abs=1e-9
    )
    assert qecc_analysis.analytical_gain_threshold("five_qubit", "depolarizing") == pytest.approx(
        0.1376275643, abs=1e-7
    )


def test_t1_f7_analytical_fidelity_polynomials(qecc_analysis, helpers) -> None:
    p_grid = np.linspace(0.0, 0.5, 51)
    expected_3q = 1.0 - 3.0 * (p_grid**2) + 2.0 * (p_grid**3)
    expected_5q = np.array([helpers.five_qubit_exact_pi(float(p)) for p in p_grid], dtype=np.float64)

    f_bf = np.asarray(qecc_analysis.analytical_fidelity("bit_flip_3", "bit_flip", p_grid), dtype=float)
    f_pf = np.asarray(qecc_analysis.analytical_fidelity("phase_flip_3", "phase_flip", p_grid), dtype=float)
    f_5q = np.asarray(qecc_analysis.analytical_fidelity("five_qubit", "depolarizing", p_grid), dtype=float)

    assert np.allclose(f_bf, expected_3q, atol=1e-12)
    assert np.allclose(f_pf, expected_3q, atol=1e-12)
    assert np.allclose(f_5q, expected_5q, atol=1e-12)


def test_t1_f7_analytical_advantage_region_ordering(qecc_analysis) -> None:
    for p in (0.02, 0.05, 0.10):
        f_5q = float(qecc_analysis.analytical_fidelity("five_qubit", "depolarizing", p))
        assert f_5q > 1.0 - p

    for p in (0.05, 0.25, 0.45):
        f_bf = float(qecc_analysis.analytical_fidelity("bit_flip_3", "bit_flip", p))
        assert f_bf > 1.0 - p

    for p in (0.15, 0.25, 0.40):
        f_5q = float(qecc_analysis.analytical_fidelity("five_qubit", "depolarizing", p))
        assert f_5q < 1.0 - p


def test_t1_f7_sweep_data_artifacts_and_cli(
    repo_root, results_dir, scripts_dir, python_executable, helpers
) -> None:
    sweep_script = scripts_dir / "run_sweeps.py"
    assert sweep_script.is_file()
    proc_script = subprocess.run(
        [str(python_executable), str(sweep_script), "--help"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc_script.returncode == 0, proc_script.stderr

    proc_cli = subprocess.run(
        [str(python_executable), "-m", "qecc.cli", "--help"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc_cli.returncode == 0, proc_cli.stderr

    json_path = results_dir / "data" / "fidelity_sweeps.json"
    csv_path = results_dir / "data" / "fidelity_sweeps.csv"
    assert json_path.is_file() and json_path.stat().st_size > 100
    assert csv_path.is_file() and csv_path.stat().st_size > 100

    data = json.loads(json_path.read_text(encoding="utf-8"))
    flat_text = helpers.flatten_json_text(data).lower()
    for token in ("bit_flip_3", "phase_flip_3", "five_qubit", "raw"):
        assert token in flat_text

    f_vals = _collect_floats_by_key(data, ("fidelity", "f_"), fidelity_only=True, helpers=helpers)
    assert len(f_vals) >= 15
    for val in f_vals:
        assert math.isfinite(val) and -1e-9 <= val <= 1.0 + 1e-9

    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) >= 15
    header_lower = {k.lower() for k in rows[0].keys()}
    p_col = next((k for k in rows[0] if k.lower() in {"p", "error_rate", "p_error"}), None)
    assert p_col is not None, f"Missing p column in {sorted(header_lower)}"

    p_values = [float(r[p_col]) for r in rows]
    assert min(p_values) == pytest.approx(0.0, abs=1e-9)
    assert max(p_values) == pytest.approx(0.5, abs=1e-6)

    fid_cols = [k for k in rows[0] if helpers.is_fidelity_metric_name(k)]
    assert len(fid_cols) >= 1
    for r in rows:
        p_val = float(r[p_col])
        for fc in fid_cols:
            raw_cell = r[fc]
            if raw_cell is None or raw_cell.strip() == "":
                continue
            fv = float(raw_cell)
            assert math.isfinite(fv) and -1e-9 <= fv <= 1.0 + 1e-9
            if abs(p_val) < 1e-12:
                assert fv > 1.0 - 1e-10


def test_t1_f7_generated_figures_valid_and_nonempty(results_dir) -> None:
    fig_dir = results_dir / "figures"
    assert fig_dir.is_dir()
    png_files = sorted(fig_dir.glob("*.png"))
    pdf_files = sorted(fig_dir.glob("*.pdf"))
    assert len(png_files) + len(pdf_files) >= 2

    for png in png_files:
        assert png.stat().st_size > 10_000
        with png.open("rb") as fh:
            assert fh.read(8) == b"\x89PNG\r\n\x1a\n"
        img = mpimg.imread(png)
        assert float(np.std(img)) > 0.01

    for pdf in pdf_files:
        assert pdf.stat().st_size > 5_000
        with pdf.open("rb") as fh:
            assert fh.read(5) == b"%PDF-"


def test_t1_f7_monte_carlo_crosses_raw_at_threshold(
    qecc_codes, qecc_channels, numpy_backend, helpers
) -> None:
    bf = qecc_codes.get_code("bit_flip_3")
    ket0 = helpers.canonical_states()["|0>"]
    res_bf_15 = numpy_backend.simulate_cycle(
        bf, ket0, channel=qecc_channels.ChannelSpec("bit_flip", 0.15), p=0.15, num_shots=10_000, seed=701
    )
    exp_bf_15 = 1.0 - 3.0 * (0.15**2) + 2.0 * (0.15**3)
    assert res_bf_15.mean_fidelity > 0.85
    assert abs(res_bf_15.mean_fidelity - exp_bf_15) <= helpers.binom_sigma(exp_bf_15, 10_000, z=4.0)

    res_bf_45 = numpy_backend.simulate_cycle(
        bf, ket0, channel=qecc_channels.ChannelSpec("bit_flip", 0.45), p=0.45, num_shots=10_000, seed=702
    )
    exp_bf_45 = 1.0 - 3.0 * (0.45**2) + 2.0 * (0.45**3)
    assert res_bf_45.mean_fidelity > 0.55
    assert abs(res_bf_45.mean_fidelity - exp_bf_45) <= helpers.binom_sigma(exp_bf_45, 10_000, z=4.0)

    fq = qecc_codes.get_code("five_qubit")
    ketp = helpers.canonical_states()["|+>"]
    res_5q_05 = numpy_backend.simulate_cycle(
        fq, ketp, channel=qecc_channels.ChannelSpec("depolarizing", 0.05), p=0.05, num_shots=10_000, seed=703
    )
    exp_5q_05 = 1.0 - (2.0 / 3.0) * (1.0 - helpers.five_qubit_exact_pi(0.05))
    assert res_5q_05.mean_fidelity > 1.0 - (2.0 / 3.0) * 0.05
    assert abs(res_5q_05.mean_fidelity - exp_5q_05) <= helpers.binom_sigma(exp_5q_05, 10_000, z=4.0)

    res_5q_30 = numpy_backend.simulate_cycle(
        fq, ketp, channel=qecc_channels.ChannelSpec("depolarizing", 0.30), p=0.30, num_shots=10_000, seed=704
    )
    exp_5q_30 = 1.0 - (2.0 / 3.0) * (1.0 - helpers.five_qubit_exact_pi(0.30))
    assert res_5q_30.mean_fidelity < 1.0 - (2.0 / 3.0) * 0.30
    assert abs(res_5q_30.mean_fidelity - exp_5q_30) <= helpers.binom_sigma(exp_5q_30, 10_000, z=4.0)


# ---------------------------------------------------------------------------
# Feature F8: SwappableBackendsCUDAAndBenchmarks
# ---------------------------------------------------------------------------


def test_t1_f8_list_available_backends_includes_numpy_and_cuda(qecc_backends) -> None:
    backends = qecc_backends.list_available_backends()
    assert isinstance(backends, list)
    assert "numpy" in backends
    assert "torch_cuda" in backends


def test_t1_f8_numpy_and_cuda_zero_error_parity(
    qecc_backends, qecc_codes, helpers
) -> None:
    np_backend = qecc_backends.get_backend("numpy")
    cu_backend = qecc_backends.get_backend("torch_cuda")
    ket_plus_i = helpers.canonical_states()["|+i>"]

    for code_name in ("bit_flip_3", "phase_flip_3", "five_qubit", "dfs_2"):
        code = qecc_codes.get_code(code_name)
        res_np = np_backend.simulate_cycle(code, ket_plus_i, channel=None, p=0.0, num_shots=100, seed=801)
        res_cu = cu_backend.simulate_cycle(code, ket_plus_i, channel=None, p=0.0, num_shots=100, seed=801)
        assert res_np.mean_fidelity > 1.0 - 1e-10
        assert res_cu.mean_fidelity > 1.0 - 1e-10
        assert abs(res_np.mean_fidelity - res_cu.mean_fidelity) < 1e-10
        assert np.allclose(res_np.rho_out, res_cu.rho_out, atol=1e-10)


def test_t1_f8_numpy_and_cuda_deterministic_error_parity(
    qecc_backends, qecc_codes, helpers
) -> None:
    np_backend = qecc_backends.get_backend("numpy")
    cu_backend = qecc_backends.get_backend("torch_cuda")
    ket_plus_i = helpers.canonical_states()["|+i>"]

    cases = [
        ("five_qubit", ["YIIII"]),
        ("five_qubit", ["IIZII"]),
        ("bit_flip_3", ["IXI"]),
    ]
    for code_name, det_errs in cases:
        code = qecc_codes.get_code(code_name)
        res_np = np_backend.simulate_cycle(
            code, ket_plus_i, channel=None, p=0.0, num_shots=1, deterministic_errors=det_errs
        )
        res_cu = cu_backend.simulate_cycle(
            code, ket_plus_i, channel=None, p=0.0, num_shots=1, deterministic_errors=det_errs
        )
        assert res_np.syndrome_counts == res_cu.syndrome_counts
        assert res_np.mean_fidelity > 1.0 - 1e-10
        assert res_cu.mean_fidelity > 1.0 - 1e-10
        assert np.allclose(res_np.rho_out, res_cu.rho_out, atol=1e-10)


def test_t1_f8_eigenkit_evaluation_doc_and_unitary_adapter(
    docs_dir, scripts_dir, qecc_backends, qecc_codes, helpers
) -> None:
    eval_doc = docs_dir / "eigenkit_evaluation.md"
    assert eval_doc.is_file()
    text = eval_doc.read_text(encoding="utf-8")
    assert len(text) > 500
    text_lower = text.lower()
    assert "eigenkit" in text_lower
    assert any(tok in text_lower for tok in ("c_if", "mid-circuit", "measurement", "partial trace", "density matrix"))
    assert any(tok in text_lower for tok in ("rtx 5080", "sm_120", "cu126", "cu130", "cuda"))
    assert (scripts_dir / "probe_eigenkit.py").is_file()

    if "eigenkit_unitary" in qecc_backends.list_available_backends():
        ek_backend = qecc_backends.get_backend("eigenkit_unitary")
        np_backend = qecc_backends.get_backend("numpy")
        code = qecc_codes.get_code("bit_flip_3")
        psi = helpers.canonical_states()["|+i>"]
        res_ek = ek_backend.simulate_cycle(code, psi, channel=None, p=0.0, num_shots=10, seed=802)
        res_np = np_backend.simulate_cycle(code, psi, channel=None, p=0.0, num_shots=10, seed=802)
        assert res_ek.mean_fidelity > 1.0 - 1e-10
        assert np.allclose(res_ek.rho_out, res_np.rho_out, atol=1e-10)


def test_t1_f8_benchmark_artifacts_and_cli(
    repo_root, results_dir, scripts_dir, python_executable, helpers
) -> None:
    bench_script = scripts_dir / "run_benchmarks.py"
    assert bench_script.is_file()
    proc = subprocess.run(
        [str(python_executable), str(bench_script), "--help"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr

    json_path = results_dir / "benchmarks" / "backend_benchmark.json"
    md_path = results_dir / "benchmarks" / "backend_benchmark.md"
    assert json_path.is_file() and json_path.stat().st_size > 100
    assert md_path.is_file() and md_path.stat().st_size > 100

    data = json.loads(json_path.read_text(encoding="utf-8"))
    flat_text = helpers.flatten_json_text(data).lower()
    assert "numpy" in flat_text
    assert "torch_cuda" in flat_text
    for shot_tok in ("1000", "10000", "100000"):
        assert shot_tok in flat_text

    runtimes = _collect_floats_by_key(data, ("elapsed", "seconds", "time", "shots_per_sec", "speedup"))
    assert len(runtimes) >= 6
    assert all(math.isfinite(v) and v > 0.0 for v in runtimes)


def test_t1_f8_simulation_result_contract_fields(qecc_backends, qecc_codes, qecc_channels, helpers) -> None:
    code = qecc_codes.get_code("bit_flip_3")
    psi = helpers.canonical_states()["|+>"]
    channel = qecc_channels.ChannelSpec("bit_flip", 0.05)

    for bname in ("numpy", "torch_cuda"):
        backend = qecc_backends.get_backend(bname)
        res = backend.simulate_cycle(code, psi, channel=channel, p=0.05, num_shots=50, seed=803)
        assert res.code_name == code.name
        assert res.backend_name == bname
        assert res.p == pytest.approx(0.05, abs=1e-12)
        assert res.num_shots == 50
        assert len(res.fidelities) == 50
        assert sum(res.syndrome_counts.values()) == 50
        assert np.asarray(res.rho_out).shape == (2, 2)
        assert 0.0 <= float(res.mean_fidelity) <= 1.0
        assert float(res.std_error) >= 0.0
        assert float(res.elapsed_seconds) > 0.0
