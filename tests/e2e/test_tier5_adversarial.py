"""Tier 5 white-box adversarial & coverage-hardening E2E tests.

Integrates deduplicated adversarial specifications covering:
- qecc.cli JSON subcommands, isolated --data-dir/--figures-dir/--output-dir flags,
  deterministic --p 0.0 semantics, and clean error reporting without tracebacks
- Exhaustive p=0 and all 21 single-qubit deterministic errors (+ heterogeneous batches
  and weight-2 uncorrectable errors) across numpy, torch_cuda, and eigenkit_unitary
- TorchCUDABackend PRNG reproducibility, max_batch_bytes chunking parity,
  torch.device input handling, and 6-seed statistical independence
- EigenkitUnitaryAdapter capability rejections and run_unitary_ops renormalization parity
- Strict 4.5*std_error audit of all 231 stored sweep CSV/JSON rows, gain thresholds,
  and coherence_at_p005
- Strict backend_benchmark.{json,md} parity and RTX 5080 sm_120 audit
- All 16 (code, channel) analytical_fidelity(..., metric="state") vs torch_cuda parity
- Substantive docs/review_rubric.md evidence & cross-file anchor resolution, plus
  docs/seminar_qecc.md 32-term ket signs, 16-syndrome table, 1024-term enumerator,
  and section 6.4 table sync
- White-box edge cases: partial_trace_batch_to_qubit0, SingleQubitTomography PSD
  projection & is_statevector_batch, negative Knill-Laflamme checks, and
  coherent_collective_dephasing_ops across all 3 backends
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.tier5


def _run_module(
    python_executable,
    repo_root,
    args: list[str],
    *,
    timeout: int = 60,
) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    return subprocess.run(
        [str(python_executable), "-m", "qecc.cli", *args],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# 1. qecc.cli Subprocess JSON Integration, Output Isolation & Error Paths
# ---------------------------------------------------------------------------


def test_t5_cli_info_verify_simulate_benchmark_json_integration(
    repo_root, scripts_dir, python_executable, qecc_analysis
) -> None:
    p_info = _run_module(python_executable, repo_root, ["info", "--json"])
    assert p_info.returncode == 0, p_info.stderr
    info_data = json.loads(p_info.stdout)
    assert set(info_data.keys()) == {
        "codes",
        "backends",
        "channels",
        "nontrivial_gain_thresholds",
    }
    assert {"bit_flip_3", "phase_flip_3", "five_qubit", "dfs_2"}.issubset(info_data["codes"])
    assert {"numpy", "torch_cuda", "eigenkit_unitary"}.issubset(info_data["backends"])

    for pair, th_val in info_data["nontrivial_gain_thresholds"].items():
        c_name, ch_name = pair.split("/", 1)
        assert th_val == pytest.approx(
            qecc_analysis.analytical_gain_threshold(c_name, ch_name), abs=1e-12
        )

    p_verify_cli = _run_module(python_executable, repo_root, ["verify", "--json"])
    assert p_verify_cli.returncode == 0, p_verify_cli.stderr
    verify_cli = json.loads(p_verify_cli.stdout)

    p_verify_script = subprocess.run(
        [str(python_executable), str(scripts_dir / "verify_analytical_thresholds.py"), "--json"],
        cwd=repo_root,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert p_verify_script.returncode == 0, p_verify_script.stderr
    assert verify_cli == json.loads(p_verify_script.stdout)

    p_sim = _run_module(
        python_executable,
        repo_root,
        [
            "simulate",
            "--code",
            "five_qubit",
            "--channel",
            "depolarizing",
            "--p",
            "0.05",
            "--shots",
            "2000",
            "--backend",
            "torch_cuda",
            "--state",
            "+i",
            "--seed",
            "42",
            "--json",
        ],
    )
    assert p_sim.returncode == 0, p_sim.stderr
    sim_data = json.loads(p_sim.stdout)
    assert sim_data["code"] == "five_qubit"
    assert sim_data["backend"] == "torch_cuda"
    assert sim_data["p"] == pytest.approx(0.05, abs=1e-15)
    assert sim_data["num_shots"] == 2000
    assert sum(sim_data["syndrome_counts"].values()) == 2000
    assert sim_data["analytical_state_fidelity"] == pytest.approx(
        float(qecc_analysis.analytical_fidelity("five_qubit", "depolarizing", 0.05, metric="state")),
        abs=1e-12,
    )
    assert abs(sim_data["mean_fidelity"] - sim_data["analytical_state_fidelity"]) < 0.02

    # Deterministic mode WITHOUT explicit --p must resolve p=0.0 (never stochastic default 0.05)
    p_det = _run_module(
        python_executable,
        repo_root,
        [
            "simulate",
            "--code",
            "five_qubit",
            "--deterministic-error",
            "YIIII",
            "--shots",
            "4",
            "--backend",
            "eigenkit_unitary",
            "--state",
            "+i",
            "--json",
        ],
    )
    assert p_det.returncode == 0, p_det.stderr
    det_data = json.loads(p_det.stdout)
    assert det_data["p"] == 0.0
    assert det_data["channel"] is None
    assert det_data["syndrome_counts"] == {"1011": 4}
    assert det_data["mean_fidelity"] > 1.0 - 1e-10
    assert det_data["analytical_entanglement_fidelity"] is None
    assert det_data["analytical_state_fidelity"] is None

    p_bench = _run_module(
        python_executable,
        repo_root,
        ["benchmark", "--quick", "--shots", "100", "--codes", "bit_flip_3", "five_qubit", "--json"],
    )
    assert p_bench.returncode == 0, p_bench.stderr
    bench_rows = json.loads(p_bench.stdout)
    assert len(bench_rows) == 4
    assert {(r["backend"], r["code"]) for r in bench_rows} == {
        ("numpy", "bit_flip_3"),
        ("numpy", "five_qubit"),
        ("torch_cuda", "bit_flip_3"),
        ("torch_cuda", "five_qubit"),
    }
    assert all(r["elapsed_seconds"] > 0.0 and 0.0 <= r["mean_fidelity"] <= 1.0 for r in bench_rows)

    # Verify CLI help exposes isolation flags on sweep and benchmark
    p_sw_help = _run_module(python_executable, repo_root, ["sweep", "--help"])
    assert p_sw_help.returncode == 0
    assert "--data-dir" in p_sw_help.stdout
    assert "--figures-dir" in p_sw_help.stdout
    assert "--skip-figures" in p_sw_help.stdout

    p_bm_help = _run_module(python_executable, repo_root, ["benchmark", "--help"])
    assert p_bm_help.returncode == 0
    assert "--output-dir" in p_bm_help.stdout


def test_t5_cli_negative_errors_and_isolated_cli_sweep(
    repo_root, results_dir, python_executable, tmp_path
) -> None:
    expected_errors = [
        (["simulate", "--backend", "eigenkit_unitary", "--p", "0.05", "--json"], "NotImplementedError"),
        (["simulate", "--deterministic-error", "YIIII", "--p", "0.1", "--json"], "ValueError"),
        (["simulate", "--p", "-0.1"], "ValueError"),
        (["simulate", "--shots", "0"], "ValueError"),
        (["simulate", "--deterministic-error", "INVALID"], "ValueError"),
        (["benchmark", "--quick", "--shots", "0"], "ValueError"),
    ]
    for bad_args, err_name in expected_errors:
        proc = _run_module(python_executable, repo_root, bad_args)
        assert proc.returncode == 1, f"Expected exit code 1 for {bad_args}, got {proc.returncode}"
        assert err_name in proc.stderr, f"Expected {err_name!r} in stderr for {bad_args}: {proc.stderr!r}"
        assert "Traceback" not in proc.stderr, f"Unexpected traceback in stderr for {bad_args}: {proc.stderr!r}"

    # Verify qecc.cli sweep --data-dir/--figures-dir/--skip-figures isolates outputs from repo results/
    repo_csv = results_dir / "data" / "fidelity_sweeps.csv"
    repo_json = results_dir / "data" / "fidelity_sweeps.json"
    before_hashes = (_sha256(repo_csv), _sha256(repo_json))

    data_dir = tmp_path / "data"
    figs_dir = tmp_path / "figs"
    proc_sw = _run_module(
        python_executable,
        repo_root,
        [
            "sweep",
            "--backend",
            "torch_cuda",
            "--shots",
            "100",
            "--num-p",
            "15",
            "--data-dir",
            str(data_dir),
            "--figures-dir",
            str(figs_dir),
            "--skip-figures",
        ],
        timeout=90,
    )
    assert proc_sw.returncode == 0, proc_sw.stderr
    summary = json.loads(proc_sw.stdout)
    assert summary["num_records"] == 11 * 15
    assert (data_dir / "fidelity_sweeps.csv").is_file()
    assert (data_dir / "fidelity_sweeps.json").is_file()
    assert not figs_dir.exists() or not any(figs_dir.iterdir())
    assert (_sha256(repo_csv), _sha256(repo_json)) == before_hashes


# ---------------------------------------------------------------------------
# 2. Exhaustive CUDA & Eigenkit Single-Qubit Error & Multi-State Parity
# ---------------------------------------------------------------------------


def test_t5_exhaustive_zero_error_multi_state_three_backend_parity(
    qecc_backends, qecc_codes, qecc_metrics
) -> None:
    np_b = qecc_backends.get_backend("numpy")
    cu_b = qecc_backends.get_backend("torch_cuda")
    ek_b = qecc_backends.get_backend("eigenkit_unitary")

    sampler = qecc_metrics.InitialStateSampler()
    states = list(sampler.cardinal_states().values()) + list(sampler.haar_random_states(4, seed=5001))

    for cname in ("bit_flip_3", "phase_flip_3", "five_qubit", "dfs_2"):
        code = qecc_codes.get_code(cname)
        expected_syn = "" if code.n_ancilla == 0 else "0" * code.n_ancilla
        for psi in states:
            r_np = np_b.simulate_cycle(code, psi, channel=None, p=0.0, num_shots=3, seed=5001)
            r_cu = cu_b.simulate_cycle(code, psi, channel=None, p=0.0, num_shots=3, seed=5001)
            r_ek = ek_b.simulate_cycle(code, psi, channel=None, p=0.0, num_shots=3, seed=5001)
            for res in (r_np, r_cu, r_ek):
                assert res.mean_fidelity > 1.0 - 1e-10
                assert res.syndrome_counts == {expected_syn: 3}
            assert np.allclose(r_cu.rho_out, r_np.rho_out, atol=1e-10)
            assert np.allclose(r_ek.rho_out, r_np.rho_out, atol=1e-10)


def test_t5_exhaustive_all_21_single_qubit_errors_and_batch_parity(
    qecc_backends, qecc_codes, qecc_channels, qecc_metrics
) -> None:
    np_b = qecc_backends.get_backend("numpy")
    cu_b = qecc_backends.get_backend("torch_cuda")
    ek_b = qecc_backends.get_backend("eigenkit_unitary")

    sampler = qecc_metrics.InitialStateSampler()
    cards = sampler.cardinal_states()
    haars = sampler.haar_random_states(2, seed=5002)
    probe_states = [cards["0"], cards["+"], cards["+i"], haars[0], haars[1]]

    injector = qecc_channels.DeterministicErrorInjector
    exhaustive_plan = [
        ("bit_flip_3", ["III"] + [injector.from_single_qubit(q, "X", 3) for q in range(3)]),
        ("phase_flip_3", ["III"] + [injector.from_single_qubit(q, "Z", 3) for q in range(3)]),
        (
            "five_qubit",
            ["IIIII"]
            + [injector.from_single_qubit(q, op, 5) for q in range(5) for op in ("X", "Y", "Z")],
        ),
        ("dfs_2", ["II", "ZZ"]),
    ]

    for cname, err_list in exhaustive_plan:
        code = qecc_codes.get_code(cname)
        for err in err_list:
            for psi in probe_states:
                r_np = np_b.simulate_cycle(
                    code, psi, channel=None, p=0.0, num_shots=2, deterministic_errors=err
                )
                r_cu = cu_b.simulate_cycle(
                    code, psi, channel=None, p=0.0, num_shots=2, deterministic_errors=err
                )
                r_ek = ek_b.simulate_cycle(
                    code, psi, channel=None, p=0.0, num_shots=2, deterministic_errors=err
                )
                for res in (r_np, r_cu, r_ek):
                    assert res.mean_fidelity > 1.0 - 1e-10
                assert r_np.syndrome_counts == r_cu.syndrome_counts == r_ek.syndrome_counts
                assert np.allclose(r_cu.rho_out, r_np.rho_out, atol=1e-10)
                assert np.allclose(r_ek.rho_out, r_np.rho_out, atol=1e-10)

    # Verify parity on uncorrectable weight-2 errors and heterogeneous batch deterministic errors
    fq = qecc_codes.get_code("five_qubit")
    for bad_err in ("XXIII", "ZZIII", "XZIII"):
        r_np = np_b.simulate_cycle(fq, haars[0], channel=None, p=0.0, num_shots=1, deterministic_errors=bad_err)
        r_cu = cu_b.simulate_cycle(fq, haars[0], channel=None, p=0.0, num_shots=1, deterministic_errors=bad_err)
        r_ek = ek_b.simulate_cycle(fq, haars[0], channel=None, p=0.0, num_shots=1, deterministic_errors=bad_err)
        assert r_np.syndrome_counts == r_cu.syndrome_counts == r_ek.syndrome_counts
        assert np.allclose(r_cu.rho_out, r_np.rho_out, atol=1e-10)
        assert np.allclose(r_ek.rho_out, r_np.rho_out, atol=1e-10)

    det_batch = ["XIIII", "IYIII", "IIZII", "IIIYI", "IIIIX", "XXIII"]
    b_np = np_b.simulate_cycle(fq, haars[1], None, 0.0, len(det_batch), deterministic_errors=det_batch)
    b_cu = cu_b.simulate_cycle(fq, haars[1], None, 0.0, len(det_batch), deterministic_errors=det_batch)
    b_ek = ek_b.simulate_cycle(fq, haars[1], None, 0.0, len(det_batch), deterministic_errors=det_batch)
    assert b_np.syndromes == b_cu.syndromes == b_ek.syndromes
    assert np.allclose(b_np.rho_out, b_cu.rho_out, atol=1e-12)
    assert np.allclose(b_np.rho_out, b_ek.rho_out, atol=1e-12)


# ---------------------------------------------------------------------------
# 3. GPU PRNG Reproducibility, Chunking, torch.device & Seed Independence
# ---------------------------------------------------------------------------


def test_t5_torch_cuda_prng_reproducibility_chunking_device_and_independence(
    qecc_backends, qecc_codes, qecc_channels, helpers
) -> None:
    import torch

    cu_b = qecc_backends.get_backend("torch_cuda")
    code = qecc_codes.get_code("five_qubit")
    psi = helpers.canonical_states()["|+i>"]
    ch = qecc_channels.ChannelSpec("depolarizing", 0.12)

    r1 = cu_b.simulate_cycle(code, psi, channel=ch, p=0.12, num_shots=3000, seed=5301)
    r2 = cu_b.simulate_cycle(code, psi, channel=ch, p=0.12, num_shots=3000, seed=5301)
    assert r1.syndromes == r2.syndromes
    assert r1.syndrome_counts == r2.syndrome_counts
    assert np.array_equal(r1.fidelities, r2.fidelities)
    assert np.array_equal(r1.rho_out, r2.rho_out)

    # Chunked GPU execution (both 1 MiB and strict 16 KiB -> chunk_size=2) must match bit-for-bit
    # Also verify TorchCUDABackend accepts torch.device objects directly
    cu_chunked_1m = qecc_backends.TorchCUDABackend(
        device=torch.device("cuda"), max_batch_bytes=1024 * 1024, warmup=False
    )
    r_chunk = cu_chunked_1m.simulate_cycle(code, psi, channel=ch, p=0.12, num_shots=3000, seed=5301)
    assert r_chunk.syndromes == r1.syndromes
    assert np.allclose(r_chunk.fidelities, r1.fidelities, atol=1e-13)
    assert np.allclose(r_chunk.rho_out, r1.rho_out, atol=1e-13)

    cu_chunked_16k = qecc_backends.TorchCUDABackend(
        device=torch.device("cuda:0"), max_batch_bytes=16_384, warmup=False
    )
    r1_small = cu_b.simulate_cycle(code, psi, channel=ch, p=0.12, num_shots=20, seed=5302)
    r_small_chunk = cu_chunked_16k.simulate_cycle(code, psi, channel=ch, p=0.12, num_shots=20, seed=5302)
    assert r_small_chunk.syndromes == r1_small.syndromes
    assert np.allclose(r_small_chunk.rho_out, r1_small.rho_out, atol=1e-14)

    # Distinct seeds produce distinct, uncorrelated shot streams
    streams: list[np.ndarray] = []
    syn_seqs: set[tuple[str, ...]] = set()
    for seed in range(5400, 5406):
        rs = cu_b.simulate_cycle(code, psi, channel=ch, p=0.12, num_shots=3000, seed=seed)
        syn_seqs.add(tuple(rs.syndromes))
        streams.append(rs.fidelities)
    assert len(syn_seqs) == 6
    corr_bound = 4.5 / math.sqrt(3000)
    for i in range(len(streams)):
        for j in range(i + 1, len(streams)):
            corr = abs(float(np.corrcoef(streams[i], streams[j])[0, 1]))
            assert corr < corr_bound


# ---------------------------------------------------------------------------
# 4. EigenkitUnitaryAdapter Clean Rejections & Renormalization Parity
# ---------------------------------------------------------------------------


def test_t5_eigenkit_adapter_rejections_and_renormalization_parity(
    qecc_backends, qecc_codes, helpers
) -> None:
    ek_b = qecc_backends.get_backend("eigenkit_unitary")
    code = qecc_codes.get_code("five_qubit")
    psi = helpers.canonical_states()["|+i>"]

    # Any p > 0 stochastic channel must raise NotImplementedError (even tiny p)
    for p_val in (1e-12, 0.05, 1.0):
        with pytest.raises(NotImplementedError):
            ek_b.simulate_cycle(code, psi, channel="depolarizing", p=p_val, num_shots=4, seed=1)

    # Invalid gates in translate_ops_to_eigenkit
    bad_op = SimpleNamespace(name="TOFFOLI", targets=(0,), controls=(1, 2), params=())
    with pytest.raises((TypeError, ValueError)):
        qecc_backends.translate_ops_to_eigenkit([bad_op], 3)

    # Out-of-bounds qubit indices
    with pytest.raises(ValueError):
        qecc_backends.translate_ops_to_eigenkit([qecc_backends.GateOp("X", targets=(3,))], 3)

    # Invalid state dimensions, non-finite values, and unnormalized states
    for bad_state in (
        np.array([1.0], dtype=np.complex128),
        np.array([1.0, 0.0, 0.0], dtype=np.complex128),
        np.eye(2, dtype=np.complex128),
        np.array([1.0, 1.0], dtype=np.complex128),
        np.array([np.nan, 0.0], dtype=np.complex128),
        np.array([np.inf, 0.0], dtype=np.complex128),
    ):
        with pytest.raises(ValueError):
            ek_b.run_unitary_ops([qecc_backends.GateOp("X", targets=(0,))], 1, bad_state)

    # Near-unit state within 1e-6 tolerance must be renormalized before execution (parity with base.py)
    near_unit = np.array([1.0 + 4e-7, 0.0], dtype=np.complex128)
    sv_out = ek_b.run_unitary_ops([qecc_backends.GateOp("X", targets=(0,))], 1, near_unit)
    assert abs(float(np.linalg.norm(sv_out)) - 1.0) < 1e-12
    assert np.allclose(sv_out, np.array([0.0, 1.0], dtype=np.complex128), atol=1e-12)

    # Invalid provider names must raise cleanly
    for bad_provider in ("nonexistent_provider", "cuda_quantum"):
        with pytest.raises((KeyError, ValueError, RuntimeError)):
            qecc_backends.EigenkitUnitaryAdapter(provider=bad_provider)


# ---------------------------------------------------------------------------
# 5. Strict 4.5-Sigma Sweep Audit, Benchmark Parity & Analytical State Parity
# ---------------------------------------------------------------------------


def test_t5_sweep_csv_and_json_strict_4_5_sigma_audit(results_dir, qecc_analysis) -> None:
    csv_path = results_dir / "data" / "fidelity_sweeps.csv"
    json_path = results_dir / "data" / "fidelity_sweeps.json"
    assert csv_path.is_file() and json_path.is_file()

    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        csv_rows = list(csv.DictReader(fh))
    jdata = json.loads(json_path.read_text(encoding="utf-8"))
    j_rows = jdata["records"]

    assert len(csv_rows) == len(j_rows) == 231

    for idx, (r, jr) in enumerate(zip(csv_rows, j_rows, strict=True)):
        code = r["code"].strip()
        channel = r["channel"].strip()
        p = float(r["p"])
        mean_f = float(r["mean_fidelity"])
        se = float(r["std_error"])
        state_mode = r["state_mode"].strip()

        assert code == jr["code"] and channel == jr["channel"]
        assert abs(p - float(jr["p"])) < 1e-15
        assert abs(mean_f - float(jr["mean_fidelity"])) < 1e-15
        assert abs(se - float(jr["std_error"])) < 1e-15

        exp_ent = float(qecc_analysis.analytical_fidelity(code, channel, p, metric="entanglement"))
        exp_st = float(qecc_analysis.analytical_fidelity(code, channel, p, metric="state"))
        exp_raw_ent = float(qecc_analysis.analytical_fidelity("raw", channel, p, metric="entanglement"))
        exp_raw_st = float(qecc_analysis.analytical_fidelity("raw", channel, p, metric="state"))
        assert abs(float(r["analytical_entanglement_fidelity"]) - exp_ent) < 1e-12
        assert abs(float(r["analytical_state_fidelity"]) - exp_st) < 1e-12
        assert abs(float(r["raw_entanglement_fidelity"]) - exp_raw_ent) < 1e-12
        assert abs(float(r["raw_state_fidelity"]) - exp_raw_st) < 1e-12

        if code == "raw":
            expected = float(qecc_analysis.analytical_fidelity("raw", channel, p, metric=state_mode))
            assert se == 0.0
            assert abs(mean_f - expected) <= 1e-12, f"Raw row {idx} mismatch"
        else:
            metric = "state" if state_mode == "cardinal_avg" else "entanglement"
            expected = float(qecc_analysis.analytical_fidelity(code, channel, p, metric=metric))
            if p == 0.0 or (code == "dfs_2" and channel == "collective_dephasing"):
                assert abs(mean_f - expected) <= 1e-10, f"Deterministic row {idx} ({code}/{channel} p={p}) mismatch"
            else:
                tol = max(4.5 * se, 1e-10)
                assert abs(mean_f - expected) <= tol, (
                    f"Row {idx} ({code}/{channel}, p={p}) outlier: "
                    f"mean={mean_f}, expected={expected}, se={se}, z={abs(mean_f - expected) / max(se, 1e-15):.3f}"
                )

    for key, val in jdata["gain_thresholds"].items():
        code_name, ch_name = key.split("/", 1)
        live = float(qecc_analysis.analytical_gain_threshold(code_name, ch_name))
        assert abs(float(val) - live) <= 1e-12, f"Threshold mismatch for {key}: {val} vs {live}"

    se_coh_bound = 0.5 / math.sqrt(float(jdata["metadata"]["num_shots"]))
    for row in jdata["coherence_at_p005"]:
        f_ent = float(qecc_analysis.analytical_fidelity(row["code"], row["channel"], row["p"], metric="entanglement"))
        if row["channel"] == "depolarizing":
            ana = 0.5 * abs(1.0 - (4.0 / 3.0) * (1.0 - f_ent))
            raw = 0.5 * (1.0 - (4.0 / 3.0) * row["p"])
        elif row["channel"] == "collective_dephasing":
            ana = 0.5 * f_ent
            raw = 0.5 * abs(1.0 - 2.0 * row["p"])
        else:
            ana = 0.5 * abs(2.0 * f_ent - 1.0)
            raw = 0.5 * abs(1.0 - 2.0 * row["p"])
        assert abs(float(row["abs_rho01_analytical"]) - ana) <= 1e-12
        assert abs(float(row["abs_rho01_raw"]) - raw) <= 1e-12
        assert abs(float(row["abs_rho01_sim"]) - ana) <= 4.5 * se_coh_bound


def test_t5_backend_benchmark_strict_parity_and_md_sync(results_dir) -> None:
    bench_json = results_dir / "benchmarks" / "backend_benchmark.json"
    bench_md = results_dir / "benchmarks" / "backend_benchmark.md"
    assert bench_json.is_file() and bench_md.is_file()

    data = json.loads(bench_json.read_text(encoding="utf-8"))
    meta = data["metadata"]
    assert meta["cuda_available"] is True
    assert "RTX 5080" in meta["gpu_name"]
    assert meta["sm_120_supported"] is True
    assert meta["eigenkit_stochastic_supported"] is False

    assert len(data["comparisons"]) == 9
    for comp in data["comparisons"]:
        assert comp["parity_passed"] is True, f"Parity failed for {comp}"
        assert comp["syndrome_p_value"] > 1e-4

    up = data["unitary_parity"]
    assert up["all_passed"] is True
    assert up["stochastic_supported"] is False
    assert len(up["checks"]) == 6
    for chk in up["checks"]:
        assert chk["passed"] is True
        assert chk["syndrome_counts_match"] is True
        assert float(chk["max_rho_diff"]) < 1e-10

    md_text = bench_md.read_text(encoding="utf-8")
    for comp in data["comparisons"]:
        assert f"{comp['cuda_speedup']:.2f}x" in md_text
        assert f"{comp['numpy_median_seconds']:.6f}" in md_text
        assert f"{comp['torch_cuda_median_seconds']:.6f}" in md_text


def test_t5_all_16_code_channel_analytical_state_fidelity_parity(
    qecc_backends, qecc_codes, qecc_analysis, qecc_metrics
) -> None:
    cu_b = qecc_backends.get_backend("torch_cuda")
    cards = list(qecc_metrics.InitialStateSampler().cardinal_states().values())
    p_val = 0.10

    for cname in ("bit_flip_3", "phase_flip_3", "five_qubit", "dfs_2"):
        code = qecc_codes.get_code(cname)
        for ch_name in qecc_analysis.VALID_CHANNELS:
            f_cards = [
                cu_b.simulate_cycle(code, psi, channel=ch_name, p=p_val, num_shots=3000, seed=5500 + idx).mean_fidelity
                for idx, psi in enumerate(cards)
            ]
            mc_state = float(np.mean(f_cards))
            ana_state = float(qecc_analysis.analytical_fidelity(cname, ch_name, p_val, metric="state"))
            assert abs(mc_state - ana_state) < 0.015


# ---------------------------------------------------------------------------
# 6. Substantive Rubric Evidence, Cross-File Anchors & Seminar Math Sync
# ---------------------------------------------------------------------------


def test_t5_review_rubric_substantive_evidence_and_anchor_validity(docs_dir, helpers) -> None:
    rubric_path = docs_dir / "review_rubric.md"
    rubric_text = rubric_path.read_text(encoding="utf-8")

    expected_ids = {
        "R-1.1", "R-1.2", "R-1.3",
        "R-2.1", "R-2.2", "R-2.3",
        "R-3.1", "R-3.2", "R-3.3", "R-3.4",
        "R-4.1", "R-4.2",
        "R-5.1",
        "R-6.1", "R-6.2", "R-6.3", "R-6.4",
        "R-7.1",
    }

    matched_ids: set[str] = set()
    for line in rubric_text.splitlines():
        if line.startswith("| `R-"):
            assert len(helpers.split_markdown_table_row(line)) == 4
            cells = [c.strip() for c in line.strip("|").split("|")]
            assert len(cells) == 4
            rid, _criterion, verdict, evidence = cells
            matched_ids.add(rid.strip("`"))
            assert "PASS" in verdict
            assert len(evidence) >= 50, f"Evidence cell too thin for {rid}: {evidence!r}"
            assert re.search(r"(\.md|\.py|\.json|\.png|[0-9]+)", evidence), (
                f"Missing concrete artifact/number in {rid}"
            )
            for m in re.finditer(r"\[[^\]]+\]\(([^)\s]+)\)", evidence):
                tgt = m.group(1).strip()
                fpart, _, frag = tgt.partition("#")
                fpath = (rubric_path.parent / fpart).resolve()
                assert fpath.exists(), f"Missing link target {tgt} in {rid}"
                if frag and fpath.suffix == ".md":
                    no_fences = helpers.strip_fenced_code_blocks(fpath.read_text(encoding="utf-8"))
                    headings = re.findall(r"(?m)^#{1,6}\s+(.+)$", no_fences)
                    slugs: set[str] = set()
                    for h in headings:
                        s = helpers.slugify_heading(h)
                        slugs.add(s)
                        slugs.add(re.sub(r"-+", "-", s).strip("-"))
                    assert frag.lower() in slugs, f"Broken cross-file anchor {tgt} in {rid}"

    assert matched_ids == expected_ids


def test_t5_seminar_logical_kets_syndromes_enumerator_and_tables_match_live_code(
    docs_dir, results_dir, qecc_codes, qecc_analysis
) -> None:
    from qecc.analysis.analytical import _logical_basis_states

    sem_text = (docs_dir / "seminar_qecc.md").read_text(encoding="utf-8")
    fq = qecc_codes.get_code("five_qubit")

    # 1. Exact |0_L> and |1_L> 16-term sign expansions must match encode_ops()
    k0, k1 = _logical_basis_states(fq)
    for prefix, ket in (("$$|0_L\\rangle =", k0), ("$$|1_L\\rangle =", k1)):
        expected_signs = {
            f"{idx:05b}": ("+" if amp.real > 0 else "-")
            for idx, amp in enumerate(ket)
            if abs(amp) > 1e-9
        }
        line = next(ln for ln in sem_text.splitlines() if ln.startswith(prefix))
        parsed_signs = {
            m.group(2): (m.group(1) or "+")
            for m in re.finditer(r"([+-])?\s*\|([01]{5})\\rangle", line)
        }
        assert parsed_signs == expected_signs

    # 2. 16-syndrome table in seminar_qecc.md must match FiveQubitPerfectCode.syndrome_table
    tbl_rows = re.findall(r"\|\s*`([01]{4})`\s*\|\s*`([IXYZ]{5})`\s*\|", sem_text)
    assert dict(tbl_rows) == fq.syndrome_table

    # 3. 1024-term weight enumerator & KL derivations
    enum = qecc_analysis.five_qubit_weight_enumerator()
    assert [enum["I"][w] for w in range(6)] == [1, 15, 0, 60, 135, 45]
    assert "135" in sem_text and "45" in sem_text and "60" in sem_text
    assert "bit_flip_3" in sem_text and "phase_flip_3" in sem_text and "dfs_2" in sem_text

    # 4. Section 6.4 numerical summary table matches fidelity_sweeps.json to 6 decimals
    sweeps = json.loads((results_dir / "data" / "fidelity_sweeps.json").read_text(encoding="utf-8"))
    lookup = {(r["code"], r["channel"], round(float(r["p"]), 6)): float(r["mean_fidelity"]) for r in sweeps["records"]}
    p_cols = [0.00, 0.05, 0.10, 0.15, 0.25, 0.50]
    for c, ch in (
        ("bit_flip_3", "bit_flip"),
        ("phase_flip_3", "phase_flip"),
        ("five_qubit", "depolarizing"),
        ("five_qubit", "bit_flip"),
        ("bit_flip_3", "depolarizing"),
        ("dfs_2", "collective_dephasing"),
        ("dfs_2", "phase_flip"),
    ):
        matches = [
            ln
            for ln in sem_text.splitlines()
            if f"`{c} / {ch}`" in ln and len(re.findall(r"`([0-9]+\.[0-9]+)`", ln)) == 6
        ]
        assert len(matches) == 1
        nums = [float(x) for x in re.findall(r"`([0-9]+\.[0-9]+)`", matches[0])]
        for pv, nv in zip(p_cols, nums, strict=True):
            assert abs(round(lookup[(c, ch, pv)], 6) - nv) <= 1e-6


# ---------------------------------------------------------------------------
# 7. White-Box Edge Cases: Partial Trace, Tomography PSD, Negative KL & Coherent DFS
# ---------------------------------------------------------------------------


def test_t5_whitebox_metrics_kl_and_coherent_dfs_edge_cases(
    qecc_codes, qecc_channels, qecc_metrics, qecc_backends
) -> None:
    import torch
    from qecc.analysis.analytical import _logical_basis_states, _pauli_string_matrix
    from qecc.backends.numpy_backend import _apply_ops_np
    from qecc.backends.torch_cuda_backend import _apply_ops_torch

    # 1. partial_trace_batch_to_qubit0 disambiguates B == 2^n
    batch_b4 = np.eye(4, dtype=np.complex128)
    pt_batched = qecc_metrics.partial_trace_batch_to_qubit0(batch_b4, n_qubits=2)
    assert pt_batched.shape == (4, 2, 2)

    # 2. SingleQubitTomography PSD projection & (2,2) statevector batch flag
    rho_proj = qecc_metrics.SingleQubitTomography.reconstruct_from_expectations(0.9, 0.9, 0.9, project_physical=True)
    rho_unproj = qecc_metrics.SingleQubitTomography.reconstruct_from_expectations(0.9, 0.9, 0.9, project_physical=False)
    assert np.min(np.linalg.eigvalsh(rho_proj)) >= -1e-12
    assert np.min(np.linalg.eigvalsh(rho_unproj)) < -0.2

    sv2 = np.array([[1.0, 0.0], [1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0)]], dtype=np.complex128)
    rho_sv2 = qecc_metrics.SingleQubitTomography.reconstruct(
        sv2, shots_per_axis=30_000, rng=np.random.default_rng(5004), is_statevector_batch=True
    )
    expected_ens = 0.5 * (np.outer(sv2[0], sv2[0].conj()) + np.outer(sv2[1], sv2[1].conj()))
    assert qecc_metrics.quantum_fidelity(expected_ens, rho_sv2) > 0.995

    # 3. Non-correctable error sets violate Knill-Laflamme conditions
    for cname, bad_errs in (
        ("bit_flip_3", ("III", "XII", "IXX")),
        ("phase_flip_3", ("III", "ZII", "IZZ")),
        ("five_qubit", ("IIIII", "IIIZI", "XXIII")),
        ("dfs_2", ("II", "ZI")),
    ):
        k0, k1 = _logical_basis_states(qecc_codes.get_code(cname))
        mats = [_pauli_string_matrix(e) for e in bad_errs]
        max_viol = max(
            max(abs(np.vdot(ma @ k0, mb @ k0) - np.vdot(ma @ k1, mb @ k1)), abs(np.vdot(ma @ k0, mb @ k1)))
            for ma in mats
            for mb in mats
        )
        assert max_viol >= 0.99

    # 4. Coherent collective dephasing Rz(phi)^2 invariance across numpy, torch_cuda & eigenkit_unitary
    dfs = qecc_codes.get_code("dfs_2")
    ek_b = qecc_backends.get_backend("eigenkit_unitary")
    psi = qecc_metrics.InitialStateSampler().haar_random_states(1, seed=5005)[0]
    init_n = np.zeros(4, dtype=np.complex128)
    init_n[0], init_n[2] = psi[0], psi[1]
    ops = dfs.encode_ops() + qecc_channels.coherent_collective_dephasing_ops(2, 1.23) + dfs.decode_ops()

    sv_ek = ek_b.run_unitary_ops(ops, 2, init_n)
    st_np = init_n[None, :].copy()
    _apply_ops_np(st_np, ops, 2)
    st_cu = torch.tensor(init_n[None, :], dtype=torch.complex128, device="cuda")
    _apply_ops_torch(st_cu, ops, 2)
    sv_cu = st_cu.cpu().numpy()[0]

    assert qecc_metrics.quantum_fidelity(psi, qecc_metrics.partial_trace_to_qubit0(sv_ek, 2)) > 1.0 - 1e-12
    assert qecc_metrics.quantum_fidelity(psi, qecc_metrics.partial_trace_to_qubit0(st_np[0], 2)) > 1.0 - 1e-12
    assert qecc_metrics.quantum_fidelity(psi, qecc_metrics.partial_trace_to_qubit0(sv_cu, 2)) > 1.0 - 1e-12
