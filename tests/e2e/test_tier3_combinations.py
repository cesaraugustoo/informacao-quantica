"""Tier 3 E2E tests: 16 pairwise cross-feature combinations + 4 seminar/artifact integrations."""

from __future__ import annotations

import csv
import json
import math

import numpy as np
import pytest

pytestmark = pytest.mark.tier3


# ---------------------------------------------------------------------------
# Pairwise Covering Array (16 tests): Backend x Code x Channel x State x Metric
# ---------------------------------------------------------------------------


def test_t3_pair_01_numpy_bit_flip_3_bit_flip_ket0_direct_dm(
    qecc_backends, qecc_codes, qecc_channels, qecc_metrics, helpers
) -> None:
    backend = qecc_backends.get_backend("numpy")
    code = qecc_codes.get_code("bit_flip_3")
    psi = helpers.canonical_states()["|0>"]
    res = backend.simulate_cycle(
        code, psi, channel=qecc_channels.ChannelSpec("bit_flip", 0.1), p=0.1, num_shots=10_000, seed=3101
    )
    f_dm = qecc_metrics.quantum_fidelity(helpers.ket_to_rho(psi), res.rho_out)
    assert f_dm > 0.90
    assert abs(f_dm - 0.972) <= helpers.binom_sigma(0.972, 10_000, z=4.5)


def test_t3_pair_02_cuda_bit_flip_3_phase_flip_ketplus_per_shot(
    qecc_backends, qecc_codes, qecc_channels, helpers
) -> None:
    backend = qecc_backends.get_backend("torch_cuda")
    code = qecc_codes.get_code("bit_flip_3")
    psi = helpers.canonical_states()["|+>"]
    res = backend.simulate_cycle(
        code, psi, channel=qecc_channels.ChannelSpec("phase_flip", 0.1), p=0.1, num_shots=5_000, seed=3102
    )
    assert res.syndrome_counts == {"00": 5_000}
    expected = 1.0 - (3.0 * 0.1 * (0.9**2) + (0.1**3))  # 0.756
    assert abs(res.mean_fidelity - expected) <= helpers.binom_sigma(expected, 5_000, z=4.5)


def test_t3_pair_03_numpy_bit_flip_3_depolarizing_ketplus_i_tomography(
    qecc_backends, qecc_codes, qecc_channels, qecc_metrics, helpers
) -> None:
    backend = qecc_backends.get_backend("numpy")
    code = qecc_codes.get_code("bit_flip_3")
    psi = helpers.canonical_states()["|+i>"]
    res = backend.simulate_cycle(
        code, psi, channel=qecc_channels.ChannelSpec("depolarizing", 0.1), p=0.1, num_shots=10_000, seed=3103
    )
    expected = 0.8254814815
    assert res.mean_fidelity < 1.0 - (2.0 / 3.0) * 0.1
    assert abs(res.mean_fidelity - expected) <= helpers.binom_sigma(expected, 10_000, z=4.5)
    rho_tomo = qecc_metrics.SingleQubitTomography.reconstruct(
        res.rho_out, shots_per_axis=20_000, rng=np.random.default_rng(3103)
    )
    assert qecc_metrics.quantum_fidelity(res.rho_out, rho_tomo) > 0.995


def test_t3_pair_04_cuda_bit_flip_3_collective_dephasing_haar_direct_dm(
    qecc_backends, qecc_codes, qecc_channels, qecc_metrics, helpers
) -> None:
    backend = qecc_backends.get_backend("torch_cuda")
    code = qecc_codes.get_code("bit_flip_3")
    haar_pool = qecc_metrics.InitialStateSampler().haar_random_states(8, seed=3104)
    psi = next(s for s in haar_pool if abs(helpers.ket_to_rho(s)[0, 1]) > 0.15)
    rho_in = helpers.ket_to_rho(psi)
    res = backend.simulate_cycle(
        code,
        psi,
        channel=qecc_channels.ChannelSpec("collective_dephasing", 0.2),
        p=0.2,
        num_shots=5_000,
        seed=3104,
    )
    assert res.syndrome_counts == {"00": 5_000}
    coherence_ratio = abs(res.rho_out[0, 1]) / abs(rho_in[0, 1])
    assert abs(coherence_ratio - 0.60) <= 0.05


def test_t3_pair_05_cuda_phase_flip_3_phase_flip_ket0_direct_dm(
    qecc_backends, qecc_codes, qecc_channels, qecc_metrics, helpers
) -> None:
    backend = qecc_backends.get_backend("torch_cuda")
    code = qecc_codes.get_code("phase_flip_3")
    psi = helpers.canonical_states()["|0>"]
    res = backend.simulate_cycle(
        code, psi, channel=qecc_channels.ChannelSpec("phase_flip", 0.1), p=0.1, num_shots=10_000, seed=3105
    )
    f_dm = qecc_metrics.quantum_fidelity(helpers.ket_to_rho(psi), res.rho_out)
    assert f_dm > 0.90
    assert abs(f_dm - 0.972) <= helpers.binom_sigma(0.972, 10_000, z=4.5)


def test_t3_pair_06_numpy_phase_flip_3_bit_flip_ketplus_per_shot(
    qecc_backends, qecc_codes, qecc_channels, helpers
) -> None:
    backend = qecc_backends.get_backend("numpy")
    code = qecc_codes.get_code("phase_flip_3")
    psi = helpers.canonical_states()["|+>"]
    res = backend.simulate_cycle(
        code, psi, channel=qecc_channels.ChannelSpec("bit_flip", 0.1), p=0.1, num_shots=5_000, seed=3106
    )
    assert res.syndrome_counts == {"00": 5_000}
    expected = 1.0 - (3.0 * 0.1 * (0.9**2) + (0.1**3))  # 0.756
    assert abs(res.mean_fidelity - expected) <= helpers.binom_sigma(expected, 5_000, z=4.5)


def test_t3_pair_07_cuda_phase_flip_3_depolarizing_ket0_tomography(
    qecc_backends, qecc_codes, qecc_channels, qecc_metrics, helpers
) -> None:
    backend = qecc_backends.get_backend("torch_cuda")
    code = qecc_codes.get_code("phase_flip_3")
    psi = helpers.canonical_states()["|0>"]
    res = backend.simulate_cycle(
        code, psi, channel=qecc_channels.ChannelSpec("depolarizing", 0.1), p=0.1, num_shots=10_000, seed=3107
    )
    # By Hadamard duality with bit_flip_3 on |+>, F(|0>) = p_I + p_Z_L = 0.987259259
    expected = 0.9872592593
    assert abs(res.mean_fidelity - expected) <= helpers.binom_sigma(expected, 10_000, z=4.5)
    rho_tomo = qecc_metrics.SingleQubitTomography.reconstruct(
        res.rho_out, shots_per_axis=20_000, rng=np.random.default_rng(3107)
    )
    assert qecc_metrics.quantum_fidelity(res.rho_out, rho_tomo) > 0.995


def test_t3_pair_08_numpy_phase_flip_3_collective_dephasing_ketplus_i_direct_dm(
    qecc_backends, qecc_codes, qecc_channels, qecc_metrics, helpers
) -> None:
    backend = qecc_backends.get_backend("numpy")
    code = qecc_codes.get_code("phase_flip_3")
    psi = helpers.canonical_states()["|+i>"]
    res = backend.simulate_cycle(
        code,
        psi,
        channel=qecc_channels.ChannelSpec("collective_dephasing", 0.2),
        p=0.2,
        num_shots=10_000,
        seed=3108,
    )
    f_dm = qecc_metrics.quantum_fidelity(helpers.ket_to_rho(psi), res.rho_out)
    assert abs(f_dm - 0.80) <= helpers.binom_sigma(0.80, 10_000, z=4.5)


def test_t3_pair_09_numpy_five_qubit_depolarizing_ket0_direct_dm(
    qecc_backends, qecc_codes, qecc_channels, qecc_metrics, helpers
) -> None:
    backend = qecc_backends.get_backend("numpy")
    code = qecc_codes.get_code("five_qubit")
    psi = helpers.canonical_states()["|0>"]
    res = backend.simulate_cycle(
        code, psi, channel=qecc_channels.ChannelSpec("depolarizing", 0.08), p=0.08, num_shots=10_000, seed=3109
    )
    expected = 1.0 - (2.0 / 3.0) * (1.0 - helpers.five_qubit_exact_pi(0.08))
    f_dm = qecc_metrics.quantum_fidelity(helpers.ket_to_rho(psi), res.rho_out)
    assert f_dm > 1.0 - (2.0 / 3.0) * 0.08
    assert abs(f_dm - expected) <= helpers.binom_sigma(expected, 10_000, z=4.5)


def test_t3_pair_10_cuda_five_qubit_bit_flip_ket0_per_shot(
    qecc_backends, qecc_codes, qecc_channels, helpers
) -> None:
    backend = qecc_backends.get_backend("torch_cuda")
    code = qecc_codes.get_code("five_qubit")
    psi = helpers.canonical_states()["|0>"]
    res = backend.simulate_cycle(
        code, psi, channel=qecc_channels.ChannelSpec("bit_flip", 0.08), p=0.08, num_shots=10_000, seed=3110
    )
    expected = 0.9727235072
    assert res.mean_fidelity > 0.92
    assert abs(res.mean_fidelity - expected) <= helpers.binom_sigma(expected, 10_000, z=4.5)


def test_t3_pair_11_numpy_five_qubit_phase_flip_ketplus_i_tomography(
    qecc_backends, qecc_codes, qecc_channels, qecc_metrics, helpers
) -> None:
    backend = qecc_backends.get_backend("numpy")
    code = qecc_codes.get_code("five_qubit")
    psi = helpers.canonical_states()["|+i>"]
    res = backend.simulate_cycle(
        code, psi, channel=qecc_channels.ChannelSpec("phase_flip", 0.08), p=0.08, num_shots=10_000, seed=3111
    )
    expected = 0.9727235072
    assert res.mean_fidelity > 0.92
    assert abs(res.mean_fidelity - expected) <= helpers.binom_sigma(expected, 10_000, z=4.5)
    rho_tomo = qecc_metrics.SingleQubitTomography.reconstruct(
        res.rho_out, shots_per_axis=20_000, rng=np.random.default_rng(3111)
    )
    assert qecc_metrics.quantum_fidelity(res.rho_out, rho_tomo) > 0.995


def test_t3_pair_12_cuda_five_qubit_collective_dephasing_haar_direct_dm(
    qecc_backends, qecc_codes, qecc_channels, qecc_metrics, helpers
) -> None:
    backend = qecc_backends.get_backend("torch_cuda")
    code = qecc_codes.get_code("five_qubit")
    psi = qecc_metrics.InitialStateSampler().haar_random_states(1, seed=3112)[0]
    p = 0.20
    res = backend.simulate_cycle(
        code,
        psi,
        channel=qecc_channels.ChannelSpec("collective_dephasing", p),
        p=p,
        num_shots=5_000,
        seed=3112,
    )
    assert res.syndrome_counts == {"0000": 5_000}
    z_mat = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    z_exp = abs(np.vdot(psi, z_mat @ psi)) ** 2
    expected = (1.0 - p) + p * float(z_exp)
    f_dm = qecc_metrics.quantum_fidelity(helpers.ket_to_rho(psi), res.rho_out)
    assert abs(f_dm - expected) <= helpers.binom_sigma(expected, 5_000, z=4.5)


def test_t3_pair_13_numpy_dfs_2_collective_dephasing_ketplus_i_direct_dm(
    qecc_backends, qecc_codes, qecc_channels, qecc_metrics, helpers
) -> None:
    backend = qecc_backends.get_backend("numpy")
    code = qecc_codes.get_code("dfs_2")
    psi = helpers.canonical_states()["|+i>"]
    res = backend.simulate_cycle(
        code,
        psi,
        channel=qecc_channels.ChannelSpec("collective_dephasing", 0.3),
        p=0.3,
        num_shots=5_000,
        seed=3113,
    )
    assert qecc_metrics.quantum_fidelity(helpers.ket_to_rho(psi), res.rho_out) > 1.0 - 1e-10


def test_t3_pair_14_cuda_dfs_2_phase_flip_ketplus_i_per_shot(
    qecc_backends, qecc_codes, qecc_channels, helpers
) -> None:
    backend = qecc_backends.get_backend("torch_cuda")
    code = qecc_codes.get_code("dfs_2")
    psi = helpers.canonical_states()["|+i>"]
    res = backend.simulate_cycle(
        code, psi, channel=qecc_channels.ChannelSpec("phase_flip", 0.1), p=0.1, num_shots=10_000, seed=3114
    )
    expected = 1.0 - 2.0 * 0.1 * 0.9
    assert res.mean_fidelity < 0.90
    assert abs(res.mean_fidelity - expected) <= helpers.binom_sigma(expected, 10_000, z=4.5)


def test_t3_pair_15_numpy_dfs_2_bit_flip_ket0_tomography(
    qecc_backends, qecc_codes, qecc_channels, qecc_metrics, helpers
) -> None:
    backend = qecc_backends.get_backend("numpy")
    code = qecc_codes.get_code("dfs_2")
    psi = helpers.canonical_states()["|0>"]
    res = backend.simulate_cycle(
        code, psi, channel=qecc_channels.ChannelSpec("bit_flip", 0.1), p=0.1, num_shots=10_000, seed=3115
    )
    assert abs(res.mean_fidelity - 0.90) <= helpers.binom_sigma(0.90, 10_000, z=4.5)
    rho_tomo = qecc_metrics.SingleQubitTomography.reconstruct(
        res.rho_out, shots_per_axis=20_000, rng=np.random.default_rng(3115)
    )
    assert qecc_metrics.quantum_fidelity(res.rho_out, rho_tomo) > 0.995


def test_t3_pair_16_cuda_dfs_2_depolarizing_ketplus_direct_dm(
    qecc_backends, qecc_codes, qecc_channels, qecc_metrics, helpers
) -> None:
    backend = qecc_backends.get_backend("torch_cuda")
    code = qecc_codes.get_code("dfs_2")
    psi = helpers.canonical_states()["|+>"]
    res = backend.simulate_cycle(
        code, psi, channel=qecc_channels.ChannelSpec("depolarizing", 0.1), p=0.1, num_shots=10_000, seed=3116
    )
    expected = 0.8755555556
    f_dm = qecc_metrics.quantum_fidelity(helpers.ket_to_rho(psi), res.rho_out)
    assert f_dm < 1.0 - (2.0 / 3.0) * 0.1
    assert abs(f_dm - expected) <= helpers.binom_sigma(expected, 10_000, z=4.5)


# ---------------------------------------------------------------------------
# Seminar / Artifact / Live-Code Integration Pairwise Tests (4 tests)
# ---------------------------------------------------------------------------


def test_t3_sweep_data_matches_seminar_threshold_and_figure_claims(
    repo_root, results_dir, docs_dir, helpers
) -> None:
    json_path = results_dir / "data" / "fidelity_sweeps.json"
    csv_path = results_dir / "data" / "fidelity_sweeps.csv"
    assert json_path.is_file() and csv_path.is_file()

    data = json.loads(json_path.read_text(encoding="utf-8"))
    flat = helpers.flatten_json_text(data).lower()
    for code_tok in ("bit_flip_3", "phase_flip_3", "five_qubit"):
        assert code_tok in flat

    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) >= 15

    # Check advantage/disadvantage ordering across p in CSV when wide or long format is used
    cols = {k.lower(): k for k in rows[0].keys()}
    p_key = next((cols[c] for c in ("p", "error_rate", "p_error") if c in cols), None)
    assert p_key is not None

    if "code" in cols or "code_name" in cols:
        code_key = cols.get("code") or cols["code_name"]
        ch_key = cols.get("channel") or cols.get("channel_type")
        fid_key = next(
            (
                orig
                for low, orig in cols.items()
                if helpers.is_fidelity_metric_name(orig)
                and "raw" not in low
                and "analytical" not in low
            ),
            None,
        )
        assert fid_key is not None
        for r in rows:
            p_val = float(r[p_key])
            c_name = r[code_key].strip().lower()
            ch_val = r[ch_key].strip().lower() if ch_key is not None and r[ch_key] is not None else ""
            f_val = float(r[fid_key])
            if 0.02 <= p_val <= 0.10:
                if c_name == "bit_flip_3" and (ch_key is None or ch_val == "bit_flip"):
                    assert f_val > 1.0 - p_val - 0.015
                elif c_name == "phase_flip_3" and (ch_key is None or ch_val == "phase_flip"):
                    assert f_val > 1.0 - p_val - 0.015
            if p_val >= 0.20 and c_name == "five_qubit":
                is_depol = ("depolar" in ch_val) if ch_key is not None else ("depolar" in str(r).lower())
                if is_depol:
                    assert f_val < 1.0 - (2.0 / 3.0) * p_val + 0.02
    else:
        def _wide_code_cols(code_tok: str) -> list[str]:
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

        bf_cols = _wide_code_cols("bit_flip_3")
        pf_cols = _wide_code_cols("phase_flip_3")
        fq_cols = _wide_code_cols("five_qubit")
        assert len(bf_cols) + len(pf_cols) + len(fq_cols) >= 1, f"No wide code fidelity columns in {list(cols.values())}"

        for r in rows:
            p_val = float(r[p_key])
            for col in bf_cols + pf_cols:
                raw_cell = r.get(col)
                if raw_cell is None or raw_cell.strip() == "":
                    continue
                f_val = float(raw_cell)
                if 0.02 <= p_val <= 0.10:
                    assert f_val > 1.0 - p_val - 0.015
            for col in fq_cols:
                raw_cell = r.get(col)
                if raw_cell is None or raw_cell.strip() == "":
                    continue
                f_val = float(raw_cell)
                if 0.02 <= p_val <= 0.10:
                    assert f_val > 1.0 - p_val - 0.015
                if p_val >= 0.20:
                    assert f_val < 1.0 - (2.0 / 3.0) * p_val + 0.02

    for md_name in ("seminar_qecc.md", "seminar_slides.md"):
        md_path = docs_dir / md_name
        targets = helpers.extract_markdown_image_targets(md_path.read_text(encoding="utf-8"))
        for tgt in targets:
            resolved = (md_path.parent / tgt).resolve()
            if not resolved.exists():
                resolved = (repo_root / tgt).resolve()
            assert resolved.is_file()


def test_t3_benchmark_json_matches_eigenkit_eval_and_seminar_docs(
    repo_root, results_dir, docs_dir, helpers
) -> None:
    bench_json = results_dir / "benchmarks" / "backend_benchmark.json"
    bench_md = results_dir / "benchmarks" / "backend_benchmark.md"
    assert bench_json.is_file() and bench_md.is_file()

    data = json.loads(bench_json.read_text(encoding="utf-8"))
    flat = helpers.flatten_json_text(data).lower()
    for tok in ("numpy", "torch_cuda", "bit_flip_3", "phase_flip_3", "five_qubit", "100000"):
        assert tok in flat

    eval_text = (docs_dir / "eigenkit_evaluation.md").read_text(encoding="utf-8").lower()
    readme_text = (repo_root / "README.md").read_text(encoding="utf-8").lower()
    for doc_text in (eval_text, readme_text):
        assert "numpy" in doc_text
        assert "torch_cuda" in doc_text or "cuda" in doc_text
        assert "eigenkit" in doc_text
        assert "5080" in doc_text


def test_t3_dfs_invariance_matches_seminar_topic3(
    qecc_backends, qecc_codes, qecc_channels, qecc_metrics, docs_dir, helpers
) -> None:
    backend = qecc_backends.get_backend("numpy")
    code = qecc_codes.get_code("dfs_2")
    cardinals = list(qecc_metrics.InitialStateSampler().cardinal_states().values())

    ch_coll = qecc_channels.ChannelSpec("collective_dephasing", 0.25)
    ch_ind = qecc_channels.ChannelSpec("phase_flip", 0.25)

    for psi in cardinals:
        res_coll = backend.simulate_cycle(code, psi, channel=ch_coll, p=0.25, num_shots=200, seed=3201)
        assert res_coll.mean_fidelity > 1.0 - 1e-10

    plus_state = helpers.canonical_states()["|+>"]
    res_ind = backend.simulate_cycle(code, plus_state, channel=ch_ind, p=0.25, num_shots=1000, seed=3202)
    assert res_ind.mean_fidelity < 1.0 - 0.25 / 2.0

    seminar_low = (docs_dir / "seminar_qecc.md").read_text(encoding="utf-8").lower()
    assert "decoherence-free" in seminar_low and "collective" in seminar_low


def test_t3_live_code_stabilizers_and_syndromes_match_seminar_tables(
    qecc_codes, docs_dir
) -> None:
    seminar_text = (docs_dir / "seminar_qecc.md").read_text(encoding="utf-8")

    bf = qecc_codes.get_code("bit_flip_3")
    pf = qecc_codes.get_code("phase_flip_3")
    fq = qecc_codes.get_code("five_qubit")

    assert set(bf.syndrome_table.keys()) == {"00", "10", "11", "01"}
    assert set(pf.syndrome_table.keys()) == {"00", "10", "11", "01"}
    assert set(fq.syndrome_table.keys()) == {f"{i:04b}" for i in range(16)}

    for gen in fq.stabilizers:
        assert gen in seminar_text
    for synd in fq.syndrome_table:
        assert synd in seminar_text
