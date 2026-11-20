"""Tier 1 E2E tests for Features F5-F6: Pauli channels, error injectors, fidelity & tomography."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

pytestmark = pytest.mark.tier1


# ---------------------------------------------------------------------------
# Feature F5: PauliChannelsAndInjectors
# ---------------------------------------------------------------------------


def test_t1_f5_bit_flip_frequency_and_independence(qecc_channels, helpers) -> None:
    p = 0.25
    n_qubits = 5
    num_shots = 20_000
    spec = qecc_channels.ChannelSpec("bit_flip", p)
    channel = qecc_channels.StochasticPauliChannel(spec)
    samples = helpers.normalize_pauli_samples(
        channel.sample(n_qubits, num_shots, np.random.default_rng(42)), n_qubits
    )

    assert set(np.unique(samples)).issubset({"I", "X"})
    tol = helpers.binom_sigma(p, num_shots, z=4.5)
    indicators = (samples != "I").astype(np.int64)
    for j in range(n_qubits):
        assert abs(float(np.mean(indicators[:, j])) - p) <= tol

    num_pairs = n_qubits * (n_qubits - 1) // 2
    corr_tol = 4.5 / math.sqrt(num_shots)
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            r_ij = float(np.corrcoef(indicators[:, i], indicators[:, j])[0, 1])
            assert abs(r_ij) < corr_tol
            table = np.array(
                [
                    [
                        int(np.sum((indicators[:, i] == 0) & (indicators[:, j] == 0))),
                        int(np.sum((indicators[:, i] == 0) & (indicators[:, j] == 1))),
                    ],
                    [
                        int(np.sum((indicators[:, i] == 1) & (indicators[:, j] == 0))),
                        int(np.sum((indicators[:, i] == 1) & (indicators[:, j] == 1))),
                    ],
                ]
            )
            _, p_val, _, _ = stats.chi2_contingency(table)
            assert p_val > 0.0005 / num_pairs


def test_t1_f5_phase_flip_frequency_and_independence(qecc_channels, helpers) -> None:
    p = 0.20
    n_qubits = 3
    num_shots = 20_000
    spec = qecc_channels.ChannelSpec("phase_flip", p)
    channel = qecc_channels.StochasticPauliChannel(spec)
    samples = helpers.normalize_pauli_samples(
        channel.sample(n_qubits, num_shots, np.random.default_rng(43)), n_qubits
    )

    assert set(np.unique(samples)).issubset({"I", "Z"})
    tol = helpers.binom_sigma(p, num_shots, z=4.5)
    indicators = (samples == "Z").astype(np.int64)
    for j in range(n_qubits):
        assert abs(float(np.mean(indicators[:, j])) - p) <= tol

    num_pairs = n_qubits * (n_qubits - 1) // 2
    corr_tol = 4.5 / math.sqrt(num_shots)
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            r_ij = float(np.corrcoef(indicators[:, i], indicators[:, j])[0, 1])
            assert abs(r_ij) < corr_tol
            table = np.array(
                [
                    [
                        int(np.sum((indicators[:, i] == 0) & (indicators[:, j] == 0))),
                        int(np.sum((indicators[:, i] == 0) & (indicators[:, j] == 1))),
                    ],
                    [
                        int(np.sum((indicators[:, i] == 1) & (indicators[:, j] == 0))),
                        int(np.sum((indicators[:, i] == 1) & (indicators[:, j] == 1))),
                    ],
                ]
            )
            _, p_val, _, _ = stats.chi2_contingency(table)
            assert p_val > 0.0005 / num_pairs


def test_t1_f5_depolarizing_components_and_independence(qecc_channels, helpers) -> None:
    p = 0.30
    n_qubits = 5
    num_shots = 30_000
    spec = qecc_channels.ChannelSpec("depolarizing", p)
    channel = qecc_channels.StochasticPauliChannel(spec)
    samples = helpers.normalize_pauli_samples(
        channel.sample(n_qubits, num_shots, np.random.default_rng(44)), n_qubits
    )

    total_tol = helpers.binom_sigma(p, num_shots, z=4.5)
    comp_tol = helpers.binom_sigma(p / 3.0, num_shots, z=4.5)
    for j in range(n_qubits):
        assert abs(float(np.mean(samples[:, j] != "I")) - p) <= total_tol
        for pauli_sym in ("X", "Y", "Z"):
            assert abs(float(np.mean(samples[:, j] == pauli_sym)) - (p / 3.0)) <= comp_tol

    num_pairs = n_qubits * (n_qubits - 1) // 2
    syms = ("I", "X", "Y", "Z")
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            table = np.array(
                [
                    [int(np.sum((samples[:, i] == a) & (samples[:, j] == b))) for b in syms]
                    for a in syms
                ]
            )
            _, p_val, _, _ = stats.chi2_contingency(table)
            assert p_val > 0.0005 / num_pairs


def test_t1_f5_collective_dephasing_correlation(qecc_channels, helpers) -> None:
    p = 0.25
    n_qubits = 4
    num_shots = 10_000
    spec = qecc_channels.ChannelSpec("collective_dephasing", p)
    channel = qecc_channels.StochasticPauliChannel(spec)
    samples = helpers.normalize_pauli_samples(
        channel.sample(n_qubits, num_shots, np.random.default_rng(45)), n_qubits
    )

    all_i = np.all(samples == "I", axis=1)
    all_z = np.all(samples == "Z", axis=1)
    assert np.all(all_i | all_z)
    tol = helpers.binom_sigma(p, num_shots, z=4.0)
    assert abs(float(np.mean(all_z)) - p) <= tol


def test_t1_f5_rng_seed_reproducibility(qecc_codes, qecc_channels, numpy_backend, helpers) -> None:
    spec = qecc_channels.ChannelSpec("depolarizing", 0.30)
    channel = qecc_channels.StochasticPauliChannel(spec)

    s1 = helpers.normalize_pauli_samples(channel.sample(5, 1000, np.random.default_rng(12345)), 5)
    s2 = helpers.normalize_pauli_samples(channel.sample(5, 1000, np.random.default_rng(12345)), 5)
    s3 = helpers.normalize_pauli_samples(channel.sample(5, 1000, np.random.default_rng(54321)), 5)
    assert np.array_equal(s1, s2)
    assert not np.array_equal(s1, s3)

    code = qecc_codes.get_code("five_qubit")
    ket_plus = helpers.canonical_states()["|+>"]
    ch15 = qecc_channels.ChannelSpec("depolarizing", 0.15)
    r1 = numpy_backend.simulate_cycle(code, ket_plus, channel=ch15, p=0.15, num_shots=500, seed=12345)
    r2 = numpy_backend.simulate_cycle(code, ket_plus, channel=ch15, p=0.15, num_shots=500, seed=12345)
    assert list(r1.syndromes) == list(r2.syndromes)
    assert abs(r1.mean_fidelity - r2.mean_fidelity) < 1e-15


def test_t1_f5_deterministic_error_injector_validation_and_execution(
    qecc_codes, qecc_channels, numpy_backend, helpers
) -> None:
    for err in ("IIIII", "XIIII", "IYIZI", "ZZZZZ"):
        qecc_channels.DeterministicErrorInjector.validate_pauli_string(err, 5)

    code = qecc_codes.get_code("five_qubit")
    ket_plus_i = helpers.canonical_states()["|+i>"]
    res = numpy_backend.simulate_cycle(
        code, ket_plus_i, channel=None, p=0.0, num_shots=1, deterministic_errors=["IXIII"]
    )
    assert res.syndrome_counts == {"1000": 1}
    assert res.mean_fidelity > 1.0 - 1e-10


# ---------------------------------------------------------------------------
# Feature F6: QuantumFidelityAndTomography
# ---------------------------------------------------------------------------


def test_t1_f6_pure_pure_fidelity(qecc_metrics, helpers) -> None:
    sampler = qecc_metrics.InitialStateSampler()
    states = list(sampler.cardinal_states().values()) + list(sampler.haar_random_states(10, seed=11))

    for psi in states:
        assert qecc_metrics.quantum_fidelity(psi, psi) == pytest.approx(1.0, abs=1e-12)

    for psi in states:
        rho_psi = helpers.ket_to_rho(psi)
        for phi in states:
            rho_phi = helpers.ket_to_rho(phi)
            expected = float(abs(np.vdot(psi, phi)) ** 2)
            assert abs(qecc_metrics.quantum_fidelity(psi, phi) - expected) < 1e-12
            assert abs(qecc_metrics.quantum_fidelity(rho_psi, rho_phi) - expected) < 1e-12


def test_t1_f6_pure_mixed_squared_uhlmann_convention(qecc_metrics, helpers) -> None:
    canonical = helpers.canonical_states()
    rho0 = helpers.ket_to_rho(canonical["|0>"])
    rho1 = helpers.ket_to_rho(canonical["|1>"])
    rhop = helpers.ket_to_rho(canonical["|+>"])
    rhom = helpers.ket_to_rho(canonical["|->"])

    for p in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0):
        sigma = (1.0 - p) * rho0 + p * rho1
        sigma_plus = (1.0 - p) * rhop + p * rhom
        assert abs(qecc_metrics.quantum_fidelity(canonical["|0>"], sigma) - (1.0 - p)) < 1e-12
        assert abs(qecc_metrics.quantum_fidelity(canonical["|+>"], sigma_plus) - (1.0 - p)) < 1e-12


def test_t1_f6_mixed_mixed_analytic_uhlmann_formula(qecc_metrics, helpers) -> None:
    pairs = [
        ((0.4, 0.2, 0.3), (-0.2, 0.5, 0.1)),
        ((0.6, -0.3, 0.2), (0.1, 0.4, -0.5)),
    ]
    for r_vec, s_vec in pairs:
        rho = helpers.bloch_to_rho(r_vec)
        sigma = helpers.bloch_to_rho(s_vec)
        expected = helpers.jozsa_single_qubit_fidelity(r_vec, s_vec)
        f_rs = qecc_metrics.quantum_fidelity(rho, sigma)
        f_sr = qecc_metrics.quantum_fidelity(sigma, rho)
        assert abs(f_rs - expected) < 1e-12
        assert abs(f_rs - f_sr) < 1e-12

    rho_diag = np.diag([0.8, 0.2]).astype(np.complex128)
    sigma_diag = np.diag([0.3, 0.7]).astype(np.complex128)
    expected_diag = (math.sqrt(0.8 * 0.3) + math.sqrt(0.2 * 0.7)) ** 2
    assert abs(qecc_metrics.quantum_fidelity(rho_diag, sigma_diag) - expected_diag) < 1e-12


def test_t1_f6_partial_trace_to_qubit0_product_vs_entangled(qecc_metrics, helpers) -> None:
    canonical = helpers.canonical_states()
    psi0 = qecc_metrics.InitialStateSampler().haar_random_states(1, seed=61)[0]
    prod_state = np.kron(np.kron(psi0, canonical["|1>"]), canonical["|+>"])
    prod_rho = np.outer(prod_state, prod_state.conj())

    rho_expected = helpers.ket_to_rho(psi0)
    assert np.allclose(qecc_metrics.partial_trace_to_qubit0(prod_state, 3), rho_expected, atol=1e-12)
    assert np.allclose(qecc_metrics.partial_trace_to_qubit0(prod_rho, 3), rho_expected, atol=1e-12)

    bell = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.complex128) / math.sqrt(2.0)
    bell_rho = np.outer(bell, bell.conj())
    ghz = np.zeros(8, dtype=np.complex128)
    ghz[0] = 1.0 / math.sqrt(2.0)
    ghz[7] = 1.0 / math.sqrt(2.0)
    ghz_rho = np.outer(ghz, ghz.conj())

    half_eye = 0.5 * np.eye(2, dtype=np.complex128)
    for st, n_q in ((bell, 2), (bell_rho, 2), (ghz, 3), (ghz_rho, 3)):
        reduced = qecc_metrics.partial_trace_to_qubit0(st, n_q)
        assert np.allclose(reduced, half_eye, atol=1e-12)
        assert abs(reduced[0, 1]) < 1e-12
        assert abs(helpers.purity(reduced) - 0.5) < 1e-12


def test_t1_f6_single_qubit_tomography_cardinal_and_haar(qecc_metrics, helpers) -> None:
    sampler = qecc_metrics.InitialStateSampler()
    mixed = np.diag([0.7, 0.3]).astype(np.complex128)
    targets = [helpers.ket_to_rho(psi) for psi in sampler.cardinal_states().values()]
    targets += [helpers.ket_to_rho(psi) for psi in sampler.haar_random_states(5, seed=77)]
    targets.append(mixed)

    rng = np.random.default_rng(77)
    for rho_true in targets:
        rho_hat = qecc_metrics.SingleQubitTomography.reconstruct(
            rho_true, shots_per_axis=20_000, rng=rng
        )
        assert rho_hat.shape == (2, 2)
        assert np.allclose(rho_hat, rho_hat.conj().T, atol=1e-10)
        assert abs(float(np.real(np.trace(rho_hat))) - 1.0) < 1e-10
        assert qecc_metrics.quantum_fidelity(rho_true, rho_hat) > 0.99
        assert float(np.linalg.norm(rho_true - rho_hat, ord="fro")) < 0.05


def test_t1_f6_initial_state_sampler_cardinal_and_haar(qecc_metrics) -> None:
    sampler = qecc_metrics.InitialStateSampler()
    cardinal = sampler.cardinal_states()
    assert len(cardinal) == 6
    vecs = list(cardinal.values())
    for v in vecs:
        assert v.shape == (2,)
        assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-12

    # Check 3 orthogonal pairs exist among the 6 cardinal states
    ortho_pairs = 0
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            if abs(np.vdot(vecs[i], vecs[j])) < 1e-12:
                ortho_pairs += 1
    assert ortho_pairs == 3

    haar1 = sampler.haar_random_states(n_states=50, seed=99)
    haar2 = sampler.haar_random_states(n_states=50, seed=99)
    assert haar1.shape == (50, 2)
    assert np.allclose(haar1, haar2, atol=1e-15)
    norms = np.linalg.norm(haar1, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-12)

    pauli_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    pauli_y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
    pauli_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    mean_bloch = np.array(
        [
            np.mean([np.real(np.vdot(v, pauli_x @ v)) for v in haar1]),
            np.mean([np.real(np.vdot(v, pauli_y @ v)) for v in haar1]),
            np.mean([np.real(np.vdot(v, pauli_z @ v)) for v in haar1]),
        ]
    )
    assert float(np.linalg.norm(mean_bloch)) < 3.0 / math.sqrt(50.0)
