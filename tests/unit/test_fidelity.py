"""Unit tests for qecc.metrics (fidelity, partial trace, states, tomography)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.linalg import sqrtm

from qecc.metrics import (
    InitialStateSampler,
    SingleQubitTomography,
    partial_trace_batch_to_qubit0,
    partial_trace_to_qubit0,
    quantum_fidelity,
)


def _reference_sqrtm_fidelity(rho: np.ndarray, sigma: np.ndarray) -> float:
    sr = sqrtm(rho)
    inner = sr @ sigma @ sr
    si = sqrtm(inner)
    val = float(np.real(np.trace(si)) ** 2)
    return float(np.clip(val, 0.0, 1.0))


def test_quantum_fidelity_pure_and_squared_convention() -> None:
    ket0 = np.array([1.0, 0.0], dtype=np.complex128)
    ket1 = np.array([0.0, 1.0], dtype=np.complex128)
    ket_plus = np.array([1.0, 1.0], dtype=np.complex128) / math.sqrt(2.0)
    ket_plus_i = np.array([1.0, 1.0j], dtype=np.complex128) / math.sqrt(2.0)

    assert math.isclose(quantum_fidelity(ket0, ket0), 1.0, abs_tol=1e-13)
    assert math.isclose(quantum_fidelity(ket0, ket1), 0.0, abs_tol=1e-13)
    assert math.isclose(quantum_fidelity(ket_plus, ket_plus_i), 0.5, abs_tol=1e-13)

    rho0 = np.outer(ket0, ket0.conj())
    rho1 = np.outer(ket1, ket1.conj())
    for p in (0.0, 0.1, 0.25, 0.5, 1.0):
        mixed = (1.0 - p) * rho0 + p * rho1
        assert math.isclose(quantum_fidelity(ket0, mixed), 1.0 - p, abs_tol=1e-12)
        assert math.isclose(quantum_fidelity(mixed, ket0), 1.0 - p, abs_tol=1e-12)


def test_quantum_fidelity_mixed_parity_and_edge_cases() -> None:
    rng = np.random.default_rng(101)
    for _ in range(30):
        a = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
        b = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
        rho = a @ a.conj().T
        rho /= np.trace(rho)
        sigma = b @ b.conj().T
        sigma /= np.trace(sigma)
        f_ours = quantum_fidelity(rho, sigma)
        f_ref = _reference_sqrtm_fidelity(rho, sigma)
        assert abs(f_ours - f_ref) < 1e-12

    eye_half = 0.5 * np.eye(2, dtype=np.complex128)
    assert math.isclose(quantum_fidelity(eye_half, eye_half), 1.0, abs_tol=1e-12)
    assert math.isclose(
        quantum_fidelity(eye_half, np.array([1.0, 0.0], dtype=np.complex128)),
        0.5,
        abs_tol=1e-12,
    )

    # Tiny negative roundoff eigenvalue within tolerance should yield clean 0.0, no NaNs
    rho_roundoff = np.array([[1.0 + 1e-14, 0.0], [0.0, -1e-14]], dtype=np.complex128)
    ket1 = np.array([0.0, 1.0], dtype=np.complex128)
    fid = quantum_fidelity(rho_roundoff, ket1)
    assert math.isfinite(fid)
    assert 0.0 <= fid <= 1e-12


def test_quantum_fidelity_validation_errors() -> None:
    ket0 = np.array([1.0, 0.0], dtype=np.complex128)
    with pytest.raises(ValueError):
        quantum_fidelity(np.array([1.0, 1.0]), ket0)
    with pytest.raises(ValueError):
        quantum_fidelity(np.array([[1.0, 0.5], [0.0, 0.0]]), ket0)
    with pytest.raises(ValueError):
        quantum_fidelity(np.array([[0.8, 0.0], [0.0, 0.8]]), ket0)
    with pytest.raises(ValueError):
        quantum_fidelity(np.array([[1.1, 0.0], [0.0, -0.1]]), ket0)
    with pytest.raises(ValueError):
        quantum_fidelity(np.array([1.0, 0.0, 0.0, 0.0]), ket0)
    with pytest.raises(ValueError):
        quantum_fidelity(np.array([float("nan"), 0.0]), ket0)


def test_quantum_fidelity_haar_rank1_vs_mixed_precision() -> None:
    states = InitialStateSampler.haar_random_states(20, seed=123)
    for i, psi in enumerate(states):
        dm_psi = np.outer(psi, psi.conj())
        psi_perp = np.array([-psi[1].conj(), psi[0].conj()], dtype=np.complex128)
        dm_perp = np.outer(psi_perp, psi_perp.conj())
        for p in (0.0, 0.1, 0.3, 0.7, 1.0):
            rho_mix = (1.0 - p) * dm_psi + p * dm_perp
            expected = 1.0 - p
            assert abs(quantum_fidelity(psi, rho_mix) - expected) < 1e-13
            assert abs(quantum_fidelity(rho_mix, psi) - expected) < 1e-13
            assert abs(quantum_fidelity(dm_psi, rho_mix) - expected) < 1e-13

        phi = states[(i + 1) % len(states)]
        dm_phi = np.outer(phi, phi.conj())
        exact_overlap = float(abs(np.vdot(psi, phi)) ** 2)
        assert abs(quantum_fidelity(dm_psi, dm_phi) - exact_overlap) < 1e-13

    # Perturbed norm within tol=1e-6 must be normalized before computing overlap/fidelity
    psi0 = states[0]
    psi_perturbed = psi0 * (1.0 + 4e-7)
    dm_psi0 = np.outer(psi0, psi0.conj())
    psi0_perp = np.array([-psi0[1].conj(), psi0[0].conj()], dtype=np.complex128)
    rho_mix0 = 0.7 * dm_psi0 + 0.3 * np.outer(psi0_perp, psi0_perp.conj())
    assert abs(quantum_fidelity(psi_perturbed, psi_perturbed) - 1.0) < 1e-14
    assert abs(quantum_fidelity(psi_perturbed, rho_mix0) - 0.7) < 1e-13
    assert abs(quantum_fidelity(rho_mix0, psi_perturbed) - 0.7) < 1e-13


def test_quantum_fidelity_high_dim_rank_deficient_precision() -> None:
    rng = np.random.default_rng(2027)
    d = 32
    for _ in range(10):
        u = rng.normal(size=d) + 1j * rng.normal(size=d)
        u /= np.linalg.norm(u)
        v = rng.normal(size=d) + 1j * rng.normal(size=d)
        v -= np.vdot(u, v) * u
        v /= np.linalg.norm(v)
        w = rng.normal(size=d) + 1j * rng.normal(size=d)
        w /= np.linalg.norm(w)

        dm_u = np.outer(u, u.conj())
        dm_v = np.outer(v, v.conj())
        dm_w = np.outer(w, w.conj())

        exact_uw = float(abs(np.vdot(u, w)) ** 2)
        assert abs(quantum_fidelity(dm_u, dm_w) - exact_uw) < 1e-13

        p = 0.35
        rho_mix = (1.0 - p) * dm_u + p * dm_v
        assert abs(quantum_fidelity(u, rho_mix) - (1.0 - p)) < 1e-13
        assert abs(quantum_fidelity(rho_mix, u) - (1.0 - p)) < 1e-13
        assert abs(quantum_fidelity(dm_u, rho_mix) - (1.0 - p)) < 1e-13

        exact_w_mix = float(np.real(np.vdot(w, rho_mix @ w)))
        assert abs(quantum_fidelity(w, rho_mix) - exact_w_mix) < 1e-13
        assert abs(quantum_fidelity(rho_mix, w) - exact_w_mix) < 1e-13
        assert abs(quantum_fidelity(dm_w, rho_mix) - exact_w_mix) < 1e-13


def test_quantum_fidelity_small_infidelity_non_diagonal_pure_state() -> None:
    cards = InitialStateSampler.cardinal_states()
    plus, minus = cards["+"], cards["-"]
    dm_plus = np.outer(plus, plus.conj())
    mixed_plus = (1.0 - 1e-9) * dm_plus + 1e-9 * np.outer(minus, minus.conj())

    for a, b in ((plus, mixed_plus), (mixed_plus, plus), (dm_plus, mixed_plus), (mixed_plus, dm_plus)):
        fid = quantum_fidelity(a, b)
        assert abs((1.0 - fid) - 1e-9) < 1e-14


def test_partial_trace_to_qubit0_big_endian_and_batches() -> None:
    psi0 = np.array([math.sqrt(0.35), np.exp(1j * 0.7) * math.sqrt(0.65)], dtype=np.complex128)
    tail_10 = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.complex128)  # |10> on qubits 1, 2
    full_state = np.kron(psi0, tail_10)
    rho0 = partial_trace_to_qubit0(full_state, n_qubits=3)
    assert rho0.shape == (2, 2)
    assert np.allclose(rho0, np.outer(psi0, psi0.conj()), atol=1e-13)

    # Reverse check: |0>_0 (x) |psi>_1 (x) |0>_2 must trace out to |0><0|
    ket0 = np.array([1.0, 0.0], dtype=np.complex128)
    swapped = np.kron(np.kron(ket0, psi0), ket0)
    rho_swapped = partial_trace_to_qubit0(swapped, n_qubits=3)
    assert np.allclose(rho_swapped, np.outer(ket0, ket0.conj()), atol=1e-13)

    # Bell state (|00> + |11>)/sqrt(2) -> I/2
    bell = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.complex128) / math.sqrt(2.0)
    assert np.allclose(partial_trace_to_qubit0(bell, 2), 0.5 * np.eye(2), atol=1e-13)
    assert np.allclose(
        partial_trace_to_qubit0(np.outer(bell, bell.conj()), 2),
        0.5 * np.eye(2),
        atol=1e-13,
    )

    # Batched statevectors including ambiguous B == 2^n case
    batch_b5 = np.tile(full_state, (5, 1))
    out_b5 = partial_trace_to_qubit0(batch_b5, n_qubits=3)
    assert out_b5.shape == (5, 2, 2)
    assert np.allclose(out_b5[0], np.outer(psi0, psi0.conj()), atol=1e-13)

    batch_b8 = np.tile(full_state, (8, 1))  # B == 2^3 == 8
    out_b8 = partial_trace_batch_to_qubit0(batch_b8, n_qubits=3)
    assert out_b8.shape == (8, 2, 2)
    assert np.allclose(out_b8[3], np.outer(psi0, psi0.conj()), atol=1e-13)

    rho_batch = np.tile(np.outer(full_state, full_state.conj()), (4, 1, 1))
    out_rho_batch = partial_trace_to_qubit0(rho_batch, n_qubits=3)
    assert out_rho_batch.shape == (4, 2, 2)
    assert np.allclose(out_rho_batch[2], np.outer(psi0, psi0.conj()), atol=1e-13)

    with pytest.raises(ValueError):
        partial_trace_to_qubit0(full_state, n_qubits=0)
    with pytest.raises(ValueError):
        partial_trace_to_qubit0(np.zeros(5), n_qubits=2)


def test_initial_state_sampler_cardinal_and_haar() -> None:
    cards = InitialStateSampler.cardinal_states()
    assert set(cards.keys()) == {"0", "1", "+", "-", "+i", "-i"}
    for v in cards.values():
        assert v.shape == (2,)
        assert v.dtype == np.complex128
        assert math.isclose(float(np.linalg.norm(v)), 1.0, abs_tol=1e-14)
    # Mutating returned array does not mutate subsequent calls
    cards["0"][0] = 999.0
    assert InitialStateSampler.cardinal_states()["0"][0] == 1.0

    s1 = InitialStateSampler.haar_random_states(10000, seed=42)
    s2 = InitialStateSampler.haar_random_states(10000, seed=42)
    assert s1.shape == (10000, 2)
    assert np.array_equal(s1, s2)
    assert np.allclose(np.linalg.norm(s1, axis=1), 1.0, atol=1e-13)

    # Bloch isotropy: E[r_i] ~ 0, E[r_i^2] ~ 1/3
    rx = 2.0 * np.real(s1[:, 0].conj() * s1[:, 1])
    ry = 2.0 * np.imag(s1[:, 0].conj() * s1[:, 1])
    rz = np.abs(s1[:, 0]) ** 2 - np.abs(s1[:, 1]) ** 2
    for r in (rx, ry, rz):
        assert abs(float(np.mean(r))) < 0.04
        assert abs(float(np.mean(r**2)) - 1.0 / 3.0) < 0.02

    normed = InitialStateSampler.normalize_statevector(np.array([3.0, 4.0j]))
    assert math.isclose(float(np.linalg.norm(normed)), 1.0, abs_tol=1e-14)

    with pytest.raises(ValueError):
        InitialStateSampler.haar_random_states(0, seed=1)
    with pytest.raises(ValueError):
        InitialStateSampler.normalize_statevector(np.array([0.0, 0.0]))


def test_single_qubit_tomography_accuracy_projection_and_batches() -> None:
    rng = np.random.default_rng(77)
    cards = InitialStateSampler.cardinal_states()
    rho_mixed = 0.75 * np.outer(cards["+"], cards["+"].conj()) + 0.25 * np.outer(
        cards["-"], cards["-"].conj()
    )

    errs = []
    for n_shots in (500, 5000, 50000):
        trial_errs = []
        for _ in range(25):
            rho_hat = SingleQubitTomography.reconstruct(rho_mixed, n_shots, rng)
            trial_errs.append(float(np.linalg.norm(rho_hat - rho_mixed, ord="fro")))
        errs.append(float(np.mean(trial_errs)))
    assert errs[0] > errs[1] > errs[2]
    assert errs[2] < 0.02

    # Unphysical expectation vector -> Smolin projection returns Bloch norm 1 PSD trace-1 matrix
    proj = SingleQubitTomography.reconstruct_from_expectations(0.9, 0.9, 0.0, project_physical=True)
    eigs = np.linalg.eigvalsh(proj)
    assert np.all(eigs >= -1e-12)
    assert math.isclose(float(np.real(np.trace(proj))), 1.0, abs_tol=1e-13)
    assert math.isclose(quantum_fidelity(proj, proj), 1.0, abs_tol=1e-12)

    # Unprojected linear inversion has unbiased sample mean
    samples = [
        SingleQubitTomography.reconstruct(rho_mixed, 2000, rng, project_physical=False)
        for _ in range(200)
    ]
    mean_rho = np.mean(samples, axis=0)
    assert np.max(np.abs(mean_rho - rho_mixed)) < 0.02

    # Batch inputs (B, 2) including B==2 ambiguity flag and (B, 2, 2)
    sv_b2 = np.vstack([cards["0"], cards["1"]])
    rho_b2 = SingleQubitTomography.reconstruct(
        sv_b2, 20000, rng, is_statevector_batch=True
    )
    assert np.max(np.abs(rho_b2 - 0.5 * np.eye(2))) < 0.03

    rho_b3 = SingleQubitTomography.reconstruct(
        np.broadcast_to(rho_mixed, (3, 2, 2)).copy(), 20000, rng
    )
    assert np.max(np.abs(rho_b3 - rho_mixed)) < 0.03


def test_quantum_fidelity_unnormalized_2d_density_matrices_within_tol() -> None:
    cards = InitialStateSampler.cardinal_states()
    eye_half = 0.5 * np.eye(2, dtype=np.complex128)
    plus = cards["+"]
    dm_plus = np.outer(plus, plus.conj())

    for delta in (-4e-7, 4e-7):
        psi_u = plus * (1.0 + delta)
        dm_u = np.outer(psi_u, psi_u.conj())
        dm_u_orig = dm_u.copy()
        dm_u_ro = dm_u.copy()
        dm_u_ro.setflags(write=False)

        assert abs(quantum_fidelity(dm_u, dm_u) - 1.0) < 1e-13
        assert abs(quantum_fidelity(dm_u, plus) - 1.0) < 1e-13
        assert abs(quantum_fidelity(plus, dm_u) - 1.0) < 1e-13
        assert abs(quantum_fidelity(dm_u, eye_half) - 0.5) < 1e-13
        assert abs(quantum_fidelity(eye_half, dm_u) - 0.5) < 1e-13
        assert abs(quantum_fidelity(dm_u_ro, eye_half) - 0.5) < 1e-13
        assert np.array_equal(dm_u, dm_u_orig)

        dm_scaled = dm_plus * (1.0 + delta)
        dm_scaled_orig = dm_scaled.copy()
        assert abs(quantum_fidelity(dm_scaled, dm_scaled) - 1.0) < 1e-13
        assert abs(quantum_fidelity(dm_scaled, dm_plus) - 1.0) < 1e-13
        assert abs(quantum_fidelity(dm_scaled, plus) - 1.0) < 1e-13
        assert abs(quantum_fidelity(dm_scaled, eye_half) - 0.5) < 1e-13
        assert np.array_equal(dm_scaled, dm_scaled_orig)

    # Mixed density matrix scaled within tol=1e-6
    rho_mix = 0.7 * dm_plus + 0.3 * np.outer(cards["-"], cards["-"].conj())
    for delta in (-4e-7, 4e-7):
        rho_mix_u = rho_mix * (1.0 + delta)
        assert abs(quantum_fidelity(rho_mix_u, rho_mix_u) - 1.0) < 1e-13
        assert abs(quantum_fidelity(rho_mix_u, rho_mix) - 1.0) < 1e-13
        assert abs(quantum_fidelity(rho_mix_u, plus) - 0.7) < 1e-13

    # Outside tol=1e-6 must still raise ValueError
    with pytest.raises(ValueError):
        quantum_fidelity(dm_plus * (1.0 + 1.5e-6), dm_plus)
    with pytest.raises(ValueError):
        quantum_fidelity(dm_plus * (1.0 - 1.5e-6), dm_plus)
    psi_too_far_for_dm = plus * (1.0 + 6e-7)  # trace ~ 1 + 1.2e-6 > 1 + 1e-6
    with pytest.raises(ValueError):
        quantum_fidelity(np.outer(psi_too_far_for_dm, psi_too_far_for_dm.conj()), dm_plus)


def test_single_qubit_tomography_unnormalized_batches_within_tol() -> None:
    cards = InitialStateSampler.cardinal_states()

    # (1 + 6e-7)-norm (B, 2) statevector batch must normalize cleanly without crashing
    sv_batch_hi = np.vstack([cards["0"] * (1.0 + 6e-7), cards["1"] * (1.0 + 6e-7)])
    sv_batch_hi_orig = sv_batch_hi.copy()
    sv_batch_hi_ro = sv_batch_hi.copy()
    sv_batch_hi_ro.setflags(write=False)

    rho_hi = SingleQubitTomography.reconstruct(
        sv_batch_hi, 20000, np.random.default_rng(1), is_statevector_batch=True
    )
    assert abs(float(np.real(np.trace(rho_hi))) - 1.0) < 1e-13
    assert np.all(np.linalg.eigvalsh(rho_hi) >= -1e-12)
    assert np.max(np.abs(rho_hi - 0.5 * np.eye(2))) < 0.03
    assert np.array_equal(sv_batch_hi, sv_batch_hi_orig)

    rho_hi_ro = SingleQubitTomography.reconstruct(
        sv_batch_hi_ro, 1000, np.random.default_rng(2), is_statevector_batch=True
    )
    assert abs(float(np.real(np.trace(rho_hi_ro))) - 1.0) < 1e-13

    # (1 - 4e-7)-norm |0> batch must produce unbiased Z expectations (p_plus == 1.0 -> r_z == 1.0)
    sv_batch_lo = np.tile(cards["0"] * (1.0 - 4e-7), (4, 1))
    sv_batch_lo_orig = sv_batch_lo.copy()
    rho_lo = SingleQubitTomography.reconstruct(
        sv_batch_lo, 10_000_000, np.random.default_rng(3), project_physical=False
    )
    assert np.array_equal(sv_batch_lo, sv_batch_lo_orig)
    assert abs(float(np.real(np.trace(rho_lo))) - 1.0) < 1e-13
    assert abs(float(np.real(rho_lo[1, 1]))) < 1e-13
    assert abs(float(np.real(rho_lo[0, 0])) - 1.0) < 1e-13

    # (B, 2, 2) density-matrix batch with opposing trace perturbations within tol=1e-6
    dm_0 = np.outer(cards["0"], cards["0"].conj())
    dm_1 = np.outer(cards["1"], cards["1"].conj())
    dm_batch = np.stack([dm_0 * (1.0 + 8e-7), dm_1 * (1.0 - 8e-7)], axis=0)
    dm_batch_orig = dm_batch.copy()
    rho_dm_batch = SingleQubitTomography.reconstruct(
        dm_batch, 100_000, np.random.default_rng(4), project_physical=False
    )
    assert np.array_equal(dm_batch, dm_batch_orig)
    assert abs(float(np.real(np.trace(rho_dm_batch))) - 1.0) < 1e-13
    assert np.max(np.abs(rho_dm_batch - 0.5 * np.eye(2))) < 0.015

    # Outside tol=1e-6 must still raise ValueError
    with pytest.raises(ValueError):
        SingleQubitTomography.reconstruct(
            np.vstack([cards["0"] * (1.0 + 1.5e-6), cards["1"]]),
            1000,
            np.random.default_rng(5),
            is_statevector_batch=True,
        )
    with pytest.raises(ValueError):
        SingleQubitTomography.reconstruct(
            np.stack([dm_0 * (1.0 + 1.5e-6), dm_1], axis=0),
            1000,
            np.random.default_rng(6),
        )
