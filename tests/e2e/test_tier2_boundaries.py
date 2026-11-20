"""Tier 2 E2E tests: boundary, corner, and error-handling cases across Features F1-F9."""

from __future__ import annotations

import inspect
import math
import re
import subprocess

import numpy as np
import pytest
from scipy import stats

pytestmark = pytest.mark.tier2


# ---------------------------------------------------------------------------
# Feature F1 Boundaries: BitFlip3QubitCode
# ---------------------------------------------------------------------------


def test_f1_bit_flip_3_plus_eigenstate_invariance_under_ex(
    qecc_codes, qecc_channels, qecc_metrics, numpy_backend, helpers
) -> None:
    code = qecc_codes.get_code("bit_flip_3")
    psi = helpers.canonical_states()["|+>"]
    rho_in = helpers.ket_to_rho(psi)

    res_mc = numpy_backend.simulate_cycle(
        code,
        psi,
        channel=qecc_channels.ChannelSpec("bit_flip", 0.3),
        p=0.3,
        num_shots=500,
        seed=105,
    )
    assert res_mc.mean_fidelity > 1.0 - 1e-10
    assert qecc_metrics.quantum_fidelity(rho_in, res_mc.rho_out) > 1.0 - 1e-10

    for err in ("XII", "XXX"):
        res_det = numpy_backend.simulate_cycle(
            code, psi, channel=None, p=0.0, num_shots=1, deterministic_errors=[err]
        )
        assert res_det.mean_fidelity > 1.0 - 1e-10
        assert qecc_metrics.quantum_fidelity(rho_in, res_det.rho_out) > 1.0 - 1e-10


def test_f1_bit_flip_3_weight_2_and_3_x_errors_cause_logical_bit_flip(
    qecc_codes, qecc_metrics, numpy_backend, helpers
) -> None:
    code = qecc_codes.get_code("bit_flip_3")
    canonical = helpers.canonical_states()
    rho0 = helpers.ket_to_rho(canonical["|0>"])
    rho1 = helpers.ket_to_rho(canonical["|1>"])
    aliased = {"XXI": "01", "XIX": "11", "IXX": "10", "XXX": "00"}

    for err, expected_synd in aliased.items():
        res0 = numpy_backend.simulate_cycle(
            code, canonical["|0>"], channel=None, p=0.0, num_shots=1, deterministic_errors=[err]
        )
        assert str(np.asarray(res0.syndromes).tolist()[0]) == expected_synd
        assert qecc_metrics.quantum_fidelity(rho0, res0.rho_out) < 1e-10
        assert qecc_metrics.quantum_fidelity(rho1, res0.rho_out) > 1.0 - 1e-10

        res1 = numpy_backend.simulate_cycle(
            code, canonical["|1>"], channel=None, p=0.0, num_shots=1, deterministic_errors=[err]
        )
        assert qecc_metrics.quantum_fidelity(rho1, res1.rho_out) < 1e-10
        assert qecc_metrics.quantum_fidelity(rho0, res1.rho_out) > 1.0 - 1e-10


def test_f1_bit_flip_3_single_z_undetected_00_syndrome(
    qecc_codes, qecc_metrics, numpy_backend, helpers
) -> None:
    code = qecc_codes.get_code("bit_flip_3")
    canonical = helpers.canonical_states()
    rhop = helpers.ket_to_rho(canonical["|+>"])
    rhom = helpers.ket_to_rho(canonical["|->"])

    for err in ("ZII", "IZI", "IIZ"):
        res = numpy_backend.simulate_cycle(
            code, canonical["|+>"], channel=None, p=0.0, num_shots=1, deterministic_errors=[err]
        )
        assert str(np.asarray(res.syndromes).tolist()[0]) == "00"
        assert qecc_metrics.quantum_fidelity(rhop, res.rho_out) < 1e-10
        assert qecc_metrics.quantum_fidelity(rhom, res.rho_out) > 1.0 - 1e-10


def test_f1_bit_flip_3_boundary_p_zero_and_half(
    qecc_codes, qecc_channels, numpy_backend, helpers
) -> None:
    code = qecc_codes.get_code("bit_flip_3")
    ket0 = helpers.canonical_states()["|0>"]

    res0 = numpy_backend.simulate_cycle(
        code, ket0, channel=qecc_channels.ChannelSpec("bit_flip", 0.0), p=0.0, num_shots=20, seed=106
    )
    assert res0.mean_fidelity > 1.0 - 1e-10
    assert res0.syndrome_counts == {"00": 20}

    res_half = numpy_backend.simulate_cycle(
        code, ket0, channel=qecc_channels.ChannelSpec("bit_flip", 0.5), p=0.5, num_shots=4000, seed=106
    )
    assert abs(res_half.mean_fidelity - 0.5) < 4.0 * math.sqrt(0.25 / 4000)


def test_f1_bit_flip_3_invalid_error_strings_and_syndrome_lookup(
    qecc_codes, qecc_channels, numpy_backend, helpers
) -> None:
    code = qecc_codes.get_code("bit_flip_3")
    ket0 = helpers.canonical_states()["|0>"]

    for bad_err in ("X", "XIII", "XIA", ""):
        with pytest.raises(ValueError):
            qecc_channels.DeterministicErrorInjector.validate_pauli_string(bad_err, 3)
        with pytest.raises(ValueError):
            numpy_backend.simulate_cycle(
                code, ket0, channel=None, p=0.0, num_shots=1, deterministic_errors=[bad_err]
            )

    for bad_synd in ("000", "invalid"):
        with pytest.raises((KeyError, ValueError)):
            code.recovery_ops(bad_synd)


# ---------------------------------------------------------------------------
# Feature F2 Boundaries: PhaseFlip3QubitCode
# ---------------------------------------------------------------------------


def test_f2_phase_flip_3_eigenstate_boundary_ket0_raw_vs_ketplus_encoded(
    qecc_codes, qecc_channels, qecc_metrics, numpy_backend, helpers
) -> None:
    code = qecc_codes.get_code("phase_flip_3")
    canonical = helpers.canonical_states()
    ket0 = canonical["|0>"]
    ketp = canonical["|+>"]
    ch = qecc_channels.ChannelSpec("phase_flip", 0.3)

    res0 = numpy_backend.simulate_cycle(code, ket0, channel=ch, p=0.3, num_shots=1000, seed=205)
    expected_0 = 1.0 - 3.0 * (0.3**2) + 2.0 * (0.3**3)
    assert res0.mean_fidelity < 0.85
    assert abs(res0.mean_fidelity - expected_0) < 4.0 * math.sqrt(expected_0 * (1.0 - expected_0) / 1000)

    res0_zzz = numpy_backend.simulate_cycle(
        code, ket0, channel=None, p=0.0, num_shots=1, deterministic_errors=["ZZZ"]
    )
    assert qecc_metrics.quantum_fidelity(helpers.ket_to_rho(ket0), res0_zzz.rho_out) < 1e-10

    resp = numpy_backend.simulate_cycle(code, ketp, channel=ch, p=0.3, num_shots=1000, seed=205)
    resp_zzz = numpy_backend.simulate_cycle(
        code, ketp, channel=None, p=0.0, num_shots=1, deterministic_errors=["ZZZ"]
    )
    assert resp.mean_fidelity > 1.0 - 1e-10
    assert resp_zzz.mean_fidelity > 1.0 - 1e-10


def test_f2_phase_flip_3_weight_2_and_3_z_errors_cause_logical_flip(
    qecc_codes, qecc_metrics, numpy_backend, helpers
) -> None:
    code = qecc_codes.get_code("phase_flip_3")
    canonical = helpers.canonical_states()
    rho0 = helpers.ket_to_rho(canonical["|0>"])
    rho1 = helpers.ket_to_rho(canonical["|1>"])
    rhopi = helpers.ket_to_rho(canonical["|+i>"])
    aliased = {"ZZI": "01", "ZIZ": "11", "IZZ": "10", "ZZZ": "00"}

    for err, expected_synd in aliased.items():
        res0 = numpy_backend.simulate_cycle(
            code, canonical["|0>"], channel=None, p=0.0, num_shots=1, deterministic_errors=[err]
        )
        assert str(np.asarray(res0.syndromes).tolist()[0]) == expected_synd
        assert qecc_metrics.quantum_fidelity(rho0, res0.rho_out) < 1e-10
        assert qecc_metrics.quantum_fidelity(rho1, res0.rho_out) > 1.0 - 1e-10

        res_pi = numpy_backend.simulate_cycle(
            code, canonical["|+i>"], channel=None, p=0.0, num_shots=1, deterministic_errors=[err]
        )
        assert qecc_metrics.quantum_fidelity(rhopi, res_pi.rho_out) < 1e-10


def test_f2_phase_flip_3_single_x_undetected_00_syndrome(
    qecc_codes, qecc_metrics, numpy_backend, helpers
) -> None:
    code = qecc_codes.get_code("phase_flip_3")
    canonical = helpers.canonical_states()
    rhop = helpers.ket_to_rho(canonical["|+>"])
    rhom = helpers.ket_to_rho(canonical["|->"])

    for err in ("XII", "IXI", "IIX"):
        res = numpy_backend.simulate_cycle(
            code, canonical["|+>"], channel=None, p=0.0, num_shots=1, deterministic_errors=[err]
        )
        assert str(np.asarray(res.syndromes).tolist()[0]) == "00"
        assert qecc_metrics.quantum_fidelity(rhop, res.rho_out) < 1e-10
        assert qecc_metrics.quantum_fidelity(rhom, res.rho_out) > 1.0 - 1e-10


def test_f2_phase_flip_3_boundary_p_zero_and_half(
    qecc_codes, qecc_channels, numpy_backend, helpers
) -> None:
    code = qecc_codes.get_code("phase_flip_3")
    ketpi = helpers.canonical_states()["|+i>"]

    res0 = numpy_backend.simulate_cycle(
        code, ketpi, channel=qecc_channels.ChannelSpec("phase_flip", 0.0), p=0.0, num_shots=20, seed=206
    )
    assert res0.mean_fidelity > 1.0 - 1e-10
    assert res0.syndrome_counts == {"00": 20}

    res_half = numpy_backend.simulate_cycle(
        code, ketpi, channel=qecc_channels.ChannelSpec("phase_flip", 0.5), p=0.5, num_shots=4000, seed=206
    )
    assert abs(res_half.mean_fidelity - 0.5) < 4.0 * math.sqrt(0.25 / 4000)


def test_f2_phase_flip_3_invalid_error_strings_and_syndrome_lookup(
    qecc_codes, qecc_channels, numpy_backend, helpers
) -> None:
    code = qecc_codes.get_code("phase_flip_3")
    ketp = helpers.canonical_states()["|+>"]

    for bad_err in ("Z", "ZIII", "ZIB", ""):
        with pytest.raises(ValueError):
            qecc_channels.DeterministicErrorInjector.validate_pauli_string(bad_err, 3)
        with pytest.raises(ValueError):
            numpy_backend.simulate_cycle(
                code, ketp, channel=None, p=0.0, num_shots=1, deterministic_errors=[bad_err]
            )

    for bad_synd in ("000", "bad"):
        with pytest.raises((KeyError, ValueError)):
            code.recovery_ops(bad_synd)


# ---------------------------------------------------------------------------
# Feature F3 Boundaries: FiveQubitPerfectCode
# ---------------------------------------------------------------------------


def test_f3_five_qubit_weight_2_errors_fail_recovery(
    qecc_codes, qecc_metrics, numpy_backend, helpers
) -> None:
    code = qecc_codes.get_code("five_qubit")
    psi = qecc_metrics.InitialStateSampler().haar_random_states(1, seed=306)[0]
    rho_in = helpers.ket_to_rho(psi)

    wt2_cases = {"XXIII": "1001", "ZZIII": "1111", "XZIII": "0100", "YYIII": "0110"}
    for err, expected_synd in wt2_cases.items():
        res = numpy_backend.simulate_cycle(
            code, psi, channel=None, p=0.0, num_shots=1, deterministic_errors=[err]
        )
        assert str(np.asarray(res.syndromes).tolist()[0]) == expected_synd
        assert res.mean_fidelity < 0.99
        assert qecc_metrics.quantum_fidelity(rho_in, res.rho_out) < 0.99

    canonical = helpers.canonical_states()
    res_xx = numpy_backend.simulate_cycle(
        code, canonical["|+>"], channel=None, p=0.0, num_shots=1, deterministic_errors=["XXIII"]
    )
    assert qecc_metrics.quantum_fidelity(helpers.ket_to_rho(canonical["|+>"]), res_xx.rho_out) < 1e-10

    for err in ("ZZIII", "XZIII"):
        res_0 = numpy_backend.simulate_cycle(
            code, canonical["|0>"], channel=None, p=0.0, num_shots=1, deterministic_errors=[err]
        )
        assert qecc_metrics.quantum_fidelity(helpers.ket_to_rho(canonical["|0>"]), res_0.rho_out) < 1e-10


def test_f3_five_qubit_weight_3_4_5_stabilizer_coset_recover_identity(
    qecc_codes, qecc_metrics, numpy_backend, helpers
) -> None:
    code = qecc_codes.get_code("five_qubit")
    haar_state = qecc_metrics.InitialStateSampler().haar_random_states(1, seed=307)[0]
    states = [haar_state, helpers.canonical_states()["|+i>"]]
    coset_cases = {"IZZXI": "0001", "XZZXI": "0000", "XZZXZ": "0100"}

    for psi in states:
        rho_in = helpers.ket_to_rho(psi)
        for err, expected_synd in coset_cases.items():
            res = numpy_backend.simulate_cycle(
                code, psi, channel=None, p=0.0, num_shots=1, deterministic_errors=[err]
            )
            assert str(np.asarray(res.syndromes).tolist()[0]) == expected_synd
            assert res.mean_fidelity > 1.0 - 1e-10
            assert qecc_metrics.quantum_fidelity(rho_in, res.rho_out) > 1.0 - 1e-10


def test_f3_five_qubit_invalid_syndrome_lookup_and_error_strings(
    qecc_codes, qecc_channels, numpy_backend, helpers
) -> None:
    code = qecc_codes.get_code("five_qubit")
    ket0 = helpers.canonical_states()["|0>"]

    for bad_err in ("XII", "IIIIIX", "QIIII", ""):
        with pytest.raises(ValueError):
            qecc_channels.DeterministicErrorInjector.validate_pauli_string(bad_err, 5)
        with pytest.raises(ValueError):
            numpy_backend.simulate_cycle(
                code, ket0, channel=None, p=0.0, num_shots=1, deterministic_errors=[bad_err]
            )

    for bad_synd in ("000", "11111", "abcd", ""):
        with pytest.raises((KeyError, ValueError)):
            code.recovery_ops(bad_synd)


def test_f3_five_qubit_boundary_p_zero(
    qecc_codes, qecc_channels, qecc_metrics, numpy_backend, helpers
) -> None:
    code = qecc_codes.get_code("five_qubit")
    ketpi = helpers.canonical_states()["|+i>"]
    res = numpy_backend.simulate_cycle(
        code, ketpi, channel=qecc_channels.ChannelSpec("depolarizing", 0.0), p=0.0, num_shots=25, seed=308
    )
    assert res.mean_fidelity > 1.0 - 1e-10
    assert qecc_metrics.quantum_fidelity(helpers.ket_to_rho(ketpi), res.rho_out) > 1.0 - 1e-10
    assert res.syndrome_counts == {"0000": 25}


def test_f3_five_qubit_boundary_p_half(
    qecc_codes, qecc_channels, qecc_analysis, numpy_backend, helpers
) -> None:
    code = qecc_codes.get_code("five_qubit")
    ketpi = helpers.canonical_states()["|+i>"]
    res = numpy_backend.simulate_cycle(
        code, ketpi, channel=qecc_channels.ChannelSpec("depolarizing", 0.5), p=0.5, num_shots=4000, seed=309
    )
    expected = float(qecc_analysis.analytical_fidelity("five_qubit", "depolarizing", 0.5, metric="state"))
    assert abs(res.mean_fidelity - expected) < 4.0 * math.sqrt(0.25 / 4000)
    assert res.mean_fidelity < 2.0 / 3.0


# ---------------------------------------------------------------------------
# Feature F4 Boundaries: TwoQubitDFSDemo
# ---------------------------------------------------------------------------


def test_f4_dfs_2_independent_dephasing_breaks_dfs(
    qecc_codes, qecc_channels, qecc_metrics, numpy_backend, helpers
) -> None:
    code = qecc_codes.get_code("dfs_2")
    ketp = helpers.canonical_states()["|+>"]
    rhop = helpers.ket_to_rho(ketp)

    res = numpy_backend.simulate_cycle(
        code, ketp, channel=qecc_channels.ChannelSpec("phase_flip", 0.3), p=0.3, num_shots=2000, seed=404
    )
    assert res.mean_fidelity < 0.70
    assert abs(res.mean_fidelity - 0.58) < 4.0 * math.sqrt(0.25 / 2000)

    for err in ("ZI", "IZ"):
        res_det = numpy_backend.simulate_cycle(
            code, ketp, channel=None, p=0.0, num_shots=1, deterministic_errors=[err]
        )
        assert qecc_metrics.quantum_fidelity(rhop, res_det.rho_out) < 1e-10


def test_f4_dfs_2_collective_bit_flip_xx_vs_bell_singlet(
    qecc_codes, qecc_metrics, numpy_backend, helpers
) -> None:
    code = qecc_codes.get_code("dfs_2")
    canonical = helpers.canonical_states()

    for key in ("|0>", "|+i>"):
        psi = canonical[key]
        res = numpy_backend.simulate_cycle(
            code, psi, channel=None, p=0.0, num_shots=1, deterministic_errors=["XX"]
        )
        assert qecc_metrics.quantum_fidelity(helpers.ket_to_rho(psi), res.rho_out) < 1e-10

    ketm = canonical["|->"]
    res_singlet = numpy_backend.simulate_cycle(
        code, ketm, channel=None, p=0.0, num_shots=1, deterministic_errors=["XX"]
    )
    assert qecc_metrics.quantum_fidelity(helpers.ket_to_rho(ketm), res_singlet.rho_out) > 1.0 - 1e-10


def test_f4_dfs_2_boundary_p_zero(
    qecc_codes, qecc_channels, qecc_metrics, numpy_backend, helpers
) -> None:
    code = qecc_codes.get_code("dfs_2")
    psi = qecc_metrics.InitialStateSampler().haar_random_states(1, seed=405)[0]
    res = numpy_backend.simulate_cycle(
        code,
        psi,
        channel=qecc_channels.ChannelSpec("collective_dephasing", 0.0),
        p=0.0,
        num_shots=25,
        seed=405,
    )
    assert res.mean_fidelity > 1.0 - 1e-10
    assert qecc_metrics.quantum_fidelity(helpers.ket_to_rho(psi), res.rho_out) > 1.0 - 1e-10


def test_f4_dfs_2_boundary_p_half_and_one(
    qecc_codes, qecc_channels, qecc_metrics, numpy_backend, helpers
) -> None:
    code = qecc_codes.get_code("dfs_2")
    psi = qecc_metrics.InitialStateSampler().haar_random_states(1, seed=406)[0]
    states = [helpers.canonical_states()["|+i>"], psi]

    for p, n_shots in ((0.5, 500), (1.0, 100)):
        ch = qecc_channels.ChannelSpec("collective_dephasing", p)
        for st in states:
            res = numpy_backend.simulate_cycle(code, st, channel=ch, p=p, num_shots=n_shots, seed=406)
            assert res.mean_fidelity > 1.0 - 1e-10
            assert qecc_metrics.quantum_fidelity(helpers.ket_to_rho(st), res.rho_out) > 1.0 - 1e-10


def test_f4_dfs_2_invalid_p_and_invalid_deterministic_errors(
    qecc_codes, qecc_channels, numpy_backend, helpers
) -> None:
    code = qecc_codes.get_code("dfs_2")
    ket0 = helpers.canonical_states()["|0>"]

    for bad_p in (-0.01, 1.01, float("nan")):
        with pytest.raises(ValueError):
            qecc_channels.ChannelSpec("collective_dephasing", bad_p)

    for bad_err in ("Z", "ZZZ", "ZQ", ""):
        with pytest.raises(ValueError):
            qecc_channels.DeterministicErrorInjector.validate_pauli_string(bad_err, 2)
        with pytest.raises(ValueError):
            numpy_backend.simulate_cycle(
                code, ket0, channel=None, p=0.0, num_shots=1, deterministic_errors=[bad_err]
            )


# ---------------------------------------------------------------------------
# Feature F5 Boundaries: PauliChannelsAndInjectors
# ---------------------------------------------------------------------------


def test_t2_f5_p_zero_100_percent_identity(qecc_channels, helpers) -> None:
    rng = np.random.default_rng(501)
    for ctype in ("bit_flip", "phase_flip", "depolarizing", "collective_dephasing"):
        ch = qecc_channels.StochasticPauliChannel(qecc_channels.ChannelSpec(ctype, 0.0))
        samples = helpers.normalize_pauli_samples(ch.sample(5, 5000, rng), 5)
        assert np.all(samples == "I")


def test_t2_f5_p_one_100_percent_non_identity(qecc_channels, helpers) -> None:
    rng = np.random.default_rng(502)
    n_shots = 5000

    s_bf = helpers.normalize_pauli_samples(
        qecc_channels.StochasticPauliChannel(qecc_channels.ChannelSpec("bit_flip", 1.0)).sample(5, n_shots, rng),
        5,
    )
    assert np.all(s_bf == "X")

    s_pf = helpers.normalize_pauli_samples(
        qecc_channels.StochasticPauliChannel(qecc_channels.ChannelSpec("phase_flip", 1.0)).sample(5, n_shots, rng),
        5,
    )
    assert np.all(s_pf == "Z")

    s_cd = helpers.normalize_pauli_samples(
        qecc_channels.StochasticPauliChannel(qecc_channels.ChannelSpec("collective_dephasing", 1.0)).sample(
            5, n_shots, rng
        ),
        5,
    )
    assert np.all(s_cd == "Z")

    s_dep = helpers.normalize_pauli_samples(
        qecc_channels.StochasticPauliChannel(qecc_channels.ChannelSpec("depolarizing", 1.0)).sample(
            5, n_shots, rng
        ),
        5,
    )
    assert not np.any(s_dep == "I")
    tol = helpers.binom_sigma(1.0 / 3.0, n_shots, z=4.5)
    for sym in ("X", "Y", "Z"):
        assert abs(float(np.mean(s_dep[:, 0] == sym)) - (1.0 / 3.0)) <= tol


def test_t2_f5_invalid_probability_raises_value_error(qecc_channels) -> None:
    for bad_p in (-0.01, -1e-9, 1.0001, 1.5, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            qecc_channels.ChannelSpec("depolarizing", bad_p)


def test_t2_f5_invalid_channel_type_raises_value_error(qecc_channels) -> None:
    for bad_ct in ("amplitude_damping", "", "DEPOLARIZING"):
        with pytest.raises(ValueError):
            qecc_channels.ChannelSpec(bad_ct, 0.1)


def test_t2_f5_deterministic_injector_invalid_strings(qecc_channels) -> None:
    for bad_str, n_q in (("", 3), ("IX", 3), ("IXYZ", 3), ("IAI", 3), ("I1I", 3), ("IxI", 3)):
        with pytest.raises(ValueError):
            qecc_channels.DeterministicErrorInjector.validate_pauli_string(bad_str, n_q)


def test_t2_f5_minimal_and_zero_qubit_shot_boundaries(qecc_channels, helpers) -> None:
    rng = np.random.default_rng(503)
    ch = qecc_channels.StochasticPauliChannel(qecc_channels.ChannelSpec("depolarizing", 0.5))
    one = helpers.normalize_pauli_samples(ch.sample(1, 1, rng), 1)
    assert one.shape == (1, 1)
    assert one[0, 0] in {"I", "X", "Y", "Z"}

    with pytest.raises(ValueError):
        ch.sample(0, 10, rng)
    with pytest.raises(ValueError):
        ch.sample(2, 0, rng)
    with pytest.raises(ValueError):
        ch.sample(2, -5, rng)


# ---------------------------------------------------------------------------
# Feature F6 Boundaries: QuantumFidelityAndTomography
# ---------------------------------------------------------------------------


def test_t2_f6_orthogonal_states_zero_fidelity(qecc_metrics, helpers) -> None:
    canonical = helpers.canonical_states()
    for a, b in (("|0>", "|1>"), ("|+>", "|->"), ("|+i>", "|-i>")):
        f_ab = qecc_metrics.quantum_fidelity(canonical[a], canonical[b])
        assert f_ab >= 0.0
        assert f_ab == pytest.approx(0.0, abs=1e-12)


def test_t2_f6_maximally_mixed_state_fidelity(qecc_metrics) -> None:
    sampler = qecc_metrics.InitialStateSampler()
    half_eye = 0.5 * np.eye(2, dtype=np.complex128)
    states = list(sampler.cardinal_states().values()) + list(sampler.haar_random_states(5, seed=601))
    for psi in states:
        assert qecc_metrics.quantum_fidelity(psi, half_eye) == pytest.approx(0.5, abs=1e-12)
    assert qecc_metrics.quantum_fidelity(half_eye, half_eye) == pytest.approx(1.0, abs=1e-12)


def test_t2_f6_rank_deficient_and_negative_roundoff_eigenvalues(qecc_metrics, helpers) -> None:
    canonical = helpers.canonical_states()
    u = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128) / math.sqrt(2.0)
    v = np.eye(2, dtype=np.complex128)
    rho_noisy = u @ np.diag([1.0 + 1e-16, -1e-16]) @ u.conj().T
    sigma_noisy = v @ np.diag([-1e-15, 1.0 + 1e-15]) @ v.conj().T

    f_val = qecc_metrics.quantum_fidelity(rho_noisy, sigma_noisy)
    assert math.isfinite(f_val) and 0.0 <= f_val <= 1.0
    clean = qecc_metrics.quantum_fidelity(canonical["|+>"], canonical["|1>"])
    assert abs(f_val - clean) < 1e-10


def test_t2_f6_zero_norm_and_invalid_state_raises(
    qecc_codes, qecc_metrics, numpy_backend
) -> None:
    ket_zero_vec = np.zeros(2, dtype=np.complex128)
    rho_zero_mat = np.zeros((2, 2), dtype=np.complex128)
    valid = np.array([1.0, 0.0], dtype=np.complex128)

    with pytest.raises(ValueError):
        qecc_metrics.quantum_fidelity(ket_zero_vec, valid)
    with pytest.raises(ValueError):
        qecc_metrics.quantum_fidelity(rho_zero_mat, valid)

    code = qecc_codes.get_code("bit_flip_3")
    with pytest.raises(ValueError):
        numpy_backend.simulate_cycle(code, ket_zero_vec, channel=None, p=0.0, num_shots=1)

    with pytest.raises(ValueError):
        qecc_metrics.InitialStateSampler().haar_random_states(n_states=0, seed=1)
    with pytest.raises(ValueError):
        qecc_metrics.InitialStateSampler().haar_random_states(n_states=-3, seed=1)


def test_t2_f6_tomography_shots_per_axis_bounds(qecc_metrics, helpers) -> None:
    rho_true = helpers.ket_to_rho(helpers.canonical_states()["|+i>"])
    errs_low: list[float] = []
    errs_high: list[float] = []
    fids_high: list[float] = []

    for seed in range(50, 60):
        hat_low = qecc_metrics.SingleQubitTomography.reconstruct(
            rho_true, shots_per_axis=100, rng=np.random.default_rng(seed)
        )
        hat_high = qecc_metrics.SingleQubitTomography.reconstruct(
            rho_true, shots_per_axis=25_000, rng=np.random.default_rng(seed + 100)
        )
        errs_low.append(float(np.linalg.norm(rho_true - hat_low, ord="fro")))
        errs_high.append(float(np.linalg.norm(rho_true - hat_high, ord="fro")))
        fids_high.append(float(qecc_metrics.quantum_fidelity(rho_true, hat_high)))

    assert float(np.mean(errs_high)) < 0.03
    assert float(np.mean(fids_high)) > 0.995
    assert float(np.mean(errs_low)) < 0.35
    assert float(np.mean(errs_high)) < float(np.mean(errs_low))

    for bad_shots in (0, -10):
        with pytest.raises(ValueError):
            qecc_metrics.SingleQubitTomography.reconstruct(
                rho_true, shots_per_axis=bad_shots, rng=np.random.default_rng(1)
            )


def test_t2_f6_partial_trace_invalid_dimension_raises(qecc_metrics) -> None:
    with pytest.raises(ValueError):
        qecc_metrics.partial_trace_to_qubit0(np.ones(5, dtype=np.complex128) / math.sqrt(5.0), 3)
    with pytest.raises(ValueError):
        qecc_metrics.partial_trace_to_qubit0(np.eye(4, dtype=np.complex128) / 4.0, 3)


# ---------------------------------------------------------------------------
# Feature F7 Boundaries: AnalyticalThresholdsSweepsAndPlots
# ---------------------------------------------------------------------------


def test_t2_f7_p_zero_fidelity_equals_one(qecc_analysis) -> None:
    for code_name, channel_type in (
        ("bit_flip_3", "bit_flip"),
        ("phase_flip_3", "phase_flip"),
        ("five_qubit", "depolarizing"),
    ):
        assert float(qecc_analysis.analytical_fidelity(code_name, channel_type, 0.0)) == pytest.approx(
            1.0, abs=1e-12
        )


def test_t2_f7_p_half_three_qubit_intersection(qecc_analysis) -> None:
    assert float(qecc_analysis.analytical_fidelity("bit_flip_3", "bit_flip", 0.5)) == pytest.approx(
        0.5, abs=1e-12
    )
    assert float(qecc_analysis.analytical_fidelity("phase_flip_3", "phase_flip", 0.5)) == pytest.approx(
        0.5, abs=1e-12
    )


def test_t2_f7_three_qubit_under_depolarizing_zero_threshold(qecc_analysis) -> None:
    assert float(qecc_analysis.analytical_gain_threshold("bit_flip_3", "depolarizing")) == pytest.approx(
        0.0, abs=1e-12
    )
    assert float(qecc_analysis.analytical_gain_threshold("phase_flip_3", "depolarizing")) == pytest.approx(
        0.0, abs=1e-12
    )


def test_t2_f7_five_qubit_pure_xz_vs_depolarizing_distinction(qecc_analysis) -> None:
    th_bf = float(qecc_analysis.analytical_gain_threshold("five_qubit", "bit_flip"))
    th_pf = float(qecc_analysis.analytical_gain_threshold("five_qubit", "phase_flip"))
    th_dep = float(qecc_analysis.analytical_gain_threshold("five_qubit", "depolarizing"))

    assert th_bf == pytest.approx(0.1311231479, abs=1e-6)
    assert th_pf == pytest.approx(0.1311231479, abs=1e-6)
    assert th_dep == pytest.approx(0.1376275643, abs=1e-6)
    assert abs(th_dep - th_bf) > 0.006


def test_t2_f7_invalid_code_or_channel_raises(qecc_analysis) -> None:
    for code_name, channel_type in (("unknown_code", "bit_flip"), ("bit_flip_3", "unknown_channel")):
        with pytest.raises(ValueError):
            qecc_analysis.analytical_fidelity(code_name, channel_type, 0.1)
        with pytest.raises(ValueError):
            qecc_analysis.analytical_gain_threshold(code_name, channel_type)


def test_t2_f7_scalar_vs_array_p_and_out_of_range(qecc_analysis) -> None:
    f_scalar = qecc_analysis.analytical_fidelity("five_qubit", "depolarizing", 0.1)
    assert isinstance(f_scalar, (float, np.floating))

    p_arr = np.array([0.0, 0.1376275643, 0.5], dtype=np.float64)
    f_arr = qecc_analysis.analytical_fidelity("five_qubit", "depolarizing", p_arr)
    assert isinstance(f_arr, np.ndarray)
    assert f_arr.shape == p_arr.shape

    for bad_p in (-0.1, 1.2):
        with pytest.raises(ValueError):
            qecc_analysis.analytical_fidelity("bit_flip_3", "bit_flip", bad_p)


# ---------------------------------------------------------------------------
# Feature F8 Boundaries: SwappableBackendsCUDAAndBenchmarks
# ---------------------------------------------------------------------------


def test_t2_f8_invalid_backend_name_raises(qecc_backends) -> None:
    with pytest.raises((ValueError, KeyError)):
        qecc_backends.get_backend("nonexistent_backend_xyz")


def test_t2_f8_num_shots_one_edge_case(
    qecc_backends, qecc_codes, qecc_channels, helpers
) -> None:
    code = qecc_codes.get_code("five_qubit")
    psi = helpers.canonical_states()["|+>"]

    for bname in ("numpy", "torch_cuda"):
        backend = qecc_backends.get_backend(bname)
        for ch, p in ((None, 0.0), (qecc_channels.ChannelSpec("depolarizing", 0.1), 0.1)):
            res = backend.simulate_cycle(code, psi, channel=ch, p=p, num_shots=1, seed=7)
            assert len(res.fidelities) == 1
            assert sum(res.syndrome_counts.values()) == 1
            assert math.isfinite(float(res.std_error)) and float(res.std_error) >= 0.0


def test_t2_f8_num_shots_zero_or_negative_raises(
    qecc_backends, qecc_codes, helpers
) -> None:
    code = qecc_codes.get_code("bit_flip_3")
    psi = helpers.canonical_states()["|0>"]

    for bname in ("numpy", "torch_cuda"):
        backend = qecc_backends.get_backend(bname)
        for bad_shots in (0, -5):
            with pytest.raises(ValueError):
                backend.simulate_cycle(code, psi, channel=None, p=0.0, num_shots=bad_shots)


def test_t2_f8_channel_none_and_deterministic_precedence(
    qecc_codes, qecc_channels, numpy_backend, helpers
) -> None:
    code = qecc_codes.get_code("five_qubit")
    psi = helpers.canonical_states()["|+i>"]

    res_none = numpy_backend.simulate_cycle(code, psi, channel=None, p=0.0, num_shots=10, seed=81)
    assert res_none.syndrome_counts == {"0000": 10}
    assert res_none.mean_fidelity > 1.0 - 1e-10

    res_det = numpy_backend.simulate_cycle(
        code, psi, channel=None, p=0.0, num_shots=10, deterministic_errors=["XIIII"]
    )
    assert res_det.syndrome_counts == {"0001": 10}
    assert res_det.mean_fidelity > 1.0 - 1e-10

    ch50 = qecc_channels.ChannelSpec("depolarizing", 0.5)
    try:
        res_conflict = numpy_backend.simulate_cycle(
            code, psi, channel=ch50, p=0.5, num_shots=10, deterministic_errors=["XIIII"]
        )
    except ValueError:
        pass
    else:
        assert res_conflict.syndrome_counts == {"0001": 10}
        assert res_conflict.mean_fidelity > 1.0 - 1e-10


def test_t2_f8_cuda_vs_numpy_parity_at_p_half(
    qecc_backends, qecc_codes, qecc_channels, helpers
) -> None:
    np_backend = qecc_backends.get_backend("numpy")
    cu_backend = qecc_backends.get_backend("torch_cuda")
    code = qecc_codes.get_code("five_qubit")
    psi = helpers.canonical_states()["|+i>"]
    ch = qecc_channels.ChannelSpec("depolarizing", 0.5)
    n_shots = 20_000

    res_np = np_backend.simulate_cycle(code, psi, channel=ch, p=0.5, num_shots=n_shots, seed=804)
    res_cu = cu_backend.simulate_cycle(code, psi, channel=ch, p=0.5, num_shots=n_shots, seed=805)

    expected = 1.0 - (2.0 / 3.0) * (1.0 - helpers.five_qubit_exact_pi(0.5))
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


def test_t2_f8_precision_complex64_vs_complex128_tolerances(
    qecc_backends, qecc_codes, helpers
) -> None:
    np_backend = qecc_backends.get_backend("numpy")
    cu_default = qecc_backends.get_backend("torch_cuda")
    code = qecc_codes.get_code("five_qubit")
    psi = helpers.canonical_states()["|+i>"]

    res_np = np_backend.simulate_cycle(
        code, psi, channel=None, p=0.0, num_shots=1, deterministic_errors=["YIIII"]
    )
    res_cu = cu_default.simulate_cycle(
        code, psi, channel=None, p=0.0, num_shots=1, deterministic_errors=["YIIII"]
    )
    assert abs(res_np.mean_fidelity - res_cu.mean_fidelity) < 1e-10
    assert np.allclose(res_np.rho_out, res_cu.rho_out, atol=1e-10)

    sig = inspect.signature(qecc_backends.get_backend)
    if "dtype" in sig.parameters or any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    ):
        try:
            cu64 = qecc_backends.get_backend("torch_cuda", dtype="complex64")
            cu128 = qecc_backends.get_backend("torch_cuda", dtype="complex128")
        except TypeError:
            return
        res64 = cu64.simulate_cycle(
            code, psi, channel=None, p=0.0, num_shots=1, deterministic_errors=["YIIII"]
        )
        res128 = cu128.simulate_cycle(
            code, psi, channel=None, p=0.0, num_shots=1, deterministic_errors=["YIIII"]
        )
        assert res64.mean_fidelity > 1.0 - 1e-5
        assert res128.mean_fidelity > 1.0 - 1e-10


# ---------------------------------------------------------------------------
# Feature F9 Boundaries: Seminar Deliverables, Rubric & Verifier
# ---------------------------------------------------------------------------


def test_f9_no_fidelity_metric_confusion_in_seminar_docs(docs_dir) -> None:
    text = (docs_dir / "seminar_qecc.md").read_text(encoding="utf-8")
    low = text.lower()
    assert "entanglement" in low
    assert "state fidelity" in low or "f_{\\text{state}}" in low or "f_state" in low
    assert "2/3" in text or r"\frac{2}{3}" in text
    assert "1 - p" in text or "1-p" in text
    assert any(
        phrase in low
        for phrase in (
            "mismatch",
            "consistent",
            "apple",
            "cannot compare",
            "do not compare",
            "must not compare",
            "not directly",
            "false advantage",
            "spurious",
        )
    )


def test_f9_latex_math_delimiters_balanced_and_no_broken_tables(docs_dir, helpers) -> None:
    for fname in ("seminar_qecc.md", "seminar_slides.md"):
        raw = (docs_dir / fname).read_text(encoding="utf-8")
        assert helpers.has_balanced_fenced_code_blocks(raw), f"Unbalanced code fences in {fname}"

        no_fences = helpers.strip_fenced_code_blocks(raw)
        no_code = helpers.strip_inline_code_spans(no_fences)
        no_escaped = helpers.strip_escaped_dollars(no_code)
        assert no_escaped.count("$$") % 2 == 0, f"Unbalanced $$ display math in {fname}"

        no_display = re.sub(r"\$\$[\s\S]*?\$\$", "", no_escaped)
        for para in re.split(r"\n\s*\n", no_display):
            assert para.count("$") % 2 == 0, f"Unbalanced inline $ in {fname}: {para[:120]!r}"

        lines = no_fences.splitlines()
        idx = 0
        while idx < len(lines):
            line = lines[idx].strip()
            if line.startswith("|") and line.endswith("|") and idx + 1 < len(lines):
                sep = lines[idx + 1].strip()
                if re.match(r"^\|[\s:\-|]+\|$", sep):
                    ncols = len(helpers.split_markdown_table_row(line))
                    assert len(helpers.split_markdown_table_row(sep)) == ncols
                    j = idx + 2
                    while j < len(lines) and lines[j].strip().startswith("|") and lines[j].strip().endswith("|"):
                        row_cols = len(helpers.split_markdown_table_row(lines[j]))
                        assert row_cols == ncols, f"Broken table column count in {fname}: {lines[j]}"
                        j += 1
                    idx = j
                    continue
            idx += 1


def test_f9_internal_markdown_links_and_anchors_valid(repo_root, docs_dir, helpers) -> None:
    md_files = [
        docs_dir / "seminar_qecc.md",
        docs_dir / "seminar_slides.md",
        docs_dir / "review_rubric.md",
        repo_root / "README.md",
    ]
    for md_path in md_files:
        raw_text = md_path.read_text(encoding="utf-8")
        no_fences = helpers.strip_fenced_code_blocks(raw_text)
        headings = re.findall(r"(?m)^#{1,6}\s+(.+)$", no_fences)
        slugs: set[str] = set()
        for h in headings:
            slug = helpers.slugify_heading(h)
            slugs.add(slug)
            slugs.add(re.sub(r"-+", "-", slug).strip("-"))
            slugs.add(slug.replace("_", ""))
        html_ids = {
            a.lower()
            for a in re.findall(r"<a\s+(?:id|name)=[\"']([^\"']+)[\"']", no_fences, re.IGNORECASE)
        }
        valid_anchors = slugs | html_ids

        link_text = helpers.strip_inline_code_spans(no_fences)
        for m in re.finditer(r"(?<!!)\[[^\]]+\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)", link_text):
            target = m.group(1).strip()
            if re.match(r"^(?:https?://|mailto:)", target, re.IGNORECASE):
                continue
            if target.startswith("#"):
                frag = target[1:].lower()
                assert frag in valid_anchors, f"Broken anchor {target!r} in {md_path.name}"
                continue
            file_part = target.split("#", 1)[0]
            resolved = (md_path.parent / file_part).resolve()
            if not resolved.exists():
                resolved = (repo_root / file_part).resolve()
            assert resolved.exists(), f"Broken relative link {target!r} in {md_path.name}"


def test_f9_review_rubric_zero_pending_or_failure_markers(docs_dir, helpers) -> None:
    text = (docs_dir / "review_rubric.md").read_text(encoding="utf-8")
    for marker in ("TBD", "TODO", "FIXME", "XXX", "- [ ]"):
        assert marker not in text
    assert not helpers.has_rubric_fail_verdict(text)

    lines = text.splitlines()
    expected_ids = (
        "R-1.1", "R-1.2", "R-1.3",
        "R-2.1", "R-2.2", "R-2.3",
        "R-3.1", "R-3.2", "R-3.3", "R-3.4",
        "R-4.1", "R-4.2",
        "R-5.1",
        "R-6.1", "R-6.2", "R-6.3", "R-6.4",
        "R-7.1",
    )
    for rid in expected_ids:
        assert any(rid in ln and "PASS" in ln for ln in lines), f"Missing PASS for {rid}"


def test_f9_manuscript_and_slides_formula_synchronization(docs_dir) -> None:
    manuscript = (docs_dir / "seminar_qecc.md").read_text(encoding="utf-8")
    slides = (docs_dir / "seminar_slides.md").read_text(encoding="utf-8")

    for doc_name, text in (("seminar_qecc.md", manuscript), ("seminar_slides.md", slides)):
        assert ("3p^2" in text or "3 p^2" in text) and "0.5" in text, f"Missing 3q threshold in {doc_name}"
        assert "0.1376" in text, f"Missing [[5,1,3]] 0.1376 threshold in {doc_name}"
        assert "willow" in text.lower() and "288" in text and "48" in text, (
            f"Missing Topic 5 metrics in {doc_name}"
        )


def test_f9_verify_script_deterministic_and_cwd_independent(
    scripts_dir, python_executable, tmp_path
) -> None:
    script = scripts_dir / "verify_analytical_thresholds.py"
    outs: list[str] = []
    for _ in range(2):
        proc = subprocess.run(
            [str(python_executable), str(script)],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr
        outs.append(proc.stdout)
        for tok in ("0.5", "0.137628", "0.131123"):
            assert tok in proc.stdout or tok[:6] in proc.stdout
    assert outs[0] == outs[1]


def test_f9_readme_referenced_files_and_scripts_exist(repo_root) -> None:
    readme_text = (repo_root / "README.md").read_text(encoding="utf-8")
    pattern = r"(?:docs|scripts|results|src|tests)/[A-Za-z0-9_./\-*{},]+"
    refs = set(re.findall(pattern, readme_text))
    assert len(refs) >= 4
    for ref in refs:
        if any(tok in ref for tok in ("{", "*", "...", "..")):
            continue
        cleaned = ref.rstrip(".,;:)`'\"}")
        assert (repo_root / cleaned).exists(), f"README references non-existent path {cleaned!r}"
