"""Tier 1 E2E tests for Features F1-F4: 3-qubit codes, [[5,1,3]] code, and 2-qubit DFS."""

from __future__ import annotations

import math

import numpy as np
import pytest

pytestmark = pytest.mark.tier1


# ---------------------------------------------------------------------------
# Feature F1: BitFlip3QubitCode ("bit_flip_3")
# ---------------------------------------------------------------------------


def test_f1_bit_flip_3_zero_error_cardinal_and_haar_fidelity(
    qecc_codes, qecc_channels, qecc_metrics, numpy_backend, helpers
) -> None:
    code = qecc_codes.get_code("bit_flip_3")
    sampler = qecc_metrics.InitialStateSampler()
    states = list(sampler.cardinal_states().values()) + list(sampler.haar_random_states(5, seed=101))
    channel_zero = qecc_channels.ChannelSpec("bit_flip", 0.0)

    for psi in states:
        rho_in = helpers.ket_to_rho(psi)
        for ch, n_shots in ((None, 1), (channel_zero, 10)):
            res = numpy_backend.simulate_cycle(code, psi, channel=ch, p=0.0, num_shots=n_shots, seed=101)
            assert res.mean_fidelity > 1.0 - 1e-10
            assert qecc_metrics.quantum_fidelity(rho_in, res.rho_out) > 1.0 - 1e-10
            assert np.all(np.asarray(res.fidelities) > 1.0 - 1e-10)
            assert res.syndrome_counts == {"00": res.num_shots}


def test_f1_bit_flip_3_deterministic_single_x_syndromes_and_recovery(
    qecc_codes, qecc_metrics, numpy_backend, helpers
) -> None:
    code = qecc_codes.get_code("bit_flip_3")
    sampler = qecc_metrics.InitialStateSampler()
    canonical = helpers.canonical_states()
    test_states = [canonical["|0>"], canonical["|+>"], canonical["|+i>"]] + list(
        sampler.haar_random_states(2, seed=102)
    )
    error_map = {"III": "00", "XII": "10", "IXI": "11", "IIX": "01"}

    for psi in test_states:
        rho_in = helpers.ket_to_rho(psi)
        observed_syndromes: set[str] = set()
        for err, expected_synd in error_map.items():
            res = numpy_backend.simulate_cycle(
                code, psi, channel=None, p=0.0, num_shots=1, deterministic_errors=[err]
            )
            synd = str(np.asarray(res.syndromes).tolist()[0])
            observed_syndromes.add(synd)
            assert synd == expected_synd
            assert res.syndrome_counts == {expected_synd: 1}
            assert res.mean_fidelity > 1.0 - 1e-10
            assert qecc_metrics.quantum_fidelity(rho_in, res.rho_out) > 1.0 - 1e-10
        assert len(observed_syndromes) == 4


def test_f1_bit_flip_3_stabilizer_properties_and_metadata(qecc_codes, helpers) -> None:
    c_factory = qecc_codes.get_code("bit_flip_3")
    c_cls = qecc_codes.BitFlip3QubitCode()

    assert c_factory.n_data == 3
    assert c_factory.n_ancilla == 2
    assert c_factory.n_total == 5
    assert len(c_factory.stabilizers) == 2
    assert c_factory.stabilizers == c_cls.stabilizers

    s0 = helpers.pauli_string_to_matrix(c_factory.stabilizers[0])
    s1 = helpers.pauli_string_to_matrix(c_factory.stabilizers[1])
    eye8 = np.eye(8, dtype=np.complex128)
    for mat in (s0, s1):
        assert mat.shape == (8, 8)
        assert np.allclose(mat, mat.conj().T, atol=1e-12)
        assert np.allclose(mat @ mat, eye8, atol=1e-12)
        assert not np.allclose(mat, eye8, atol=1e-12)
    assert not np.allclose(s0, s1, atol=1e-12)
    assert np.allclose(s0 @ s1, s1 @ s0, atol=1e-12)

    assert len(c_factory.syndrome_table) == 4
    assert set(c_factory.syndrome_table.keys()) == {"00", "10", "11", "01"}
    assert len(c_factory.recovery_ops("00")) == 0
    for synd in ("10", "11", "01"):
        assert len(c_factory.recovery_ops(synd)) > 0


def test_f1_bit_flip_3_non_destructive_ancilla_preserves_superposition_coherence(
    qecc_codes, qecc_metrics, numpy_backend, helpers
) -> None:
    code = qecc_codes.get_code("bit_flip_3")
    canonical = helpers.canonical_states()
    haar_state = qecc_metrics.InitialStateSampler().haar_random_states(1, seed=103)[0]
    superpositions = [canonical["|+>"], canonical["|-i>"], haar_state]

    for psi in superpositions:
        rho_in = helpers.ket_to_rho(psi)
        for err in ("XII", "IXI", "IIX"):
            res = numpy_backend.simulate_cycle(
                code, psi, channel=None, p=0.0, num_shots=1, deterministic_errors=[err]
            )
            synd = str(np.asarray(res.syndromes).tolist()[0])
            assert synd != "00"
            assert helpers.purity(res.rho_out) > 1.0 - 1e-10
            assert abs(res.rho_out[0, 1] - rho_in[0, 1]) < 1e-10
            assert qecc_metrics.quantum_fidelity(rho_in, res.rho_out) > 1.0 - 1e-10


def test_f1_bit_flip_3_circuit_ops_and_low_p_stochastic_gain(
    qecc_codes, qecc_channels, numpy_backend, helpers
) -> None:
    code = qecc_codes.get_code("bit_flip_3")
    psi = helpers.canonical_states()["|0>"]
    channel = qecc_channels.ChannelSpec("bit_flip", 0.05)

    assert len(code.encode_ops()) > 0
    assert len(code.syndrome_ops()) > 0
    assert len(code.decode_ops()) > 0

    res = numpy_backend.simulate_cycle(code, psi, channel=channel, p=0.05, num_shots=1000, seed=104)
    assert res.code_name == code.name
    assert res.mean_fidelity > (1.0 - 0.05) + 0.02
    assert len(res.syndrome_counts) >= 2


# ---------------------------------------------------------------------------
# Feature F2: PhaseFlip3QubitCode ("phase_flip_3")
# ---------------------------------------------------------------------------


def test_f2_phase_flip_3_zero_error_cardinal_and_haar_fidelity(
    qecc_codes, qecc_channels, qecc_metrics, numpy_backend, helpers
) -> None:
    code = qecc_codes.get_code("phase_flip_3")
    sampler = qecc_metrics.InitialStateSampler()
    states = list(sampler.cardinal_states().values()) + list(sampler.haar_random_states(5, seed=201))
    channel_zero = qecc_channels.ChannelSpec("phase_flip", 0.0)

    for psi in states:
        rho_in = helpers.ket_to_rho(psi)
        for ch, n_shots in ((None, 1), (channel_zero, 10)):
            res = numpy_backend.simulate_cycle(code, psi, channel=ch, p=0.0, num_shots=n_shots, seed=201)
            assert res.mean_fidelity > 1.0 - 1e-10
            assert qecc_metrics.quantum_fidelity(rho_in, res.rho_out) > 1.0 - 1e-10
            assert res.syndrome_counts == {"00": res.num_shots}


def test_f2_phase_flip_3_deterministic_single_z_syndromes_and_recovery(
    qecc_codes, qecc_metrics, numpy_backend, helpers
) -> None:
    code = qecc_codes.get_code("phase_flip_3")
    sampler = qecc_metrics.InitialStateSampler()
    canonical = helpers.canonical_states()
    test_states = [canonical["|+>"], canonical["|0>"], canonical["|+i>"]] + list(
        sampler.haar_random_states(2, seed=202)
    )
    error_map = {"III": "00", "ZII": "10", "IZI": "11", "IIZ": "01"}

    for psi in test_states:
        rho_in = helpers.ket_to_rho(psi)
        observed_syndromes: set[str] = set()
        for err, expected_synd in error_map.items():
            res = numpy_backend.simulate_cycle(
                code, psi, channel=None, p=0.0, num_shots=1, deterministic_errors=[err]
            )
            synd = str(np.asarray(res.syndromes).tolist()[0])
            observed_syndromes.add(synd)
            assert synd == expected_synd
            assert res.mean_fidelity > 1.0 - 1e-10
            assert qecc_metrics.quantum_fidelity(rho_in, res.rho_out) > 1.0 - 1e-10
        assert len(observed_syndromes) == 4


def test_f2_phase_flip_3_hadamard_duality_with_bit_flip_3(
    qecc_codes, qecc_channels, qecc_metrics, numpy_backend
) -> None:
    bf = qecc_codes.get_code("bit_flip_3")
    pf = qecc_codes.get_code("phase_flip_3")
    haar_states = qecc_metrics.InitialStateSampler().haar_random_states(4, seed=203)
    dual_pairs = [("XII", "ZII"), ("IXI", "IZI"), ("IIX", "IIZ")]

    for psi in haar_states:
        for err_bf, err_pf in dual_pairs:
            res_bf = numpy_backend.simulate_cycle(
                bf, psi, channel=None, p=0.0, num_shots=1, deterministic_errors=[err_bf]
            )
            res_pf = numpy_backend.simulate_cycle(
                pf, psi, channel=None, p=0.0, num_shots=1, deterministic_errors=[err_pf]
            )
            assert np.asarray(res_bf.syndromes).tolist()[0] == np.asarray(res_pf.syndromes).tolist()[0]
            assert res_bf.mean_fidelity > 1.0 - 1e-10
            assert res_pf.mean_fidelity > 1.0 - 1e-10

        stoch_bf = numpy_backend.simulate_cycle(
            bf, psi, channel=qecc_channels.ChannelSpec("bit_flip", 0.15), p=0.15, num_shots=2000, seed=203
        )
        stoch_pf = numpy_backend.simulate_cycle(
            pf, psi, channel=qecc_channels.ChannelSpec("phase_flip", 0.15), p=0.15, num_shots=2000, seed=203
        )
        assert abs(stoch_bf.mean_fidelity - stoch_pf.mean_fidelity) < 0.05


def test_f2_phase_flip_3_stabilizer_properties_and_metadata(qecc_codes, helpers) -> None:
    c_factory = qecc_codes.get_code("phase_flip_3")
    c_cls = qecc_codes.PhaseFlip3QubitCode()

    assert c_factory.n_data == 3
    assert c_factory.n_ancilla == 2
    assert c_factory.n_total == 5
    assert len(c_factory.stabilizers) == 2
    assert c_factory.stabilizers == c_cls.stabilizers

    s0 = helpers.pauli_string_to_matrix(c_factory.stabilizers[0])
    s1 = helpers.pauli_string_to_matrix(c_factory.stabilizers[1])
    eye8 = np.eye(8, dtype=np.complex128)
    for mat in (s0, s1):
        assert mat.shape == (8, 8)
        assert np.allclose(mat, mat.conj().T, atol=1e-12)
        assert np.allclose(mat @ mat, eye8, atol=1e-12)
        assert not np.allclose(mat, eye8, atol=1e-12)
    assert not np.allclose(s0, s1, atol=1e-12)
    assert np.allclose(s0 @ s1, s1 @ s0, atol=1e-12)

    assert set(c_factory.syndrome_table.keys()) == {"00", "10", "11", "01"}
    assert len(c_factory.recovery_ops("00")) == 0
    for synd in ("10", "11", "01"):
        assert len(c_factory.recovery_ops(synd)) > 0


def test_f2_phase_flip_3_non_destructive_ancilla_and_low_p_gain(
    qecc_codes, qecc_channels, qecc_metrics, numpy_backend, helpers
) -> None:
    code = qecc_codes.get_code("phase_flip_3")
    canonical = helpers.canonical_states()
    haar_state = qecc_metrics.InitialStateSampler().haar_random_states(1, seed=204)[0]

    for psi in (canonical["|+i>"], canonical["|0>"], haar_state):
        rho_in = helpers.ket_to_rho(psi)
        for err in ("ZII", "IZI", "IIZ"):
            res = numpy_backend.simulate_cycle(
                code, psi, channel=None, p=0.0, num_shots=1, deterministic_errors=[err]
            )
            assert helpers.purity(res.rho_out) > 1.0 - 1e-10
            assert qecc_metrics.quantum_fidelity(rho_in, res.rho_out) > 1.0 - 1e-10

    res_mc = numpy_backend.simulate_cycle(
        code,
        canonical["|+>"],
        channel=qecc_channels.ChannelSpec("phase_flip", 0.05),
        p=0.05,
        num_shots=1000,
        seed=204,
    )
    assert res_mc.mean_fidelity > (1.0 - 0.05) + 0.02


# ---------------------------------------------------------------------------
# Feature F3: FiveQubitPerfectCode ("five_qubit")
# ---------------------------------------------------------------------------


def test_f3_five_qubit_zero_error_cardinal_and_haar_fidelity(
    qecc_codes, qecc_channels, qecc_metrics, numpy_backend, helpers
) -> None:
    code = qecc_codes.get_code("five_qubit")
    sampler = qecc_metrics.InitialStateSampler()
    states = list(sampler.cardinal_states().values()) + list(sampler.haar_random_states(5, seed=301))
    channel_zero = qecc_channels.ChannelSpec("depolarizing", 0.0)

    for psi in states:
        rho_in = helpers.ket_to_rho(psi)
        for ch, n_shots in ((None, 1), (channel_zero, 10)):
            res = numpy_backend.simulate_cycle(code, psi, channel=ch, p=0.0, num_shots=n_shots, seed=301)
            assert res.mean_fidelity > 1.0 - 1e-10
            assert qecc_metrics.quantum_fidelity(rho_in, res.rho_out) > 1.0 - 1e-10
            assert res.syndrome_counts == {"0000": res.num_shots}


def test_f3_five_qubit_all_15_single_qubit_pauli_syndromes_and_recovery(
    qecc_codes, qecc_metrics, numpy_backend, helpers
) -> None:
    code = qecc_codes.get_code("five_qubit")
    psi = qecc_metrics.InitialStateSampler().haar_random_states(1, seed=302)[0]
    rho_in = helpers.ket_to_rho(psi)

    expected_map = {
        "IIIII": "0000",
        "XIIII": "0001",
        "ZIIII": "1010",
        "YIIII": "1011",
        "IXIII": "1000",
        "IZIII": "0101",
        "IYIII": "1101",
        "IIXII": "1100",
        "IIZII": "0010",
        "IIYII": "1110",
        "IIIXI": "0110",
        "IIIZI": "1001",
        "IIIYI": "1111",
        "IIIIX": "0011",
        "IIIIZ": "0100",
        "IIIIY": "0111",
    }

    observed_syndromes: set[str] = set()
    for err, expected_synd in expected_map.items():
        res = numpy_backend.simulate_cycle(
            code, psi, channel=None, p=0.0, num_shots=1, deterministic_errors=[err]
        )
        synd = str(np.asarray(res.syndromes).tolist()[0])
        observed_syndromes.add(synd)
        assert synd == expected_synd
        assert res.mean_fidelity > 1.0 - 1e-10
        assert qecc_metrics.quantum_fidelity(rho_in, res.rho_out) > 1.0 - 1e-10

    assert len(observed_syndromes) == 16


def test_f3_five_qubit_stabilizer_and_logical_operator_algebra(qecc_codes, helpers) -> None:
    code = qecc_codes.get_code("five_qubit")
    c_cls = qecc_codes.FiveQubitPerfectCode()

    assert code.n_data == 5
    assert code.n_ancilla == 4
    assert code.n_total == 9
    assert len(code.stabilizers) == 4
    assert code.stabilizers == c_cls.stabilizers

    stab_mats = [helpers.pauli_string_to_matrix(s) for s in code.stabilizers]
    xl = helpers.pauli_string_to_matrix("XXXXX")
    zl = helpers.pauli_string_to_matrix("ZZZZZ")
    eye32 = np.eye(32, dtype=np.complex128)

    for mat in stab_mats:
        assert mat.shape == (32, 32)
        assert np.allclose(mat, mat.conj().T, atol=1e-12)
        assert np.allclose(mat @ mat, eye32, atol=1e-12)
        assert np.allclose(mat @ xl, xl @ mat, atol=1e-12)
        assert np.allclose(mat @ zl, zl @ mat, atol=1e-12)

    for i in range(len(stab_mats)):
        for j in range(i + 1, len(stab_mats)):
            assert np.allclose(stab_mats[i] @ stab_mats[j], stab_mats[j] @ stab_mats[i], atol=1e-12)

    assert np.allclose(xl @ zl, -zl @ xl, atol=1e-12)
    assert len(code.syndrome_table) == 16
    assert len(code.recovery_ops("0000")) == 0
    for idx in range(1, 16):
        assert len(code.recovery_ops(f"{idx:04b}")) > 0


def _apply_gate_ops_statevector(ops: list[object], psi_q0: np.ndarray, n_qubits: int) -> np.ndarray:
    """Apply a sequence of GateOp instances to |psi_q0> tensor |0>^{n_qubits-1}."""
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    mats_1q: dict[str, np.ndarray] = {
        "I": np.eye(2, dtype=np.complex128),
        "X": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128),
        "Y": np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128),
        "Z": np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128),
        "H": np.array([[inv_sqrt2, inv_sqrt2], [inv_sqrt2, -inv_sqrt2]], dtype=np.complex128),
        "S": np.array([[1.0, 0.0], [0.0, 1.0j]], dtype=np.complex128),
        "Sdg": np.array([[1.0, 0.0], [0.0, -1.0j]], dtype=np.complex128),
    }

    state = np.zeros(1 << n_qubits, dtype=np.complex128)
    state[0] = complex(psi_q0[0])
    state[1 << (n_qubits - 1)] = complex(psi_q0[1])
    tensor = state.reshape((2,) * n_qubits)

    for op in ops:
        name = str(op.name)
        targets = tuple(int(q) for q in op.targets)
        controls = tuple(int(q) for q in op.controls)
        params = tuple(float(a) for a in op.params)

        if name in mats_1q:
            u = mats_1q[name]
            q = targets[0]
            tensor = np.tensordot(u, tensor, axes=([1], [q]))
            tensor = np.moveaxis(tensor, 0, q)
        elif name in {"RX", "RY", "RZ"}:
            theta = params[0]
            c = math.cos(0.5 * theta)
            s = math.sin(0.5 * theta)
            if name == "RX":
                u = np.array([[c, -1.0j * s], [-1.0j * s, c]], dtype=np.complex128)
            elif name == "RY":
                u = np.array([[c, -s], [s, c]], dtype=np.complex128)
            else:
                u = np.array([[math.cos(-0.5 * theta) + 1.0j * math.sin(-0.5 * theta), 0.0],
                              [0.0, math.cos(0.5 * theta) + 1.0j * math.sin(0.5 * theta)]], dtype=np.complex128)
            q = targets[0]
            tensor = np.tensordot(u, tensor, axes=([1], [q]))
            tensor = np.moveaxis(tensor, 0, q)
        elif name in {"CNOT", "CZ"}:
            ctrl, tgt = controls[0], targets[0]
            u = mats_1q["X"] if name == "CNOT" else mats_1q["Z"]
            idx_c1 = [slice(None)] * n_qubits
            idx_c1[ctrl] = 1
            sub = tensor[tuple(idx_c1)]
            t_adj = tgt if tgt < ctrl else tgt - 1
            sub_out = np.tensordot(u, sub, axes=([1], [t_adj]))
            sub_out = np.moveaxis(sub_out, 0, t_adj)
            tensor[tuple(idx_c1)] = sub_out
        else:
            raise ValueError(f"Unsupported GateOp {name!r}")

    return tensor.reshape(-1)


def test_f3_five_qubit_relative_phase_z0_encoder_check(
    qecc_codes, qecc_metrics, numpy_backend, helpers
) -> None:
    code = qecc_codes.get_code("five_qubit")
    canonical = helpers.canonical_states()
    equatorial = [canonical["|+>"], canonical["|->"], canonical["|+i>"], canonical["|-i>"]]

    for psi in equatorial:
        rho_in = helpers.ket_to_rho(psi)
        for err in (None, "XIIII", "YIIII", "ZIIII"):
            det = [err] if err is not None else None
            res = numpy_backend.simulate_cycle(
                code, psi, channel=None, p=0.0, num_shots=1, deterministic_errors=det
            )
            assert qecc_metrics.quantum_fidelity(rho_in, res.rho_out) > 1.0 - 1e-10
            assert abs(res.rho_out[0, 1] - rho_in[0, 1]) < 1e-10

    # Verify encode_ops() directly so symmetric omission of Z(0) from encode & decode cannot escape
    psi_enc = _apply_gate_ops_statevector(list(code.encode_ops()), canonical["|+>"], code.n_total)
    anc_eye = np.eye(1 << code.n_ancilla, dtype=np.complex128)
    for stab in code.stabilizers:
        op_mat = np.kron(helpers.pauli_string_to_matrix(stab), anc_eye)
        exp_val = float(np.real(np.vdot(psi_enc, op_mat @ psi_enc)))
        assert abs(exp_val - 1.0) < 1e-10
    xl_mat = np.kron(helpers.pauli_string_to_matrix("XXXXX"), anc_eye)
    assert abs(float(np.real(np.vdot(psi_enc, xl_mat @ psi_enc))) - 1.0) < 1e-10


def test_f3_five_qubit_non_destructive_ancilla_and_low_p_depolarizing_gain(
    qecc_codes, qecc_channels, numpy_backend, helpers
) -> None:
    code = qecc_codes.get_code("five_qubit")
    psi = helpers.canonical_states()["|+i>"]
    res = numpy_backend.simulate_cycle(
        code,
        psi,
        channel=qecc_channels.ChannelSpec("depolarizing", 0.05),
        p=0.05,
        num_shots=10_000,
        seed=305,
    )
    raw_state_fidelity = 1.0 - (2.0 / 3.0) * 0.05
    assert res.mean_fidelity > raw_state_fidelity + 0.010
    assert len(res.syndrome_counts) > 4


# ---------------------------------------------------------------------------
# Feature F4: TwoQubitDFSDemo ("dfs_2")
# ---------------------------------------------------------------------------


def test_f4_dfs_2_subspace_encoding_decoding_and_metadata(
    qecc_codes, qecc_metrics, numpy_backend, helpers
) -> None:
    code = qecc_codes.get_code("dfs_2")
    c_cls = qecc_codes.TwoQubitDFSDemo()
    assert code.n_data == 2
    assert code.n_ancilla == 0
    assert code.n_total == 2
    assert code.name == c_cls.name
    assert len(code.encode_ops()) > 0
    assert len(code.decode_ops()) > 0

    canonical = helpers.canonical_states()
    for key in ("|0>", "|1>", "|+>", "|+i>"):
        psi = canonical[key]
        res_zz = numpy_backend.simulate_cycle(
            code, psi, channel=None, p=0.0, num_shots=1, deterministic_errors=["ZZ"]
        )
        assert qecc_metrics.quantum_fidelity(helpers.ket_to_rho(psi), res_zz.rho_out) > 1.0 - 1e-10

    for key in ("|0>", "|1>"):
        psi = canonical[key]
        res_zi = numpy_backend.simulate_cycle(
            code, psi, channel=None, p=0.0, num_shots=1, deterministic_errors=["ZI"]
        )
        assert qecc_metrics.quantum_fidelity(helpers.ket_to_rho(psi), res_zi.rho_out) > 1.0 - 1e-10

    res_zi_plus = numpy_backend.simulate_cycle(
        code, canonical["|+>"], channel=None, p=0.0, num_shots=1, deterministic_errors=["ZI"]
    )
    assert qecc_metrics.quantum_fidelity(helpers.ket_to_rho(canonical["|+>"]), res_zi_plus.rho_out) < 1e-10


def test_f4_dfs_2_zero_error_cardinal_and_haar_fidelity(
    qecc_codes, qecc_channels, qecc_metrics, numpy_backend, helpers
) -> None:
    code = qecc_codes.get_code("dfs_2")
    sampler = qecc_metrics.InitialStateSampler()
    states = list(sampler.cardinal_states().values()) + list(sampler.haar_random_states(5, seed=401))
    channel_zero = qecc_channels.ChannelSpec("collective_dephasing", 0.0)

    for psi in states:
        rho_in = helpers.ket_to_rho(psi)
        for ch, n_shots in ((None, 1), (channel_zero, 10)):
            res = numpy_backend.simulate_cycle(code, psi, channel=ch, p=0.0, num_shots=n_shots, seed=401)
            assert res.mean_fidelity > 1.0 - 1e-10
            assert qecc_metrics.quantum_fidelity(rho_in, res.rho_out) > 1.0 - 1e-10


def test_f4_dfs_2_collective_dephasing_invariance_across_states_and_p(
    qecc_codes, qecc_channels, qecc_metrics, numpy_backend, helpers
) -> None:
    code = qecc_codes.get_code("dfs_2")
    canonical = helpers.canonical_states()
    haar_states = qecc_metrics.InitialStateSampler().haar_random_states(3, seed=402)
    states = [canonical["|0>"], canonical["|+>"], canonical["|+i>"]] + list(haar_states)

    for p in (0.1, 0.25, 0.5):
        channel = qecc_channels.ChannelSpec("collective_dephasing", p)
        for psi in states:
            rho_in = helpers.ket_to_rho(psi)
            res = numpy_backend.simulate_cycle(code, psi, channel=channel, p=p, num_shots=100, seed=402)
            assert res.mean_fidelity > 1.0 - 1e-10
            assert np.all(np.asarray(res.fidelities) > 1.0 - 1e-10)
            assert qecc_metrics.quantum_fidelity(rho_in, res.rho_out) > 1.0 - 1e-10


def test_f4_dfs_2_deterministic_collective_zz_invariance(
    qecc_codes, qecc_metrics, numpy_backend, helpers
) -> None:
    code = qecc_codes.get_code("dfs_2")
    sampler = qecc_metrics.InitialStateSampler()
    states = list(sampler.cardinal_states().values()) + list(sampler.haar_random_states(3, seed=403))

    for psi in states:
        rho_in = helpers.ket_to_rho(psi)
        res = numpy_backend.simulate_cycle(
            code, psi, channel=None, p=0.0, num_shots=1, deterministic_errors=["ZZ"]
        )
        assert res.mean_fidelity > 1.0 - 1e-10
        assert qecc_metrics.quantum_fidelity(rho_in, res.rho_out) > 1.0 - 1e-10


def test_f4_dfs_2_no_ancillas_or_syndrome_extraction_needed(qecc_codes) -> None:
    code = qecc_codes.get_code("dfs_2")
    assert code.n_ancilla == 0
    assert code.n_total == code.n_data == 2
    assert len(code.stabilizers) == 0
    assert len(code.syndrome_ops()) == 0
    assert len(code.syndrome_table) == 0
