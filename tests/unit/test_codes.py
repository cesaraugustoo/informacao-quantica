"""Unit tests for qecc.codes (BitFlip3, PhaseFlip3, FiveQubit, TwoQubitDFS, get_code)."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from qecc import (
    BitFlip3QubitCode,
    FiveQubitPerfectCode,
    PhaseFlip3QubitCode,
    TwoQubitDFSDemo,
    get_code as top_get_code,
    list_available_codes as top_list_available_codes,
)
from qecc.backends import QECCCode, get_backend
from qecc.backends.numpy_backend import _apply_ops_np
from qecc.channels import ChannelSpec, coherent_collective_dephasing_ops
from qecc.codes import get_code, list_available_codes
from qecc.metrics import InitialStateSampler, partial_trace_to_qubit0, quantum_fidelity

_PAULI_MATS: dict[str, np.ndarray] = {
    "I": np.eye(2, dtype=np.complex128),
    "X": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128),
    "Y": np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128),
    "Z": np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128),
}


def _pauli_kron(pauli_str: str) -> np.ndarray:
    mat = _PAULI_MATS[pauli_str[0]]
    for ch in pauli_str[1:]:
        mat = np.kron(mat, _PAULI_MATS[ch])
    return mat


def test_code_registry_exports_protocol_and_defensive_copies() -> None:
    expected = ["bit_flip_3", "phase_flip_3", "five_qubit", "dfs_2"]
    assert list_available_codes() == expected
    assert top_list_available_codes() == expected

    cls_map = {
        "bit_flip_3": BitFlip3QubitCode,
        "phase_flip_3": PhaseFlip3QubitCode,
        "five_qubit": FiveQubitPerfectCode,
        "dfs_2": TwoQubitDFSDemo,
    }
    for name, cls in cls_map.items():
        c1 = get_code(name)
        c2 = top_get_code(name)
        assert isinstance(c1, cls)
        assert isinstance(c2, cls)
        assert c1 is not c2
        assert isinstance(c1, QECCCode)
        assert c1.name == name
        assert c1.n_total == c1.n_data + c1.n_ancilla

        # Verify syndrome_table and *_ops() return fresh copies
        tbl = c1.syndrome_table
        tbl["corrupted"] = "XXXXX"
        assert "corrupted" not in c1.syndrome_table

        enc = c1.encode_ops()
        orig_len = len(enc)
        enc.clear()
        assert len(c1.encode_ops()) == orig_len

        syn = c1.syndrome_ops()
        orig_syn_len = len(syn)
        syn.clear()
        assert len(c1.syndrome_ops()) == orig_syn_len

        dec = c1.decode_ops()
        orig_dec_len = len(dec)
        dec.clear()
        assert len(c1.decode_ops()) == orig_dec_len

    for bad_name in ("unknown", "", "bit_flip", None, 123):
        with pytest.raises((ValueError, KeyError)):
            get_code(bad_name)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("code_name", "valid_empty", "bad_syndromes"),
    [
        ("bit_flip_3", "00", ("", "0", "000", "12", "invalid", None, 0)),
        ("phase_flip_3", "00", ("", "0", "000", "ab", "invalid", None, 1)),
        ("five_qubit", "0000", ("", "000", "00000", "0002", "abcd", None, 0)),
        ("dfs_2", "", ("0", "00", "invalid", None, 0, [])),
    ],
)
def test_recovery_ops_validation(
    code_name: str,
    valid_empty: str,
    bad_syndromes: tuple[object, ...],
) -> None:
    code = get_code(code_name)
    assert code.recovery_ops(valid_empty) == []
    for bad in bad_syndromes:
        with pytest.raises((ValueError, KeyError)):
            code.recovery_ops(bad)  # type: ignore[arg-type]


@pytest.mark.parametrize("code_name", ["bit_flip_3", "phase_flip_3", "five_qubit", "dfs_2"])
def test_zero_error_all_backends_cardinal_and_haar(code_name: str) -> None:
    code = get_code(code_name)
    cards = InitialStateSampler.cardinal_states()
    haars = InitialStateSampler.haar_random_states(2, seed=42)
    states = list(cards.values()) + list(haars)
    backends = [get_backend("numpy"), get_backend("torch_cuda"), get_backend("eigenkit_unitary")]

    expected_syn = "" if code.n_ancilla == 0 else "0" * code.n_ancilla

    for psi in states:
        results = [
            b.simulate_cycle(code, psi, channel=None, p=0.0, num_shots=3, seed=7)
            for b in backends
        ]
        for res in results:
            assert res.mean_fidelity > 1.0 - 1e-12
            assert res.syndrome_counts == {expected_syn: 3}
        assert np.allclose(results[0].rho_out, results[1].rho_out, atol=1e-12)
        assert np.allclose(results[0].rho_out, results[2].rho_out, atol=1e-12)


def test_bit_flip_3_single_errors_coherence_and_boundaries() -> None:
    code = get_code("bit_flip_3")
    assert code.stabilizers == ("ZZI", "IZZ")
    assert code.syndrome_table == {"00": "III", "10": "XII", "11": "IXI", "01": "IIX"}

    backends = [get_backend("numpy"), get_backend("torch_cuda"), get_backend("eigenkit_unitary")]
    psi_haar = InitialStateSampler.haar_random_states(1, seed=101)[0]
    rho_in = np.outer(psi_haar, psi_haar.conj())

    for syn, err in code.syndrome_table.items():
        results = [
            b.simulate_cycle(code, psi_haar, None, 0.0, 4, seed=11, deterministic_errors=err)
            for b in backends
        ]
        for res in results:
            assert res.syndrome_counts == {syn: 4}
            assert res.mean_fidelity > 1.0 - 1e-12
            assert abs(res.rho_out[0, 1] - rho_in[0, 1]) < 1e-12
        assert np.allclose(results[0].rho_out, results[1].rho_out, atol=1e-12)
        assert np.allclose(results[0].rho_out, results[2].rho_out, atol=1e-12)

    np_b = backends[0]
    cards = InitialStateSampler.cardinal_states()
    # Weight-2 and weight-3 X errors fail on |0>
    for bad_err, bad_syn in (("XXI", "01"), ("XIX", "11"), ("IXX", "10"), ("XXX", "00")):
        res = np_b.simulate_cycle(
            code, cards["0"], None, 0.0, 2, seed=1, deterministic_errors=bad_err
        )
        assert res.syndrome_counts == {bad_syn: 2}
        assert res.mean_fidelity < 1e-12
        assert quantum_fidelity(cards["1"], res.rho_out) > 1.0 - 1e-12

    # Uncorrectable Z error goes undetected and flips |+> -> |->
    res_z = np_b.simulate_cycle(
        code, cards["+"], None, 0.0, 2, seed=1, deterministic_errors="ZII"
    )
    assert res_z.syndrome_counts == {"00": 2}
    assert res_z.mean_fidelity < 1e-12
    assert quantum_fidelity(cards["-"], res_z.rho_out) > 1.0 - 1e-12


def test_phase_flip_3_single_errors_duality_and_boundaries() -> None:
    bf = get_code("bit_flip_3")
    pf = get_code("phase_flip_3")
    assert pf.stabilizers == ("XXI", "IXX")
    assert pf.syndrome_table == {"00": "III", "10": "ZII", "11": "IZI", "01": "IIZ"}

    backends = [get_backend("numpy"), get_backend("torch_cuda"), get_backend("eigenkit_unitary")]
    psi_haar = InitialStateSampler.haar_random_states(1, seed=202)[0]

    for syn, err in pf.syndrome_table.items():
        results = [
            b.simulate_cycle(pf, psi_haar, None, 0.0, 4, seed=12, deterministic_errors=err)
            for b in backends
        ]
        for res in results:
            assert res.syndrome_counts == {syn: 4}
            assert res.mean_fidelity > 1.0 - 1e-12
        assert np.allclose(results[0].rho_out, results[1].rho_out, atol=1e-12)
        assert np.allclose(results[0].rho_out, results[2].rho_out, atol=1e-12)

    # Hadamard duality: syndrome of Z on phase_flip_3 equals syndrome of X on bit_flip_3
    np_b = backends[0]
    for x_err, z_err in (("XII", "ZII"), ("IXI", "IZI"), ("IIX", "IIZ")):
        r_bf = np_b.simulate_cycle(bf, psi_haar, None, 0.0, 1, seed=3, deterministic_errors=x_err)
        r_pf = np_b.simulate_cycle(pf, psi_haar, None, 0.0, 1, seed=3, deterministic_errors=z_err)
        assert r_bf.syndromes == r_pf.syndromes

    cards = InitialStateSampler.cardinal_states()
    # Weight-2 and weight-3 Z errors act as logical ZZZ (swapping |0_L> <-> |1_L>),
    # so |0> flips to |1> while |+> remains invariant.
    for bad_err, bad_syn in (("ZZI", "01"), ("ZIZ", "11"), ("IZZ", "10"), ("ZZZ", "00")):
        res_0 = np_b.simulate_cycle(
            pf, cards["0"], None, 0.0, 2, seed=1, deterministic_errors=bad_err
        )
        assert res_0.syndrome_counts == {bad_syn: 2}
        assert res_0.mean_fidelity < 1e-12
        assert quantum_fidelity(cards["1"], res_0.rho_out) > 1.0 - 1e-12

        res_plus = np_b.simulate_cycle(
            pf, cards["+"], None, 0.0, 2, seed=1, deterministic_errors=bad_err
        )
        assert res_plus.mean_fidelity > 1.0 - 1e-12

    # Single X error goes undetected and flips |+> -> |->
    res_x = np_b.simulate_cycle(
        pf, cards["+"], None, 0.0, 2, seed=1, deterministic_errors="XII"
    )
    assert res_x.syndrome_counts == {"00": 2}
    assert res_x.mean_fidelity < 1e-12


def test_five_qubit_stabilizer_algebra_and_logical_relative_phase() -> None:
    code = FiveQubitPerfectCode()
    assert code.stabilizers == ("XZZXI", "IXZZX", "XIXZZ", "ZXIXZ")
    assert len(code.syndrome_table) == 16

    stab_mats = [_pauli_kron(s) for s in code.stabilizers]
    xl = _pauli_kron("XXXXX")
    yl = _pauli_kron("YYYYY")
    zl = _pauli_kron("ZZZZZ")
    eye32 = np.eye(32, dtype=np.complex128)

    for s in stab_mats:
        assert np.allclose(s, s.conj().T)
        assert np.allclose(s @ s, eye32)
        assert np.allclose(s @ xl, xl @ s)
        assert np.allclose(s @ zl, zl @ s)
    for i in range(4):
        for j in range(i + 1, 4):
            assert np.allclose(stab_mats[i] @ stab_mats[j], stab_mats[j] @ stab_mats[i])
    assert np.allclose(xl @ zl, -zl @ xl)

    # Verify relative-phase preservation on encoded |0>, |+>, and |+i>
    cards = InitialStateSampler.cardinal_states()
    for state_name, expected_bloch in (
        ("0", (0.0, 0.0, 1.0)),
        ("1", (0.0, 0.0, -1.0)),
        ("+", (1.0, 0.0, 0.0)),
        ("-", (-1.0, 0.0, 0.0)),
        ("+i", (0.0, 1.0, 0.0)),
        ("-i", (0.0, -1.0, 0.0)),
    ):
        psi = cards[state_name]
        sv = np.zeros((1, 32), dtype=np.complex128)
        sv[0, 0] = psi[0]
        sv[0, 16] = psi[1]
        _apply_ops_np(sv, code.encode_ops(), 5)
        vec = sv[0]

        for s in stab_mats:
            exp_s = float(np.real(vec.conj() @ s @ vec))
            assert abs(exp_s - 1.0) < 1e-12

        bloch = tuple(
            float(np.real(vec.conj() @ op @ vec)) for op in (xl, yl, zl)
        )
        assert np.allclose(bloch, expected_bloch, atol=1e-12)


def test_five_qubit_all_weight1_weight2_and_coset_errors() -> None:
    code = get_code("five_qubit")
    backends = [get_backend("numpy"), get_backend("torch_cuda"), get_backend("eigenkit_unitary")]
    psi_haar = InitialStateSampler.haar_random_states(1, seed=303)[0]

    # All 16 single-qubit errors (including IIIII) across all 3 backends
    for syn, err in code.syndrome_table.items():
        results = [
            b.simulate_cycle(code, psi_haar, None, 0.0, 2, seed=15, deterministic_errors=err)
            for b in backends
        ]
        for res in results:
            assert res.syndrome_counts == {syn: 2}
            assert res.mean_fidelity > 1.0 - 1e-12
        assert np.allclose(results[0].rho_out, results[1].rho_out, atol=1e-12)
        assert np.allclose(results[0].rho_out, results[2].rho_out, atol=1e-12)

    # Weight-2 errors exceed distance-3 correction capacity
    np_b = backends[0]
    cards = InitialStateSampler.cardinal_states()
    for err_w2, syn_w2 in (("XXIII", "1001"), ("ZZIII", "1111"), ("XZIII", "0100"), ("YYIII", "0110")):
        res_w2 = np_b.simulate_cycle(
            code, psi_haar, None, 0.0, 2, seed=15, deterministic_errors=err_w2
        )
        assert res_w2.syndrome_counts == {syn_w2: 2}
        assert res_w2.mean_fidelity < 0.99

    res_xx_plus = np_b.simulate_cycle(
        code, cards["+"], None, 0.0, 2, seed=15, deterministic_errors="XXIII"
    )
    assert res_xx_plus.mean_fidelity < 1e-12

    # Stabilizer coset errors (E = stabilizer * weight-1) recover cleanly
    for coset_err, coset_syn in (("IZZXI", "0001"), ("XZZXI", "0000"), ("XZZXZ", "0100")):
        res_c = np_b.simulate_cycle(
            code, psi_haar, None, 0.0, 2, seed=15, deterministic_errors=coset_err
        )
        assert res_c.syndrome_counts == {coset_syn: 2}
        assert res_c.mean_fidelity > 1.0 - 1e-12


def test_dfs_2_collective_dephasing_invariance_and_failure_modes() -> None:
    code = get_code("dfs_2")
    assert code.n_data == 2
    assert code.n_ancilla == 0
    assert code.n_total == 2
    assert code.stabilizers == ()
    assert code.syndrome_table == {}
    assert code.syndrome_ops() == []

    backends = [get_backend("numpy"), get_backend("torch_cuda"), get_backend("eigenkit_unitary")]
    cards = InitialStateSampler.cardinal_states()
    haars = InitialStateSampler.haar_random_states(2, seed=404)
    states = list(cards.values()) + list(haars)

    # Deterministic collective Z (ZZ) leaves all states invariant across all 3 backends
    for psi in states:
        results = [
            b.simulate_cycle(code, psi, None, 0.0, 3, seed=21, deterministic_errors="ZZ")
            for b in backends
        ]
        for res in results:
            assert res.syndrome_counts == {"": 3}
            assert res.mean_fidelity > 1.0 - 1e-12

    # Coherent collective dephasing Rz(phi) x Rz(phi) preserves all states across all 3 backends
    ek = backends[2]
    for phi in (0.0, 0.73, math.pi, -1.41):
        circuit = code.encode_ops() + coherent_collective_dephasing_ops(2, phi) + code.decode_ops()
        for psi in (cards["+"], cards["+i"], haars[0]):
            sv_np = np.zeros((1, 4), dtype=np.complex128)
            sv_np[0, 0] = psi[0]
            sv_np[0, 2] = psi[1]
            _apply_ops_np(sv_np, circuit, 2)
            rho_np = partial_trace_to_qubit0(sv_np[0], 2)
            assert quantum_fidelity(psi, rho_np) > 1.0 - 1e-12

            init_ek = np.zeros(4, dtype=np.complex128)
            init_ek[0] = psi[0]
            init_ek[2] = psi[1]
            sv_ek = ek.run_unitary_ops(circuit, n_qubits=2, initial_state=init_ek)
            rho_ek = partial_trace_to_qubit0(sv_ek, 2)
            assert quantum_fidelity(psi, rho_ek) > 1.0 - 1e-12

    # Independent ZI / IZ flips |+> -> |->, while XX preserves singlet |-> and flips |0>
    np_b = backends[0]
    for ind_z in ("ZI", "IZ"):
        res_plus = np_b.simulate_cycle(code, cards["+"], None, 0.0, 2, deterministic_errors=ind_z)
        assert res_plus.mean_fidelity < 1e-12
        res_zero = np_b.simulate_cycle(code, cards["0"], None, 0.0, 2, deterministic_errors=ind_z)
        assert res_zero.mean_fidelity > 1.0 - 1e-12

    res_xx_zero = np_b.simulate_cycle(code, cards["0"], None, 0.0, 2, deterministic_errors="XX")
    assert res_xx_zero.mean_fidelity < 1e-12
    res_xx_minus = np_b.simulate_cycle(code, cards["-"], None, 0.0, 2, deterministic_errors="XX")
    assert res_xx_minus.mean_fidelity > 1.0 - 1e-12

    # Stochastic collective dephasing vs independent phase_flip
    for b in backends[:2]:
        res_cd = b.simulate_cycle(
            code, cards["+i"], channel=ChannelSpec("collective_dephasing", 0.35), p=0.35, num_shots=500, seed=77
        )
        assert res_cd.syndrome_counts == {"": 500}
        assert res_cd.mean_fidelity > 1.0 - 1e-12

        res_pf = b.simulate_cycle(
            code, cards["+"], channel=ChannelSpec("phase_flip", 0.2), p=0.2, num_shots=4000, seed=77
        )
        expected_pf = 1.0 - 2.0 * 0.2 * 0.8  # 0.68
        assert abs(res_pf.mean_fidelity - expected_pf) < 4.5 * max(res_pf.std_error, 1e-3)


def test_stochastic_monte_carlo_agreement_numpy_and_cuda() -> None:
    cards = InitialStateSampler.cardinal_states()
    backends = [get_backend("numpy"), get_backend("torch_cuda")]

    # bit_flip_3 under bit_flip(0.1): 1 - 3p^2 + 2p^3 = 0.972
    bf = get_code("bit_flip_3")
    for b in backends:
        res = b.simulate_cycle(bf, cards["0"], "bit_flip", 0.1, num_shots=4000, seed=801)
        assert abs(res.mean_fidelity - 0.972) < 4.5 * max(res.std_error, 1e-3)

    # phase_flip_3 under phase_flip(0.1) on |0>: 0.972
    pf = get_code("phase_flip_3")
    for b in backends:
        res = b.simulate_cycle(pf, cards["0"], "phase_flip", 0.1, num_shots=4000, seed=802)
        assert abs(res.mean_fidelity - 0.972) < 4.5 * max(res.std_error, 1e-3)

    # five_qubit under bit_flip(0.08) on |0>: exact probability that E * rec(s(E)) commutes with ZZZZZ
    def _mul_pauli(a: str, b: str) -> str:
        tbl = {
            ("I", "I"): "I", ("I", "X"): "X", ("I", "Y"): "Y", ("I", "Z"): "Z",
            ("X", "I"): "X", ("X", "X"): "I", ("X", "Y"): "Z", ("X", "Z"): "Y",
            ("Y", "I"): "Y", ("Y", "X"): "Z", ("Y", "Y"): "I", ("Y", "Z"): "X",
            ("Z", "I"): "Z", ("Z", "X"): "Y", ("Z", "Y"): "X", ("Z", "Z"): "I",
        }
        return "".join(tbl[(x, y)] for x, y in zip(a, b))

    def _anticomm(a: str, b: str) -> int:
        return sum(1 for x, y in zip(a, b) if x != "I" and y != "I" and x != y) % 2

    fq = get_code("five_qubit")
    p_bf = 0.08
    expected_fq_bf = 0.0
    for bits in itertools.product("IX", repeat=5):
        err = "".join(bits)
        syn = "".join(str(_anticomm(err, g)) for g in fq.stabilizers)
        rec = fq.syndrome_table[syn]
        if _anticomm(_mul_pauli(err, rec), "ZZZZZ") == 0:
            w = err.count("X")
            expected_fq_bf += (p_bf**w) * ((1.0 - p_bf) ** (5 - w))

    for b in backends:
        res = b.simulate_cycle(fq, cards["0"], "bit_flip", p_bf, num_shots=5000, seed=803)
        assert abs(res.mean_fidelity - expected_fq_bf) < 4.5 * max(res.std_error, 1e-3)

    # five_qubit under depolarizing(0.08) on |+i>: exact 256-coset entanglement & state fidelity
    stabs = ["IIIII"]
    for gen in fq.stabilizers:
        stabs = stabs + [_mul_pauli(s, gen) for s in stabs]
    correctable = {
        _mul_pauli(s, rec) for s in stabs for rec in fq.syndrome_table.values()
    }
    p_dep = 0.08
    expected_fq_dep_ent = sum(
        ((1.0 - p_dep) ** err.count("I")) * ((p_dep / 3.0) ** (5 - err.count("I")))
        for err in correctable
    )
    expected_fq_dep_state = 1.0 - (2.0 / 3.0) * (1.0 - expected_fq_dep_ent)
    for b in backends:
        res = b.simulate_cycle(fq, cards["+i"], "depolarizing", p_dep, num_shots=5000, seed=804)
        assert abs(res.mean_fidelity - expected_fq_dep_state) < 4.5 * max(res.std_error, 1e-3)
