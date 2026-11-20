"""Unit tests for qecc.backends (NumPy, TorchCUDA RTX 5080, EigenkitUnitaryAdapter)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from qecc.backends import (
    EigenkitUnitaryAdapter,
    GateOp,
    NumPyCPUBackend,
    QECCCode,
    TorchCUDABackend,
    get_backend,
    list_available_backends,
)
from qecc.backends.base import validate_and_prepare_cycle_inputs
from qecc.backends.eigenkit_adapter import translate_ops_to_eigenkit
from qecc.backends.numpy_backend import _apply_ops_np
from qecc.backends.torch_cuda_backend import _apply_ops_torch
from qecc.channels import ChannelSpec
from qecc.metrics import InitialStateSampler, quantum_fidelity


class _MockBitFlip3Code:
    """Minimal 3-qubit bit-flip code (3 data + 2 ancilla) for M1 backend tests."""

    name = "mock_bit_flip_3"
    n_data = 3
    n_ancilla = 2
    n_total = 5
    stabilizers = ("ZZI", "IZZ")
    syndrome_table = {"00": "III", "10": "XII", "11": "IXI", "01": "IIX"}

    def encode_ops(self) -> list[GateOp]:
        return [
            GateOp("CNOT", targets=(1,), controls=(0,)),
            GateOp("CNOT", targets=(2,), controls=(0,)),
        ]

    def syndrome_ops(self) -> list[GateOp]:
        return [
            GateOp("CNOT", targets=(3,), controls=(0,)),
            GateOp("CNOT", targets=(3,), controls=(1,)),
            GateOp("CNOT", targets=(4,), controls=(1,)),
            GateOp("CNOT", targets=(4,), controls=(2,)),
        ]

    def recovery_ops(self, syndrome: str) -> list[GateOp]:
        mapping = {
            "00": [],
            "10": [GateOp("X", targets=(0,))],
            "11": [GateOp("X", targets=(1,))],
            "01": [GateOp("X", targets=(2,))],
        }
        return mapping[syndrome]

    def decode_ops(self) -> list[GateOp]:
        return [
            GateOp("CNOT", targets=(2,), controls=(0,)),
            GateOp("CNOT", targets=(1,), controls=(0,)),
        ]


class _MockDFS2Code:
    """Minimal 0-ancilla 2-qubit collective-dephasing DFS code."""

    name = "mock_dfs_2"
    n_data = 2
    n_ancilla = 0
    n_total = 2
    stabilizers = ("ZZ",)
    syndrome_table = {"": "II"}

    def encode_ops(self) -> list[GateOp]:
        return [
            GateOp("X", targets=(1,)),
            GateOp("CNOT", targets=(1,), controls=(0,)),
        ]

    def syndrome_ops(self) -> list[GateOp]:
        return []

    def recovery_ops(self, syndrome: str) -> list[GateOp]:
        return []

    def decode_ops(self) -> list[GateOp]:
        return [
            GateOp("CNOT", targets=(1,), controls=(0,)),
            GateOp("X", targets=(1,)),
        ]


class _MockRaw1Code:
    """Minimal 1-qubit unprotected baseline."""

    name = "mock_raw_1"
    n_data = 1
    n_ancilla = 0
    n_total = 1
    stabilizers: tuple[str, ...] = ()
    syndrome_table = {"": "I"}

    def encode_ops(self) -> list[GateOp]:
        return []

    def syndrome_ops(self) -> list[GateOp]:
        return []

    def recovery_ops(self, syndrome: str) -> list[GateOp]:
        return []

    def decode_ops(self) -> list[GateOp]:
        return []


def test_backend_registry_and_protocol() -> None:
    avail = list_available_backends()
    assert "numpy" in avail
    assert "eigenkit_unitary" in avail
    assert torch.cuda.is_available()
    assert "torch_cuda" in avail

    assert isinstance(get_backend("numpy"), NumPyCPUBackend)
    assert isinstance(get_backend("torch_cuda"), TorchCUDABackend)
    assert isinstance(get_backend("eigenkit_unitary"), EigenkitUnitaryAdapter)
    assert isinstance(_MockBitFlip3Code(), QECCCode)

    with pytest.raises(ValueError):
        get_backend("nonexistent_backend")


def test_all_gates_parity_numpy_cuda_eigenkit() -> None:
    ops = [
        GateOp("I", targets=(0,)),
        GateOp("H", targets=(0,)),
        GateOp("X", targets=(1,)),
        GateOp("Y", targets=(2,)),
        GateOp("Z", targets=(0,)),
        GateOp("S", targets=(1,)),
        GateOp("Sdg", targets=(2,)),
        GateOp("RX", targets=(0,), params=(0.73,)),
        GateOp("RY", targets=(1,), params=(1.19,)),
        GateOp("RZ", targets=(2,), params=(-0.61,)),
        GateOp("CNOT", targets=(2,), controls=(0,)),
        GateOp("CNOT", targets=(0,), controls=(2,)),
        GateOp("CZ", targets=(1,), controls=(0,)),
        GateOp("CZ", targets=(0,), controls=(2,)),
    ]
    rng = np.random.default_rng(17)
    init = rng.normal(size=8) + 1j * rng.normal(size=8)
    init /= np.linalg.norm(init)
    init = init.astype(np.complex128)

    state_np = init[None, :].copy()
    _apply_ops_np(state_np, ops, n_total=3)

    state_t = torch.tensor(init[None, :], dtype=torch.complex128, device="cuda")
    _apply_ops_torch(state_t, ops, n_total=3)
    state_cuda = state_t.cpu().numpy()

    ek = EigenkitUnitaryAdapter()
    state_ek = ek.run_unitary_ops(ops, n_qubits=3, initial_state=init)

    assert np.allclose(state_np[0], state_cuda[0], atol=1e-13)
    assert np.allclose(state_np[0], state_ek, atol=1e-13)


def test_zero_error_across_cardinal_states_all_backends() -> None:
    code = _MockBitFlip3Code()
    cards = InitialStateSampler.cardinal_states()
    backends = [get_backend("numpy"), get_backend("torch_cuda"), get_backend("eigenkit_unitary")]

    for b in backends:
        for state in cards.values():
            res = b.simulate_cycle(code, state, channel=None, p=0.0, num_shots=8, seed=11)
            assert res.mean_fidelity > 1.0 - 1e-12
            assert res.syndrome_counts == {"00": 8}
            res_ch0 = b.simulate_cycle(
                code, state, channel=ChannelSpec("bit_flip", 0.0), p=0.0, num_shots=4, seed=11
            )
            assert res_ch0.mean_fidelity > 1.0 - 1e-12


def test_deterministic_single_qubit_errors_all_backends() -> None:
    code = _MockBitFlip3Code()
    psi = np.array([math.sqrt(0.3), np.exp(1j * 0.8) * math.sqrt(0.7)], dtype=np.complex128)
    expected_syndromes = {"III": "00", "XII": "10", "IXI": "11", "IIX": "01"}

    np_b = get_backend("numpy")
    cu_b = get_backend("torch_cuda")
    ek_b = get_backend("eigenkit_unitary")

    for err_str, syn in expected_syndromes.items():
        r_np = np_b.simulate_cycle(code, psi, None, 0.0, 5, seed=3, deterministic_errors=err_str)
        r_cu = cu_b.simulate_cycle(code, psi, None, 0.0, 5, seed=3, deterministic_errors=err_str)
        r_ek = ek_b.simulate_cycle(code, psi, None, 0.0, 5, seed=3, deterministic_errors=err_str)
        for r in (r_np, r_cu, r_ek):
            assert r.mean_fidelity > 1.0 - 1e-12
            assert r.syndrome_counts == {syn: 5}
        assert np.allclose(r_np.rho_out, r_cu.rho_out, atol=1e-12)
        assert np.allclose(r_np.rho_out, r_ek.rho_out, atol=1e-12)

    # Per-shot list of deterministic errors
    shot_errs = ["XII", "IXI", "IIX", "III"]
    r_np_list = np_b.simulate_cycle(code, psi, None, 0.0, 4, seed=5, deterministic_errors=shot_errs)
    r_cu_list = cu_b.simulate_cycle(code, psi, None, 0.0, 4, seed=5, deterministic_errors=shot_errs)
    r_ek_list = ek_b.simulate_cycle(code, psi, None, 0.0, 4, seed=5, deterministic_errors=shot_errs)
    for r in (r_np_list, r_cu_list, r_ek_list):
        assert r.syndromes == ["10", "11", "01", "00"]
        assert r.mean_fidelity > 1.0 - 1e-12


def test_stochastic_bit_flip_matches_analytical_numpy_and_cuda() -> None:
    code = _MockBitFlip3Code()
    ket0 = InitialStateSampler.cardinal_states()["0"]
    p = 0.1
    n_shots = 5000
    expected_f = 1.0 - 3.0 * (p**2) + 2.0 * (p**3)  # 0.972

    for name in ("numpy", "torch_cuda"):
        backend = get_backend(name)
        res = backend.simulate_cycle(
            code, ket0, channel=ChannelSpec("bit_flip", p), p=p, num_shots=n_shots, seed=2026
        )
        assert res.num_shots == n_shots
        assert sum(res.syndrome_counts.values()) == n_shots
        assert abs(res.mean_fidelity - expected_f) < 4.0 * max(res.std_error, 1e-3)


def test_zero_ancilla_dfs_and_raw_codes() -> None:
    dfs = _MockDFS2Code()
    raw = _MockRaw1Code()
    psi = InitialStateSampler.cardinal_states()["+"]

    for b in (get_backend("numpy"), get_backend("torch_cuda")):
        res_dfs = b.simulate_cycle(
            dfs, psi, channel="collective_dephasing", p=0.25, num_shots=200, seed=9
        )
        assert res_dfs.mean_fidelity > 1.0 - 1e-12
        assert res_dfs.syndromes == [""] * 200
        assert res_dfs.syndrome_counts == {"": 200}

        res_raw = b.simulate_cycle(raw, psi, channel="phase_flip", p=0.2, num_shots=4000, seed=9)
        assert abs(res_raw.mean_fidelity - 0.8) < 4.0 * max(res_raw.std_error, 1e-3)


def test_eigenkit_adapter_rejects_stochastic_mode() -> None:
    ek = get_backend("eigenkit_unitary")
    code = _MockBitFlip3Code()
    ket0 = InitialStateSampler.cardinal_states()["0"]
    with pytest.raises(NotImplementedError):
        ek.simulate_cycle(code, ket0, channel=ChannelSpec("bit_flip", 0.1), p=0.1, num_shots=10)


def test_torch_cuda_chunking_matches_single_chunk() -> None:
    code = _MockBitFlip3Code()
    psi = InitialStateSampler.cardinal_states()["+i"]
    b_single = TorchCUDABackend(max_batch_bytes=1024**3, warmup=True)
    b_chunked = TorchCUDABackend(max_batch_bytes=4096, warmup=False)

    r1 = b_single.simulate_cycle(code, psi, channel="bit_flip", p=0.12, num_shots=128, seed=55)
    r2 = b_chunked.simulate_cycle(code, psi, channel="bit_flip", p=0.12, num_shots=128, seed=55)

    assert r1.syndromes == r2.syndromes
    assert r1.syndrome_counts == r2.syndrome_counts
    assert np.allclose(r1.fidelities, r2.fidelities, atol=1e-13)
    assert np.allclose(r1.rho_out, r2.rho_out, atol=1e-13)


def test_validation_errors_in_gate_op_and_simulate_cycle() -> None:
    with pytest.raises(ValueError):
        GateOp("X", targets=(0, 1))
    with pytest.raises(ValueError):
        GateOp("CNOT", targets=(0,), controls=(0,))
    with pytest.raises(ValueError):
        GateOp("RX", targets=(0,), params=())
    with pytest.raises(ValueError):
        GateOp("INVALID", targets=(0,))  # type: ignore[arg-type]

    code = _MockBitFlip3Code()
    ket0 = InitialStateSampler.cardinal_states()["0"]
    np_b = get_backend("numpy")

    with pytest.raises(ValueError):
        np_b.simulate_cycle(code, np.array([1.0, 1.0]), None, 0.0, 4)
    with pytest.raises(ValueError):
        np_b.simulate_cycle(code, ket0, None, p=0.1, num_shots=4)
    with pytest.raises(ValueError):
        np_b.simulate_cycle(code, ket0, ChannelSpec("bit_flip", 0.1), p=0.2, num_shots=4)
    with pytest.raises(ValueError):
        np_b.simulate_cycle(
            code, ket0, ChannelSpec("bit_flip", 0.0), p=0.0, num_shots=4, deterministic_errors="XII"
        )
    with pytest.raises(ValueError):
        np_b.simulate_cycle(code, ket0, None, 0.0, 4, deterministic_errors="II")
    with pytest.raises(ValueError):
        np_b.simulate_cycle(code, ket0, None, 0.0, 4, deterministic_errors=["XII", "IXI"])
    with pytest.raises(ValueError):
        np_b.simulate_cycle(code, ket0, None, -0.1, 4)


def test_np_bool_rejected_in_gate_op_and_cycle_inputs() -> None:
    with pytest.raises(ValueError):
        GateOp("RX", targets=(0,), params=(np.bool_(True),))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        GateOp("RX", targets=(0,), params=(np.bool_(False),))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        GateOp("X", targets=(np.bool_(True),))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        GateOp("CNOT", targets=(1,), controls=(np.bool_(False),))  # type: ignore[arg-type]

    code = _MockBitFlip3Code()
    ket0 = InitialStateSampler.cardinal_states()["0"]
    for bad_p in (np.bool_(True), np.bool_(False)):
        with pytest.raises(ValueError):
            validate_and_prepare_cycle_inputs(code, ket0, None, p=bad_p, num_shots=4)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        validate_and_prepare_cycle_inputs(
            code, ket0, None, p=0.0, num_shots=np.bool_(True)  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError):
        validate_and_prepare_cycle_inputs(
            code, ket0, None, p=0.0, num_shots=4, seed=np.bool_(True)  # type: ignore[arg-type]
        )


def test_slightly_unnormalized_initial_state_normalized_across_backends() -> None:
    code = _MockBitFlip3Code()
    base_psi = InitialStateSampler.cardinal_states()["+"]

    np_b = get_backend("numpy")
    cu_b = get_backend("torch_cuda")
    ek_b = get_backend("eigenkit_unitary")

    for scale in (1.0 + 4e-7, 1.0 - 4e-7):
        psi_u = base_psi * scale

        for b in (np_b, cu_b):
            res_stoch = b.simulate_cycle(
                code, psi_u, channel="depolarizing", p=0.2, num_shots=200, seed=1
            )
            tr_stoch = np.trace(res_stoch.rho_out)
            assert abs(float(tr_stoch.real) - 1.0) < 1e-13
            assert abs(float(tr_stoch.imag)) < 1e-13
            assert abs(res_stoch.mean_fidelity - quantum_fidelity(psi_u, res_stoch.rho_out)) < 1e-13

        for b in (np_b, cu_b, ek_b):
            res_det = b.simulate_cycle(
                code,
                psi_u,
                channel=None,
                p=0.0,
                num_shots=2,
                seed=1,
                deterministic_errors=["III", "XXZ"],
            )
            tr_det = np.trace(res_det.rho_out)
            assert abs(float(tr_det.real) - 1.0) < 1e-13
            assert abs(float(tr_det.imag)) < 1e-13
            assert abs(res_det.mean_fidelity - quantum_fidelity(psi_u, res_det.rho_out)) < 1e-13


class _SuperpositionSyndromeCode:
    """Minimal 2-qubit code producing a non-deterministic syndrome superposition."""

    name = "mock_superposition_syndrome"
    n_data = 1
    n_ancilla = 1
    n_total = 2
    stabilizers = ("Z",)
    syndrome_table = {"0": "I", "1": "X"}

    def encode_ops(self) -> list[GateOp]:
        return [GateOp("RY", targets=(0,), params=(1.2,))]

    def syndrome_ops(self) -> list[GateOp]:
        return [GateOp("CNOT", targets=(1,), controls=(0,))]

    def recovery_ops(self, syndrome: str) -> list[GateOp]:
        if syndrome == "0":
            return []
        if syndrome == "1":
            return [GateOp("X", targets=(0,))]
        raise ValueError(f"Unknown syndrome {syndrome!r}")

    def decode_ops(self) -> list[GateOp]:
        return []


def test_torch_cuda_unseeded_produces_distinct_syndromes() -> None:
    code = _SuperpositionSyndromeCode()
    ket0 = InitialStateSampler.cardinal_states()["0"]
    cu_b = get_backend("torch_cuda")

    runs = [
        cu_b.simulate_cycle(code, ket0, channel=None, p=0.0, num_shots=32, seed=None)
        for _ in range(3)
    ]
    unique_sequences = {tuple(r.syndromes) for r in runs}
    assert len(unique_sequences) > 1


def test_eigenkit_adapter_dfs2_mcz_fallback_duplicate_det_and_superposition() -> None:
    ek_b = EigenkitUnitaryAdapter()
    np_b = NumPyCPUBackend()
    ket_plus = InitialStateSampler.cardinal_states()["+"]
    ket0 = InitialStateSampler.cardinal_states()["0"]

    # 1. Zero-ancilla dfs_2 path on EigenkitUnitaryAdapter (p=0 and deterministic "ZZ")
    dfs = _MockDFS2Code()
    r_dfs_0 = ek_b.simulate_cycle(dfs, ket_plus, channel=None, p=0.0, num_shots=8, seed=3)
    r_dfs_zz = ek_b.simulate_cycle(
        dfs, ket_plus, channel=None, p=0.0, num_shots=8, seed=3, deterministic_errors="ZZ"
    )
    for r in (r_dfs_0, r_dfs_zz):
        assert r.mean_fidelity > 1.0 - 1e-12
        assert r.syndromes == [""] * 8
        assert r.syndrome_counts == {"": 8}

    # 2. translate_ops_to_eigenkit(..., supports_mcz=False) CZ decomposition parity
    cz_ops = [GateOp("CZ", targets=(1,), controls=(0,))]
    builder_no_mcz = translate_ops_to_eigenkit(cz_ops, n_qubits=2, supports_mcz=False)
    gate_names = [g.name for g in builder_no_mcz.build().gates]
    assert gate_names == ["H", "CNOT", "H"]
    from eigenkit import Circuit

    rng = np.random.default_rng(123)
    init = rng.normal(size=4) + 1j * rng.normal(size=4)
    init /= np.linalg.norm(init)
    init = init.astype(np.complex128)
    sv_no_mcz = np.asarray(
        Circuit(builder_no_mcz, observables=None, provider="numpy", interface="numpy").statevector(
            np.empty(0, dtype=np.float64), inputs=None, initial_state=init
        ),
        dtype=np.complex128,
    )
    sv_mcz = ek_b.run_unitary_ops(cz_ops, n_qubits=2, initial_state=init)
    assert np.allclose(sv_no_mcz, sv_mcz, atol=1e-13)

    # 3. Duplicate deterministic_errors rows (cnt > 1)
    bf3 = _MockBitFlip3Code()
    dup_errs = ["XII", "IXI", "XII", "IXI"]
    r_ek_dup = ek_b.simulate_cycle(
        bf3, ket_plus, channel=None, p=0.0, num_shots=4, seed=19, deterministic_errors=dup_errs
    )
    r_np_dup = np_b.simulate_cycle(
        bf3, ket_plus, channel=None, p=0.0, num_shots=4, seed=19, deterministic_errors=dup_errs
    )
    assert r_ek_dup.syndromes == ["10", "11", "10", "11"]
    assert r_ek_dup.mean_fidelity > 1.0 - 1e-12
    assert np.allclose(r_ek_dup.rho_out, r_np_dup.rho_out, atol=1e-12)

    # 4. Syndrome-superposition circuit parity vs NumPy
    sup = _SuperpositionSyndromeCode()
    r_ek_sup = ek_b.simulate_cycle(sup, ket0, channel=None, p=0.0, num_shots=2000, seed=7)
    r_np_sup = np_b.simulate_cycle(sup, ket0, channel=None, p=0.0, num_shots=2000, seed=7)
    assert r_ek_sup.syndrome_counts == r_np_sup.syndrome_counts
    assert np.allclose(r_ek_sup.rho_out, r_np_sup.rho_out, atol=1e-12)


def test_torch_cuda_and_base_boundary_validation() -> None:
    with pytest.raises(ValueError):
        TorchCUDABackend(max_batch_bytes=512, warmup=False)
    with pytest.raises(ValueError):
        TorchCUDABackend(max_batch_bytes=True, warmup=False)  # type: ignore[arg-type]

    # n_ancilla == 0 chunked execution matches single-chunk
    dfs = _MockDFS2Code()
    psi = InitialStateSampler.cardinal_states()["+i"]
    b_single = TorchCUDABackend(max_batch_bytes=1024**3, warmup=False)
    b_chunked = TorchCUDABackend(max_batch_bytes=1024, warmup=False)
    r1 = b_single.simulate_cycle(dfs, psi, channel="depolarizing", p=0.15, num_shots=64, seed=42)
    r2 = b_chunked.simulate_cycle(dfs, psi, channel="depolarizing", p=0.15, num_shots=64, seed=42)
    assert r1.syndromes == r2.syndromes
    assert np.allclose(r1.fidelities, r2.fidelities, atol=1e-13)
    assert np.allclose(r1.rho_out, r2.rho_out, atol=1e-13)

    # base.py invalid input boundaries
    code = _MockBitFlip3Code()
    ket0 = InitialStateSampler.cardinal_states()["0"]
    with pytest.raises(TypeError):
        validate_and_prepare_cycle_inputs("not_a_code", ket0, None, 0.0, 4)  # type: ignore[arg-type]
    for bad_state in (
        np.array([np.nan, 0.0]),
        np.array([np.inf, 0.0]),
        np.eye(2, dtype=np.complex128),
    ):
        with pytest.raises(ValueError):
            validate_and_prepare_cycle_inputs(code, bad_state, None, 0.0, 4)
    for bad_p in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            validate_and_prepare_cycle_inputs(code, ket0, None, p=bad_p, num_shots=4)
    with pytest.raises(ValueError):
        validate_and_prepare_cycle_inputs(code, ket0, None, p=0.0, num_shots=4, seed=-1)
