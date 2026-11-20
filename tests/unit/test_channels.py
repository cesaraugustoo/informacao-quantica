"""Unit tests for qecc.channels.pauli."""

from __future__ import annotations

import math

import numpy as np
import pytest

from qecc.backends.base import GateOp
from qecc.channels import (
    PAULI_I,
    PAULI_X,
    PAULI_Y,
    PAULI_Z,
    ChannelSpec,
    DeterministicErrorInjector,
    StochasticPauliChannel,
    codes_to_strings,
    coherent_collective_dephasing_ops,
)


def test_channel_spec_construction_and_immutability() -> None:
    spec_pos = ChannelSpec("bit_flip", 0.1)
    spec_kw = ChannelSpec(channel_type="depolarizing", p=0.25)
    assert spec_pos.channel_type == "bit_flip"
    assert spec_pos.p == 0.1
    assert spec_kw.channel_type == "depolarizing"
    assert spec_kw.p == 0.25
    assert ChannelSpec("phase_flip", 0.0).p == 0.0
    assert ChannelSpec("collective_dephasing", 1.0).p == 1.0

    with pytest.raises(Exception):
        spec_pos.p = 0.5  # type: ignore[misc]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"channel_type": "bit_flip", "p": -0.01},
        {"channel_type": "bit_flip", "p": 1.01},
        {"channel_type": "bit_flip", "p": float("nan")},
        {"channel_type": "bit_flip", "p": float("inf")},
        {"channel_type": "bit_flip", "p": True},
        {"channel_type": "bit_flip", "p": np.bool_(True)},
        {"channel_type": "bit_flip", "p": np.bool_(False)},
        {"channel_type": "bit_flip", "p": "not_a_number"},
        {"channel_type": "amplitude_damping", "p": 0.1},
    ],
)
def test_channel_spec_rejects_invalid_inputs(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        ChannelSpec(**kwargs)


def test_channel_spec_rejects_extra_fields() -> None:
    with pytest.raises(ValueError):
        ChannelSpec(channel_type="bit_flip", p=0.1, extra_field=42)  # type: ignore[call-arg]


def test_stochastic_channel_boundary_probabilities_and_validation() -> None:
    rng = np.random.default_rng(123)
    for ctype in ("bit_flip", "phase_flip", "depolarizing", "collective_dephasing"):
        zeros = StochasticPauliChannel(ChannelSpec(ctype, 0.0)).sample(4, 200, rng)  # type: ignore[arg-type]
        assert zeros.shape == (200, 4)
        assert zeros.dtype == np.uint8
        assert np.all(zeros == PAULI_I)

    bf1 = StochasticPauliChannel(ChannelSpec("bit_flip", 1.0)).sample(3, 50, rng)
    assert np.all(bf1 == PAULI_X)

    pf1 = StochasticPauliChannel(ChannelSpec("phase_flip", 1.0)).sample(3, 50, rng)
    assert np.all(pf1 == PAULI_Z)

    cd1 = StochasticPauliChannel(ChannelSpec("collective_dephasing", 1.0)).sample(3, 50, rng)
    assert np.all(cd1 == PAULI_Z)

    dep1 = StochasticPauliChannel(ChannelSpec("depolarizing", 1.0)).sample(4, 500, rng)
    assert not np.any(dep1 == PAULI_I)
    assert set(np.unique(dep1).tolist()) == {PAULI_X, PAULI_Y, PAULI_Z}

    ch = StochasticPauliChannel(ChannelSpec("bit_flip", 0.1))
    with pytest.raises(ValueError):
        ch.sample(0, 10, rng)
    with pytest.raises(ValueError):
        ch.sample(2, 0, rng)
    with pytest.raises(ValueError):
        ch.sample(True, 10, rng)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        ch.sample(2, 10, rng="bad")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("channel_type", "expected_code"),
    [("bit_flip", PAULI_X), ("phase_flip", PAULI_Z)],
)
def test_bit_and_phase_flip_statistical_independence(channel_type: str, expected_code: int) -> None:
    p = 0.15
    n_shots = 40000
    n_qubits = 4
    rng = np.random.Generator(np.random.PCG64(2026))
    codes = StochasticPauliChannel(ChannelSpec(channel_type, p)).sample(n_qubits, n_shots, rng)  # type: ignore[arg-type]

    assert set(np.unique(codes).tolist()).issubset({PAULI_I, expected_code})
    sigma_single = math.sqrt(p * (1.0 - p) / n_shots)
    for q in range(n_qubits):
        emp = float(np.mean(codes[:, q] == expected_code))
        assert abs(emp - p) < 4.0 * sigma_single

    p2 = p * p
    sigma_pair = math.sqrt(p2 * (1.0 - p2) / n_shots)
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            joint = float(np.mean((codes[:, i] != PAULI_I) & (codes[:, j] != PAULI_I)))
            assert abs(joint - p2) < 4.0 * sigma_pair


def test_depolarizing_channel_statistics_and_independence() -> None:
    p = 0.24
    n_shots = 40000
    n_qubits = 4
    rng = np.random.Generator(np.random.PCG64(99))
    codes = StochasticPauliChannel(ChannelSpec("depolarizing", p)).sample(n_qubits, n_shots, rng)

    sigma_p3 = math.sqrt((p / 3.0) * (1.0 - p / 3.0) / n_shots)
    sigma_id = math.sqrt((1.0 - p) * p / n_shots)
    for q in range(n_qubits):
        assert abs(float(np.mean(codes[:, q] == PAULI_I)) - (1.0 - p)) < 4.0 * sigma_id
        for code in (PAULI_X, PAULI_Y, PAULI_Z):
            assert abs(float(np.mean(codes[:, q] == code)) - (p / 3.0)) < 4.0 * sigma_p3

    p2 = p * p
    sigma_pair = math.sqrt(p2 * (1.0 - p2) / n_shots)
    joint = float(np.mean((codes[:, 0] != PAULI_I) & (codes[:, 2] != PAULI_I)))
    assert abs(joint - p2) < 4.0 * sigma_pair


def test_collective_dephasing_statistics_and_coherent_ops() -> None:
    p = 0.3
    n_shots = 30000
    n_qubits = 4
    rng = np.random.Generator(np.random.PCG64(7))
    codes = StochasticPauliChannel(ChannelSpec("collective_dephasing", p)).sample(
        n_qubits, n_shots, rng
    )
    assert np.all(codes == codes[:, [0]])
    assert set(np.unique(codes).tolist()).issubset({PAULI_I, PAULI_Z})
    emp = float(np.mean(codes[:, 0] == PAULI_Z))
    sigma = math.sqrt(p * (1.0 - p) / n_shots)
    assert abs(emp - p) < 4.0 * sigma

    ops = coherent_collective_dephasing_ops(3, 0.42)
    assert ops == [
        GateOp(name="RZ", targets=(0,), params=(0.42,)),
        GateOp(name="RZ", targets=(1,), params=(0.42,)),
        GateOp(name="RZ", targets=(2,), params=(0.42,)),
    ]
    with pytest.raises(ValueError):
        coherent_collective_dephasing_ops(0, 0.1)
    with pytest.raises(ValueError):
        coherent_collective_dephasing_ops(2, float("nan"))
    with pytest.raises(ValueError):
        coherent_collective_dephasing_ops(2, True)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        coherent_collective_dephasing_ops(2, np.bool_(True))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        coherent_collective_dephasing_ops(2, np.bool_(False))  # type: ignore[arg-type]


def test_deterministic_error_injector_and_codes_to_strings() -> None:
    assert DeterministicErrorInjector.validate_pauli_string("IXYZ", 4) == "IXYZ"
    assert DeterministicErrorInjector.from_single_qubit(2, "Z", 5) == "IIZII"

    arr = DeterministicErrorInjector.to_codes("IXYZ")
    assert arr.dtype == np.uint8
    assert np.array_equal(arr, np.array([0, 1, 2, 3], dtype=np.uint8))
    assert codes_to_strings(arr) == ["IXYZ"]

    batch = DeterministicErrorInjector.batch_to_codes(["XII", "IZI"], 3)
    assert batch.shape == (2, 3)
    assert codes_to_strings(batch) == ["XII", "IZI"]

    ops = DeterministicErrorInjector.to_gate_ops("IXZ", 3)
    assert ops == [GateOp("X", (1,)), GateOp("Z", (2,))]

    ops_with_id = DeterministicErrorInjector.to_gate_ops((1, "Y"), 3, include_identity=True)
    assert ops_with_id == [GateOp("I", (0,)), GateOp("Y", (1,)), GateOp("I", (2,))]

    for bad_str, nq in [("XI", 3), ("xII", 3), ("AII", 3), ("", 1)]:
        with pytest.raises(ValueError):
            DeterministicErrorInjector.validate_pauli_string(bad_str, nq)

    with pytest.raises(ValueError):
        DeterministicErrorInjector.from_single_qubit(3, "X", 3)
    with pytest.raises(ValueError):
        DeterministicErrorInjector.from_single_qubit(-1, "X", 3)
    with pytest.raises(ValueError):
        DeterministicErrorInjector.from_single_qubit(True, "X", 3)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        DeterministicErrorInjector.batch_to_codes([], 3)
