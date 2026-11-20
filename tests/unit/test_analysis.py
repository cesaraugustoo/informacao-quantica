"""Unit tests for qecc.analysis, qecc.cli, qecc exports, and threshold verification script."""

from __future__ import annotations

import itertools
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import qecc
from qecc.analysis import (
    VALID_CHANNELS,
    VALID_CODES,
    analytical_fidelity,
    analytical_gain_threshold,
    five_qubit_single_axis_weight_enumerator,
    five_qubit_weight_enumerator,
    verify_knill_laflamme,
)
from qecc.backends import get_backend
from qecc.cli import main as cli_main
from qecc.codes import get_code
from qecc.metrics import InitialStateSampler

REPO_ROOT = Path(__file__).resolve().parents[2]


def _exhaustive_circuit_state_fidelity(
    code_name: str, channel_type: str, p: float
) -> float:
    """Compute exact 6-cardinal state fidelity via exhaustive Pauli string enumeration."""
    backend = get_backend("numpy")
    states = list(InitialStateSampler.cardinal_states().values())

    if code_name == "raw":
        f_ent = 1.0 - p
        return 1.0 - (2.0 / 3.0) * (1.0 - f_ent)

    code = get_code(code_name)
    n_data = code.n_data

    if channel_type == "collective_dephasing":
        patterns = [("I" * n_data, 1.0 - p), ("Z" * n_data, p)]
    elif channel_type == "bit_flip":
        patterns = []
        for bits in itertools.product(("I", "X"), repeat=n_data):
            w = sum(1 for ch in bits if ch != "I")
            patterns.append(("".join(bits), (p**w) * ((1.0 - p) ** (n_data - w))))
    elif channel_type == "phase_flip":
        patterns = []
        for bits in itertools.product(("I", "Z"), repeat=n_data):
            w = sum(1 for ch in bits if ch != "I")
            patterns.append(("".join(bits), (p**w) * ((1.0 - p) ** (n_data - w))))
    elif channel_type == "depolarizing":
        patterns = []
        for paulis in itertools.product(("I", "X", "Y", "Z"), repeat=n_data):
            w = sum(1 for ch in paulis if ch != "I")
            patterns.append(
                ("".join(paulis), ((p / 3.0) ** w) * ((1.0 - p) ** (n_data - w)))
            )
    else:
        raise AssertionError(f"Unexpected channel_type: {channel_type}")

    error_strings = [pat for pat, _ in patterns]
    probs = np.array([pr for _, pr in patterns], dtype=np.float64)

    avg_by_error = np.zeros(len(error_strings), dtype=np.float64)
    for psi in states:
        res = backend.simulate_cycle(
            code=code,
            initial_state=psi,
            channel=None,
            p=0.0,
            num_shots=len(error_strings),
            deterministic_errors=error_strings,
        )
        avg_by_error += res.fidelities / len(states)

    return float(np.dot(probs, avg_by_error))


@pytest.mark.parametrize("code_name", VALID_CODES)
@pytest.mark.parametrize("channel_type", VALID_CHANNELS)
def test_analytical_fidelity_matches_exhaustive_circuit_all_20_pairs(
    code_name: str, channel_type: str
) -> None:
    """Every (code, channel) formula matches exhaustive 4^n circuit simulation."""
    if code_name == "five_qubit" and channel_type == "depolarizing":
        p_test = 0.17
        # Verify 1024-term depolarizing enumerator against analytical_fidelity directly
        cosets = five_qubit_weight_enumerator()
        f_enum = sum(
            cnt * ((1.0 - p_test) ** (5 - w)) * ((p_test / 3.0) ** w)
            for w, cnt in cosets["I"].items()
        )
        f_ana_ent = analytical_fidelity(code_name, channel_type, p_test, metric="entanglement")
        f_ana_state = analytical_fidelity(code_name, channel_type, p_test, metric="state")
        assert f_ana_ent == pytest.approx(f_enum, abs=1e-13)
        assert f_ana_state == pytest.approx(1.0 - (2.0 / 3.0) * (1.0 - f_enum), abs=1e-13)
        # Also spot-check high-shot Monte Carlo circuit simulation
        backend = get_backend("numpy")
        code = get_code(code_name)
        states = list(InitialStateSampler.cardinal_states().values())
        fids = [
            backend.simulate_cycle(code, psi, channel_type, p_test, 1000, seed=100 + i).mean_fidelity
            for i, psi in enumerate(states)
        ]
        assert np.mean(fids) == pytest.approx(f_ana_state, abs=0.025)
        return

    p_test = 0.20
    exact_state = _exhaustive_circuit_state_fidelity(code_name, channel_type, p_test)
    exact_ent = 1.0 - 1.5 * (1.0 - exact_state)

    f_state = analytical_fidelity(code_name, channel_type, p_test, metric="state")
    f_ent = analytical_fidelity(code_name, channel_type, p_test, metric="entanglement")

    assert isinstance(f_state, float)
    assert isinstance(f_ent, float)
    assert f_state == pytest.approx(exact_state, abs=1e-12)
    assert f_ent == pytest.approx(exact_ent, abs=1e-12)


def test_analytical_fidelity_polynomials_and_shapes() -> None:
    p_arr = np.linspace(0.0, 0.5, 51)
    expected_3q = 1.0 - 3.0 * (p_arr**2) + 2.0 * (p_arr**3)
    assert np.allclose(analytical_fidelity("bit_flip_3", "bit_flip", p_arr), expected_3q, atol=1e-12)
    assert np.allclose(analytical_fidelity("phase_flip_3", "phase_flip", p_arr), expected_3q, atol=1e-12)

    q = 1.0 - p_arr
    expected_5q = (
        q**5
        + 5.0 * p_arr * (q**4)
        + (20.0 / 9.0) * (p_arr**3) * (q**2)
        + (5.0 / 3.0) * (p_arr**4) * q
        + (5.0 / 27.0) * (p_arr**5)
    )
    f_5q = analytical_fidelity("five_qubit", "depolarizing", p_arr)
    assert isinstance(f_5q, np.ndarray)
    assert f_5q.dtype == np.float64
    assert f_5q.shape == p_arr.shape
    assert np.allclose(f_5q, expected_5q, atol=1e-12)

    # 2D and 0D ndarray shape preservation
    p_2d = np.array([[0.0, 0.1], [0.25, 0.5]])
    f_2d = analytical_fidelity("dfs_2", "collective_dephasing", p_2d)
    assert isinstance(f_2d, np.ndarray)
    assert f_2d.shape == (2, 2)
    assert np.allclose(f_2d, np.ones((2, 2)), atol=1e-12)

    p_0d = np.array(0.25)
    f_0d = analytical_fidelity("raw", "bit_flip", p_0d)
    assert isinstance(f_0d, np.ndarray)
    assert f_0d.shape == ()
    assert float(f_0d) == pytest.approx(0.75, abs=1e-12)

    # Scalar Python & NumPy scalars return float
    assert isinstance(analytical_fidelity("raw", "bit_flip", 0), float)
    assert isinstance(analytical_fidelity("raw", "bit_flip", np.float64(0.1)), float)

    # List input returns ndarray
    f_list = analytical_fidelity("raw", "depolarizing", [0.0, 0.3], metric="state")
    assert isinstance(f_list, np.ndarray)
    assert np.allclose(f_list, [1.0, 0.8], atol=1e-12)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"code_name": "invalid", "channel_type": "bit_flip", "p": 0.1},
        {"code_name": "bit_flip_3", "channel_type": "invalid", "p": 0.1},
        {"code_name": "bit_flip_3", "channel_type": "bit_flip", "p": 0.1, "metric": "bad"},
        {"code_name": "bit_flip_3", "channel_type": "bit_flip", "p": -0.01},
        {"code_name": "bit_flip_3", "channel_type": "bit_flip", "p": 1.01},
        {"code_name": "bit_flip_3", "channel_type": "bit_flip", "p": float("nan")},
        {"code_name": "bit_flip_3", "channel_type": "bit_flip", "p": float("inf")},
        {"code_name": "bit_flip_3", "channel_type": "bit_flip", "p": True},
        {"code_name": "bit_flip_3", "channel_type": "bit_flip", "p": np.bool_(False)},
        {"code_name": "bit_flip_3", "channel_type": "bit_flip", "p": [0.1, True]},
        {"code_name": "bit_flip_3", "channel_type": "bit_flip", "p": np.array([True, False])},
        {"code_name": "bit_flip_3", "channel_type": "bit_flip", "p": np.array([0.1, -0.2])},
        {"code_name": "bit_flip_3", "channel_type": "bit_flip", "p": np.array([0.1, np.nan])},
        {"code_name": "bit_flip_3", "channel_type": "bit_flip", "p": 0.1 + 0.0j},
        {"code_name": "bit_flip_3", "channel_type": "bit_flip", "p": "0.1"},
    ],
)
def test_analytical_fidelity_strict_validation(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        analytical_fidelity(**kwargs)  # type: ignore[arg-type]


def test_analytical_gain_threshold_values_sign_flips_and_validation() -> None:
    assert analytical_gain_threshold("bit_flip_3", "bit_flip") == pytest.approx(0.5, abs=1e-12)
    assert analytical_gain_threshold("phase_flip_3", "phase_flip") == pytest.approx(0.5, abs=1e-12)
    assert analytical_gain_threshold("dfs_2", "collective_dephasing") == pytest.approx(1.0, abs=1e-12)

    p_5q_dep = analytical_gain_threshold("five_qubit", "depolarizing")
    p_5q_bf = analytical_gain_threshold("five_qubit", "bit_flip")
    p_5q_pf = analytical_gain_threshold("five_qubit", "phase_flip")

    assert p_5q_dep == pytest.approx(0.137627564304, abs=1e-9)
    assert p_5q_bf == pytest.approx(0.131123147904, abs=1e-9)
    assert p_5q_pf == pytest.approx(0.131123147904, abs=1e-9)

    # Non-gaining pairs return 0.0
    for pair in (
        ("raw", "bit_flip"),
        ("bit_flip_3", "depolarizing"),
        ("bit_flip_3", "phase_flip"),
        ("phase_flip_3", "depolarizing"),
        ("five_qubit", "collective_dephasing"),
        ("dfs_2", "phase_flip"),
        ("dfs_2", "depolarizing"),
    ):
        assert analytical_gain_threshold(*pair) == pytest.approx(0.0, abs=1e-12)

    # Verify strict sign flip across nontrivial thresholds in (0, 0.5)
    delta = 1e-4
    for code_name, channel_type, p_star in (
        ("five_qubit", "depolarizing", p_5q_dep),
        ("five_qubit", "bit_flip", p_5q_bf),
        ("five_qubit", "phase_flip", p_5q_pf),
    ):
        assert analytical_fidelity(code_name, channel_type, p_star - delta) > 1.0 - (p_star - delta)
        assert analytical_fidelity(code_name, channel_type, p_star + delta) < 1.0 - (p_star + delta)
        assert analytical_fidelity(code_name, channel_type, p_star, metric="state") == pytest.approx(
            1.0 - (2.0 / 3.0) * p_star, abs=1e-12
        )

    with pytest.raises(ValueError):
        analytical_gain_threshold("unknown", "bit_flip")
    with pytest.raises(ValueError):
        analytical_gain_threshold("bit_flip_3", "unknown")


def test_five_qubit_enumerators_and_knill_laflamme_verification() -> None:
    cosets = five_qubit_weight_enumerator()
    assert set(cosets.keys()) == {"I", "X", "Y", "Z"}
    assert sum(sum(d.values()) for d in cosets.values()) == 1024
    for op in ("I", "X", "Y", "Z"):
        assert sum(cosets[op].values()) == 256
    assert [cosets["I"][w] for w in range(6)] == [1, 15, 0, 60, 135, 45]

    assert five_qubit_single_axis_weight_enumerator("X") == {0: 1, 1: 5, 2: 0, 3: 0, 4: 0, 5: 0}
    assert five_qubit_single_axis_weight_enumerator("Z") == {0: 1, 1: 5, 2: 0, 3: 0, 4: 0, 5: 0}

    for code_name in ("bit_flip_3", "phase_flip_3", "five_qubit", "dfs_2"):
        kl = verify_knill_laflamme(code_name)
        assert kl.passed is True
        assert kl.max_violation < 1e-12
        assert np.allclose(kl.c_matrix, kl.c_matrix.conj().T, atol=1e-12)

    with pytest.raises(ValueError):
        verify_knill_laflamme("raw")


def test_top_level_qecc_exports_preserved_and_extended() -> None:
    expected_24 = {
        "__version__",
        "BitFlip3QubitCode",
        "ChannelSpec",
        "DeterministicErrorInjector",
        "EigenkitUnitaryAdapter",
        "FiveQubitPerfectCode",
        "GateOp",
        "InitialStateSampler",
        "NumPyCPUBackend",
        "PhaseFlip3QubitCode",
        "QECCBackend",
        "QECCCode",
        "SimulationResult",
        "SingleQubitTomography",
        "StochasticPauliChannel",
        "TorchCUDABackend",
        "TwoQubitDFSDemo",
        "get_backend",
        "get_code",
        "list_available_backends",
        "list_available_codes",
        "partial_trace_batch_to_qubit0",
        "partial_trace_to_qubit0",
        "quantum_fidelity",
    }
    exported = set(qecc.__all__)
    assert expected_24.issubset(exported)
    assert {"analytical_fidelity", "analytical_gain_threshold"}.issubset(exported)
    assert qecc.analytical_fidelity is analytical_fidelity
    assert qecc.analytical_gain_threshold is analytical_gain_threshold


def test_qecc_cli_help_and_subcommands(capsys: pytest.CaptureFixture[str]) -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "qecc.cli", "--help"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc.returncode == 0
    assert "simulate" in proc.stdout
    assert "verify" in proc.stdout
    assert "sweep" in proc.stdout

    assert cli_main(["info", "--json"]) == 0
    info_data = json.loads(capsys.readouterr().out)
    assert "five_qubit" in info_data["codes"]
    assert math.isclose(info_data["nontrivial_gain_thresholds"]["five_qubit/depolarizing"], 0.1376275643, abs_tol=1e-7)

    assert cli_main(["simulate", "--code", "bit_flip_3", "--channel", "bit_flip", "--p", "0.0", "--shots", "50", "--backend", "numpy", "--state", "0", "--json"]) == 0
    sim_data = json.loads(capsys.readouterr().out)
    assert sim_data["mean_fidelity"] > 1.0 - 1e-10

    assert cli_main(["verify", "--json"]) == 0
    verify_data = json.loads(capsys.readouterr().out)
    assert verify_data["five_qubit_enumerator"]["total_paulis"] == 1024

    assert cli_main(["benchmark", "--quick", "--shots", "200", "--codes", "bit_flip_3", "--json"]) == 0
    bench_data = json.loads(capsys.readouterr().out)
    assert len(bench_data) >= 1


def test_verify_analytical_thresholds_script_cwd_independent_and_deterministic(
    tmp_path: Path,
) -> None:
    script = REPO_ROOT / "scripts" / "verify_analytical_thresholds.py"
    assert script.is_file()

    outputs: list[str] = []
    for _ in range(2):
        proc = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(tmp_path),
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr
        outputs.append(proc.stdout)

    assert outputs[0] == outputs[1]
    out = outputs[0]
    assert "Knill-Laflamme" in out
    assert "1024" in out
    assert "0.5" in out
    assert "0.137628" in out
    assert "0.131123" in out
