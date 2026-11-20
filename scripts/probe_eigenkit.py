#!/usr/bin/env python3
"""Live technical probe of eigenkit v1.3.0 QECC primitive coverage and gaps."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import typing
from pathlib import Path
from typing import Any

import numpy as np


def run_probe() -> dict[str, Any]:
    results: dict[str, Any] = {"groups": {}, "all_passed": False}
    failures: list[str] = []

    def record(group: str, name: str, status: str, detail: Any) -> None:
        results["groups"].setdefault(group, {})[name] = {
            "status": status,
            "detail": detail,
        }
        if status == "FAIL":
            failures.append(f"{group}.{name}: {detail}")

    # ------------------------------------------------------------------
    # Group A: Package & Lazy Import Isolation Checks
    # ------------------------------------------------------------------
    import eigenkit

    torch_loaded_after_eigenkit_import = "torch" in sys.modules
    record(
        "A_import_isolation",
        "eigenkit_version",
        "PASS",
        getattr(eigenkit, "__version__", "unknown"),
    )
    record(
        "A_import_isolation",
        "no_torch_imported_eagerly",
        "PASS" if not torch_loaded_after_eigenkit_import else "FAIL",
        {"torch_in_sys_modules": torch_loaded_after_eigenkit_import},
    )

    from eigenkit.core.backends.cudaquantum_backend import (
        CudaQuantumBackend,
        CudaQuantumCompiled,
    )
    from eigenkit.core.backends.numpy_backend import NumpySimulator
    from eigenkit.core.builder import CircuitBuilder
    from eigenkit.core.ir import ALL_GATES, CircuitSpec, GateName
    from eigenkit.core.provider import registered_names

    reg_names = sorted(registered_names())
    record(
        "A_import_isolation",
        "registered_names",
        "PASS" if "numpy" in reg_names else "FAIL",
        reg_names,
    )

    # ------------------------------------------------------------------
    # Group B: IR & CircuitBuilder Gate Vocabulary Checks
    # ------------------------------------------------------------------
    expected_ir_gates = {
        "RX",
        "RY",
        "RZ",
        "H",
        "CNOT",
        "X",
        "Z",
        "S",
        "Sdg",
        "MCZ",
        "MCX",
        "PhaseShift",
        "SWAP",
        "CRY",
        "CPHASE",
        "AmplitudeEncoding",
        "QubitUnitary",
        "ControlledQubitUnitary",
    }
    gate_name_args = set(typing.get_args(GateName))
    all_gates_set = set(ALL_GATES)
    record(
        "B_ir_and_builder",
        "ir_gate_vocabulary",
        "PASS"
        if (gate_name_args == expected_ir_gates and all_gates_set == expected_ir_gates)
        else "FAIL",
        sorted(all_gates_set),
    )

    spec_fields = tuple(f.name for f in dataclasses.fields(CircuitSpec))
    has_dynamic_fields = any(
        k in spec_fields for k in ("clbits", "measurements", "noise", "resets", "conditions")
    )
    record(
        "B_ir_and_builder",
        "circuit_spec_fields",
        "PASS"
        if (spec_fields == ("n_qubits", "n_weights", "n_inputs", "gates") and not has_dynamic_fields)
        else "FAIL",
        list(spec_fields),
    )

    present_methods = ["h", "x", "z", "s", "sdg", "cnot", "rx", "ry", "rz", "mcx", "mcz"]
    absent_methods = ["y", "cz", "cy", "measure", "reset", "c_if", "barrier", "kraus", "noise"]
    missing_expected = [m for m in present_methods if not hasattr(CircuitBuilder, m)]
    unexpected_present = [m for m in absent_methods if hasattr(CircuitBuilder, m)]
    record(
        "B_ir_and_builder",
        "builder_methods_present",
        "PASS" if not missing_expected else "FAIL",
        {"checked": present_methods, "missing": missing_expected},
    )
    record(
        "B_ir_and_builder",
        "builder_qecc_gaps_absent",
        "CONFIRMED_GAP" if not unexpected_present else "FAIL",
        {"confirmed_absent": absent_methods, "unexpected_present": unexpected_present},
    )

    # Test Y = S * X * Sdg (builder.sdg -> x -> s) and CZ = H(t) CNOT(c,t) H(t) == MCZ(c,t)
    psi_1 = np.array([math.sqrt(0.3), math.sqrt(0.7) * np.exp(1j * 0.4)], dtype=np.complex128)
    b_y = CircuitBuilder(1).sdg(0).x(0).s(0)
    sv_y = eigenkit.Circuit(b_y).statevector(np.empty(0), initial_state=psi_1)
    y_mat = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
    y_err = float(np.max(np.abs(sv_y - (y_mat @ psi_1))))

    psi_2 = np.array([0.3, -0.4j, 0.5, math.sqrt(0.5)], dtype=np.complex128)
    psi_2 /= np.linalg.norm(psi_2)
    sv_mcz = eigenkit.Circuit(CircuitBuilder(2).mcz(0, 1)).statevector(
        np.empty(0), initial_state=psi_2
    )
    sv_h_cnot_h = eigenkit.Circuit(CircuitBuilder(2).h(1).cnot(0, 1).h(1)).statevector(
        np.empty(0), initial_state=psi_2
    )
    cz_err = float(np.max(np.abs(sv_mcz - sv_h_cnot_h)))
    record(
        "B_ir_and_builder",
        "y_and_cz_emulation_equivalence",
        "PASS" if (y_err < 1e-12 and cz_err < 1e-12) else "FAIL",
        {"y_max_abs_err": y_err, "cz_max_abs_err": cz_err},
    )

    # ------------------------------------------------------------------
    # Group C: Backend Capabilities, CUDA-Q Gaps & PyTorch RTX 5080 Check
    # ------------------------------------------------------------------
    np_caps = NumpySimulator().capabilities
    np_caps_ok = (
        np_caps.supports_batching is True
        and np_caps.supports_shots is False
        and np_caps.max_qubits is None
        and set(np_caps.supported_gates) == ALL_GATES
    )
    record(
        "C_capabilities_and_cuda",
        "numpy_simulator_capabilities",
        "PASS" if np_caps_ok else "FAIL",
        {
            "supports_batching": np_caps.supports_batching,
            "supports_shots": np_caps.supports_shots,
            "max_qubits": np_caps.max_qubits,
            "n_supported_gates": len(np_caps.supported_gates),
        },
    )

    cq_caps = CudaQuantumBackend().capabilities
    missing_from_cq = sorted(ALL_GATES - set(cq_caps.supported_gates))
    expected_cq_missing = [
        "AmplitudeEncoding",
        "CPHASE",
        "ControlledQubitUnitary",
        "MCX",
        "MCZ",
        "QubitUnitary",
    ]
    cq_has_sv = hasattr(CudaQuantumCompiled, "statevector")
    cq_has_zsp = hasattr(CudaQuantumCompiled, "zero_state_probability")
    cq_gap_ok = (
        missing_from_cq == expected_cq_missing
        and cq_caps.supports_shots is True
        and not cq_has_sv
        and cq_has_zsp
    )
    record(
        "C_capabilities_and_cuda",
        "cuda_quantum_backend_gaps",
        "CONFIRMED_GAP" if cq_gap_ok else "FAIL",
        {
            "supports_shots": cq_caps.supports_shots,
            "supports_batching": cq_caps.supports_batching,
            "missing_gates_vs_all": missing_from_cq,
            "compiled_has_statevector": cq_has_sv,
            "compiled_has_zero_state_probability": cq_has_zsp,
        },
    )

    import torch

    cuda_avail = torch.cuda.is_available()
    dev_name = torch.cuda.get_device_name(0) if cuda_avail else "none"
    dev_cap = list(torch.cuda.get_device_capability(0)) if cuda_avail else []
    arch_list = torch.cuda.get_arch_list() if cuda_avail else []
    has_sm120 = "sm_120" in arch_list
    c128_gpu_ok = False
    if cuda_avail:
        t_a = torch.tensor([1.0 + 2.0j, -0.5j], dtype=torch.complex128, device="cuda:0")
        t_b = torch.tensor([0.5 - 1.0j, 1.0 + 0.0j], dtype=torch.complex128, device="cuda:0")
        c128_gpu_ok = bool(torch.isfinite(torch.vdot(t_a, t_b)).item())
        torch.cuda.synchronize()
    record(
        "C_capabilities_and_cuda",
        "host_torch_cuda_sm120",
        "PASS" if (cuda_avail and "+cu130" in torch.__version__ and has_sm120 and c128_gpu_ok) else "FAIL",
        {
            "torch_version": torch.__version__,
            "cuda_available": cuda_avail,
            "device_name": dev_name,
            "device_capability": dev_cap,
            "has_sm_120": has_sm120,
            "complex128_gpu_ok": c128_gpu_ok,
        },
    )

    # ------------------------------------------------------------------
    # Group D: Statevector, MSB Ordering, initial_state & 5-Qubit Unitary QECC
    # ------------------------------------------------------------------
    sv_msb = eigenkit.Circuit(CircuitBuilder(2).h(0).x(1)).statevector(np.empty(0))
    expected_msb = np.array([0.0, 1.0 / math.sqrt(2.0), 0.0, 1.0 / math.sqrt(2.0)], dtype=np.complex128)
    msb_err = float(np.max(np.abs(sv_msb - expected_msb)))
    record(
        "D_statevector_and_qecc_unitary",
        "qubit0_msb_ordering",
        "PASS" if msb_err < 1e-12 else "FAIL",
        {"max_abs_err": msb_err, "sv_real_imag": [[float(z.real), float(z.imag)] for z in sv_msb]},
    )

    # 5-qubit unitary 3-qubit bit-flip encode + X(1) + syndrome-entangle
    psi_logical = np.array(
        [math.sqrt(0.3), np.exp(1j * math.pi / 4.0) * math.sqrt(0.7)],
        dtype=np.complex128,
    )
    zero_4 = np.zeros(16, dtype=np.complex128)
    zero_4[0] = 1.0
    init_5 = np.kron(psi_logical, zero_4)

    b_qecc = (
        CircuitBuilder(5)
        .cnot(0, 1)
        .cnot(0, 2)
        .x(1)
        .cnot(0, 3)
        .cnot(1, 3)
        .cnot(1, 4)
        .cnot(2, 4)
    )
    sv_5 = eigenkit.Circuit(b_qecc).statevector(np.empty(0), initial_state=init_5)
    norm_5 = float(np.linalg.norm(sv_5))
    # Ancilla bits are q3, q4 (lowest 2 bits in big-endian 5-qubit index)
    idx_all = np.arange(32)
    prob_syn11 = float(np.sum(np.abs(sv_5[(idx_all & 0b00011) == 0b00011]) ** 2))
    prob_other = float(np.sum(np.abs(sv_5[(idx_all & 0b00011) != 0b00011]) ** 2))
    qecc_5_ok = (
        sv_5.shape == (32,)
        and abs(norm_5 - 1.0) < 1e-12
        and abs(prob_syn11 - 1.0) < 1e-12
        and prob_other < 1e-15
    )
    record(
        "D_statevector_and_qecc_unitary",
        "bit_flip_5qubit_unitary_syndrome",
        "PASS" if qecc_5_ok else "FAIL",
        {
            "shape": list(sv_5.shape),
            "norm": norm_5,
            "prob_syndrome_11_for_X1": prob_syn11,
            "prob_other_syndromes": prob_other,
        },
    )

    # Batched (B, 2**n) statevector check
    b_batch = CircuitBuilder(2)
    x_in = b_batch.add_inputs(1)
    b_batch.ry(x_in[0], 0).cnot(0, 1)
    circ_batch = eigenkit.Circuit(b_batch)
    angles = np.array([[0.0], [0.4], [1.1], [2.3]], dtype=np.float64)
    init_batch = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128), (4, 1))
    sv_batch = circ_batch.statevector(np.empty(0), inputs=angles, initial_state=init_batch)
    max_batch_err = 0.0
    for i in range(4):
        sv_single = circ_batch.statevector(np.empty(0), inputs=angles[i], initial_state=init_batch[i])
        max_batch_err = max(max_batch_err, float(np.max(np.abs(sv_batch[i] - sv_single))))
    record(
        "D_statevector_and_qecc_unitary",
        "batched_statevector",
        "PASS" if (sv_batch.shape == (4, 4) and max_batch_err < 1e-12) else "FAIL",
        {"shape": list(sv_batch.shape), "max_abs_err_vs_single": max_batch_err},
    )

    # ------------------------------------------------------------------
    # Group E: Error-Handling & Confirmed Primitive Gap Tests
    # ------------------------------------------------------------------
    circ_np = eigenkit.Circuit(CircuitBuilder(2).h(0).cnot(0, 1), provider="numpy")
    try:
        circ_np.sample(np.empty(0))
        record("E_error_handling_and_gaps", "numpy_sample_not_implemented", "FAIL", "Did not raise")
    except NotImplementedError as exc:
        record("E_error_handling_and_gaps", "numpy_sample_not_implemented", "CONFIRMED_GAP", str(exc))

    try:
        eigenkit.Circuit(CircuitBuilder(2).h(0), provider=CudaQuantumBackend())
        record("E_error_handling_and_gaps", "cudaq_missing_import_error", "FAIL", "Did not raise")
    except ImportError as exc:
        record("E_error_handling_and_gaps", "cudaq_missing_import_error", "CONFIRMED_GAP", str(exc))

    try:
        circ_np.statevector(np.empty(0), initial_state=np.array([1.0, 1.0, 0.0, 0.0]))
        record("E_error_handling_and_gaps", "unnormalized_initial_state_rejected", "FAIL", "Did not raise")
    except ValueError as exc:
        record("E_error_handling_and_gaps", "unnormalized_initial_state_rejected", "PASS", str(exc))

    try:
        _ = getattr(eigenkit, "HQNC")
        record("E_error_handling_and_gaps", "lazy_hqnc_optional_extra", "PASS", "HQNC imported (optional deps present)")
    except ModuleNotFoundError as exc:
        record("E_error_handling_and_gaps", "lazy_hqnc_optional_extra", "PASS", str(exc))

    import eigenkit.core as ek_core
    import eigenkit.core.backends.numpy_backend as ek_np_mod

    absent_symbols = ["partial_trace", "density_matrix", "kraus", "c_if", "mid_circuit"]
    found_symbols = [
        s for s in absent_symbols if hasattr(ek_core, s) or hasattr(ek_np_mod, s) or hasattr(eigenkit, s)
    ]
    record(
        "E_error_handling_and_gaps",
        "qecc_runtime_primitives_absent",
        "CONFIRMED_GAP" if not found_symbols else "FAIL",
        {"checked_absent": absent_symbols, "unexpected_found": found_symbols},
    )

    results["all_passed"] = len(failures) == 0
    results["failures"] = failures
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe eigenkit v1.3.0 capabilities and QECC gaps.")
    parser.add_argument("--json", dest="json_path", type=Path, default=None, help="Optional path to write JSON output.")
    parser.add_argument("--quiet", action="store_true", help="Print only compact JSON output.")
    args = parser.parse_args()

    report = run_probe()
    json_text = json.dumps(report, indent=2, sort_keys=False)

    if args.json_path is not None:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(json_text + "\n", encoding="utf-8")

    if args.quiet:
        print(json_text)
    else:
        print("=== eigenkit v1.3.0 Live Probe Summary ===")
        for group_name, checks in report["groups"].items():
            print(f"\n[{group_name}]")
            for check_name, info in checks.items():
                print(f"  - {check_name}: {info['status']}")
        print("\n=== Full JSON Report ===")
        print(json_text)

    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
