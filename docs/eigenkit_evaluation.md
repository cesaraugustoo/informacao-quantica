# Technical Evaluation of `eigenkit` v1.3.0 for Dynamic QECC Simulation & RTX 5080 CUDA Acceleration

## 1. Executive Summary & R4 Verdict

**Verdict:** `eigenkit` v1.3.0 is a **pure unitary variational quantum circuit (VQC) and observable expectation-value library**. It provides clean, reproducible big-endian (qubit 0 MSB) unitary statevector simulation (`Circuit(..., provider="numpy").statevector(..., initial_state=...)`) and a clean provider abstraction, making it valuable for **unitary sub-circuit cross-verification** via `qecc.backends.EigenkitUnitaryAdapter` (`"eigenkit_unitary"`).

However, **`eigenkit` v1.3.0 alone lacks essential primitives required by R2 and R3**:
1. No mid-circuit projective measurement or post-measurement state collapse.
2. No classical bit registers or syndrome-conditioned `c_if` feed-forward recovery operations.
3. No stochastic Pauli or Kraus noise channels (and `NumpySimulator` does not implement `.sample()`).
4. No reduced density matrices, partial trace (`partial_trace`), or Uhlmann-Jozsa fidelity utilities.
5. `eigenkit`'s GPU backend (`CudaQuantumBackend`) lacks `.statevector()`, omits `MCX`/`MCZ`/`CPHASE`/`QubitUnitary`/`ControlledQubitUnitary`/`AmplitudeEncoding`, and relies on `cuda-quantum` rather than native PyTorch CUDA kernels; additionally, `eigenkit`'s repository `uv` config pins `pytorch-cu126`, which lacks native `sm_120` (Blackwell) kernels for the local **NVIDIA GeForce RTX 5080**.

Per Requirement **R4**, this documented technical check establishes the exact primitive gaps of `eigenkit` v1.3.0 and motivates our hybrid swappable `QECCBackend` architecture:
- **`"numpy"` (`NumPyCPUBackend`)**: Vectorized `(B, 2^n)` `complex128` CPU engine supporting full dynamic QECC cycles.
- **`"torch_cuda"` (`TorchCUDABackend`)**: Native `torch==2.14.1+cu130` (`sm_120`) batched GPU engine on RTX 5080 matching `NumPyCPUBackend` semantics.
- **`"eigenkit_unitary"` (`EigenkitUnitaryAdapter`)**: Wraps `eigenkit.Circuit` (`provider="numpy"`) for unitary sub-circuit execution (`p=0` and deterministic single/multi-qubit error verification) while explicitly raising `NotImplementedError` on stochastic Monte Carlo QECC cycles.

---

## 2. Environment & RTX 5080 Compatibility Findings

- **Host Hardware & Toolchain**:
  - GPU: **NVIDIA GeForce RTX 5080** (`compute capability 12.0` / `sm_120`, 16,275 MiB PyTorch CUDA / 16,303 MiB driver VRAM, driver `617.14`, CUDA `13.2` toolkit).
  - Python & Packaging: `Python 3.12.3`, `uv 0.11.3`, `torch==2.14.1+cu130`, `eigenkit==1.3.0`, `numpy==2.5.5`, `scipy==1.18.1`, `matplotlib==3.11.2`, `pydantic==2.14.3`, `pytest==9.2.0`.
- **Dependency Isolation (`pytorch-cu126` vs `cu130` `sm_120`)**:
  - `eigenkit==1.3.0` core PyPI metadata requires only `numpy`, `scipy`, and `pydantic`; its `pytorch-cu126` pin lives only in its repository-local `[tool.uv.sources]` for optional extras.
  - Because PyPI unpinned `torch` resolves to `2.15.0` (CUDA 13.2) while the `cu130` wheel index serves `torch==2.14.1+cu130`, `pyproject.toml` declares an `explicit = true` `[[tool.uv.index]]` (`pytorch-cu130`) bound via `[tool.uv.sources]` for `torch`.
  - Live runtime verification confirms `torch.__version__ == "2.14.1+cu130"`, `torch.cuda.is_available() == True`, `torch.cuda.get_device_capability(0) == (12, 0)`, `"sm_120" in torch.cuda.get_arch_list()`, and `complex128` tensor execution succeeds on `cuda:0`.
  - Additionally, `eigenkit`'s `cudaquantum_backend.py` warns of LLVM/MLIR symbol collisions between `cudaq >= 0.15` and `triton` (`torch`'s Linux compiler), further favoring native PyTorch `cu130` execution on RTX 5080.

---

## 3. Live Probe Results (`scripts/probe_eigenkit.py`)

Executed live inside `/home/cesar/teamwork_projects/qecc_simulation`:
```bash
.venv/bin/python scripts/probe_eigenkit.py
```

### 3.1 Summary of Live Probe Output

```text
=== eigenkit v1.3.0 Live Probe Summary ===

[A_import_isolation]
  - eigenkit_version: PASS
  - no_torch_imported_eagerly: PASS
  - registered_names: PASS

[B_ir_and_builder]
  - ir_gate_vocabulary: PASS
  - circuit_spec_fields: PASS
  - builder_methods_present: PASS
  - builder_qecc_gaps_absent: CONFIRMED_GAP
  - y_and_cz_emulation_equivalence: PASS

[C_capabilities_and_cuda]
  - numpy_simulator_capabilities: PASS
  - cuda_quantum_backend_gaps: CONFIRMED_GAP
  - host_torch_cuda_sm120: PASS

[D_statevector_and_qecc_unitary]
  - qubit0_msb_ordering: PASS
  - bit_flip_5qubit_unitary_syndrome: PASS
  - batched_statevector: PASS

[E_error_handling_and_gaps]
  - numpy_sample_not_implemented: CONFIRMED_GAP
  - cudaq_missing_import_error: CONFIRMED_GAP
  - unnormalized_initial_state_rejected: PASS
  - lazy_hqnc_optional_extra: PASS
  - qecc_runtime_primitives_absent: CONFIRMED_GAP
```

### 3.2 Measured Quantitative & Structural Details

1. **Package & Import Isolation (`Group A`)**:
   - `eigenkit.__version__`: `"1.3.0"`
   - `"torch" in sys.modules` immediately after `import eigenkit`: `false`
   - `eigenkit.core.provider.registered_names()`: `["numpy"]`
2. **IR & `CircuitBuilder` Vocabulary (`Group B`)**:
   - `ALL_GATES` (18 gates): `AmplitudeEncoding`, `CNOT`, `CPHASE`, `CRY`, `ControlledQubitUnitary`, `H`, `MCX`, `MCZ`, `PhaseShift`, `QubitUnitary`, `RX`, `RY`, `RZ`, `S`, `SWAP`, `Sdg`, `X`, `Z`.
   - `CircuitSpec` fields: `["n_qubits", "n_weights", "n_inputs", "gates"]` (no classical register, measurement, reset, or noise fields).
   - `CircuitBuilder` confirmed absent methods: `["y", "cz", "cy", "measure", "reset", "c_if", "barrier", "kraus", "noise"]`.
   - Emulation accuracy verified via live `Circuit.statevector(...)`:
     - `Y` emulated via `builder.sdg(q).x(q).s(q)` ($S X S^\dagger = Y$, zero global phase): `y_max_abs_err = 0.0`.
     - `CZ` emulated via `builder.h(t).cnot(c, t).h(t)` vs `builder.mcz(c, t)`: `cz_max_abs_err = 1.1102230246251565e-16`.
3. **Backend Capabilities & Host CUDA (`Group C`)**:
   - `NumpySimulator().capabilities`: `supports_batching=True`, `supports_shots=False`, `max_qubits=None`, `n_supported_gates=18`.
   - `CudaQuantumBackend().capabilities`: `supports_shots=True`, `supports_batching=True`, `missing_gates_vs_all=["AmplitudeEncoding", "CPHASE", "ControlledQubitUnitary", "MCX", "MCZ", "QubitUnitary"]`.
   - `CudaQuantumCompiled`: `hasattr(..., "statevector") == False`, `hasattr(..., "zero_state_probability") == True`.
   - Host PyTorch CUDA: `torch_version="2.14.1+cu130"`, `cuda_available=True`, `device_name="NVIDIA GeForce RTX 5080"`, `device_capability=[12, 0]`, `has_sm_120=True`, `complex128_gpu_ok=True`.
4. **Statevector Semantics & Unitary QECC Cross-Check (`Group D`)**:
   - Qubit 0 MSB test (`H(0) -> X(1)`): `max_abs_err = 0.0` against `[0, 1/sqrt(2), 0, 1/sqrt(2)]`.
   - 5-qubit unitary 3-qubit bit-flip encode + `X(1)` + ancilla syndrome entanglement (`CNOT(0->3), CNOT(1->3), CNOT(1->4), CNOT(2->4)`): `shape=[32]`, `norm=1.0`, `prob_syndrome_11_for_X1=1.0`, `prob_other_syndromes=0.0`.
   - Batched `statevector` (`B=4`, 2 qubits): `shape=[4, 4]`, `max_abs_err_vs_single=0.0`.
5. **Confirmed Exceptions & Absent Runtime Primitives (`Group E`)**:
   - `Circuit(..., provider="numpy").sample(...)` -> `NotImplementedError`: `"backend 'numpy' does not implement sample(). It returns exact values rather than shots; use probabilities()."`
   - `Circuit(..., provider=CudaQuantumBackend())` -> `ImportError`: `"cuda-quantum is not installed. Install it via \`pip install eigenkit[cudaquantum]\`."`
   - Unnormalized `initial_state` -> `ValueError`: `"initial_state must be L2-normalized (got squared norm 2)"`
   - Absent runtime symbols across `eigenkit`, `eigenkit.core`, `eigenkit.core.backends.numpy_backend`: `partial_trace`, `density_matrix`, `kraus`, `c_if`, `mid_circuit`.

---

## 4. QECC Primitive Gap Matrix (R2 & R3 vs `eigenkit` v1.3.0)

| Requirement | QECC Primitive Needed | `eigenkit` v1.3.0 Status | Impact / Mitigation in `qecc` |
|---|---|---|---|
| R2: Syndrome Extraction | Mid-circuit projective measurement & ancilla collapse | **Missing** (`No measure/reset`, `supports_shots=False` on `numpy`) | Implemented vectorized `(B, 2^n_data, 2^n_ancilla)` projective collapse in `NumPyCPUBackend` and `TorchCUDABackend`. |
| R2: Recovery | Classical registers & syndrome-conditioned `c_if` feed-forward | **Missing** (`CircuitSpec` has only unitary `gates`) | Implemented syndrome-grouped conditional `recovery_ops(syndrome)` execution in `NumPyCPUBackend` and `TorchCUDABackend`. |
| R2: Error Channel | Stochastic Pauli / Kraus noise injection per shot | **Missing** (`No kraus/noise`, `sample()` raises `NotImplementedError`) | Implemented PRNG `StochasticPauliChannel` (`uint8` codes) with branchless `where` injection on CPU & CUDA. |
| R2: Gate Set | Direct `Y` and `CZ` gates | **Absent in `CircuitBuilder`/`GateName`** | Supported directly in `NumPyCPUBackend` & `TorchCUDABackend`; translated via `Sdg-X-S` and `MCZ`/`H-CNOT-H` in `EigenkitUnitaryAdapter`. |
| R3: Readout & Fidelity | Partial trace (`partial_trace`), density matrix, squared Uhlmann fidelity | **Missing** (no `partial_trace` or `density_matrix` APIs) | Implemented big-endian `partial_trace_to_qubit0`, `quantum_fidelity`, and `SingleQubitTomography` in `qecc.metrics`. |
| R4: GPU Acceleration | RTX 5080 (`sm_120`) batched CUDA simulation | **Incompatible/Incomplete** (`CudaQuantumBackend` lacks `statevector()` & `MCZ`; `cu126` lacks `sm_120`) | Implemented `TorchCUDABackend` on `torch==2.14.1+cu130` with chunked `complex128` kernels and warm GPU timing. |

---

## 5. Resulting Hybrid Architecture (`src/qecc/backends/`)

All simulation backends conform to `qecc.backends.QECCBackend` and share `validate_and_prepare_cycle_inputs(...)`:
- **`"numpy"` (`NumPyCPUBackend`)**: Vectorized batched `complex128` CPU engine supporting all 12 QECC gates (`{I, X, Y, Z, H, S, Sdg, RX, RY, RZ, CNOT, CZ}`), stochastic and deterministic Pauli injection, projective ancilla collapse, classical recovery, and partial trace.
- **`"torch_cuda"` (`TorchCUDABackend`)**: Native RTX 5080 `cu130` (`sm_120`) GPU engine with automatic memory-safe chunking (`max_batch_bytes`), branchless `torch.where` Pauli injection, GPU-resident `torch.bmm` partial trace, and `warmup=True` to exclude one-time CUDA JIT overhead.
- **`"eigenkit_unitary"` (`EigenkitUnitaryAdapter`)**: Translates `GateOp` sequences to `eigenkit.CircuitBuilder` (`Y -> Sdg-X-S`, `CZ -> MCZ` / `H-CNOT-H`) and executes pure unitary sub-circuits via `eigenkit.Circuit(..., provider="numpy").statevector(...)` for noiseless (`p=0`) and deterministic error cross-verification; rejects stochastic `p > 0` channels with `NotImplementedError`.

### Reproduction
```bash
cd /home/cesar/teamwork_projects/qecc_simulation
.venv/bin/python scripts/probe_eigenkit.py
```
