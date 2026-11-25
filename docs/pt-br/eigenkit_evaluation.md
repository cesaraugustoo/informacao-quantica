# Avaliação Técnica do `eigenkit` v1.3.0 para Simulação Dinâmica de QECC e Aceleração CUDA na RTX 5080

## 1. Resumo Executivo e Veredito do Requisito R4

**Veredito:** O `eigenkit` v1.3.0 é uma **biblioteca puramente unitária para circuitos quânticos variacionais (VQC) e cálculo de valores esperados de observáveis**. Ele fornece uma simulação limpa e reprodutível de vetor de estado unitário em convenção *big-endian* (qubit 0 como MSB) (`Circuit(..., provider="numpy").statevector(..., initial_state=...)`) e uma abstração bem estruturada de provedores (*providers*), tornando-o valioso para **verificação cruzada de subcircuitos unitários** por meio de `qecc.backends.EigenkitUnitaryAdapter` (`"eigenkit_unitary"`).

Contudo, **o `eigenkit` v1.3.0 isoladamente carece de primitivas essenciais exigidas por R2 e R3**:
1. Ausência de medição projetiva no meio do circuito (*mid-circuit*) ou colapso de estado pós-medição.
2. Ausência de registradores de bits clássicos ou operações de recuperação por realimentação clássica (*feed-forward*) condicionadas à síndrome (`c_if`).
3. Ausência de canais estocásticos de Pauli ou de Kraus (e o `NumpySimulator` não implementa `.sample()`).
4. Ausência de matrizes densidade reduzidas, traço parcial (`partial_trace`) ou utilitários de fidelidade de Uhlmann–Jozsa.
5. O backend de GPU do `eigenkit` (`CudaQuantumBackend`) não possui `.statevector()`, omite `MCX`/`MCZ`/`CPHASE`/`QubitUnitary`/`ControlledQubitUnitary`/`AmplitudeEncoding` e depende de `cuda-quantum` em vez de kernels CUDA nativos do PyTorch; além disso, a configuração `uv` do repositório do `eigenkit` fixa `pytorch-cu126`, que carece de kernels `sm_120` (Blackwell) nativos para a **NVIDIA GeForce RTX 5080** local.

Conforme o Requisito **R4**, esta verificação técnica documentada estabelece exatamente as lacunas de primitivas do `eigenkit` v1.3.0 e motiva nossa arquitetura híbrida intercambiável `QECCBackend`:
- **`"numpy"` (`NumPyCPUBackend`)**: Motor de CPU vetorizado `(B, 2^n)` em `complex128` com suporte a ciclos dinâmicos completos de QECC.
- **`"torch_cuda"` (`TorchCUDABackend`)**: Motor de GPU em lote (*batched*) nativo em `torch==2.14.1+cu130` (`sm_120`) na RTX 5080, com semântica equivalente à do `NumPyCPUBackend`.
- **`"eigenkit_unitary"` (`EigenkitUnitaryAdapter`)**: Encapsula `eigenkit.Circuit` (`provider="numpy"`) para execução de subcircuitos unitários (`p=0` e verificação determinística de erros de um ou múltiplos qubits), lançando explicitamente `NotImplementedError` em ciclos estocásticos de Monte Carlo de QECC.

---

## 2. Achados de Ambiente e Compatibilidade com a RTX 5080

- **Hardware e Cadeia de Ferramentas do Host**:
  - GPU: **NVIDIA GeForce RTX 5080** (`compute capability 12.0` / `sm_120`, 16.275 MiB de VRAM no PyTorch CUDA / 16.303 MiB no driver, driver `617.14`, toolkit CUDA `13.2`).
  - Python e Empacotamento: `Python 3.12.3`, `uv 0.11.3`, `torch==2.14.1+cu130`, `eigenkit==1.3.0`, `numpy==2.5.5`, `scipy==1.18.1`, `matplotlib==3.11.2`, `pydantic==2.14.3`, `pytest==9.2.0`.
- **Isolamento de Dependências (`pytorch-cu126` vs. `cu130` `sm_120`)**:
  - Os metadados principais do `eigenkit==1.3.0` no PyPI requerem apenas `numpy`, `scipy` e `pydantic`; sua fixação em `pytorch-cu126` reside somente no `[tool.uv.sources]` local de seu repositório para extras opcionais.
  - Como o `torch` sem fixação no PyPI resolve para `2.15.0` (CUDA 13.2), enquanto o índice de *wheels* `cu130` fornece `torch==2.14.1+cu130`, o arquivo `pyproject.toml` declara um `[[tool.uv.index]]` com `explicit = true` (`pytorch-cu130`) vinculado via `[tool.uv.sources]` para o `torch`.
  - A verificação em tempo de execução confirma `torch.__version__ == "2.14.1+cu130"`, `torch.cuda.is_available() == True`, `torch.cuda.get_device_capability(0) == (12, 0)`, `"sm_120" in torch.cuda.get_arch_list()` e execução bem-sucedida de tensores `complex128` em `cuda:0`.
  - Adicionalmente, o arquivo `cudaquantum_backend.py` do `eigenkit` alerta sobre colisões de símbolos LLVM/MLIR entre `cudaq >= 0.15` e `triton` (o compilador Linux do `torch`), reforçando a preferência pela execução nativa com PyTorch `cu130` na RTX 5080.

---

## 3. Resultados da Sonda ao Vivo (`scripts/probe_eigenkit.py`)

Executado diretamente em `/home/cesar/teamwork_projects/qecc_simulation`:
```bash
.venv/bin/python scripts/probe_eigenkit.py
```

### 3.1 Resumo da Saída da Sonda ao Vivo

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

### 3.2 Detalhes Quantitativos e Estruturais Medidos

1. **Pacote e Isolamento de Importação (Grupo A)**:
   - `eigenkit.__version__`: `"1.3.0"`
   - `"torch" in sys.modules` imediatamente após `import eigenkit`: `false`
   - `eigenkit.core.provider.registered_names()`: `["numpy"]`
2. **IR e Vocabulário de `CircuitBuilder` (Grupo B)**:
   - `ALL_GATES` (18 portas): `AmplitudeEncoding`, `CNOT`, `CPHASE`, `CRY`, `ControlledQubitUnitary`, `H`, `MCX`, `MCZ`, `PhaseShift`, `QubitUnitary`, `RX`, `RY`, `RZ`, `S`, `SWAP`, `Sdg`, `X`, `Z`.
   - Campos de `CircuitSpec`: `["n_qubits", "n_weights", "n_inputs", "gates"]` (sem campos para registrador clássico, medição, *reset* ou ruído).
   - Métodos confirmadamente ausentes em `CircuitBuilder`: `["y", "cz", "cy", "measure", "reset", "c_if", "barrier", "kraus", "noise"]`.
   - Precisão de emulação verificada ao vivo via `Circuit.statevector(...)`:
     - `Y` emulada via `builder.sdg(q).x(q).s(q)` ($S X S^\dagger = Y$, fase global nula): `y_max_abs_err = 0.0`.
     - `CZ` emulada via `builder.h(t).cnot(c, t).h(t)` versus `builder.mcz(c, t)`: `cz_max_abs_err = 1.1102230246251565e-16`.
3. **Capacidades dos Backends e CUDA do Host (Grupo C)**:
   - `NumpySimulator().capabilities`: `supports_batching=True`, `supports_shots=False`, `max_qubits=None`, `n_supported_gates=18`.
   - `CudaQuantumBackend().capabilities`: `supports_shots=True`, `supports_batching=True`, `missing_gates_vs_all=["AmplitudeEncoding", "CPHASE", "ControlledQubitUnitary", "MCX", "MCZ", "QubitUnitary"]`.
   - `CudaQuantumCompiled`: `hasattr(..., "statevector") == False`, `hasattr(..., "zero_state_probability") == True`.
   - PyTorch CUDA do host: `torch_version="2.14.1+cu130"`, `cuda_available=True`, `device_name="NVIDIA GeForce RTX 5080"`, `device_capability=[12, 0]`, `has_sm_120=True`, `complex128_gpu_ok=True`.
4. **Semântica do Vetor de Estado e Verificação Cruzada de QECC Unitário (Grupo D)**:
   - Teste de qubit 0 como MSB (`H(0) -> X(1)`): `max_abs_err = 0.0` em relação a `[0, 1/sqrt(2), 0, 1/sqrt(2)]`.
   - Codificação unitária de 5 qubits para inversão de bit (*bit-flip*) de 3 qubits + `X(1)` + emaranhamento de síndrome por ancilla (`CNOT(0->3), CNOT(1->3), CNOT(1->4), CNOT(2->4)`): `shape=[32]`, `norm=1.0`, `prob_syndrome_11_for_X1=1.0`, `prob_other_syndromes=0.0`.
   - `statevector` em lote (`B=4`, 2 qubits): `shape=[4, 4]`, `max_abs_err_vs_single=0.0`.
5. **Exceções Confirmadas e Primitivas de Execução Ausentes (Grupo E)**:
   - `Circuit(..., provider="numpy").sample(...)` -> `NotImplementedError`: `"backend 'numpy' does not implement sample(). It returns exact values rather than shots; use probabilities()."`
   - `Circuit(..., provider=CudaQuantumBackend())` -> `ImportError`: `"cuda-quantum is not installed. Install it via \`pip install eigenkit[cudaquantum]\`."`
   - `initial_state` não normalizado -> `ValueError`: `"initial_state must be L2-normalized (got squared norm 2)"`
   - Símbolos de tempo de execução ausentes em `eigenkit`, `eigenkit.core` e `eigenkit.core.backends.numpy_backend`: `partial_trace`, `density_matrix`, `kraus`, `c_if`, `mid_circuit`.

---

## 4. Matriz de Lacunas de Primitivas de QECC (R2 e R3 vs. `eigenkit` v1.3.0)

| Requisito | Primitiva de QECC Necessária | Status no `eigenkit` v1.3.0 | Impacto / Mitigação em `qecc` |
|---|---|---|---|
| R2: Extração de Síndrome | Medição projetiva no meio do circuito e colapso da ancilla | **Ausente** (sem `measure`/`reset`, `supports_shots=False` em `numpy`) | Implementado colapso projetivo vetorizado `(B, 2^n_data, 2^n_ancilla)` em `NumPyCPUBackend` e `TorchCUDABackend`. |
| R2: Recuperação | Registradores clássicos e realimentação clássica (*feed-forward*) `c_if` condicionada à síndrome | **Ausente** (`CircuitSpec` possui apenas `gates` unitários) | Implementada execução condicional agrupada por síndrome `recovery_ops(syndrome)` em `NumPyCPUBackend` e `TorchCUDABackend`. |
| R2: Canal de Erros | Injeção estocástica de ruído de Pauli / Kraus por *shot* | **Ausente** (sem `kraus`/`noise`, `sample()` lança `NotImplementedError`) | Implementado `StochasticPauliChannel` com PRNG (códigos `uint8`) e injeção sem desvios (*branchless*) via `where` em CPU e CUDA. |
| R2: Conjunto de Portas | Portas diretas `Y` e `CZ` | **Ausentes em `CircuitBuilder`/`GateName`** | Suportadas diretamente em `NumPyCPUBackend` e `TorchCUDABackend`; traduzidas via `Sdg-X-S` e `MCZ`/`H-CNOT-H` em `EigenkitUnitaryAdapter`. |
| R3: Leitura e Fidelidade | Traço parcial (`partial_trace`), matriz densidade e fidelidade quadrática de Uhlmann | **Ausente** (sem APIs `partial_trace` ou `density_matrix`) | Implementados `partial_trace_to_qubit0` (*big-endian*), `quantum_fidelity` e `SingleQubitTomography` em `qecc.metrics`. |
| R4: Aceleração por GPU | Simulação CUDA em lote na RTX 5080 (`sm_120`) | **Incompatível/Incompleto** (`CudaQuantumBackend` não possui `statevector()` nem `MCZ`; `cu126` não possui `sm_120`) | Implementado `TorchCUDABackend` em `torch==2.14.1+cu130` com kernels `complex128` particionados em blocos (*chunked*) e cronometragem com GPU aquecida. |

---

## 5. Arquitetura Híbrida Resultante (`src/qecc/backends/`)

Todos os backends de simulação seguem a interface `qecc.backends.QECCBackend` e compartilham `validate_and_prepare_cycle_inputs(...)`:
- **`"numpy"` (`NumPyCPUBackend`)**: Motor de CPU vetorizado em lote (`complex128`) com suporte a todas as 12 portas de QECC (`{I, X, Y, Z, H, S, Sdg, RX, RY, RZ, CNOT, CZ}`), injeção estocástica e determinística de erros de Pauli, colapso projetivo da ancilla, recuperação clássica e traço parcial.
- **`"torch_cuda"` (`TorchCUDABackend`)**: Motor de GPU nativo `cu130` (`sm_120`) na RTX 5080 com particionamento automático seguro para memória (`max_batch_bytes`), injeção de Pauli sem desvios (*branchless*) com `torch.where`, traço parcial residente em GPU via `torch.bmm` e `warmup=True` para excluir a sobrecarga única de compilação JIT em CUDA.
- **`"eigenkit_unitary"` (`EigenkitUnitaryAdapter`)**: Traduz sequências de `GateOp` para `eigenkit.CircuitBuilder` (`Y -> Sdg-X-S`, `CZ -> MCZ` / `H-CNOT-H`) e executa subcircuitos puramente unitários via `eigenkit.Circuit(..., provider="numpy").statevector(...)` para verificação cruzada sem ruído (`p=0`) e com erros determinísticos; rejeita canais estocásticos com `p > 0` lançando `NotImplementedError`.

### Reprodução
```bash
cd /home/cesar/teamwork_projects/qecc_simulation
.venv/bin/python scripts/probe_eigenkit.py
```
