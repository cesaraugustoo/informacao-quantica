# Quantum Error Correction Codes (QECC) Seminar & Simulation Suite

A graduate-level academic seminar and Python/CUDA simulation suite investigating quantum error correction codes (`bit_flip_3`, `phase_flip_3`, `[[5,1,3]]` `five_qubit`), off-diagonal coherence protection, passive two-qubit decoherence-free subspaces (`dfs_2`), non-destructive ancilla syndrome extraction, analytical gain thresholds, and RTX 5080 (`sm_120`) GPU-accelerated Monte Carlo sweeps.

---

## Key Results & Highlights

- **Implemented QECC & DFS Codes (`src/qecc/codes/`):**
  - `bit_flip_3` (`[[3,1,1]]`): stabilizers `ZZI`, `IZZ`; exact `brentq` gain threshold $p^* = 0.5$ under `bit_flip`.
  - `phase_flip_3` (`[[3,1,1]]`): Hadamard dual stabilizers `XXI`, `IXX`; exact threshold $p^* = 0.5$ under `phase_flip`.
  - `five_qubit` (`[[5,1,3]]` Laflamme perfect code): stabilizers `XZZXI`, `IXZZX`, `XIXZZ`, `ZXIXZ`, 16 bijective syndromes, 1024-term weight enumerator (`[1, 15, 0, 60, 135, 45]` in logical-`I` coset), exact `brentq` thresholds $p^* = 0.137628$ (`depolarizing`) and $p^* = 0.131123$ (`bit_flip` / `phase_flip`).
  - `dfs_2` (`span{|01>, |10>}`): passive decoherence-free subspace invariant under collective dephasing ($F = 1.0$, $p^* = 1.0$) and coherent $R_z(\phi)\otimes R_z(\phi)$.
- **Swappable Backends & RTX 5080 CUDA Acceleration (`src/qecc/backends/`):**
  - `numpy` (`NumPyCPUBackend`), `torch_cuda` (`TorchCUDABackend` on `torch==2.14.1+cu130`, NVIDIA GeForce RTX 5080 `sm_120`), and `eigenkit_unitary` (`EigenkitUnitaryAdapter` wrapping `eigenkit==1.3.0`).
  - Achieves **`24.40x` CUDA speedup** on 9-qubit `five_qubit` at `100,000` shots (`0.413090 s` GPU vs. `10.080841 s` CPU) with `9/9` CPU/GPU statistical parity and `6/6` `eigenkit_unitary` unitary parity (`Max |Δρ| <= 1.06e-15` overall; `<= 1.52e-17` `eigenkit_unitary` vs. `numpy`).
  - See [docs/eigenkit_evaluation.md](docs/eigenkit_evaluation.md) for the live `eigenkit` primitive gap analysis (`measure`, `c_if`, `kraus`, `partial_trace`, and `CudaQuantumBackend` `sm_120` gaps).

---

## Project Layout

```text
qecc_simulation/
├── pyproject.toml                       # Python 3.12+, uv + pytorch-cu130 index
├── uv.lock
├── README.md
├── docs/
│   ├── seminar_qecc.md                  # Full graduate seminar manuscript (Topics 1–6)
│   ├── seminar_slides.md                # 20-slide presentation deck (title slide + 19 numbered slides)
│   ├── review_rubric.md                 # 18-check independent review rubric (R-1.1..R-7.1)
│   └── eigenkit_evaluation.md           # Technical probe of eigenkit v1.3.0 & RTX 5080
├── src/qecc/
│   ├── cli.py                           # Unified CLI (info, simulate, verify, sweep, benchmark)
│   ├── backends/                        # numpy, torch_cuda, eigenkit_unitary
│   ├── channels/                        # StochasticPauliChannel, coherent_collective_dephasing_ops
│   ├── codes/                           # bit_flip_3, phase_flip_3, five_qubit, dfs_2
│   ├── metrics/                         # quantum_fidelity, partial_trace_to_qubit0, tomography
│   └── analysis/                        # Knill-Laflamme verifier, weight enumerators, brentq thresholds
├── scripts/
│   ├── probe_eigenkit.py                # Live probe of eigenkit v1.3.0 & RTX 5080 sm_120
│   ├── verify_analytical_thresholds.py  # Verifies KL conditions, 1024-term enumerator & thresholds
│   ├── run_sweeps.py                    # 18,000-shot Monte Carlo sweeps + figures
│   └── run_benchmarks.py                # CPU vs. RTX 5080 CUDA & eigenkit parity benchmarks
├── results/
│   ├── data/                            # fidelity_sweeps.csv, fidelity_sweeps.json
│   ├── figures/                         # PNG/PDF plots embedded in seminar docs
│   └── benchmarks/                      # backend_benchmark.json, backend_benchmark.md
└── tests/
    ├── unit/                            # Unit tests for backends, channels, codes, fidelity, analysis
    └── e2e/                             # Tier 1–5 end-to-end verification suite
```

---

## Environment Setup (`uv` / `.venv`)

The project requires Python `>=3.12,<3.13` and uses `uv` with an explicit `pytorch-cu130` index (`pyproject.toml`) for native RTX 5080 (`sm_120`) CUDA 13.0 support:

```bash
# Synchronize dependencies into .venv via uv
uv sync

# Verify Python, PyTorch CUDA 13.0, and RTX 5080 sm_120 visibility
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
```

---

## Reproducing Analysis, Sweeps, Benchmarks & Tests

### 1. Run Automated Unit & E2E Tests (`pytest`)
```bash
# Full unit + E2E test suite
PYTHONDONTWRITEBYTECODE=1 .venv/bin/pytest -p no:cacheprovider tests/unit tests/e2e -v

# Individual tracks
PYTHONDONTWRITEBYTECODE=1 .venv/bin/pytest tests/unit -v
PYTHONDONTWRITEBYTECODE=1 .venv/bin/pytest tests/e2e -v
```

### 2. Run Verification & Simulation Scripts
```bash
# Analytical Knill-Laflamme, 1024-term enumerator, and brentq threshold checks
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/verify_analytical_thresholds.py

# Live eigenkit v1.3.0 & RTX 5080 capability probe
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/probe_eigenkit.py

# Regenerate 18,000-shot sweeps (results/data/) and PNG/PDF figures (results/figures/)
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/run_sweeps.py

# Regenerate CPU vs. RTX 5080 CUDA & eigenkit parity benchmarks (results/benchmarks/)
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/run_benchmarks.py
```

### 3. Unified CLI (`python -m qecc.cli`)
```bash
.venv/bin/python -m qecc.cli info
.venv/bin/python -m qecc.cli simulate --code five_qubit --channel depolarizing --p 0.05 --shots 4000 --backend torch_cuda --state +i
.venv/bin/python -m qecc.cli verify
.venv/bin/python -m qecc.cli sweep --backend torch_cuda --shots 18000 --num-p 21
# Quick subset benchmark (omit --quick to delegate to scripts/run_benchmarks.py with full defaults)
.venv/bin/python -m qecc.cli benchmark --quick --shots 10000 --codes bit_flip_3 five_qubit
```

---

## Seminar Deliverables & Documentation (`docs/`)

- **[docs/seminar_qecc.md](docs/seminar_qecc.md)** — Full graduate-level seminar manuscript covering no-cloning, continuous $SU(2)$ noise, projective collapse, Pauli discretization, Knill–Laflamme conditions, 3-qubit codes, `[[5,1,3]]` stabilizers/16 syndromes/1024-term enumerator, coherence $\rho_{01}$ protection, passive `dfs_2`, non-destructive ancilla syndrome extraction, recent QEC developments (Willow, Harvard 48-logical, IBM 288-qubit qLDPC, cat qubits), analytical gain thresholds, and RTX 5080 results.
- **[docs/seminar_slides.md](docs/seminar_slides.md)** — 20-slide presentation deck (title slide + 19 numbered slides) synchronized with all formulas, figures, and benchmark tables.
- **[docs/review_rubric.md](docs/review_rubric.md)** — Independent 18-check review rubric (`R-1.1` through `R-7.1`) with verified evidence.
- **[docs/eigenkit_evaluation.md](docs/eigenkit_evaluation.md)** — Technical evaluation of `eigenkit` v1.3.0 and the hybrid 3-backend design.

---

## Generated Figures & Empirical Artifacts (`results/`)

| Artifact | Path | Description |
|---|---|---|
| Fidelity vs. Error Rate | [results/figures/fidelity_vs_error_rate.png](results/figures/fidelity_vs_error_rate.png) | 18,000-shot MC vs. analytical curves for 3-qubit and `[[5,1,3]]` codes |
| Gain Threshold Comparison | [results/figures/gain_threshold_comparison.png](results/figures/gain_threshold_comparison.png) | Net entanglement gain $\Delta F_{\text{ent}}$ and exact `brentq` $p^*$ bar chart |
| Coherence & DFS Protection | [results/figures/coherence_and_dfs_protection.png](results/figures/coherence_and_dfs_protection.png) | Passive `dfs_2` vs. active `phase_flip_3` and $\vert\rho_{01}\vert$ coherence at $p = 0.05$ |
| Sweep Dataset | [results/data/fidelity_sweeps.json](results/data/fidelity_sweeps.json) / [results/data/fidelity_sweeps.csv](results/data/fidelity_sweeps.csv) | 231 records across 21 $p$-points (`torch_cuda`, 18,000 shots) |
| Benchmark Report | [results/benchmarks/backend_benchmark.md](results/benchmarks/backend_benchmark.md) / [results/benchmarks/backend_benchmark.json](results/benchmarks/backend_benchmark.json) | `numpy` vs. `torch_cuda` (RTX 5080) & `eigenkit_unitary` parity results |
