# CPU vs. RTX 5080 CUDA Backend Benchmark

## Hardware & Software Environment

- **Generated (UTC)**: `2026-11-19T02:10:11Z`
- **GPU**: `NVIDIA GeForce RTX 5080` (Compute Capability `12.0`, `sm_120` supported: `True`, VRAM: `16275.4 MiB`)
- **CPU**: `AMD Ryzen 9 9950X3D 16-Core Processor` (`32` logical cores)
- **Toolchain**: `Python 3.12.3`, `torch==2.14.1+cu130`, `numpy==2.5.5`, `eigenkit==1.3.0`
- **Protocol**: state `|+i>`, error probability `p=0.05`, `warmup_runs=2`, `repeats=3`, `seed=20261118` (explicit `torch.cuda.synchronize()` before and after each GPU measurement)

## CPU (`numpy`) vs. GPU (`torch_cuda`) Throughput & Speedup

| Code | Channel | Shots | CPU Median (s) | GPU Median (s) | CPU (shots/s) | GPU (shots/s) | CUDA Speedup | CPU F | GPU F | Expected F | Analytical State F | Analytical Ent. F | Max |Δρ| | Syndrome p | Parity |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| `bit_flip_3` | `bit_flip` | 1,000 | 0.001245 | 0.001706 | 802,970.3 | 586,257.9 | 0.73x | 0.99267 | 0.99033 | 0.99275 | 0.99517 | 0.99275 | 2.33e-03 | 0.9239 | PASS |
| `bit_flip_3` | `bit_flip` | 10,000 | 0.010041 | 0.003505 | 995,959.8 | 2,853,277.1 | 2.86x | 0.99203 | 0.99293 | 0.99275 | 0.99517 | 0.99275 | 9.00e-04 | 0.1632 | PASS |
| `bit_flip_3` | `bit_flip` | 100,000 | 0.179347 | 0.022497 | 557,579.5 | 4,444,942.3 | 7.97x | 0.99268 | 0.99285 | 0.99275 | 0.99517 | 0.99275 | 1.70e-04 | 0.9254 | PASS |
| `phase_flip_3` | `phase_flip` | 1,000 | 0.002169 | 0.002260 | 461,117.6 | 442,457.1 | 0.96x | 0.99433 | 0.99467 | 0.99275 | 0.99517 | 0.99275 | 3.33e-04 | 0.0072 | PASS |
| `phase_flip_3` | `phase_flip` | 10,000 | 0.017269 | 0.004048 | 579,068.9 | 2,470,378.3 | 4.27x | 0.99277 | 0.99293 | 0.99275 | 0.99517 | 0.99275 | 1.67e-04 | 0.8963 | PASS |
| `phase_flip_3` | `phase_flip` | 100,000 | 0.334624 | 0.026219 | 298,842.7 | 3,814,018.4 | 12.76x | 0.99263 | 0.99278 | 0.99275 | 0.99517 | 0.99275 | 1.53e-04 | 0.5921 | PASS |
| `five_qubit` | `depolarizing` | 1,000 | 0.035527 | 0.006986 | 28,147.6 | 143,134.8 | 5.09x | 0.98200 | 0.98333 | 0.98511 | 0.98511 | 0.97767 | 1.33e-03 | 0.2803 | PASS |
| `five_qubit` | `depolarizing` | 10,000 | 0.911476 | 0.064071 | 10,971.2 | 156,077.4 | 14.23x | 0.98480 | 0.98523 | 0.98511 | 0.98511 | 0.97767 | 4.33e-04 | 0.6456 | PASS |
| `five_qubit` | `depolarizing` | 100,000 | 10.080841 | 0.413090 | 9,919.8 | 242,077.7 | 24.40x | 0.98505 | 0.98505 | 0.98511 | 0.98511 | 0.97767 | 6.67e-06 | 0.6394 | PASS |

### Performance & Fidelity Notes

- **Launch-latency crossover**: At small shot counts (`shots=1,000`) on 5-qubit codes (`bit_flip_3`, `phase_flip_3`, `2^5 = 32` amplitudes/shot), GPU kernel launch and host-device synchronization overhead (~2-3 ms) is comparable to vectorized NumPy CPU runtime, resulting in near-1x or sub-1x speedup. At `10,000`–`100,000` shots and on the 9-qubit `five_qubit` code (`2^9 = 512` amplitudes/shot), parallel GPU statevector execution delivers substantial multi-x speedup.
- **Fidelity convention**: `expected_state_fidelity` evaluates the exact state-specific logical Pauli channel fidelity `sum_L p_L |<psi|L|psi>|^2` for the benchmarked `(code, channel, state)`, while `analytical_state_fidelity` reports the Haar-averaged Pauli-channel state fidelity `1 - (2/3)*(1 - P_I(p))` and `analytical_entanglement_fidelity` reports `P_I(p)`.

## `eigenkit_unitary` Cross-Verification & Stochastic Gap

- **Stochastic support**: `False` (probes raise `NotImplementedError`: `EigenkitUnitaryAdapter cannot run stochastic QECC sweeps natively: eigenkit v1.3.0 lacks native stochastic Pauli/Kraus noise channels, mid-circuit projective measurement, and classical conditional c_if feed-forward (see docs/eigenkit_evaluation.md). Use backend='numpy' or 'torch_cuda'.`; see `docs/eigenkit_evaluation.md`).

| Code | Mode | Shots | `numpy` F | `torch_cuda` F | `eigenkit_unitary` F | Max |Δρ| | Syndromes Match | `eigenkit` (s) | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | :---: | ---: | :---: |
| `bit_flip_3` | `p=0` | 100 | 1.000000000000 | 1.000000000000 | 1.000000000000 | 0.00e+00 | YES | 0.000509 | PASS |
| `bit_flip_3` | `deterministic:XII` | 100 | 1.000000000000 | 1.000000000000 | 1.000000000000 | 0.00e+00 | YES | 0.000456 | PASS |
| `phase_flip_3` | `p=0` | 100 | 1.000000000000 | 1.000000000000 | 1.000000000000 | 4.44e-16 | YES | 0.000322 | PASS |
| `phase_flip_3` | `deterministic:ZII` | 100 | 1.000000000000 | 1.000000000000 | 1.000000000000 | 4.44e-16 | YES | 0.000380 | PASS |
| `five_qubit` | `p=0` | 100 | 1.000000000000 | 1.000000000000 | 1.000000000000 | 1.05e-15 | YES | 0.000717 | PASS |
| `five_qubit` | `deterministic:YIIII` | 100 | 1.000000000000 | 1.000000000000 | 1.000000000000 | 1.05e-15 | YES | 0.000744 | PASS |
