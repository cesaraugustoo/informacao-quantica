"""PyTorch CUDA batched complex128 backend for RTX 5080 (cu130 / sm_120)."""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from qecc.backends.base import (
    GateOp,
    QECCCode,
    SimulationResult,
    validate_and_prepare_cycle_inputs,
    validate_ops_qubits,
)
from qecc.channels.pauli import PAULI_CHARS, ChannelSpec, StochasticPauliChannel

if TYPE_CHECKING:
    import torch

_INV_SQRT2 = 1.0 / math.sqrt(2.0)


def _single_qubit_matrix_entries(op: GateOp) -> tuple[complex, complex, complex, complex]:
    name = op.name
    if name == "H":
        return (_INV_SQRT2, _INV_SQRT2, _INV_SQRT2, -_INV_SQRT2)
    if name == "Y":
        return (0.0 + 0.0j, -1.0j, 1.0j, 0.0 + 0.0j)
    if name == "RX":
        half = 0.5 * op.params[0]
        c = math.cos(half)
        s = math.sin(half)
        return (c + 0.0j, -1.0j * s, -1.0j * s, c + 0.0j)
    if name == "RY":
        half = 0.5 * op.params[0]
        c = math.cos(half)
        s = math.sin(half)
        return (c + 0.0j, -s + 0.0j, s + 0.0j, c + 0.0j)
    if name == "RZ":
        half = 0.5 * op.params[0]
        e0 = math.cos(half) - 1.0j * math.sin(half)
        e1 = math.cos(half) + 1.0j * math.sin(half)
        return (e0, 0.0 + 0.0j, 0.0 + 0.0j, e1)
    raise ValueError(f"Unexpected gate in _single_qubit_matrix_entries: {name}")


def _apply_ops_torch(state: Any, ops: Sequence[GateOp], n_total: int) -> None:
    """Apply GateOps in-place to a contiguous (B, 2**n_total) torch.complex128 tensor."""
    batch = state.shape[0]
    tview = None

    for op in ops:
        name = op.name
        if name == "I":
            continue
        if name in ("CNOT", "CZ"):
            if tview is None:
                tview = state.reshape((batch,) + (2,) * n_total)
            c = op.controls[0]
            t = op.targets[0]
            if name == "CNOT":
                sel0 = [slice(None)] * (n_total + 1)
                sel0[c + 1] = 1
                sel0[t + 1] = 0
                sel1 = [slice(None)] * (n_total + 1)
                sel1[c + 1] = 1
                sel1[t + 1] = 1
                k0, k1 = tuple(sel0), tuple(sel1)
                tmp = tview[k0].clone()
                tview[k0] = tview[k1]
                tview[k1] = tmp
            else:
                sel = [slice(None)] * (n_total + 1)
                sel[c + 1] = 1
                sel[t + 1] = 1
                tview[tuple(sel)] *= -1.0
            continue

        q = op.targets[0]
        view = state.reshape(batch * (1 << q), 2, 1 << (n_total - 1 - q))
        if name == "Z":
            view[:, 1, :] *= -1.0
        elif name == "S":
            view[:, 1, :] *= 1.0j
        elif name == "Sdg":
            view[:, 1, :] *= -1.0j
        elif name == "X":
            tmp = view[:, 0, :].clone()
            view[:, 0, :] = view[:, 1, :]
            view[:, 1, :] = tmp
        else:
            u00, u01, u10, u11 = _single_qubit_matrix_entries(op)
            s0 = view[:, 0, :]
            s1 = view[:, 1, :]
            new0 = u00 * s0 + u01 * s1
            new1 = u10 * s0 + u11 * s1
            view[:, 0, :] = new0
            view[:, 1, :] = new1


class TorchCUDABackend:
    """Batched PyTorch CUDA complex128 statevector backend."""

    name: str = "torch_cuda"

    def __init__(
        self,
        device: str | torch.device = "cuda",
        max_batch_bytes: int = 1024**3,
        warmup: bool = True,
    ) -> None:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "PyTorch is required for TorchCUDABackend; install torch with CUDA support."
            ) from exc

        if isinstance(max_batch_bytes, bool) or not isinstance(max_batch_bytes, (int, np.integer)):
            raise ValueError(f"max_batch_bytes must be an integer >= 1024, got {max_batch_bytes!r}")
        if int(max_batch_bytes) < 1024:
            raise ValueError(f"max_batch_bytes must be >= 1024, got {max_batch_bytes}")

        resolved_device = torch.device(device)
        if resolved_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "TorchCUDABackend requires a CUDA-capable GPU, but torch.cuda.is_available() is False."
            )

        self._torch = torch
        self.device = resolved_device
        self.max_batch_bytes = int(max_batch_bytes)

        self._phase0_lut = torch.tensor(
            [1.0 + 0.0j, 1.0 + 0.0j, -1.0j, 1.0 + 0.0j],
            dtype=torch.complex128,
            device=self.device,
        )
        self._phase1_lut = torch.tensor(
            [1.0 + 0.0j, 1.0 + 0.0j, 1.0j, -1.0 + 0.0j],
            dtype=torch.complex128,
            device=self.device,
        )

        if warmup and self.device.type == "cuda":
            self._run_warmup()

    def _run_warmup(self) -> None:
        torch = self._torch
        with torch.no_grad():
            st = torch.zeros((2, 4), dtype=torch.complex128, device=self.device)
            st[:, 0] = 1.0
            ops = [GateOp("H", (0,)), GateOp("CNOT", (1,), controls=(0,))]
            _apply_ops_torch(st, ops, 2)
            paulis = torch.tensor([[1, 2], [3, 0]], dtype=torch.int64, device=self.device)
            self._apply_pauli_batch_torch(st, paulis, 2)
            da = st.reshape(2, 2, 2)
            probs = (da.real**2 + da.imag**2).sum(dim=1)
            _ = torch.cumsum(probs, dim=1)
            psi_mat = st.reshape(2, 2, 2)
            rho = torch.bmm(psi_mat, psi_mat.conj().transpose(1, 2))
            vec = torch.tensor([1.0 + 0.0j, 0.0 + 0.0j], dtype=torch.complex128, device=self.device)
            _ = torch.einsum("i,bij,j->b", vec.conj(), rho, vec)
            torch.cuda.synchronize(self.device)

    def _apply_pauli_batch_torch(self, state: Any, paulis: Any, n_total: int) -> None:
        torch = self._torch
        batch, n_data = paulis.shape
        for q in range(n_data):
            ops_q = paulis[:, q]
            view = state.reshape(batch, 1 << q, 2, 1 << (n_total - 1 - q))
            swap = ((ops_q == 1) | (ops_q == 2))[:, None, None]
            p0 = self._phase0_lut[ops_q][:, None, None]
            p1 = self._phase1_lut[ops_q][:, None, None]
            s0 = view[:, :, 0, :]
            s1 = view[:, :, 1, :]
            new0 = p0 * torch.where(swap, s1, s0)
            new1 = p1 * torch.where(swap, s0, s1)
            view[:, :, 0, :] = new0
            view[:, :, 1, :] = new1

    def _sync_if_cuda(self) -> None:
        if self.device.type == "cuda":
            self._torch.cuda.synchronize(self.device)

    def simulate_cycle(
        self,
        code: QECCCode,
        initial_state: np.ndarray,
        channel: ChannelSpec | str | None,
        p: float,
        num_shots: int,
        seed: int | None = None,
        deterministic_errors: list[str] | str | None = None,
    ) -> SimulationResult:
        torch = self._torch
        prep = validate_and_prepare_cycle_inputs(
            code=code,
            initial_state=initial_state,
            channel=channel,
            p=p,
            num_shots=num_shots,
            seed=seed,
            deterministic_errors=deterministic_errors,
        )

        n_data = prep.n_data
        n_ancilla = prep.n_ancilla
        n_total = prep.n_total
        shots = prep.num_shots

        bytes_per_shot = (1 << n_total) * 16
        chunk_size = max(1, self.max_batch_bytes // bytes_per_shot)

        cuda_gen: Any = None
        if n_ancilla > 0:
            cuda_gen = torch.Generator(device=self.device)
            cuda_gen.manual_seed(int(prep.rng.integers(0, 2**63 - 1)))

        self._sync_if_cuda()
        t0 = time.perf_counter()

        with torch.no_grad():
            psi_gpu = torch.as_tensor(prep.psi, dtype=torch.complex128, device=self.device)

            paulis_all: np.ndarray | None = None
            det_single_ops: list[GateOp] | None = None
            if prep.error_mode == "deterministic":
                assert prep.deterministic_codes is not None
                if prep.deterministic_codes.shape[0] == 1:
                    row = prep.deterministic_codes[0]
                    det_single_ops = [
                        GateOp(name=PAULI_CHARS[int(c)], targets=(q,))  # type: ignore[arg-type]
                        for q, c in enumerate(row)
                        if c != 0
                    ]
                else:
                    paulis_all = prep.deterministic_codes
            elif prep.error_mode == "stochastic":
                assert prep.channel_spec is not None
                paulis_all = StochasticPauliChannel(prep.channel_spec).sample(
                    n_data, shots, prep.rng
                )

            u_all: Any = None
            if n_ancilla > 0:
                u_all = torch.rand(
                    shots, dtype=torch.float64, device=self.device, generator=cuda_gen
                )

            rho_sum = torch.zeros((2, 2), dtype=torch.complex128, device=self.device)
            fidelities_chunks: list[np.ndarray] = []
            m_chunks: list[np.ndarray] = []

            for start in range(0, shots, chunk_size):
                end = min(start + chunk_size, shots)
                cur_b = end - start

                state = torch.zeros(
                    (cur_b, 1 << n_total), dtype=torch.complex128, device=self.device
                )
                state[:, 0] = psi_gpu[0]
                state[:, 1 << (n_total - 1)] = psi_gpu[1]

                _apply_ops_torch(state, prep.encode_ops, n_total)

                if det_single_ops is not None:
                    if det_single_ops:
                        _apply_ops_torch(state, det_single_ops, n_total)
                elif paulis_all is not None:
                    pauli_chunk = torch.as_tensor(
                        paulis_all[start:end], dtype=torch.int64, device=self.device
                    )
                    self._apply_pauli_batch_torch(state, pauli_chunk, n_total)

                _apply_ops_torch(state, prep.syndrome_ops, n_total)

                if n_ancilla == 0:
                    rec_ops = list(code.recovery_ops(""))
                    validate_ops_qubits(rec_ops, n_total)
                    if rec_ops:
                        _apply_ops_torch(state, rec_ops, n_total)
                else:
                    da = state.reshape(cur_b, 1 << n_data, 1 << n_ancilla)
                    probs = (da.real**2 + da.imag**2).sum(dim=1)
                    probs = probs / probs.sum(dim=1, keepdim=True).clamp(min=1e-30)
                    cdf = torch.cumsum(probs, dim=1)
                    cdf[:, -1] = 1.0
                    u = u_all[start:end]
                    m = (cdf <= u[:, None]).sum(dim=1).clamp(0, (1 << n_ancilla) - 1)

                    b_idx = torch.arange(cur_b, device=self.device)
                    surv = da[b_idx, :, m] / torch.sqrt(
                        probs[b_idx, m].clamp(min=1e-30)
                    )[:, None]
                    da.zero_()
                    da[b_idx, :, m] = surv

                    m_cpu = m.cpu().numpy()
                    m_chunks.append(m_cpu)

                    unique_m, inverse = np.unique(m_cpu, return_inverse=True)
                    if len(unique_m) == 1:
                        s_only = format(int(unique_m[0]), f"0{n_ancilla}b")
                        rec_ops = list(code.recovery_ops(s_only))
                        validate_ops_qubits(rec_ops, n_total)
                        if rec_ops:
                            _apply_ops_torch(state, rec_ops, n_total)
                    else:
                        for group_id, m_val in enumerate(unique_m):
                            s = format(int(m_val), f"0{n_ancilla}b")
                            rec_ops = list(code.recovery_ops(s))
                            validate_ops_qubits(rec_ops, n_total)
                            if rec_ops:
                                idx_np = np.flatnonzero(inverse == group_id)
                                idx_t = torch.as_tensor(
                                    idx_np, dtype=torch.int64, device=self.device
                                )
                                sub = state.index_select(0, idx_t)
                                _apply_ops_torch(sub, rec_ops, n_total)
                                state.index_copy_(0, idx_t, sub)

                _apply_ops_torch(state, prep.decode_ops, n_total)

                psi_mat = state.reshape(cur_b, 2, 1 << (n_total - 1))
                rho_shots = torch.bmm(psi_mat, psi_mat.conj().transpose(1, 2))
                rho_sum += rho_shots.sum(dim=0)
                fid_chunk = (
                    torch.einsum("i,bij,j->b", psi_gpu.conj(), rho_shots, psi_gpu)
                    .real.clamp(0.0, 1.0)
                    .cpu()
                    .numpy()
                )
                fidelities_chunks.append(fid_chunk)

            rho_mean = (rho_sum / float(shots)).cpu().numpy()
            self._sync_if_cuda()

        rho_out = np.asarray(0.5 * (rho_mean + rho_mean.conj().T), dtype=np.complex128)
        fidelities = np.concatenate(fidelities_chunks).astype(np.float64, copy=False)
        mean_fidelity = float(np.mean(fidelities))
        std_error = (
            0.0
            if shots == 1
            else float(np.std(fidelities, ddof=1) / math.sqrt(shots))
        )

        if n_ancilla == 0:
            syndromes = [""] * shots
            syndrome_counts = {"": shots}
        else:
            all_m = np.concatenate(m_chunks)
            unique_m, counts = np.unique(all_m, return_counts=True)
            m_to_str = {int(val): format(int(val), f"0{n_ancilla}b") for val in unique_m}
            syndromes = [m_to_str[int(val)] for val in all_m]
            syndrome_counts = {
                m_to_str[int(val)]: int(cnt) for val, cnt in zip(unique_m, counts)
            }

        elapsed = time.perf_counter() - t0
        return SimulationResult(
            code_name=code.name,
            backend_name=self.name,
            p=prep.p,
            num_shots=shots,
            syndromes=syndromes,
            syndrome_counts=syndrome_counts,
            rho_out=rho_out,
            fidelities=fidelities,
            mean_fidelity=mean_fidelity,
            std_error=std_error,
            elapsed_seconds=elapsed,
        )
