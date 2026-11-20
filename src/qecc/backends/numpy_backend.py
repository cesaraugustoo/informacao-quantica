"""Vectorized NumPy CPU statevector backend for QECC simulation."""

from __future__ import annotations

import math
import time
from typing import Sequence

import numpy as np

from qecc.backends.base import (
    GateOp,
    QECCCode,
    SimulationResult,
    validate_and_prepare_cycle_inputs,
    validate_ops_qubits,
)
from qecc.channels.pauli import PAULI_CHARS, ChannelSpec, StochasticPauliChannel

_INV_SQRT2 = 1.0 / math.sqrt(2.0)
_PHASE0_LUT = np.array([1.0 + 0.0j, 1.0 + 0.0j, -1.0j, 1.0 + 0.0j], dtype=np.complex128)
_PHASE1_LUT = np.array([1.0 + 0.0j, 1.0 + 0.0j, 1.0j, -1.0 + 0.0j], dtype=np.complex128)


def _single_qubit_matrix(op: GateOp) -> np.ndarray:
    name = op.name
    if name == "H":
        return np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128) * _INV_SQRT2
    if name == "Y":
        return np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
    if name == "RX":
        half = 0.5 * op.params[0]
        c = math.cos(half)
        s = math.sin(half)
        return np.array([[c, -1.0j * s], [-1.0j * s, c]], dtype=np.complex128)
    if name == "RY":
        half = 0.5 * op.params[0]
        c = math.cos(half)
        s = math.sin(half)
        return np.array([[c, -s], [s, c]], dtype=np.complex128)
    if name == "RZ":
        half = 0.5 * op.params[0]
        return np.array(
            [[math.cos(half) - 1.0j * math.sin(half), 0.0], [0.0, math.cos(half) + 1.0j * math.sin(half)]],
            dtype=np.complex128,
        )
    raise ValueError(f"Unexpected gate in _single_qubit_matrix: {name}")


def _apply_ops_np(state: np.ndarray, ops: Sequence[GateOp], n_total: int) -> None:
    """Apply a sequence of GateOps in-place to a contiguous (B, 2**n_total) state array."""
    batch = state.shape[0]
    tview: np.ndarray | None = None

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
                tmp = tview[k0].copy()
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
            tmp = view[:, 0, :].copy()
            view[:, 0, :] = view[:, 1, :]
            view[:, 1, :] = tmp
        else:
            mat = _single_qubit_matrix(op)
            s0 = view[:, 0, :]
            s1 = view[:, 1, :]
            new0 = mat[0, 0] * s0 + mat[0, 1] * s1
            new1 = mat[1, 0] * s0 + mat[1, 1] * s1
            view[:, 0, :] = new0
            view[:, 1, :] = new1


def _apply_pauli_batch_np(state: np.ndarray, paulis: np.ndarray, n_total: int) -> None:
    """Apply per-shot (B, n_data) uint8 Pauli operations branchlessly in-place."""
    batch, n_data = paulis.shape
    for q in range(n_data):
        ops_q = paulis[:, q]
        if not np.any(ops_q):
            continue
        view = state.reshape(batch, 1 << q, 2, 1 << (n_total - 1 - q))
        swap = ((ops_q == 1) | (ops_q == 2))[:, None, None]
        p0 = _PHASE0_LUT[ops_q][:, None, None]
        p1 = _PHASE1_LUT[ops_q][:, None, None]
        s0 = view[:, :, 0, :]
        s1 = view[:, :, 1, :]
        new0 = p0 * np.where(swap, s1, s0)
        new1 = p1 * np.where(swap, s0, s1)
        view[:, :, 0, :] = new0
        view[:, :, 1, :] = new1


class NumPyCPUBackend:
    """Vectorized batched complex128 NumPy CPU backend."""

    name: str = "numpy"

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
        prep = validate_and_prepare_cycle_inputs(
            code=code,
            initial_state=initial_state,
            channel=channel,
            p=p,
            num_shots=num_shots,
            seed=seed,
            deterministic_errors=deterministic_errors,
        )

        t0 = time.perf_counter()

        n_data = prep.n_data
        n_ancilla = prep.n_ancilla
        n_total = prep.n_total
        shots = prep.num_shots

        state = np.zeros((shots, 1 << n_total), dtype=np.complex128)
        state[:, 0] = prep.psi[0]
        state[:, 1 << (n_total - 1)] = prep.psi[1]

        _apply_ops_np(state, prep.encode_ops, n_total)

        if prep.error_mode == "deterministic":
            assert prep.deterministic_codes is not None
            if prep.deterministic_codes.shape[0] == 1:
                row = prep.deterministic_codes[0]
                det_ops = [
                    GateOp(name=PAULI_CHARS[int(c)], targets=(q,))  # type: ignore[arg-type]
                    for q, c in enumerate(row)
                    if c != 0
                ]
                _apply_ops_np(state, det_ops, n_total)
            else:
                _apply_pauli_batch_np(state, prep.deterministic_codes, n_total)
        elif prep.error_mode == "stochastic":
            assert prep.channel_spec is not None
            paulis = StochasticPauliChannel(prep.channel_spec).sample(
                n_data, shots, prep.rng
            )
            _apply_pauli_batch_np(state, paulis, n_total)

        _apply_ops_np(state, prep.syndrome_ops, n_total)

        if n_ancilla == 0:
            syndromes = [""] * shots
            syndrome_counts = {"": shots}
            rec_ops = list(code.recovery_ops(""))
            validate_ops_qubits(rec_ops, n_total)
            if rec_ops:
                _apply_ops_np(state, rec_ops, n_total)
        else:
            da = state.reshape(shots, 1 << n_data, 1 << n_ancilla)
            probs = (da.real ** 2 + da.imag ** 2).sum(axis=1)
            probs /= np.maximum(probs.sum(axis=1, keepdims=True), 1e-30)
            cdf = np.cumsum(probs, axis=1)
            cdf[:, -1] = 1.0
            u = prep.rng.random(shots)
            m = np.clip((cdf <= u[:, None]).sum(axis=1), 0, (1 << n_ancilla) - 1)

            b_idx = np.arange(shots)
            surv = da[b_idx, :, m] / np.sqrt(np.maximum(probs[b_idx, m], 1e-30))[:, None]
            da.fill(0.0)
            da[b_idx, :, m] = surv

            unique_m, inverse, counts = np.unique(m, return_inverse=True, return_counts=True)
            m_to_str = {int(val): format(int(val), f"0{n_ancilla}b") for val in unique_m}
            syndromes = [m_to_str[int(val)] for val in m]
            syndrome_counts = {
                m_to_str[int(val)]: int(cnt) for val, cnt in zip(unique_m, counts)
            }

            if len(unique_m) == 1:
                s_only = m_to_str[int(unique_m[0])]
                rec_ops = list(code.recovery_ops(s_only))
                validate_ops_qubits(rec_ops, n_total)
                if rec_ops:
                    _apply_ops_np(state, rec_ops, n_total)
            else:
                for group_id, m_val in enumerate(unique_m):
                    s = m_to_str[int(m_val)]
                    rec_ops = list(code.recovery_ops(s))
                    validate_ops_qubits(rec_ops, n_total)
                    if rec_ops:
                        idx = np.flatnonzero(inverse == group_id)
                        sub = np.ascontiguousarray(state[idx])
                        _apply_ops_np(sub, rec_ops, n_total)
                        state[idx] = sub

        _apply_ops_np(state, prep.decode_ops, n_total)

        psi_mat = state.reshape(shots, 2, 1 << (n_total - 1))
        rho_shots = np.einsum("bij,bkj->bik", psi_mat, psi_mat.conj())
        rho_mean = rho_shots.mean(axis=0)
        rho_out = np.asarray(0.5 * (rho_mean + rho_mean.conj().T), dtype=np.complex128)

        fidelities = np.clip(
            np.real(np.einsum("i,bij,j->b", prep.psi.conj(), rho_shots, prep.psi)),
            0.0,
            1.0,
        ).astype(np.float64)
        mean_fidelity = float(np.mean(fidelities))
        std_error = (
            0.0
            if shots == 1
            else float(np.std(fidelities, ddof=1) / math.sqrt(shots))
        )

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
