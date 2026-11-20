"""Eigenkit unitary sub-circuit adapter for QECC verification."""

from __future__ import annotations

import math
import time
from typing import Any, Sequence

import numpy as np

from qecc.backends.base import (
    GateOp,
    QECCCode,
    SimulationResult,
    validate_and_prepare_cycle_inputs,
    validate_ops_qubits,
)
from qecc.channels.pauli import PAULI_CHARS, ChannelSpec


def translate_ops_to_eigenkit(
    ops: Sequence[GateOp],
    n_qubits: int,
    *,
    supports_mcz: bool = True,
) -> Any:
    """Translate GateOp sequence into an eigenkit.CircuitBuilder (big-endian qubit 0 MSB)."""
    from eigenkit.core.builder import CircuitBuilder

    validate_ops_qubits(ops, n_qubits)
    builder = CircuitBuilder(n_qubits)

    for op in ops:
        name = op.name
        t = op.targets[0]
        if name == "I":
            continue
        if name == "X":
            builder.x(t)
        elif name == "Z":
            builder.z(t)
        elif name == "H":
            builder.h(t)
        elif name == "S":
            builder.s(t)
        elif name == "Sdg":
            builder.sdg(t)
        elif name == "RX":
            builder.rx(op.params[0], t)
        elif name == "RY":
            builder.ry(op.params[0], t)
        elif name == "RZ":
            builder.rz(op.params[0], t)
        elif name == "CNOT":
            builder.cnot(op.controls[0], t)
        elif name == "Y":
            # Matrix product in circuit order: S * X * Sdg == Y (zero global phase).
            builder.sdg(t).x(t).s(t)
        elif name == "CZ":
            if supports_mcz:
                builder.mcz(op.controls[0], t)
            else:
                builder.h(t).cnot(op.controls[0], t).h(t)
        else:
            raise ValueError(f"Unsupported gate for eigenkit translation: {name!r}")

    return builder


class EigenkitUnitaryAdapter:
    """Adapter executing pure unitary stages via eigenkit.Circuit.statevector."""

    name: str = "eigenkit_unitary"

    def __init__(self, provider: str = "numpy") -> None:
        try:
            from eigenkit import Circuit
            from eigenkit.core.provider import resolve_provider
        except ImportError as exc:
            raise RuntimeError(
                "eigenkit is required for EigenkitUnitaryAdapter."
            ) from exc

        self._Circuit = Circuit
        self.provider = provider
        resolved = resolve_provider(provider)
        self._supports_mcz = "MCZ" in resolved.capabilities.supported_gates

    def run_unitary_ops(
        self,
        ops: Sequence[GateOp],
        n_qubits: int,
        initial_state: np.ndarray,
    ) -> np.ndarray:
        """Run a pure unitary GateOp sequence from initial_state using eigenkit."""
        if isinstance(n_qubits, bool) or not isinstance(n_qubits, (int, np.integer)) or int(n_qubits) < 1:
            raise ValueError(f"n_qubits must be >= 1, got {n_qubits!r}")
        nq = int(n_qubits)
        dim = 1 << nq

        init = np.asarray(initial_state, dtype=np.complex128)
        if init.shape != (dim,) or not np.all(np.isfinite(init)):
            raise ValueError(f"initial_state must have finite shape ({dim},), got {init.shape}")
        norm_sq = float(np.vdot(init, init).real)
        if abs(norm_sq - 1.0) > 1e-6:
            raise ValueError("initial_state must be unit-normalized")
        init = init / math.sqrt(norm_sq)

        builder = translate_ops_to_eigenkit(ops, nq, supports_mcz=self._supports_mcz)
        circ = self._Circuit(
            builder, observables=None, provider=self.provider, interface="numpy"
        )
        sv = circ.statevector(
            np.empty(0, dtype=np.float64), inputs=None, initial_state=init
        )
        return np.asarray(sv, dtype=np.complex128)

    def _run_deterministic_pattern(
        self,
        code: QECCCode,
        init_n: np.ndarray,
        encode_ops: list[GateOp],
        error_row: np.ndarray | None,
        syndrome_ops: list[GateOp],
        decode_ops: list[GateOp],
        n_data: int,
        n_ancilla: int,
        n_total: int,
        count: int,
        rng: np.random.Generator,
    ) -> tuple[list[str], np.ndarray]:
        """Execute a deterministic error pattern across `count` shots."""
        state = self.run_unitary_ops(encode_ops, n_total, init_n)
        if error_row is not None and np.any(error_row):
            err_ops = [
                GateOp(name=PAULI_CHARS[int(c)], targets=(q,))  # type: ignore[arg-type]
                for q, c in enumerate(error_row)
                if c != 0
            ]
            state = self.run_unitary_ops(err_ops, n_total, state)

        if syndrome_ops:
            state = self.run_unitary_ops(syndrome_ops, n_total, state)

        if n_ancilla == 0:
            rec_ops = list(code.recovery_ops(""))
            validate_ops_qubits(rec_ops, n_total)
            if rec_ops:
                state = self.run_unitary_ops(rec_ops, n_total, state)
            if decode_ops:
                state = self.run_unitary_ops(decode_ops, n_total, state)
            psi_mat = state.reshape(2, 1 << (n_total - 1))
            rho_single = psi_mat @ psi_mat.conj().T
            rho_batch = np.broadcast_to(rho_single, (count, 2, 2)).copy()
            return [""] * count, rho_batch

        da = state.reshape(1 << n_data, 1 << n_ancilla)
        probs = (da.real**2 + da.imag**2).sum(axis=0)
        probs /= max(float(probs.sum()), 1e-30)

        max_m = int(np.argmax(probs))
        if probs[max_m] >= 1.0 - 1e-12:
            # Deterministic single syndrome branch: reuse across all `count` identical shots.
            s = format(max_m, f"0{n_ancilla}b")
            surv = da[:, max_m] / math.sqrt(max(float(probs[max_m]), 1e-30))
            collapsed = np.zeros_like(da)
            collapsed[:, max_m] = surv
            post = collapsed.reshape(1 << n_total)
            rec_ops = list(code.recovery_ops(s))
            validate_ops_qubits(rec_ops, n_total)
            if rec_ops:
                post = self.run_unitary_ops(rec_ops, n_total, post)
            if decode_ops:
                post = self.run_unitary_ops(decode_ops, n_total, post)
            psi_mat = post.reshape(2, 1 << (n_total - 1))
            rho_single = psi_mat @ psi_mat.conj().T
            rho_batch = np.broadcast_to(rho_single, (count, 2, 2)).copy()
            return [s] * count, rho_batch

        # Superposition over syndromes: sample per shot and evaluate each realized branch once.
        cdf = np.cumsum(probs)
        cdf[-1] = 1.0
        u = rng.random(count)
        m_vals = np.clip(np.searchsorted(cdf, u, side="right"), 0, (1 << n_ancilla) - 1)

        branch_cache: dict[int, tuple[str, np.ndarray]] = {}
        rho_batch = np.empty((count, 2, 2), dtype=np.complex128)
        syndromes: list[str] = []
        for idx, m_int in enumerate(m_vals):
            mv = int(m_int)
            if mv not in branch_cache:
                s = format(mv, f"0{n_ancilla}b")
                surv = da[:, mv] / math.sqrt(max(float(probs[mv]), 1e-30))
                collapsed = np.zeros_like(da)
                collapsed[:, mv] = surv
                post = collapsed.reshape(1 << n_total)
                rec_ops = list(code.recovery_ops(s))
                validate_ops_qubits(rec_ops, n_total)
                if rec_ops:
                    post = self.run_unitary_ops(rec_ops, n_total, post)
                if decode_ops:
                    post = self.run_unitary_ops(decode_ops, n_total, post)
                psi_mat = post.reshape(2, 1 << (n_total - 1))
                branch_cache[mv] = (s, psi_mat @ psi_mat.conj().T)
            s, rho_s = branch_cache[mv]
            syndromes.append(s)
            rho_batch[idx] = rho_s
        return syndromes, rho_batch

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

        if prep.error_mode == "stochastic":
            raise NotImplementedError(
                "EigenkitUnitaryAdapter cannot run stochastic QECC sweeps natively: "
                "eigenkit v1.3.0 lacks native stochastic Pauli/Kraus noise channels, "
                "mid-circuit projective measurement, and classical conditional c_if feed-forward "
                "(see docs/eigenkit_evaluation.md). Use backend='numpy' or 'torch_cuda'."
            )

        t0 = time.perf_counter()

        n_data = prep.n_data
        n_ancilla = prep.n_ancilla
        n_total = prep.n_total
        shots = prep.num_shots

        init_n = np.zeros(1 << n_total, dtype=np.complex128)
        init_n[0] = prep.psi[0]
        init_n[1 << (n_total - 1)] = prep.psi[1]

        if prep.error_mode == "none" or (
            prep.deterministic_codes is not None and prep.deterministic_codes.shape[0] == 1
        ):
            err_row = None if prep.deterministic_codes is None else prep.deterministic_codes[0]
            syndromes, rho_shots = self._run_deterministic_pattern(
                code=code,
                init_n=init_n,
                encode_ops=prep.encode_ops,
                error_row=err_row,
                syndrome_ops=prep.syndrome_ops,
                decode_ops=prep.decode_ops,
                n_data=n_data,
                n_ancilla=n_ancilla,
                n_total=n_total,
                count=shots,
                rng=prep.rng,
            )
        else:
            assert prep.deterministic_codes is not None
            syndromes = [""] * shots
            rho_shots = np.empty((shots, 2, 2), dtype=np.complex128)

            unique_rows, inverse, counts = np.unique(
                prep.deterministic_codes, axis=0, return_inverse=True, return_counts=True
            )
            for gid in range(len(unique_rows)):
                row = unique_rows[gid]
                cnt = int(counts[gid])
                syn_g, rho_g = self._run_deterministic_pattern(
                    code=code,
                    init_n=init_n,
                    encode_ops=prep.encode_ops,
                    error_row=row,
                    syndrome_ops=prep.syndrome_ops,
                    decode_ops=prep.decode_ops,
                    n_data=n_data,
                    n_ancilla=n_ancilla,
                    n_total=n_total,
                    count=cnt,
                    rng=prep.rng,
                )
                indices = np.flatnonzero(inverse == gid)
                for local_i, shot_i in enumerate(indices):
                    syndromes[int(shot_i)] = syn_g[local_i]
                    rho_shots[int(shot_i)] = rho_g[local_i]

        syndrome_counts: dict[str, int] = {}
        for s in syndromes:
            syndrome_counts[s] = syndrome_counts.get(s, 0) + 1

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
