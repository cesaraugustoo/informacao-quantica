"""Squared Uhlmann-Jozsa quantum fidelity and big-endian partial trace."""

from __future__ import annotations

import numpy as np


def _validate_state_or_dm(state_or_rho: np.ndarray, *, tol: float = 1e-6) -> tuple[np.ndarray, bool]:
    """Validate a 1D statevector or 2D density matrix, returning (normalized_array, is_statevector)."""
    arr = np.asarray(state_or_rho, dtype=np.complex128)
    if not np.all(np.isfinite(arr)):
        raise ValueError("Input state/density matrix contains non-finite entries")

    if arr.ndim == 1:
        if arr.shape[0] < 2:
            raise ValueError(f"Statevector dimension must be >= 2, got {arr.shape[0]}")
        norm = float(np.linalg.norm(arr))
        if abs(norm - 1.0) > tol:
            raise ValueError(f"Statevector must be unit-normalized (norm={norm})")
        return arr / norm, True

    if arr.ndim == 2:
        if arr.shape[0] != arr.shape[1] or arr.shape[0] < 2:
            raise ValueError(f"Density matrix must be square (d, d) with d >= 2, got {arr.shape}")
        if float(np.max(np.abs(arr - arr.conj().T))) > tol:
            raise ValueError("Density matrix is not Hermitian within tolerance")
        rho = 0.5 * (arr + arr.conj().T)
        tr = np.trace(rho)
        tr_val = float(np.real(tr))
        if abs(tr_val - 1.0) > tol or abs(float(np.imag(tr))) > tol:
            raise ValueError(f"Density matrix trace must equal 1, got {tr}")
        min_eig = float(np.min(np.linalg.eigvalsh(rho)))
        if min_eig < -tol:
            raise ValueError(f"Density matrix is not PSD (min eigenvalue {min_eig})")
        rho = rho / tr_val
        return np.asarray(rho, dtype=np.complex128), False

    raise ValueError(f"Expected 1D statevector or 2D density matrix, got ndim={arr.ndim}")


def _to_density_matrix(state_or_rho: np.ndarray, *, tol: float = 1e-6) -> np.ndarray:
    """Validate and convert a statevector (d,) or density matrix (d, d) into a (d, d) density matrix."""
    arr, is_statevector = _validate_state_or_dm(state_or_rho, tol=tol)
    if is_statevector:
        return np.outer(arr, arr.conj())
    return arr


def quantum_fidelity(rho: np.ndarray, sigma: np.ndarray) -> float:
    """Compute squared Uhlmann-Jozsa fidelity F(rho, sigma) = (Tr sqrt(sqrt(rho) sigma sqrt(rho)))^2."""
    arr_a, a_is_vec = _validate_state_or_dm(rho, tol=1e-6)
    arr_b, b_is_vec = _validate_state_or_dm(sigma, tol=1e-6)

    dim_a = arr_a.shape[0]
    dim_b = arr_b.shape[0]
    if dim_a != dim_b:
        raise ValueError(
            f"State dimensions do not match: {(dim_a, dim_a)} vs {(dim_b, dim_b)}"
        )

    if a_is_vec and b_is_vec:
        overlap = abs(np.vdot(arr_a, arr_b)) ** 2
        return float(np.clip(overlap, 0.0, 1.0))

    if a_is_vec:
        fid = float(np.real(np.vdot(arr_a, arr_b @ arr_a)))
        return float(np.clip(fid, 0.0, 1.0))

    if b_is_vec:
        fid = float(np.real(np.vdot(arr_b, arr_a @ arr_b)))
        return float(np.clip(fid, 0.0, 1.0))

    eps = float(np.finfo(np.float64).eps)
    w_r, v_r = np.linalg.eigh(arr_a)
    tol_r = 16.0 * len(w_r) * eps * max(float(np.max(w_r)), 1.0)
    w_r_clamped = np.where(w_r > tol_r, w_r, 0.0)
    sqrt_rho = (v_r * np.sqrt(w_r_clamped)) @ v_r.conj().T

    w_s, v_s = np.linalg.eigh(arr_b)
    tol_s = 16.0 * len(w_s) * eps * max(float(np.max(w_s)), 1.0)
    w_s_clamped = np.where(w_s > tol_s, w_s, 0.0)
    sqrt_sigma = (v_s * np.sqrt(w_s_clamped)) @ v_s.conj().T

    s = np.linalg.svdvals(sqrt_rho @ sqrt_sigma)
    fid = float(np.sum(s)) ** 2
    return float(np.clip(fid, 0.0, 1.0))


def partial_trace_to_qubit0(
    state_or_rho: np.ndarray,
    n_qubits: int,
    *,
    batched: bool | None = None,
) -> np.ndarray:
    """Trace out qubits 1..n_qubits-1 in big-endian (qubit 0 MSB) order."""
    if isinstance(n_qubits, bool) or not isinstance(n_qubits, (int, np.integer)):
        raise ValueError(f"n_qubits must be an integer >= 1, got {n_qubits!r}")
    nq = int(n_qubits)
    if nq < 1:
        raise ValueError(f"n_qubits must be >= 1, got {nq}")

    dim = 1 << nq
    rest_dim = 1 << (nq - 1)

    arr = np.asarray(state_or_rho, dtype=np.complex128)
    if not np.all(np.isfinite(arr)):
        raise ValueError("Input array contains non-finite entries")

    if batched is None:
        if arr.shape == (dim,):
            psi_mat = arr.reshape(2, rest_dim)
            out = psi_mat @ psi_mat.conj().T
        elif arr.shape == (dim, dim):
            if nq == 1:
                out = arr.copy()
            else:
                out = np.trace(arr.reshape(2, rest_dim, 2, rest_dim), axis1=1, axis2=3)
        elif arr.ndim == 2 and arr.shape[1] == dim and arr.shape[0] != dim and arr.shape[0] >= 1:
            batch = arr.shape[0]
            psi_mat = arr.reshape(batch, 2, rest_dim)
            out = np.einsum("bij,bkj->bik", psi_mat, psi_mat.conj())
        elif arr.ndim == 3 and arr.shape[1:] == (dim, dim) and arr.shape[0] >= 1:
            batch = arr.shape[0]
            if nq == 1:
                out = arr.copy()
            else:
                out = np.trace(
                    arr.reshape(batch, 2, rest_dim, 2, rest_dim),
                    axis1=2,
                    axis2=4,
                )
        else:
            raise ValueError(
                f"Cannot disambiguate/accept shape {arr.shape} for n_qubits={nq} (dim={dim})"
            )
    elif batched is True:
        if arr.ndim == 2 and arr.shape[1] == dim and arr.shape[0] >= 1:
            batch = arr.shape[0]
            psi_mat = arr.reshape(batch, 2, rest_dim)
            out = np.einsum("bij,bkj->bik", psi_mat, psi_mat.conj())
        elif arr.ndim == 3 and arr.shape[1:] == (dim, dim) and arr.shape[0] >= 1:
            batch = arr.shape[0]
            if nq == 1:
                out = arr.copy()
            else:
                out = np.trace(
                    arr.reshape(batch, 2, rest_dim, 2, rest_dim),
                    axis1=2,
                    axis2=4,
                )
        else:
            raise ValueError(
                f"Expected batched shape (B, {dim}) or (B, {dim}, {dim}), got {arr.shape}"
            )
    elif batched is False:
        if arr.shape == (dim,):
            psi_mat = arr.reshape(2, rest_dim)
            out = psi_mat @ psi_mat.conj().T
        elif arr.shape == (dim, dim):
            if nq == 1:
                out = arr.copy()
            else:
                out = np.trace(arr.reshape(2, rest_dim, 2, rest_dim), axis1=1, axis2=3)
        else:
            raise ValueError(
                f"Expected single shape ({dim},) or ({dim}, {dim}), got {arr.shape}"
            )
    else:
        raise ValueError(f"batched must be None, True, or False, got {batched!r}")

    out = 0.5 * (out + out.swapaxes(-1, -2).conj())
    return np.asarray(out, dtype=np.complex128)


def partial_trace_batch_to_qubit0(states_or_rhos: np.ndarray, n_qubits: int) -> np.ndarray:
    """Explicitly batched partial trace to qubit 0 (handles B == 2^n unambiguously)."""
    return partial_trace_to_qubit0(states_or_rhos, n_qubits, batched=True)
