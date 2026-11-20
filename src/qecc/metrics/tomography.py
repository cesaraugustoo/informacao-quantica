"""Single-qubit Pauli-basis quantum state tomography with Smolin PSD projection."""

from __future__ import annotations

import numpy as np

from qecc.metrics.fidelity import _to_density_matrix

_PAULI_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
_PAULI_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
_PAULI_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
_EYE_2 = np.eye(2, dtype=np.complex128)


class SingleQubitTomography:
    """Simulate X/Y/Z Pauli tomography and reconstruct a 2x2 density matrix."""

    @staticmethod
    def reconstruct_from_expectations(
        rx: float,
        ry: float,
        rz: float,
        *,
        project_physical: bool = True,
    ) -> np.ndarray:
        vec = np.array([float(rx), float(ry), float(rz)], dtype=np.float64)
        if not np.all(np.isfinite(vec)):
            raise ValueError("Expectation values must be finite")
        if project_physical:
            norm = float(np.linalg.norm(vec))
            if norm > 1.0:
                vec = vec / norm
        rho_hat = 0.5 * (
            _EYE_2 + vec[0] * _PAULI_X + vec[1] * _PAULI_Y + vec[2] * _PAULI_Z
        )
        rho_hat = 0.5 * (rho_hat + rho_hat.conj().T)
        return np.asarray(rho_hat, dtype=np.complex128)

    @staticmethod
    def reconstruct(
        states_or_rhos: np.ndarray,
        shots_per_axis: int,
        rng: np.random.Generator | None = None,
        *,
        project_physical: bool = True,
        is_statevector_batch: bool = False,
    ) -> np.ndarray:
        if isinstance(shots_per_axis, bool) or not isinstance(shots_per_axis, (int, np.integer)):
            raise ValueError(f"shots_per_axis must be an int >= 1, got {shots_per_axis!r}")
        n_shots = int(shots_per_axis)
        if n_shots < 1:
            raise ValueError(f"shots_per_axis must be >= 1, got {n_shots}")

        if rng is None:
            rng = np.random.default_rng()
        elif not isinstance(rng, np.random.Generator):
            raise TypeError(f"rng must be an np.random.Generator or None, got {type(rng).__name__}")

        arr = np.asarray(states_or_rhos, dtype=np.complex128)
        if not np.all(np.isfinite(arr)):
            raise ValueError("states_or_rhos contains non-finite values")

        if arr.shape == (2,):
            rho_ens = _to_density_matrix(arr, tol=1e-6)
        elif arr.shape == (2, 2) and not is_statevector_batch:
            rho_ens = _to_density_matrix(arr, tol=1e-6)
        elif arr.ndim == 2 and arr.shape[1] == 2 and arr.shape[0] >= 1:
            norms = np.linalg.norm(arr, axis=1)
            if np.any(np.abs(norms - 1.0) > 1e-6):
                raise ValueError("All statevectors in batch must be unit-normalized")
            arr_normed = arr / norms[:, None]
            rho_ens = (arr_normed.T @ arr_normed.conj()) / float(arr_normed.shape[0])
            rho_ens = _to_density_matrix(rho_ens, tol=1e-6)
        elif arr.ndim == 3 and arr.shape[1:] == (2, 2) and arr.shape[0] >= 1:
            normalized_rhos = [
                _to_density_matrix(arr[idx], tol=1e-6) for idx in range(arr.shape[0])
            ]
            rho_ens = np.mean(normalized_rhos, axis=0)
            rho_ens = _to_density_matrix(rho_ens, tol=1e-6)
        else:
            raise ValueError(
                f"Unsupported states_or_rhos shape {arr.shape} (is_statevector_batch={is_statevector_batch})"
            )

        r_hats: list[float] = []
        for pauli in (_PAULI_X, _PAULI_Y, _PAULI_Z):
            exp_val = float(np.real(np.trace(rho_ens @ pauli)))
            p_plus = float(np.clip(0.5 * (1.0 + exp_val), 0.0, 1.0))
            k_plus = int(rng.binomial(n_shots, p_plus))
            r_hats.append((2.0 * k_plus) / float(n_shots) - 1.0)

        return SingleQubitTomography.reconstruct_from_expectations(
            r_hats[0],
            r_hats[1],
            r_hats[2],
            project_physical=project_physical,
        )
