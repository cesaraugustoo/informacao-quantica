"""Initial single-qubit state sampling utilities (cardinal and Haar-random)."""

from __future__ import annotations

import math

import numpy as np


class InitialStateSampler:
    """Provides normalized cardinal and Haar-random single-qubit states."""

    @staticmethod
    def normalize_statevector(psi: np.ndarray) -> np.ndarray:
        arr = np.asarray(psi, dtype=np.complex128)
        if arr.ndim != 1 or arr.shape[0] < 1 or not np.all(np.isfinite(arr)):
            raise ValueError("psi must be a non-empty finite 1D array")
        norm = float(np.linalg.norm(arr))
        if norm <= 1e-15:
            raise ValueError("Cannot normalize zero-norm statevector")
        return (arr / norm).astype(np.complex128)

    @staticmethod
    def cardinal_states() -> dict[str, np.ndarray]:
        inv_sqrt2 = 1.0 / math.sqrt(2.0)
        return {
            "0": np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128),
            "1": np.array([0.0 + 0.0j, 1.0 + 0.0j], dtype=np.complex128),
            "+": np.array([inv_sqrt2, inv_sqrt2], dtype=np.complex128),
            "-": np.array([inv_sqrt2, -inv_sqrt2], dtype=np.complex128),
            "+i": np.array([inv_sqrt2, 1.0j * inv_sqrt2], dtype=np.complex128),
            "-i": np.array([inv_sqrt2, -1.0j * inv_sqrt2], dtype=np.complex128),
        }

    @staticmethod
    def haar_random_states(
        n_states: int,
        seed: int | np.random.Generator | None = None,
    ) -> np.ndarray:
        if isinstance(n_states, bool) or not isinstance(n_states, (int, np.integer)):
            raise ValueError(f"n_states must be a positive integer, got {n_states!r}")
        count = int(n_states)
        if count <= 0:
            raise ValueError(f"n_states must be > 0, got {count}")

        if isinstance(seed, np.random.Generator):
            rng = seed
        elif seed is None or (
            isinstance(seed, (int, np.integer)) and not isinstance(seed, bool)
        ):
            rng = np.random.default_rng(None if seed is None else int(seed))
        else:
            raise TypeError(f"seed must be int, Generator, or None, got {type(seed).__name__}")

        z = rng.normal(size=(count, 2)) + 1j * rng.normal(size=(count, 2))
        norms = np.linalg.norm(z, axis=1, keepdims=True)
        attempts = 0
        while np.any(norms <= 1e-15):
            attempts += 1
            if attempts > 16:
                raise ValueError("RNG repeatedly produced zero-norm complex Gaussian vectors")
            mask = (norms <= 1e-15).ravel()
            n_bad = int(np.sum(mask))
            z[mask] = rng.normal(size=(n_bad, 2)) + 1j * rng.normal(size=(n_bad, 2))
            norms = np.linalg.norm(z, axis=1, keepdims=True)

        return (z / norms).astype(np.complex128)
