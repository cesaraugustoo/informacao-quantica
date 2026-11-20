"""Swappable QECC simulation backends and backend registry."""

from __future__ import annotations

import importlib.util
from typing import Any

from qecc.backends.base import GateName, GateOp, QECCBackend, QECCCode, SimulationResult
from qecc.backends.eigenkit_adapter import EigenkitUnitaryAdapter, translate_ops_to_eigenkit
from qecc.backends.numpy_backend import NumPyCPUBackend
from qecc.backends.torch_cuda_backend import TorchCUDABackend


def list_available_backends() -> list[str]:
    """Return names of backends available in the current environment."""
    backends = ["numpy"]
    if importlib.util.find_spec("torch") is not None:
        import torch

        if torch.cuda.is_available():
            backends.append("torch_cuda")
    if importlib.util.find_spec("eigenkit") is not None:
        backends.append("eigenkit_unitary")
    return backends


def get_backend(name: str, **kwargs: Any) -> QECCBackend:
    """Instantiate a simulation backend by name."""
    if name == "numpy":
        return NumPyCPUBackend(**kwargs)
    if name == "torch_cuda":
        return TorchCUDABackend(**kwargs)
    if name == "eigenkit_unitary":
        return EigenkitUnitaryAdapter(**kwargs)
    raise ValueError(
        f"Unknown backend {name!r}. Available backends: {list_available_backends()}"
    )


__all__ = [
    "EigenkitUnitaryAdapter",
    "GateName",
    "GateOp",
    "NumPyCPUBackend",
    "QECCBackend",
    "QECCCode",
    "SimulationResult",
    "TorchCUDABackend",
    "get_backend",
    "list_available_backends",
    "translate_ops_to_eigenkit",
]
