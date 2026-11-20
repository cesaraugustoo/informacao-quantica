"""Quantum Error Correction Codes (QECC) simulation suite."""

from qecc.analysis import analytical_fidelity, analytical_gain_threshold
from qecc.backends import (
    EigenkitUnitaryAdapter,
    GateOp,
    NumPyCPUBackend,
    QECCBackend,
    QECCCode,
    SimulationResult,
    TorchCUDABackend,
    get_backend,
    list_available_backends,
)
from qecc.channels import (
    ChannelSpec,
    DeterministicErrorInjector,
    StochasticPauliChannel,
)
from qecc.codes import (
    BitFlip3QubitCode,
    FiveQubitPerfectCode,
    PhaseFlip3QubitCode,
    TwoQubitDFSDemo,
    get_code,
    list_available_codes,
)
from qecc.metrics import (
    InitialStateSampler,
    SingleQubitTomography,
    partial_trace_batch_to_qubit0,
    partial_trace_to_qubit0,
    quantum_fidelity,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "analytical_fidelity",
    "analytical_gain_threshold",
    "BitFlip3QubitCode",
    "ChannelSpec",
    "DeterministicErrorInjector",
    "EigenkitUnitaryAdapter",
    "FiveQubitPerfectCode",
    "GateOp",
    "InitialStateSampler",
    "NumPyCPUBackend",
    "PhaseFlip3QubitCode",
    "QECCBackend",
    "QECCCode",
    "SimulationResult",
    "SingleQubitTomography",
    "StochasticPauliChannel",
    "TorchCUDABackend",
    "TwoQubitDFSDemo",
    "get_backend",
    "get_code",
    "list_available_backends",
    "list_available_codes",
    "partial_trace_batch_to_qubit0",
    "partial_trace_to_qubit0",
    "quantum_fidelity",
]
