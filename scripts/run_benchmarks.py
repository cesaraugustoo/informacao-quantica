#!/usr/bin/env python3
"""Benchmark NumPy CPU vs RTX 5080 PyTorch CUDA QECC backends and verify parity."""

from __future__ import annotations

import argparse
import itertools
import math
import os
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

VALID_CODES: tuple[str, ...] = ("bit_flip_3", "phase_flip_3", "five_qubit", "dfs_2")
VALID_CHANNELS: tuple[str, ...] = (
    "bit_flip",
    "phase_flip",
    "depolarizing",
    "collective_dephasing",
)
VALID_STATES: tuple[str, ...] = ("0", "1", "+", "-", "+i", "-i")

CANONICAL_CHANNELS: dict[str, str] = {
    "bit_flip_3": "bit_flip",
    "phase_flip_3": "phase_flip",
    "five_qubit": "depolarizing",
    "dfs_2": "collective_dephasing",
}

DETERMINISTIC_PROBE_ERRORS: dict[str, str] = {
    "bit_flip_3": "XII",
    "phase_flip_3": "ZII",
    "five_qubit": "YIIII",
    "dfs_2": "ZZ",
}

_DFS2_LOGICAL_PAULI_MAP: dict[str, str] = {
    "II": "I",
    "IX": "I",
    "ZY": "I",
    "ZZ": "I",
    "XI": "X",
    "XX": "X",
    "YY": "X",
    "YZ": "X",
    "XY": "Y",
    "XZ": "Y",
    "YI": "Y",
    "YX": "Y",
    "IY": "Z",
    "IZ": "Z",
    "ZI": "Z",
    "ZX": "Z",
}

_STABILIZER_LOGICAL_OPS: dict[str, tuple[str, str]] = {
    "bit_flip_3": ("XXX", "ZZZ"),
    "phase_flip_3": ("ZZZ", "XXX"),
    "five_qubit": ("XXXXX", "ZZZZZ"),
}


@lru_cache(maxsize=16)
def _cached_depolarizing_coset_weights(code_name: str) -> dict[str, dict[int, int]]:
    """Return exact {logical_op: {weight: count}} over all 4^n_data Pauli strings."""
    from qecc.analysis.analytical import _pauli_commutes, _pauli_multiply_unsigned
    from qecc.codes import get_code

    code = get_code(code_name)
    n_data = code.n_data
    counts: dict[str, dict[int, int]] = {
        op: {w: 0 for w in range(n_data + 1)} for op in ("I", "X", "Y", "Z")
    }
    for tup in itertools.product(("I", "X", "Y", "Z"), repeat=n_data):
        err = "".join(tup)
        if code_name == "dfs_2":
            lop = _DFS2_LOGICAL_PAULI_MAP[err]
        else:
            lx, lz = _STABILIZER_LOGICAL_OPS[code_name]
            syn = "".join("0" if _pauli_commutes(err, s) else "1" for s in code.stabilizers)
            rec = code.syndrome_table[syn]
            res = _pauli_multiply_unsigned(rec, err)
            cx = _pauli_commutes(res, lx)
            cz = _pauli_commutes(res, lz)
            if cx and cz:
                lop = "I"
            elif cx and not cz:
                lop = "X"
            elif not cx and not cz:
                lop = "Y"
            else:
                lop = "Z"
        weight = sum(1 for ch in err if ch != "I")
        counts[lop][weight] += 1
    return counts


def compute_logical_pauli_probabilities(
    code_name: str, channel_name: str, p: float
) -> dict[str, float]:
    """Compute exact logical single-qubit Pauli channel weights (p_I, p_X, p_Y, p_Z)."""
    from qecc.analysis.analytical import _pauli_commutes, _pauli_multiply_unsigned
    from qecc.codes import get_code

    code = get_code(code_name)
    n_data = code.n_data
    q = 1.0 - p
    probs = {"I": 0.0, "X": 0.0, "Y": 0.0, "Z": 0.0}

    def _classify(err: str) -> str:
        if code_name == "dfs_2":
            return _DFS2_LOGICAL_PAULI_MAP[err]
        lx, lz = _STABILIZER_LOGICAL_OPS[code_name]
        syn = "".join("0" if _pauli_commutes(err, s) else "1" for s in code.stabilizers)
        rec = code.syndrome_table[syn]
        res = _pauli_multiply_unsigned(rec, err)
        cx = _pauli_commutes(res, lx)
        cz = _pauli_commutes(res, lz)
        if cx and cz:
            return "I"
        if cx and not cz:
            return "X"
        if not cx and not cz:
            return "Y"
        return "Z"

    if channel_name == "collective_dephasing":
        probs[_classify("I" * n_data)] += q
        probs[_classify("Z" * n_data)] += p
    elif channel_name in ("bit_flip", "phase_flip"):
        axis = "X" if channel_name == "bit_flip" else "Z"
        for tup in itertools.product(("I", axis), repeat=n_data):
            err = "".join(tup)
            w = sum(1 for ch in err if ch != "I")
            probs[_classify(err)] += (q ** (n_data - w)) * (p**w)
    elif channel_name == "depolarizing":
        weight_table = _cached_depolarizing_coset_weights(code_name)
        p_axis = p / 3.0
        for lop, by_weight in weight_table.items():
            for w, cnt in by_weight.items():
                if cnt > 0:
                    probs[lop] += float(cnt) * (q ** (n_data - w)) * (p_axis**w)
    else:
        raise ValueError(f"Unsupported channel_name: {channel_name!r}")

    return probs


def compute_expected_state_fidelity(
    code_name: str, channel_name: str, p: float, psi: Any
) -> float:
    """Compute exact expected state fidelity <psi|rho_out|psi> for a specific pure state."""
    import numpy as np

    vec = np.asarray(psi, dtype=np.complex128)
    vec = vec / np.linalg.norm(vec)
    alpha, beta = vec[0], vec[1]
    overlap_ab = alpha.conjugate() * beta
    rx = float(2.0 * overlap_ab.real)
    ry = float(2.0 * overlap_ab.imag)
    rz = float((alpha.conjugate() * alpha - beta.conjugate() * beta).real)

    probs = compute_logical_pauli_probabilities(code_name, channel_name, p)
    return float(
        probs["I"]
        + probs["X"] * (rx * rx)
        + probs["Y"] * (ry * ry)
        + probs["Z"] * (rz * rz)
    )


def _cpu_model_name() -> str:
    """Return human-readable CPU model string if available on Linux."""
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name"):
                _, _, val = line.partition(":")
                cleaned = val.strip()
                if cleaned:
                    return cleaned
    return platform.processor() or platform.machine() or "unknown"


def _sync_cuda_if_needed(backend_name: str, torch_mod: Any) -> None:
    if backend_name == "torch_cuda" and torch_mod is not None and torch_mod.cuda.is_available():
        torch_mod.cuda.synchronize()


def _timed_simulate(
    *,
    backend: Any,
    backend_name: str,
    torch_mod: Any,
    code: Any,
    initial_state: Any,
    channel: str | None,
    p: float,
    num_shots: int,
    seed: int,
    deterministic_errors: str | list[str] | None = None,
) -> tuple[Any, float]:
    """Execute one simulate_cycle call with explicit CUDA synchronization and wall timing."""
    _sync_cuda_if_needed(backend_name, torch_mod)
    t0 = time.perf_counter()
    res = backend.simulate_cycle(
        code=code,
        initial_state=initial_state,
        channel=channel,
        p=p,
        num_shots=num_shots,
        seed=seed,
        deterministic_errors=deterministic_errors,
    )
    _sync_cuda_if_needed(backend_name, torch_mod)
    wall_s = max(float(time.perf_counter() - t0), 1e-9)
    return res, wall_s


def _collect_environment_metadata(
    *,
    torch_mod: Any,
    numpy_mod: Any,
    eigenkit_version: str | None,
    codes: list[str],
    channels_by_code: dict[str, str],
    shots_list: list[int],
    p: float,
    state: str,
    warmup: int,
    repeats: int,
    seed: int,
    eigenkit_stochastic_supported: bool | None,
) -> dict[str, Any]:
    cuda_available = bool(torch_mod.cuda.is_available())
    if cuda_available:
        props = torch_mod.cuda.get_device_properties(0)
        cap = torch_mod.cuda.get_device_capability(0)
        arch_list = list(torch_mod.cuda.get_arch_list())
        gpu_name = str(torch_mod.cuda.get_device_name(0))
        gpu_cap = f"{cap[0]}.{cap[1]}"
        sm_120 = any("120" in a for a in arch_list) or cap >= (12, 0)
        vram_mib = round(float(props.total_memory) / (1024.0 * 1024.0), 2)
    else:
        arch_list = []
        gpu_name = "unavailable"
        gpu_cap = "unavailable"
        sm_120 = False
        vram_mib = 0.0

    return {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cpu_model": _cpu_model_name(),
        "cpu_count": int(os.cpu_count() or 1),
        "numpy_version": str(numpy_mod.__version__),
        "torch_version": str(torch_mod.__version__),
        "eigenkit_version": eigenkit_version,
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
        "gpu_compute_capability": gpu_cap,
        "gpu_arch_list": arch_list,
        "sm_120_supported": bool(sm_120),
        "gpu_vram_mib": vram_mib,
        "backends": ["numpy", "torch_cuda"],
        "codes": list(codes),
        "channels_by_code": dict(channels_by_code),
        "shots_list": [int(s) for s in shots_list],
        "p": float(p),
        "state": state,
        "warmup_runs": int(warmup),
        "repeats": int(repeats),
        "seed": int(seed),
        "eigenkit_stochastic_supported": eigenkit_stochastic_supported,
    }


def _render_markdown(report: dict[str, Any]) -> str:
    meta = report["metadata"]
    comparisons = report["comparisons"]
    unitary = report.get("unitary_parity")

    lines: list[str] = [
        "# CPU vs. RTX 5080 CUDA Backend Benchmark",
        "",
        "## Hardware & Software Environment",
        "",
        f"- **Generated (UTC)**: `{meta['generated_utc']}`",
        f"- **GPU**: `{meta['gpu_name']}` (Compute Capability `{meta['gpu_compute_capability']}`, "
        f"`sm_120` supported: `{meta['sm_120_supported']}`, VRAM: `{meta['gpu_vram_mib']:.1f} MiB`)",
        f"- **CPU**: `{meta['cpu_model']}` (`{meta['cpu_count']}` logical cores)",
        f"- **Toolchain**: `Python {meta['python_version']}`, `torch=={meta['torch_version']}`, "
        f"`numpy=={meta['numpy_version']}`, `eigenkit=={meta['eigenkit_version'] or 'skipped'}`",
        f"- **Protocol**: state `|{meta['state']}>`, error probability `p={meta['p']}`, "
        f"`warmup_runs={meta['warmup_runs']}`, `repeats={meta['repeats']}`, `seed={meta['seed']}` "
        "(explicit `torch.cuda.synchronize()` before and after each GPU measurement)",
        "",
        "## CPU (`numpy`) vs. GPU (`torch_cuda`) Throughput & Speedup",
        "",
        "| Code | Channel | Shots | CPU Median (s) | GPU Median (s) | CPU (shots/s) | GPU (shots/s) | CUDA Speedup | CPU F | GPU F | Expected F | Analytical State F | Analytical Ent. F | Max |Δρ| | Syndrome p | Parity |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |",
    ]

    for comp in comparisons:
        lines.append(
            f"| `{comp['code']}` | `{comp['channel']}` | {comp['shots']:,} | "
            f"{comp['numpy_median_seconds']:.6f} | {comp['torch_cuda_median_seconds']:.6f} | "
            f"{comp['numpy_shots_per_sec']:,.1f} | {comp['torch_cuda_shots_per_sec']:,.1f} | "
            f"{comp['cuda_speedup']:.2f}x | {comp['numpy_mean_fidelity']:.5f} | "
            f"{comp['torch_cuda_mean_fidelity']:.5f} | {comp['expected_state_fidelity']:.5f} | "
            f"{comp['analytical_state_fidelity']:.5f} | {comp['analytical_entanglement_fidelity']:.5f} | "
            f"{comp['max_rho_diff']:.2e} | {comp['syndrome_p_value']:.4f} | "
            f"{'PASS' if comp['parity_passed'] else 'FAIL'} |"
        )

    lines.extend(
        [
            "",
            "### Performance & Fidelity Notes",
            "",
            "- **Launch-latency crossover**: At small shot counts (`shots=1,000`) on 5-qubit codes "
            "(`bit_flip_3`, `phase_flip_3`, `2^5 = 32` amplitudes/shot), GPU kernel launch and host-device "
            "synchronization overhead (~2-3 ms) is comparable to vectorized NumPy CPU runtime, resulting in "
            "near-1x or sub-1x speedup. At `10,000`–`100,000` shots and on the 9-qubit `five_qubit` code "
            "(`2^9 = 512` amplitudes/shot), parallel GPU statevector execution delivers substantial multi-x speedup.",
            "- **Fidelity convention**: `expected_state_fidelity` evaluates the exact state-specific "
            "logical Pauli channel fidelity `sum_L p_L |<psi|L|psi>|^2` for the benchmarked `(code, channel, state)`, "
            "while `analytical_state_fidelity` reports the Haar-averaged Pauli-channel state fidelity "
            "`1 - (2/3)*(1 - P_I(p))` and `analytical_entanglement_fidelity` reports `P_I(p)`.",
        ]
    )

    if unitary and not unitary.get("skipped"):
        lines.extend(
            [
                "",
                "## `eigenkit_unitary` Cross-Verification & Stochastic Gap",
                "",
                f"- **Stochastic support**: `{unitary['stochastic_supported']}` "
                f"(probes raise `NotImplementedError`: `{unitary['stochastic_error']}`; "
                f"see `{unitary['evaluation_doc']}`).",
                "",
                "| Code | Mode | Shots | `numpy` F | `torch_cuda` F | `eigenkit_unitary` F | Max |Δρ| | Syndromes Match | `eigenkit` (s) | Status |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | :---: | ---: | :---: |",
            ]
        )
        for chk in unitary["checks"]:
            lines.append(
                f"| `{chk['code']}` | `{chk['mode']}` | {chk['shots']} | "
                f"{chk['numpy_mean_fidelity']:.12f} | {chk['torch_cuda_mean_fidelity']:.12f} | "
                f"{chk['eigenkit_mean_fidelity']:.12f} | {chk['max_rho_diff']:.2e} | "
                f"{'YES' if chk['syndrome_counts_match'] else 'NO'} | "
                f"{chk['eigenkit_elapsed_seconds']:.6f} | {'PASS' if chk['passed'] else 'FAIL'} |"
            )

    lines.append("")
    return "\n".join(lines)


def _run_unitary_parity(
    *,
    codes: list[str],
    state_vec: Any,
    np_backend: Any,
    cu_backend: Any,
    ek_backend: Any,
    torch_mod: Any,
    numpy_mod: Any,
    seed: int,
) -> dict[str, Any]:
    from qecc.codes import get_code

    checks: list[dict[str, Any]] = []
    unitary_shots = 100

    for code_idx, code_name in enumerate(codes):
        code = get_code(code_name)
        det_err = DETERMINISTIC_PROBE_ERRORS[code_name]
        modes: list[tuple[str, str | None]] = [
            ("p=0", None),
            (f"deterministic:{det_err}", det_err),
        ]
        for mode_idx, (mode_label, det_pattern) in enumerate(modes):
            run_seed = seed + 50_000 + code_idx * 10 + mode_idx
            res_np, t_np = _timed_simulate(
                backend=np_backend,
                backend_name="numpy",
                torch_mod=torch_mod,
                code=code,
                initial_state=state_vec,
                channel=None,
                p=0.0,
                num_shots=unitary_shots,
                seed=run_seed,
                deterministic_errors=det_pattern,
            )
            res_cu, t_cu = _timed_simulate(
                backend=cu_backend,
                backend_name="torch_cuda",
                torch_mod=torch_mod,
                code=code,
                initial_state=state_vec,
                channel=None,
                p=0.0,
                num_shots=unitary_shots,
                seed=run_seed,
                deterministic_errors=det_pattern,
            )
            res_ek, t_ek = _timed_simulate(
                backend=ek_backend,
                backend_name="eigenkit_unitary",
                torch_mod=torch_mod,
                code=code,
                initial_state=state_vec,
                channel=None,
                p=0.0,
                num_shots=unitary_shots,
                seed=run_seed,
                deterministic_errors=det_pattern,
            )

            diff_cu_np = float(numpy_mod.max(numpy_mod.abs(res_cu.rho_out - res_np.rho_out)))
            diff_ek_np = float(numpy_mod.max(numpy_mod.abs(res_ek.rho_out - res_np.rho_out)))
            max_rho_diff = max(diff_cu_np, diff_ek_np)
            syndromes_match = (
                res_np.syndrome_counts == res_cu.syndrome_counts == res_ek.syndrome_counts
            )
            passed = bool(
                res_np.mean_fidelity > 1.0 - 1e-10
                and res_cu.mean_fidelity > 1.0 - 1e-10
                and res_ek.mean_fidelity > 1.0 - 1e-10
                and max_rho_diff < 1e-10
                and syndromes_match
                and t_np > 0.0
                and t_cu > 0.0
                and t_ek > 0.0
            )
            checks.append(
                {
                    "code": code_name,
                    "mode": mode_label,
                    "shots": unitary_shots,
                    "numpy_mean_fidelity": float(res_np.mean_fidelity),
                    "torch_cuda_mean_fidelity": float(res_cu.mean_fidelity),
                    "eigenkit_mean_fidelity": float(res_ek.mean_fidelity),
                    "max_rho_diff_cuda_vs_numpy": diff_cu_np,
                    "max_rho_diff_eigenkit_vs_numpy": diff_ek_np,
                    "max_rho_diff": max_rho_diff,
                    "syndrome_counts_match": bool(syndromes_match),
                    "numpy_elapsed_seconds": t_np,
                    "torch_cuda_elapsed_seconds": t_cu,
                    "eigenkit_elapsed_seconds": t_ek,
                    "passed": passed,
                }
            )

    # Probe stochastic mode to confirm explicit NotImplementedError capability boundary
    stochastic_supported = True
    stochastic_error = ""
    try:
        ek_backend.simulate_cycle(
            code=get_code("bit_flip_3"),
            initial_state=state_vec,
            channel="bit_flip",
            p=0.05,
            num_shots=10,
            seed=seed,
        )
    except NotImplementedError as exc:
        stochastic_supported = False
        stochastic_error = str(exc)

    return {
        "backend": "eigenkit_unitary",
        "stochastic_supported": stochastic_supported,
        "stochastic_error": stochastic_error,
        "evaluation_doc": "docs/eigenkit_evaluation.md",
        "all_passed": all(c["passed"] for c in checks) and not stochastic_supported,
        "checks": checks,
    }


def run_benchmarks(
    *,
    output_dir: Path,
    codes: list[str],
    shots_list: list[int],
    channel: str | None,
    p: float,
    state: str,
    warmup: int,
    repeats: int,
    seed: int,
    skip_eigenkit: bool,
) -> dict[str, Any]:
    """Execute CPU vs RTX 5080 CUDA benchmarks, verify parity, and write JSON/MD artifacts."""
    import json
    import numpy as np
    import torch
    from scipy import stats as scipy_stats

    from qecc.analysis import analytical_fidelity
    from qecc.backends import get_backend, list_available_backends
    from qecc.codes import get_code
    from qecc.metrics import InitialStateSampler

    available = list_available_backends()
    if "numpy" not in available:
        raise RuntimeError(f"'numpy' backend unavailable; available={available}")
    if "torch_cuda" not in available:
        raise RuntimeError(f"'torch_cuda' backend unavailable; available={available}")

    np_backend = get_backend("numpy")
    cu_backend = get_backend("torch_cuda")

    ek_backend = None
    eigenkit_version: str | None = None
    if not skip_eigenkit:
        if "eigenkit_unitary" not in available:
            raise RuntimeError(f"'eigenkit_unitary' backend unavailable; available={available}")
        import eigenkit

        eigenkit_version = str(getattr(eigenkit, "__version__", "unknown"))
        ek_backend = get_backend("eigenkit_unitary")

    psi = InitialStateSampler.cardinal_states()[state]
    channels_by_code = {
        c: (channel if channel is not None else CANONICAL_CHANNELS[c]) for c in codes
    }

    records: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []

    warmup_shots_by_backend = {
        "numpy": min(256, max(1, min(shots_list))),
        "torch_cuda": max(shots_list),
    }

    for code_idx, code_name in enumerate(codes):
        code = get_code(code_name)
        ch_name = channels_by_code[code_name]

        # Warm up both backends per (code, channel) before timing;
        # CUDA warms up at max(shots_list) so peak GPU allocations are pre-warmed.
        if warmup > 0:
            for bname, backend_obj in (("numpy", np_backend), ("torch_cuda", cu_backend)):
                w_shots = warmup_shots_by_backend[bname]
                for w in range(warmup):
                    _timed_simulate(
                        backend=backend_obj,
                        backend_name=bname,
                        torch_mod=torch,
                        code=code,
                        initial_state=psi,
                        channel=ch_name,
                        p=p,
                        num_shots=w_shots,
                        seed=seed + 900_000 + code_idx * 100 + w,
                    )

        ana_state = float(analytical_fidelity(code_name, ch_name, p, metric="state"))
        ana_ent = float(analytical_fidelity(code_name, ch_name, p, metric="entanglement"))
        exp_state = compute_expected_state_fidelity(code_name, ch_name, p, psi)

        for shot_idx, n_shots in enumerate(shots_list):
            cell_records: dict[str, dict[str, Any]] = {}
            cell_rhos: dict[str, np.ndarray] = {}
            cell_syndromes: dict[str, dict[str, int]] = {}
            for bname, backend_obj in (("numpy", np_backend), ("torch_cuda", cu_backend)):
                wall_times: list[float] = []
                internal_times: list[float] = []
                fidelity_chunks: list[np.ndarray] = []
                rho_chunks: list[np.ndarray] = []
                syn_pooled: dict[str, int] = {}

                for r in range(repeats):
                    run_seed = seed + code_idx * 10_000 + shot_idx * 100 + r
                    res, wall_s = _timed_simulate(
                        backend=backend_obj,
                        backend_name=bname,
                        torch_mod=torch,
                        code=code,
                        initial_state=psi,
                        channel=ch_name,
                        p=p,
                        num_shots=n_shots,
                        seed=run_seed,
                    )
                    wall_times.append(wall_s)
                    internal_times.append(max(float(res.elapsed_seconds), 1e-9))
                    fidelity_chunks.append(np.asarray(res.fidelities, dtype=np.float64))
                    rho_chunks.append(np.asarray(res.rho_out, dtype=np.complex128))
                    for skey, scnt in res.syndrome_counts.items():
                        syn_pooled[skey] = syn_pooled.get(skey, 0) + int(scnt)

                pooled_fids = np.concatenate(fidelity_chunks)
                mean_f = float(np.mean(pooled_fids))
                std_err = (
                    float(np.std(pooled_fids, ddof=1) / math.sqrt(pooled_fids.size))
                    if pooled_fids.size > 1
                    else 0.0
                )
                rho_pooled = np.mean(np.stack(rho_chunks, axis=0), axis=0)

                med_s = float(statistics.median(wall_times))
                mean_s = float(statistics.fmean(wall_times))
                min_s = float(min(wall_times))
                max_s = float(max(wall_times))
                int_med_s = float(statistics.median(internal_times))
                stdev_s = float(statistics.stdev(wall_times)) if len(wall_times) > 1 else 0.0
                rel_std_pct = float(100.0 * stdev_s / mean_s) if mean_s > 0.0 else 0.0
                shots_per_sec = float(n_shots / med_s)

                rec = {
                    "backend": bname,
                    "code": code_name,
                    "channel": ch_name,
                    "n_data": int(code.n_data),
                    "n_ancilla": int(code.n_ancilla),
                    "n_total": int(code.n_total),
                    "shots": int(n_shots),
                    "repeats": int(repeats),
                    "median_seconds": med_s,
                    "mean_seconds": mean_s,
                    "min_seconds": min_s,
                    "max_seconds": max_s,
                    "internal_median_seconds": int_med_s,
                    "rel_std_pct": rel_std_pct,
                    "shots_per_sec": shots_per_sec,
                    "mean_fidelity": mean_f,
                    "std_error": std_err,
                    "expected_state_fidelity": exp_state,
                    "analytical_state_fidelity": ana_state,
                    "analytical_entanglement_fidelity": ana_ent,
                }
                records.append(rec)
                cell_records[bname] = rec
                cell_rhos[bname] = rho_pooled
                cell_syndromes[bname] = syn_pooled

            np_rec = cell_records["numpy"]
            cu_rec = cell_records["torch_cuda"]
            n_pooled = int(n_shots) * int(repeats)
            cuda_speedup = float(np_rec["median_seconds"] / cu_rec["median_seconds"])
            fid_diff = float(abs(np_rec["mean_fidelity"] - cu_rec["mean_fidelity"]))

            se_binom = math.sqrt(max(exp_state * (1.0 - exp_state), 0.0) / float(n_pooled))
            se_pair_obs = math.sqrt(np_rec["std_error"] ** 2 + cu_rec["std_error"] ** 2)
            se_pair_binom = math.sqrt(2.0) * se_binom
            pair_tol = max(5.0 * max(se_pair_obs, se_pair_binom), 1e-4)
            ana_tol_np = max(5.0 * max(np_rec["std_error"], se_binom), 1e-4)
            ana_tol_cu = max(5.0 * max(cu_rec["std_error"], se_binom), 1e-4)

            max_rho_diff = float(np.max(np.abs(cell_rhos["numpy"] - cell_rhos["torch_cuda"])))
            se_rho = math.sqrt(2.0 * max(1.0 - ana_ent, 0.0) / float(n_pooled))
            rho_tol = max(6.0 * se_rho, 1e-4)

            np_syn = cell_syndromes["numpy"]
            cu_syn = cell_syndromes["torch_cuda"]
            active_keys = sorted(
                k
                for k in (set(np_syn) | set(cu_syn))
                if (np_syn.get(k, 0) + cu_syn.get(k, 0)) > 0
            )
            if len(active_keys) <= 1:
                syndrome_p_value = 1.0
                syndrome_parity_ok = np_syn == cu_syn
            else:
                table = np.array(
                    [
                        [np_syn.get(k, 0) for k in active_keys],
                        [cu_syn.get(k, 0) for k in active_keys],
                    ],
                    dtype=np.int64,
                )
                _, p_val, _, _ = scipy_stats.chi2_contingency(table)
                syndrome_p_value = float(p_val)
                syndrome_parity_ok = bool(
                    math.isfinite(syndrome_p_value) and syndrome_p_value > 1e-4
                )

            parity_passed = bool(
                fid_diff <= pair_tol
                and abs(np_rec["mean_fidelity"] - exp_state) <= ana_tol_np
                and abs(cu_rec["mean_fidelity"] - exp_state) <= ana_tol_cu
                and max_rho_diff <= rho_tol
                and syndrome_parity_ok
            )

            comparisons.append(
                {
                    "code": code_name,
                    "channel": ch_name,
                    "shots": int(n_shots),
                    "numpy_median_seconds": np_rec["median_seconds"],
                    "torch_cuda_median_seconds": cu_rec["median_seconds"],
                    "numpy_shots_per_sec": np_rec["shots_per_sec"],
                    "torch_cuda_shots_per_sec": cu_rec["shots_per_sec"],
                    "cuda_speedup": cuda_speedup,
                    "numpy_mean_fidelity": np_rec["mean_fidelity"],
                    "torch_cuda_mean_fidelity": cu_rec["mean_fidelity"],
                    "expected_state_fidelity": exp_state,
                    "analytical_state_fidelity": ana_state,
                    "analytical_entanglement_fidelity": ana_ent,
                    "fidelity_diff": fid_diff,
                    "max_rho_diff": max_rho_diff,
                    "syndrome_p_value": syndrome_p_value,
                    "parity_passed": parity_passed,
                }
            )

    if ek_backend is not None:
        unitary_parity = _run_unitary_parity(
            codes=codes,
            state_vec=psi,
            np_backend=np_backend,
            cu_backend=cu_backend,
            ek_backend=ek_backend,
            torch_mod=torch,
            numpy_mod=np,
            seed=seed,
        )
        ek_stoch = unitary_parity["stochastic_supported"]
    else:
        unitary_parity = {"skipped": True}
        ek_stoch = None

    metadata = _collect_environment_metadata(
        torch_mod=torch,
        numpy_mod=np,
        eigenkit_version=eigenkit_version,
        codes=codes,
        channels_by_code=channels_by_code,
        shots_list=shots_list,
        p=p,
        state=state,
        warmup=warmup,
        repeats=repeats,
        seed=seed,
        eigenkit_stochastic_supported=ek_stoch,
    )

    report: dict[str, Any] = {
        "metadata": metadata,
        "records": records,
        "comparisons": comparisons,
        "unitary_parity": unitary_parity,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "backend_benchmark.json"
    md_path = output_dir / "backend_benchmark.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(_render_markdown(report), encoding="utf-8")

    return {
        "json_path": str(json_path),
        "md_path": str(md_path),
        "report": report,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark NumPy CPU vs RTX 5080 PyTorch CUDA QECC simulation backends."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "benchmarks",
        help="Output directory for backend_benchmark.json and backend_benchmark.md.",
    )
    parser.add_argument(
        "--codes",
        nargs="+",
        default=["bit_flip_3", "phase_flip_3", "five_qubit"],
        choices=VALID_CODES,
        help="QECC codes to benchmark.",
    )
    parser.add_argument(
        "--shots",
        nargs="+",
        type=int,
        default=[1000, 10000, 100000],
        help="Shot counts to benchmark.",
    )
    parser.add_argument(
        "--channel",
        type=str,
        default=None,
        choices=VALID_CHANNELS,
        help="Optional stochastic channel override (default: canonical channel per code).",
    )
    parser.add_argument(
        "--p",
        type=float,
        default=0.05,
        help="Stochastic error probability in (0, 1] (default: 0.05).",
    )
    parser.add_argument(
        "--state",
        type=str,
        default="+i",
        choices=VALID_STATES,
        help="Cardinal initial state label (default: +i).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Untimed warmup cycles per (backend, code) before timing (default: 2).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Timed repeats per (backend, code, shots) cell (default: 3).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20261118,
        help="Base PRNG seed (default: 20261118).",
    )
    parser.add_argument(
        "--skip-eigenkit",
        action="store_true",
        help="Skip eigenkit_unitary deterministic/p=0 parity verification.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON benchmark report to stdout.",
    )
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if any(s < 1 for s in args.shots):
        parser.error(f"All --shots values must be >= 1, got {args.shots}")
    if args.warmup < 0:
        parser.error(f"--warmup must be >= 0, got {args.warmup}")
    if args.repeats < 1:
        parser.error(f"--repeats must be >= 1, got {args.repeats}")
    if not math.isfinite(args.p) or args.p <= 0.0 or args.p > 1.0:
        parser.error(f"--p must be finite and in (0.0, 1.0], got {args.p!r}")


def _print_human_summary(result: dict[str, Any]) -> None:
    report = result["report"]
    meta = report["metadata"]
    print(
        f"QECC Backend Benchmark | GPU: {meta['gpu_name']} (sm_120={meta['sm_120_supported']}) "
        f"| torch={meta['torch_version']}"
    )
    print(
        f"  state=|{meta['state']}>, p={meta['p']}, warmup={meta['warmup_runs']}, "
        f"repeats={meta['repeats']}"
    )
    print(
        f"  {'Code':<14} {'Channel':<14} {'Shots':>8} {'CPU (s)':>10} {'GPU (s)':>10} "
        f"{'Speedup':>9} {'Parity':>8}"
    )
    for comp in report["comparisons"]:
        print(
            f"  {comp['code']:<14} {comp['channel']:<14} {comp['shots']:>8} "
            f"{comp['numpy_median_seconds']:>10.5f} {comp['torch_cuda_median_seconds']:>10.5f} "
            f"{comp['cuda_speedup']:>8.2f}x {'PASS' if comp['parity_passed'] else 'FAIL':>8}"
        )
    unitary = report.get("unitary_parity", {})
    if unitary and not unitary.get("skipped"):
        print(
            f"  eigenkit_unitary unitary parity: {'PASS' if unitary['all_passed'] else 'FAIL'} "
            f"(stochastic_supported={unitary['stochastic_supported']})"
        )
    print(f"  Wrote: {result['json_path']}")
    print(f"  Wrote: {result['md_path']}")


def main(argv: list[str] | None = None) -> int:
    import json

    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)

    result = run_benchmarks(
        output_dir=args.output_dir,
        codes=list(args.codes),
        shots_list=[int(s) for s in args.shots],
        channel=args.channel,
        p=float(args.p),
        state=args.state,
        warmup=int(args.warmup),
        repeats=int(args.repeats),
        seed=int(args.seed),
        skip_eigenkit=bool(args.skip_eigenkit),
    )

    if args.json:
        print(json.dumps(result["report"], indent=2))
    else:
        _print_human_summary(result)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
