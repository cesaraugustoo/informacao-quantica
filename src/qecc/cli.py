"""Command-line interface for the QECC simulation and analysis suite."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Sequence


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_backend_name(requested: str) -> str:
    from qecc.backends import list_available_backends

    avail = list_available_backends()
    if requested == "auto":
        return "torch_cuda" if "torch_cuda" in avail else "numpy"
    if requested not in avail:
        raise ValueError(f"Requested backend {requested!r} not in available backends {avail}")
    return requested


def _cmd_info(args: argparse.Namespace) -> int:
    from qecc.analysis import VALID_CHANNELS, analytical_gain_threshold
    from qecc.backends import list_available_backends
    from qecc.codes import list_available_codes

    codes = list_available_codes()
    backends = list_available_backends()
    thresholds = {
        f"{code}/{ch}": analytical_gain_threshold(code, ch)
        for code in ("bit_flip_3", "phase_flip_3", "five_qubit", "dfs_2")
        for ch in VALID_CHANNELS
        if analytical_gain_threshold(code, ch) > 0.0
    }
    payload = {
        "codes": codes,
        "backends": backends,
        "channels": list(VALID_CHANNELS),
        "nontrivial_gain_thresholds": thresholds,
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("QECC Simulation Suite")
        print(f"  Codes    : {', '.join(codes)}")
        print(f"  Backends : {', '.join(backends)}")
        print(f"  Channels : {', '.join(VALID_CHANNELS)}")
        print("  Gain thresholds (p*):")
        for pair, val in thresholds.items():
            print(f"    - {pair:<32}: {val:.6f}")
    return 0


def _cmd_simulate(args: argparse.Namespace) -> int:
    from qecc.analysis import analytical_fidelity
    from qecc.backends import get_backend
    from qecc.codes import get_code
    from qecc.metrics import InitialStateSampler

    backend_name = _resolve_backend_name(args.backend)
    backend = get_backend(backend_name)
    code = get_code(args.code)
    cardinals = InitialStateSampler.cardinal_states()
    psi = cardinals[args.state]

    if args.deterministic_error is not None:
        if args.p is not None and args.p != 0.0:
            raise ValueError(
                f"Non-zero --p ({args.p}) cannot be combined with --deterministic-error; "
                "deterministic error mode requires --p 0.0 or omitting --p."
            )
        p_val = 0.0
    else:
        p_val = 0.05 if args.p is None else float(args.p)

    res = backend.simulate_cycle(
        code=code,
        initial_state=psi,
        channel=None if args.deterministic_error else args.channel,
        p=p_val,
        num_shots=args.shots,
        seed=args.seed,
        deterministic_errors=args.deterministic_error,
    )
    ana_ent = None
    ana_state = None
    if args.deterministic_error is None:
        ana_ent = float(analytical_fidelity(args.code, args.channel, p_val, metric="entanglement"))
        ana_state = float(analytical_fidelity(args.code, args.channel, p_val, metric="state"))

    payload = {
        "code": res.code_name,
        "backend": res.backend_name,
        "channel": args.channel if args.deterministic_error is None else None,
        "deterministic_error": args.deterministic_error,
        "state": args.state,
        "p": res.p,
        "num_shots": res.num_shots,
        "mean_fidelity": res.mean_fidelity,
        "std_error": res.std_error,
        "analytical_entanglement_fidelity": ana_ent,
        "analytical_state_fidelity": ana_state,
        "syndrome_counts": res.syndrome_counts,
        "elapsed_seconds": res.elapsed_seconds,
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(
            f"code={res.code_name} backend={res.backend_name} state=|{args.state}> "
            f"p={res.p:.4f} shots={res.num_shots}"
        )
        print(
            f"  mean_fidelity = {res.mean_fidelity:.6f} +/- {res.std_error:.6f} "
            f"(elapsed={res.elapsed_seconds:.4f}s)"
        )
        if ana_ent is not None and ana_state is not None:
            print(f"  analytical    = {ana_ent:.6f} (entanglement), {ana_state:.6f} (state)")
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    verify_script = _repo_root() / "scripts" / "verify_analytical_thresholds.py"
    if verify_script.is_file():
        cmd = [sys.executable, str(verify_script)]
        if args.json:
            cmd.append("--json")
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.stdout:
            sys.stdout.write(proc.stdout)
        if proc.stderr:
            sys.stderr.write(proc.stderr)
        return int(proc.returncode)

    from qecc.analysis import (
        analytical_gain_threshold,
        five_qubit_weight_enumerator,
        verify_knill_laflamme,
    )

    kl = {
        c: verify_knill_laflamme(c).passed
        for c in ("bit_flip_3", "phase_flip_3", "five_qubit", "dfs_2")
    }
    enum = five_qubit_weight_enumerator()
    out = {
        "knill_laflamme": kl,
        "five_qubit_logical_I_weights": enum["I"],
        "thresholds": {
            "bit_flip_3/bit_flip": analytical_gain_threshold("bit_flip_3", "bit_flip"),
            "phase_flip_3/phase_flip": analytical_gain_threshold("phase_flip_3", "phase_flip"),
            "five_qubit/depolarizing": analytical_gain_threshold("five_qubit", "depolarizing"),
            "five_qubit/bit_flip": analytical_gain_threshold("five_qubit", "bit_flip"),
        },
    }
    if args.json:
        print(json.dumps(out, indent=2, sort_keys=True))
    else:
        print(out)
    return 0 if all(kl.values()) else 1


def _cmd_sweep(args: argparse.Namespace) -> int:
    sweep_script = _repo_root() / "scripts" / "run_sweeps.py"
    if not sweep_script.is_file():
        print(f"Sweep script not found: {sweep_script}", file=sys.stderr)
        return 1
    cmd = [
        sys.executable,
        str(sweep_script),
        "--backend",
        args.backend,
        "--shots",
        str(args.shots),
        "--num-p",
        str(args.num_p),
        "--seed",
        str(args.seed),
    ]
    if args.data_dir is not None:
        cmd.extend(["--data-dir", str(args.data_dir)])
    if args.figures_dir is not None:
        cmd.extend(["--figures-dir", str(args.figures_dir)])
    if args.skip_figures:
        cmd.append("--skip-figures")
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.stdout:
        sys.stdout.write(proc.stdout)
    if proc.stderr:
        sys.stderr.write(proc.stderr)
    return int(proc.returncode)


def _cmd_benchmark(args: argparse.Namespace) -> int:
    bench_script = _repo_root() / "scripts" / "run_benchmarks.py"
    if bench_script.is_file() and not args.quick:
        cmd = [sys.executable, str(bench_script)]
        if args.output_dir is not None:
            cmd.extend(["--output-dir", str(args.output_dir)])
        if args.json:
            cmd.append("--json")
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.stdout:
            sys.stdout.write(proc.stdout)
        if proc.stderr:
            sys.stderr.write(proc.stderr)
        return int(proc.returncode)

    from qecc.backends import get_backend, list_available_backends
    from qecc.codes import get_code
    from qecc.metrics import InitialStateSampler

    psi = InitialStateSampler.cardinal_states()["+"]
    backends = list_available_backends()
    target_backends = [b for b in ("numpy", "torch_cuda") if b in backends]
    results: list[dict[str, object]] = []
    for bname in target_backends:
        backend = get_backend(bname)
        for cname in args.codes:
            code = get_code(cname)
            res = backend.simulate_cycle(
                code=code,
                initial_state=psi,
                channel="depolarizing",
                p=0.05,
                num_shots=args.shots,
                seed=args.seed,
            )
            results.append(
                {
                    "backend": bname,
                    "code": cname,
                    "shots": args.shots,
                    "elapsed_seconds": res.elapsed_seconds,
                    "mean_fidelity": res.mean_fidelity,
                }
            )
    if args.json:
        print(json.dumps(results, indent=2, sort_keys=True))
    else:
        print(f"Quick Backend Benchmark (shots={args.shots}, p=0.05, depolarizing):")
        for row in results:
            print(
                f"  - {row['backend']:<10} | {row['code']:<12} | "
                f"{row['elapsed_seconds']:.4f}s | F={row['mean_fidelity']:.4f}"
            )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m qecc.cli",
        description="QECC simulation, analytical verification, sweep, and benchmark CLI.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_info = sub.add_parser("info", help="List available codes, backends, and gain thresholds.")
    p_info.add_argument("--json", action="store_true", help="Emit JSON output.")
    p_info.set_defaults(func=_cmd_info)

    p_sim = sub.add_parser("simulate", help="Simulate a single QECC cycle.")
    p_sim.add_argument(
        "--code",
        default="five_qubit",
        choices=("bit_flip_3", "phase_flip_3", "five_qubit", "dfs_2"),
        help="Code name.",
    )
    p_sim.add_argument(
        "--channel",
        default="depolarizing",
        choices=("bit_flip", "phase_flip", "depolarizing", "collective_dephasing"),
        help="Stochastic noise channel.",
    )
    p_sim.add_argument(
        "--p",
        type=float,
        default=None,
        help="Error probability in [0, 1] (default: 0.05 stochastic, 0.0 with --deterministic-error).",
    )
    p_sim.add_argument("--shots", type=int, default=4000, help="Number of Monte Carlo shots.")
    p_sim.add_argument(
        "--backend",
        default="auto",
        choices=("auto", "numpy", "torch_cuda", "eigenkit_unitary"),
        help="Simulation backend.",
    )
    p_sim.add_argument(
        "--state",
        default="+",
        choices=("0", "1", "+", "-", "+i", "-i"),
        help="Cardinal initial state.",
    )
    p_sim.add_argument("--seed", type=int, default=42, help="PRNG seed.")
    p_sim.add_argument(
        "--deterministic-error",
        default=None,
        help="Optional deterministic Pauli error string (disables stochastic channel).",
    )
    p_sim.add_argument("--json", action="store_true", help="Emit JSON output.")
    p_sim.set_defaults(func=_cmd_simulate)

    p_verify = sub.add_parser(
        "verify",
        help="Verify Knill-Laflamme conditions, 1024-term enumerator, and thresholds.",
    )
    p_verify.add_argument("--json", action="store_true", help="Emit JSON output.")
    p_verify.set_defaults(func=_cmd_verify)

    p_sweep = sub.add_parser("sweep", help="Run Monte Carlo fidelity sweeps and generate figures.")
    p_sweep.add_argument(
        "--backend",
        default="auto",
        choices=("auto", "numpy", "torch_cuda"),
        help="Simulation backend.",
    )
    p_sweep.add_argument("--shots", type=int, default=18000, help="Shots per p point.")
    p_sweep.add_argument("--num-p", type=int, default=21, help="Number of p points in [0, 0.5].")
    p_sweep.add_argument("--seed", type=int, default=20261118, help="Base PRNG seed.")
    p_sweep.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Optional output directory for sweep CSV/JSON.",
    )
    p_sweep.add_argument(
        "--figures-dir",
        type=Path,
        default=None,
        help="Optional output directory for sweep PNG/PDF figures.",
    )
    p_sweep.add_argument("--skip-figures", action="store_true", help="Skip figure generation.")
    p_sweep.set_defaults(func=_cmd_sweep)

    p_bench = sub.add_parser("benchmark", help="Run backend performance benchmark.")
    p_bench.add_argument("--shots", type=int, default=10000, help="Shots for quick benchmark.")
    p_bench.add_argument(
        "--codes",
        nargs="+",
        default=["bit_flip_3", "five_qubit"],
        choices=("bit_flip_3", "phase_flip_3", "five_qubit", "dfs_2"),
        help="Codes to benchmark in quick mode.",
    )
    p_bench.add_argument("--seed", type=int, default=42, help="PRNG seed.")
    p_bench.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory for benchmark JSON and Markdown (non-quick mode).",
    )
    p_bench.add_argument("--quick", action="store_true", help="Run quick inline comparison.")
    p_bench.add_argument("--json", action="store_true", help="Emit JSON output.")
    p_bench.set_defaults(func=_cmd_benchmark)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        return int(args.func(args))
    except (ValueError, NotImplementedError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
