#!/usr/bin/env python3
"""Run Monte Carlo fidelity sweeps over p in [0, 0.5] and generate data & figures."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from qecc.analysis import analytical_fidelity, analytical_gain_threshold  # noqa: E402
from qecc.backends import QECCBackend, get_backend, list_available_backends  # noqa: E402
from qecc.codes import get_code  # noqa: E402
from qecc.metrics import InitialStateSampler  # noqa: E402

CSV_COLUMNS = [
    "code",
    "channel",
    "p",
    "mean_fidelity",
    "std_error",
    "analytical_entanglement_fidelity",
    "analytical_state_fidelity",
    "raw_entanglement_fidelity",
    "raw_state_fidelity",
    "num_shots",
    "backend",
    "state_mode",
]


def _resolve_backend(requested: str) -> tuple[str, QECCBackend]:
    avail = list_available_backends()
    if requested == "auto":
        name = "torch_cuda" if "torch_cuda" in avail else "numpy"
    elif requested in avail:
        name = requested
    else:
        raise ValueError(f"Backend {requested!r} unavailable; available={avail}")
    return name, get_backend(name)


def _simulate_point(
    backend: QECCBackend,
    code_name: str,
    channel_type: str,
    p: float,
    num_shots: int,
    state_mode: str,
    seed: int,
) -> tuple[float, float]:
    """Run Monte Carlo simulation for one (code, channel, p) point."""
    code = get_code(code_name)
    cardinals = InitialStateSampler.cardinal_states()

    if state_mode == "cardinal_avg":
        state_keys = ("0", "1", "+", "-", "+i", "-i")
        base_shots = num_shots // len(state_keys)
        rem = num_shots % len(state_keys)
        all_fids: list[np.ndarray] = []
        for idx, skey in enumerate(state_keys):
            s_shots = base_shots + (1 if idx < rem else 0)
            if s_shots <= 0:
                continue
            res = backend.simulate_cycle(
                code=code,
                initial_state=cardinals[skey],
                channel=channel_type,
                p=float(p),
                num_shots=s_shots,
                seed=seed + idx * 1009,
            )
            all_fids.append(res.fidelities)
        combined = np.concatenate(all_fids)
        mean_f = float(np.mean(combined))
        std_err = float(np.std(combined, ddof=1) / math.sqrt(combined.size)) if combined.size > 1 else 0.0
        return mean_f, std_err

    state_key = state_mode.strip("|>")
    psi = cardinals[state_key]
    res = backend.simulate_cycle(
        code=code,
        initial_state=psi,
        channel=channel_type,
        p=float(p),
        num_shots=num_shots,
        seed=seed,
    )
    return float(res.mean_fidelity), float(res.std_error)


def _simulate_coherence_comparison(
    backend: QECCBackend,
    num_shots: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Measure off-diagonal coherence |rho_01| at p=0.05 for |+i> initial state."""
    p = 0.05
    psi_plus_i = InitialStateSampler.cardinal_states()["+i"]
    cases = [
        ("bit_flip_3", "bit_flip", "Bit-Flip 3\n(bit_flip)"),
        ("phase_flip_3", "phase_flip", "Phase-Flip 3\n(phase_flip)"),
        ("five_qubit", "depolarizing", "5-Qubit [[5,1,3]]\n(depolarizing)"),
        ("dfs_2", "collective_dephasing", "DFS-2\n(coll. dephasing)"),
    ]
    rows: list[dict[str, Any]] = []
    for idx, (code_name, channel_type, label) in enumerate(cases):
        code = get_code(code_name)
        res = backend.simulate_cycle(
            code=code,
            initial_state=psi_plus_i,
            channel=channel_type,
            p=p,
            num_shots=num_shots,
            seed=seed + 90_000 + idx * 17,
        )
        coh_sim = float(abs(res.rho_out[0, 1]))
        # Exact Pauli channel action on |+i>:
        # bit_flip preserves X/Z differently: |+i> has <X>=0, <Y>=1, <Z>=0.
        # Under depolarizing: rho_01 decays by 1 - (4/3)*(1 - P_I)
        # Under bit_flip (X_L) or phase_flip (Z_L): Y -> -Y so |rho_01| = |P_I - (1 - P_I)| * 0.5 = |2*P_I - 1| * 0.5
        f_ent = float(analytical_fidelity(code_name, channel_type, p, metric="entanglement"))
        if channel_type == "depolarizing":
            coh_ana = 0.5 * abs(1.0 - (4.0 / 3.0) * (1.0 - f_ent))
            raw_coh = 0.5 * (1.0 - (4.0 / 3.0) * p)
        elif channel_type == "collective_dephasing":
            coh_ana = 0.5 * f_ent
            raw_coh = 0.5 * abs(1.0 - 2.0 * p)
        else:
            coh_ana = 0.5 * abs(2.0 * f_ent - 1.0)
            raw_coh = 0.5 * abs(1.0 - 2.0 * p)
        rows.append(
            {
                "code": code_name,
                "channel": channel_type,
                "label": label,
                "p": p,
                "abs_rho01_sim": coh_sim,
                "abs_rho01_analytical": coh_ana,
                "abs_rho01_raw": raw_coh,
            }
        )
    return rows


def run_sweeps(
    *,
    backend_name: str = "auto",
    num_shots: int = 18_000,
    num_p: int = 21,
    seed: int = 20261118,
    data_dir: Path | None = None,
    figures_dir: Path | None = None,
    skip_figures: bool = False,
) -> dict[str, Any]:
    """Execute full Monte Carlo sweep and write JSON, CSV, and figure artifacts."""
    if num_p < 15:
        raise ValueError(f"num_p must be >= 15, got {num_p}")
    if num_shots < 100:
        raise ValueError(f"num_shots must be >= 100, got {num_shots}")

    out_data = data_dir or (REPO_ROOT / "results" / "data")
    out_figs = figures_dir or (REPO_ROOT / "results" / "figures")
    out_data.mkdir(parents=True, exist_ok=True)
    if not skip_figures:
        out_figs.mkdir(parents=True, exist_ok=True)

    resolved_backend_name, backend = _resolve_backend(backend_name)
    p_grid = np.linspace(0.0, 0.5, num_p)

    # Warm up CUDA backend once so first-call JIT does not skew per-curve flow
    if resolved_backend_name == "torch_cuda":
        _simulate_point(backend, "bit_flip_3", "bit_flip", 0.0, 32, "|0>", seed)

    sweep_specs: list[tuple[str, str, str]] = [
        ("bit_flip_3", "bit_flip", "|0>"),
        ("phase_flip_3", "phase_flip", "|0>"),
        ("five_qubit", "depolarizing", "cardinal_avg"),
        ("five_qubit", "bit_flip", "cardinal_avg"),
        ("bit_flip_3", "depolarizing", "cardinal_avg"),
        ("phase_flip_3", "depolarizing", "cardinal_avg"),
        ("dfs_2", "collective_dephasing", "|+i>"),
        ("dfs_2", "phase_flip", "|+i>"),
        ("phase_flip_3", "collective_dephasing", "|+i>"),
    ]

    records: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = {}

    for spec_idx, (code_name, channel_type, state_mode) in enumerate(sweep_specs):
        key = f"{code_name}__{channel_type}"
        grouped[key] = []
        for p_idx, p_val in enumerate(p_grid):
            p_f = float(p_val)
            mean_f, std_err = _simulate_point(
                backend=backend,
                code_name=code_name,
                channel_type=channel_type,
                p=p_f,
                num_shots=num_shots,
                state_mode=state_mode,
                seed=seed + spec_idx * 1000 + p_idx,
            )
            f_ana_ent = float(
                analytical_fidelity(code_name, channel_type, p_f, metric="entanglement")
            )
            f_ana_state = float(
                analytical_fidelity(code_name, channel_type, p_f, metric="state")
            )
            f_raw_ent = float(
                analytical_fidelity("raw", channel_type, p_f, metric="entanglement")
            )
            f_raw_state = float(
                analytical_fidelity("raw", channel_type, p_f, metric="state")
            )
            row = {
                "code": code_name,
                "channel": channel_type,
                "p": p_f,
                "mean_fidelity": mean_f,
                "std_error": std_err,
                "analytical_entanglement_fidelity": f_ana_ent,
                "analytical_state_fidelity": f_ana_state,
                "raw_entanglement_fidelity": f_raw_ent,
                "raw_state_fidelity": f_raw_state,
                "num_shots": num_shots,
                "backend": resolved_backend_name,
                "state_mode": state_mode,
            }
            records.append(row)
            grouped[key].append(row)

    # Add analytical raw reference rows (bit_flip entanglement & depolarizing state)
    for raw_channel, metric in (("bit_flip", "entanglement"), ("depolarizing", "state")):
        key = f"raw__{raw_channel}"
        grouped[key] = []
        for p_val in p_grid:
            p_f = float(p_val)
            f_raw_ent = float(analytical_fidelity("raw", raw_channel, p_f, metric="entanglement"))
            f_raw_state = float(analytical_fidelity("raw", raw_channel, p_f, metric="state"))
            mean_f = f_raw_ent if metric == "entanglement" else f_raw_state
            row = {
                "code": "raw",
                "channel": raw_channel,
                "p": p_f,
                "mean_fidelity": mean_f,
                "std_error": 0.0,
                "analytical_entanglement_fidelity": f_raw_ent,
                "analytical_state_fidelity": f_raw_state,
                "raw_entanglement_fidelity": f_raw_ent,
                "raw_state_fidelity": f_raw_state,
                "num_shots": 0,
                "backend": "analytical",
                "state_mode": metric,
            }
            records.append(row)
            grouped[key].append(row)

    coherence_rows = _simulate_coherence_comparison(backend, num_shots=num_shots, seed=seed)

    gain_thresholds = {
        "bit_flip_3/bit_flip": analytical_gain_threshold("bit_flip_3", "bit_flip"),
        "phase_flip_3/phase_flip": analytical_gain_threshold("phase_flip_3", "phase_flip"),
        "five_qubit/depolarizing": analytical_gain_threshold("five_qubit", "depolarizing"),
        "five_qubit/bit_flip": analytical_gain_threshold("five_qubit", "bit_flip"),
        "five_qubit/phase_flip": analytical_gain_threshold("five_qubit", "phase_flip"),
        "dfs_2/collective_dephasing": analytical_gain_threshold("dfs_2", "collective_dephasing"),
        "bit_flip_3/depolarizing": analytical_gain_threshold("bit_flip_3", "depolarizing"),
        "phase_flip_3/depolarizing": analytical_gain_threshold("phase_flip_3", "depolarizing"),
    }

    csv_path = out_data / "fidelity_sweeps.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in records:
            writer.writerow(row)

    json_payload: dict[str, Any] = {
        "metadata": {
            "backend": resolved_backend_name,
            "num_shots": num_shots,
            "num_p": num_p,
            "p_min": float(p_grid[0]),
            "p_max": float(p_grid[-1]),
            "seed": seed,
            "codes": ["raw", "bit_flip_3", "phase_flip_3", "five_qubit", "dfs_2"],
            "channels": ["bit_flip", "phase_flip", "depolarizing", "collective_dephasing"],
        },
        "gain_thresholds": gain_thresholds,
        "coherence_at_p005": coherence_rows,
        "records": records,
    }
    json_path = out_data / "fidelity_sweeps.json"
    json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")

    generated_figures: list[Path] = []
    if not skip_figures:
        generated_figures = _generate_figures(
            out_figs=out_figs,
            p_grid=p_grid,
            grouped=grouped,
            gain_thresholds=gain_thresholds,
            coherence_rows=coherence_rows,
            num_shots=num_shots,
        )

    return {
        "csv_path": str(csv_path),
        "json_path": str(json_path),
        "figures": [str(p) for p in generated_figures],
        "num_records": len(records),
        "backend": resolved_backend_name,
    }


def _save_fig(fig: plt.Figure, out_figs: Path, stem: str) -> list[Path]:
    png_path = out_figs / f"{stem}.png"
    pdf_path = out_figs / f"{stem}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return [png_path, pdf_path]


def _generate_figures(
    *,
    out_figs: Path,
    p_grid: np.ndarray,
    grouped: dict[str, list[dict[str, Any]]],
    gain_thresholds: dict[str, float],
    coherence_rows: list[dict[str, Any]],
    num_shots: int,
) -> list[Path]:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )

    p_dense = np.linspace(0.0, 0.5, 251)
    saved: list[Path] = []

    # ------------------------------------------------------------------
    # Figure 1: Fidelity vs Error Rate (3-qubit codes & [[5,1,3]] code)
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.8), sharey=True)

    # Panel A: 3-qubit codes
    f_bf3_ana = analytical_fidelity("bit_flip_3", "bit_flip", p_dense, metric="entanglement")
    f_raw_ent = analytical_fidelity("raw", "bit_flip", p_dense, metric="entanglement")
    f_bf3_dep_ana = analytical_fidelity("bit_flip_3", "depolarizing", p_dense, metric="state")
    f_raw_state = analytical_fidelity("raw", "depolarizing", p_dense, metric="state")

    ax1.plot(p_dense, f_bf3_ana, color="#1f77b4", lw=2.0, label="bit_flip_3 / phase_flip_3 (analytical)")
    ax1.plot(p_dense, f_raw_ent, color="#444444", ls="--", lw=1.6, label=r"Raw physical $F_{\mathrm{ent}} = 1 - p$")
    ax1.plot(p_dense, f_bf3_dep_ana, color="#d62728", ls="-.", lw=1.6, label="bit_flip_3 under depolarizing (state)")
    ax1.plot(p_dense, f_raw_state, color="#888888", ls=":", lw=1.5, label=r"Raw state $1 - \frac{2}{3}p$")
    ax1.fill_between(
        p_dense,
        f_raw_ent,
        f_bf3_ana,
        where=f_bf3_ana > f_raw_ent,
        color="#1f77b4",
        alpha=0.12,
        label="Gain region ($0 < p < 0.5$)",
    )

    for key, marker, color, lbl in (
        ("bit_flip_3__bit_flip", "o", "#1f77b4", f"bit_flip_3 MC ({num_shots} shots)"),
        ("phase_flip_3__phase_flip", "s", "#2ca02c", f"phase_flip_3 MC ({num_shots} shots)"),
        ("bit_flip_3__depolarizing", "^", "#d62728", "bit_flip_3 depolarizing MC"),
    ):
        rows = grouped[key]
        ps = [r["p"] for r in rows]
        ys = [r["mean_fidelity"] for r in rows]
        es = [2.0 * r["std_error"] for r in rows]
        ax1.errorbar(ps, ys, yerr=es, fmt=marker, ms=4.5, color=color, capsize=2.5, elinewidth=1.0, label=lbl)

    ax1.axvline(0.5, color="#1f77b4", ls=":", lw=1.2)
    ax1.annotate(r"$p^* = 0.500$", xy=(0.5, 0.5), xytext=(0.37, 0.44), arrowprops={"arrowstyle": "->"})
    ax1.set_title("(a) 3-Qubit Repetition Codes")
    ax1.set_xlabel("Physical Error Probability $p$")
    ax1.set_ylabel("Fidelity $F$")
    ax1.set_xlim(0.0, 0.51)
    ax1.set_ylim(0.30, 1.03)
    ax1.legend(loc="lower left")

    # Panel B: [[5,1,3]] 5-qubit perfect code
    f_5q_dep_state = analytical_fidelity("five_qubit", "depolarizing", p_dense, metric="state")
    f_5q_dep_ent = analytical_fidelity("five_qubit", "depolarizing", p_dense, metric="entanglement")
    f_5q_bf_state = analytical_fidelity("five_qubit", "bit_flip", p_dense, metric="state")
    p_star_dep = gain_thresholds["five_qubit/depolarizing"]

    ax2.plot(p_dense, f_5q_dep_state, color="#9467bd", lw=2.0, label="five_qubit depolarizing (state)")
    ax2.plot(p_dense, f_5q_dep_ent, color="#9467bd", ls=":", lw=1.5, label="five_qubit depolarizing (entanglement)")
    ax2.plot(p_dense, f_5q_bf_state, color="#ff7f0e", lw=1.8, label="five_qubit bit_flip (state)")
    ax2.plot(p_dense, f_raw_state, color="#444444", ls="--", lw=1.6, label=r"Raw state $1 - \frac{2}{3}p$")
    ax2.fill_between(
        p_dense,
        f_raw_state,
        f_5q_dep_state,
        where=f_5q_dep_state > f_raw_state,
        color="#9467bd",
        alpha=0.14,
        label=rf"Depolarizing gain ($p^* = {p_star_dep:.4f}$)",
    )

    for key, marker, color, lbl in (
        ("five_qubit__depolarizing", "o", "#9467bd", f"five_qubit depolarizing MC"),
        ("five_qubit__bit_flip", "D", "#ff7f0e", f"five_qubit bit_flip MC"),
    ):
        rows = grouped[key]
        ps = [r["p"] for r in rows]
        ys = [r["mean_fidelity"] for r in rows]
        es = [2.0 * r["std_error"] for r in rows]
        ax2.errorbar(ps, ys, yerr=es, fmt=marker, ms=4.5, color=color, capsize=2.5, elinewidth=1.0, label=lbl)

    f_at_star = float(analytical_fidelity("five_qubit", "depolarizing", p_star_dep, metric="state"))
    ax2.axvline(p_star_dep, color="#9467bd", ls="--", lw=1.2)
    ax2.annotate(
        rf"$p^* = {p_star_dep:.6f}$",
        xy=(p_star_dep, f_at_star),
        xytext=(0.20, 0.92),
        arrowprops={"arrowstyle": "->"},
    )
    ax2.set_title(r"(b) Five-Qubit $[[5,1,3]]$ Perfect Code")
    ax2.set_xlabel("Physical Error Probability $p$")
    ax2.set_xlim(0.0, 0.51)
    ax2.legend(loc="lower left")

    fig.suptitle("Quantum Error Correction Fidelity vs. Physical Error Rate", y=1.02, fontsize=12)
    fig.tight_layout()
    saved.extend(_save_fig(fig, out_figs, "fidelity_vs_error_rate"))

    # ------------------------------------------------------------------
    # Figure 2: Gain Threshold Comparison (Delta F zoom + p* bar chart)
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.6))
    p_zoom = np.linspace(0.0, 0.25, 251)

    gain_curves = [
        ("bit_flip_3", "bit_flip", "entanglement", "#1f77b4", "-", "bit_flip_3 / bit_flip ($p^*=0.500$)"),
        ("five_qubit", "depolarizing", "entanglement", "#9467bd", "-", "five_qubit / depolarizing ($p^*=0.1376$)"),
        ("five_qubit", "bit_flip", "entanglement", "#ff7f0e", "--", "five_qubit / bit_flip ($p^*=0.1311$)"),
        ("bit_flip_3", "depolarizing", "entanglement", "#d62728", "-.", "bit_flip_3 / depolarizing ($p^*=0.0$)"),
    ]
    for code_name, ch_name, metric, color, ls, lbl in gain_curves:
        f_code = analytical_fidelity(code_name, ch_name, p_zoom, metric=metric)
        f_raw = analytical_fidelity("raw", ch_name, p_zoom, metric=metric)
        ax1.plot(p_zoom, f_code - f_raw, color=color, ls=ls, lw=2.0, label=lbl)

    # Overlay MC points for five_qubit / depolarizing gain (scaled to entanglement gain = 1.5 * state gain)
    rows_5q = [r for r in grouped["five_qubit__depolarizing"] if r["p"] <= 0.255]
    ps_5q = [r["p"] for r in rows_5q]
    gains_5q = [1.5 * (r["mean_fidelity"] - r["raw_state_fidelity"]) for r in rows_5q]
    errs_5q = [1.5 * 2.0 * r["std_error"] for r in rows_5q]
    ax1.errorbar(ps_5q, gains_5q, yerr=errs_5q, fmt="o", ms=4.5, color="#9467bd", capsize=2.5, label="five_qubit dep. MC")

    ax1.axhline(0.0, color="#222222", lw=1.2)
    ax1.axvline(gain_thresholds["five_qubit/depolarizing"], color="#9467bd", ls=":", lw=1.2)
    ax1.axvline(gain_thresholds["five_qubit/bit_flip"], color="#ff7f0e", ls=":", lw=1.2)
    ax1.set_title(r"(a) Net Fidelity Gain $\Delta F_{\mathrm{ent}} = F_{\mathrm{QECC}} - F_{\mathrm{raw}}$")
    ax1.set_xlabel("Physical Error Probability $p$")
    ax1.set_ylabel(r"Entanglement Fidelity Gain $\Delta F_{\mathrm{ent}}$")
    ax1.set_xlim(0.0, 0.25)
    ax1.legend(loc="lower left")

    bar_labels = [
        "bit_flip_3\n(bit_flip)",
        "phase_flip_3\n(phase_flip)",
        "five_qubit\n(depolarizing)",
        "five_qubit\n(bit/phase_flip)",
        "dfs_2\n(coll. deph.)",
    ]
    bar_vals = [
        gain_thresholds["bit_flip_3/bit_flip"],
        gain_thresholds["phase_flip_3/phase_flip"],
        gain_thresholds["five_qubit/depolarizing"],
        gain_thresholds["five_qubit/bit_flip"],
        gain_thresholds["dfs_2/collective_dephasing"],
    ]
    bar_colors = ["#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]
    bars = ax2.bar(bar_labels, bar_vals, color=bar_colors, edgecolor="#222222", linewidth=0.7)
    for bar, val in zip(bars, bar_vals, strict=True):
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            val + 0.02,
            f"{val:.4f}" if val < 0.4 else f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )
    ax2.set_title(r"(b) Analytical Gain Thresholds $p^*$")
    ax2.set_ylabel("Gain Threshold $p^*$")
    ax2.set_ylim(0.0, 1.15)

    fig.suptitle("Gain Threshold Regions Across Codes and Noise Channels", y=1.02, fontsize=12)
    fig.tight_layout()
    saved.extend(_save_fig(fig, out_figs, "gain_threshold_comparison"))

    # ------------------------------------------------------------------
    # Figure 3: Coherence Protection & DFS vs Active QECC
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.6))

    f_dfs_coll = analytical_fidelity("dfs_2", "collective_dephasing", p_dense)
    f_pf3_coll = analytical_fidelity("phase_flip_3", "collective_dephasing", p_dense)
    f_pf3_ind = analytical_fidelity("phase_flip_3", "phase_flip", p_dense)
    f_dfs_ind = analytical_fidelity("dfs_2", "phase_flip", p_dense)

    ax1.plot(p_dense, f_dfs_coll, color="#17becf", lw=2.2, label="dfs_2 (collective dephasing)")
    ax1.plot(p_dense, f_pf3_coll, color="#d62728", ls="--", lw=1.8, label="phase_flip_3 (collective dephasing)")
    ax1.plot(p_dense, f_pf3_ind, color="#2ca02c", lw=2.0, label="phase_flip_3 (independent phase_flip)")
    ax1.plot(p_dense, f_dfs_ind, color="#ff7f0e", ls="-.", lw=1.8, label="dfs_2 (independent phase_flip)")

    for key, marker, color in (
        ("dfs_2__collective_dephasing", "o", "#17becf"),
        ("phase_flip_3__collective_dephasing", "s", "#d62728"),
        ("dfs_2__phase_flip", "^", "#ff7f0e"),
    ):
        rows = grouped[key]
        ps = [r["p"] for r in rows]
        ys = [r["mean_fidelity"] for r in rows]
        es = [2.0 * r["std_error"] for r in rows]
        ax1.errorbar(ps, ys, yerr=es, fmt=marker, ms=4.5, color=color, capsize=2.5)

    ax1.set_title(r"(a) Passive DFS vs. Active QECC on $|+i\rangle$")
    ax1.set_xlabel("Physical Error Probability $p$")
    ax1.set_ylabel("Fidelity $F$")
    ax1.set_xlim(0.0, 0.51)
    ax1.set_ylim(0.45, 1.04)
    ax1.legend(loc="lower left")

    x_pos = np.arange(len(coherence_rows))
    width = 0.28
    sim_vals = [r["abs_rho01_sim"] for r in coherence_rows]
    ana_vals = [r["abs_rho01_analytical"] for r in coherence_rows]
    raw_vals = [r["abs_rho01_raw"] for r in coherence_rows]
    lbls = [r["label"] for r in coherence_rows]

    ax2.bar(x_pos - width, sim_vals, width=width, color="#1f77b4", label="Simulated (MC)")
    ax2.bar(x_pos, ana_vals, width=width, color="#2ca02c", label="Analytical")
    ax2.bar(x_pos + width, raw_vals, width=width, color="#7f7f7f", label="Unprotected raw")
    ax2.axhline(0.5, color="#222222", ls=":", lw=1.1, label=r"Ideal $|\rho_{01}| = 0.5$")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(lbls)
    ax2.set_ylim(0.40, 0.52)
    ax2.set_title(r"(b) Off-Diagonal Coherence $|\rho_{01}|$ at $p=0.05$ ($|+i\rangle$)")
    ax2.set_ylabel(r"$|\rho_{01}|$")
    ax2.legend(loc="lower right")

    fig.suptitle("Coherence Protection & Decoherence-Free Subspace Tradeoffs", y=1.02, fontsize=12)
    fig.tight_layout()
    saved.extend(_save_fig(fig, out_figs, "coherence_and_dfs_protection"))

    return saved


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run QECC Monte Carlo fidelity sweeps and generate CSV/JSON/figures."
    )
    parser.add_argument(
        "--backend",
        default="auto",
        choices=("auto", "numpy", "torch_cuda"),
        help="Simulation backend (default: auto).",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=18_000,
        help="Monte Carlo shots per p value (default: 18000).",
    )
    parser.add_argument(
        "--num-p",
        type=int,
        default=21,
        help="Number of p points in [0.0, 0.5] (default: 21).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20261118,
        help="Base PRNG seed.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Optional output directory for CSV/JSON.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=None,
        help="Optional output directory for PNG/PDF figures.",
    )
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Skip generating PNG/PDF figures.",
    )
    args = parser.parse_args(argv)
    summary = run_sweeps(
        backend_name=args.backend,
        num_shots=args.shots,
        num_p=args.num_p,
        seed=args.seed,
        data_dir=args.data_dir,
        figures_dir=args.figures_dir,
        skip_figures=args.skip_figures,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
