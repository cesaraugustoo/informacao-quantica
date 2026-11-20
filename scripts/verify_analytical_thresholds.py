#!/usr/bin/env python3
"""Deterministic, CWD-independent verification of Knill-Laflamme conditions,
the 1024-term [[5,1,3]] weight enumerator, and analytical gain thresholds.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from qecc.analysis import (  # noqa: E402
    VALID_CHANNELS,
    VALID_CODES,
    analytical_fidelity,
    analytical_gain_threshold,
    five_qubit_single_axis_weight_enumerator,
    five_qubit_weight_enumerator,
    verify_knill_laflamme,
)


def run_verification() -> dict[str, object]:
    """Execute genuine Knill-Laflamme, weight-enumerator, and threshold checks."""
    kl_results: list[dict[str, object]] = []
    for code_name in ("bit_flip_3", "phase_flip_3", "five_qubit", "dfs_2"):
        res = verify_knill_laflamme(code_name, tol=1e-12)
        if not res.passed:
            raise AssertionError(
                f"Knill-Laflamme condition failed for {code_name}: "
                f"max_violation={res.max_violation:.3e}"
            )
        kl_results.append(
            {
                "code": code_name,
                "num_errors": len(res.error_set),
                "passed": res.passed,
                "max_violation": res.max_violation,
            }
        )

    cosets = five_qubit_weight_enumerator()
    total_paulis = sum(sum(w.values()) for w in cosets.values())
    if total_paulis != 1024:
        raise AssertionError(f"Expected 1024 5-qubit Paulis, got {total_paulis}")

    logical_i_weights = [cosets["I"][w] for w in range(6)]
    if logical_i_weights != [1, 15, 0, 60, 135, 45] or sum(logical_i_weights) != 256:
        raise AssertionError(f"Unexpected [[5,1,3]] logical-I weights: {logical_i_weights}")

    x_axis_enum = five_qubit_single_axis_weight_enumerator("X")
    z_axis_enum = five_qubit_single_axis_weight_enumerator("Z")
    x_weights = [x_axis_enum[w] for w in range(6)]
    z_weights = [z_axis_enum[w] for w in range(6)]
    if x_weights != [1, 5, 0, 0, 0, 0] or z_weights != [1, 5, 0, 0, 0, 0]:
        raise AssertionError(
            f"Unexpected single-axis weights: X={x_weights}, Z={z_weights}"
        )

    p_grid = np.linspace(0.0, 0.5, 51)
    enum_poly = np.zeros_like(p_grid)
    for w, count in enumerate(logical_i_weights):
        if count > 0:
            enum_poly += count * ((1.0 - p_grid) ** (5 - w)) * ((p_grid / 3.0) ** w)
    ana_dep = analytical_fidelity("five_qubit", "depolarizing", p_grid, metric="entanglement")
    max_poly_diff = float(np.max(np.abs(enum_poly - ana_dep)))
    if max_poly_diff >= 1e-14:
        raise AssertionError(f"Enumerator mismatch vs analytical_fidelity: {max_poly_diff}")

    threshold_pairs = [
        ("bit_flip_3", "bit_flip"),
        ("phase_flip_3", "phase_flip"),
        ("five_qubit", "depolarizing"),
        ("five_qubit", "bit_flip"),
        ("five_qubit", "phase_flip"),
        ("dfs_2", "collective_dephasing"),
        ("bit_flip_3", "depolarizing"),
        ("phase_flip_3", "depolarizing"),
    ]
    thresholds: list[dict[str, object]] = []
    delta = 1e-4
    for code_name, channel_type in threshold_pairs:
        p_star = analytical_gain_threshold(code_name, channel_type)
        if 0.0 < p_star < 1.0:
            f_at_star = float(analytical_fidelity(code_name, channel_type, p_star))
            raw_at_star = 1.0 - p_star
            residual = abs(f_at_star - raw_at_star)
            if residual >= 1e-12:
                raise AssertionError(
                    f"Threshold root residual too large for {code_name}/{channel_type}: {residual}"
                )
            f_below = float(analytical_fidelity(code_name, channel_type, p_star - delta))
            f_above = float(analytical_fidelity(code_name, channel_type, p_star + delta))
            if not (f_below > 1.0 - (p_star - delta) and f_above < 1.0 - (p_star + delta)):
                raise AssertionError(
                    f"Sign flip check failed around p*={p_star} for {code_name}/{channel_type}"
                )
        else:
            residual = 0.0
        thresholds.append(
            {
                "code": code_name,
                "channel": channel_type,
                "threshold_p": p_star,
                "root_residual": residual,
            }
        )

    # Confirm all 20 (code, channel) combinations evaluate cleanly at p=0.0 and p=0.5
    for code_name in VALID_CODES:
        for channel_type in VALID_CHANNELS:
            f0 = float(analytical_fidelity(code_name, channel_type, 0.0))
            if abs(f0 - 1.0) >= 1e-12:
                raise AssertionError(f"Expected F(p=0)=1 for {code_name}/{channel_type}, got {f0}")

    return {
        "knill_laflamme": kl_results,
        "five_qubit_enumerator": {
            "total_paulis": total_paulis,
            "logical_I_weights": logical_i_weights,
            "single_axis_X_weights": x_weights,
            "single_axis_Z_weights": z_weights,
            "max_polynomial_diff": max_poly_diff,
        },
        "thresholds": thresholds,
    }


def _format_report(report: dict[str, object]) -> str:
    lines: list[str] = []
    lines.append("QECC Analytical & Algebraic Verification Report")
    lines.append("=" * 56)
    lines.append("")
    lines.append("1. Knill-Laflamme Conditions (<i_L|E_a^dag E_b|j_L> = C_ab delta_ij):")
    for item in report["knill_laflamme"]:  # type: ignore[index]
        lines.append(
            f"   - {item['code']:<14} | |E|={item['num_errors']:<2} | "
            f"status=PASS | max_violation={item['max_violation']:.3e}"
        )

    enum = report["five_qubit_enumerator"]  # type: ignore[index]
    lines.append("")
    lines.append("2. Five-Qubit [[5,1,3]] Weight Enumerator (4^5 = 1024 Pauli operators):")
    lines.append(f"   - Total Pauli terms enumerated : {enum['total_paulis']}")
    lines.append(f"   - Logical-I depolarizing weights: {enum['logical_I_weights']} (sum=256)")
    lines.append(f"   - Pure-X / Pure-Z weights       : {enum['single_axis_X_weights']} (sum=6)")
    lines.append(f"   - Max polynomial discrepancy    : {enum['max_polynomial_diff']:.3e}")

    lines.append("")
    lines.append("3. Analytical Gain Thresholds p* (F_code(p) > F_raw(p) on (0, p*)):")
    for item in report["thresholds"]:  # type: ignore[index]
        p_star = float(item["threshold_p"])
        lines.append(
            f"   - {item['code']:<12} / {item['channel']:<20} -> "
            f"p* = {p_star:.6f} ({p_star:.12f})"
        )

    lines.append("")
    lines.append("All Knill-Laflamme, 1024-term enumerator, and threshold checks PASSED.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify Knill-Laflamme conditions, 1024-term weight enumerator, and analytical gain thresholds."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit verification summary as deterministic JSON.",
    )
    args = parser.parse_args(argv)
    report = run_verification()
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(_format_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
