"""Unit tests for scripts/run_benchmarks.py and qecc.cli benchmark integration."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from qecc.cli import main as cli_main

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH_SCRIPT = REPO_ROOT / "scripts" / "run_benchmarks.py"
TIMING_KEY_SUBSTRINGS = ("elapsed", "seconds", "time", "shots_per_sec", "speedup")


def _collect_floats_by_key(obj: Any, substrings: tuple[str, ...]) -> list[float]:
    collected: list[float] = []
    if isinstance(obj, dict):
        for key, val in obj.items():
            key_low = str(key).lower()
            matched = any(sub in key_low for sub in substrings)
            if matched and isinstance(val, (int, float)) and not isinstance(val, bool):
                collected.append(float(val))
            elif matched and isinstance(val, list):
                for item in val:
                    if isinstance(item, (int, float)) and not isinstance(item, bool):
                        collected.append(float(item))
            else:
                collected.extend(_collect_floats_by_key(val, substrings))
    elif isinstance(obj, list):
        for item in obj:
            collected.extend(_collect_floats_by_key(item, substrings))
    return collected


@pytest.mark.parametrize("use_tmp_cwd", [False, True])
def test_run_benchmarks_help_and_cwd_independence(tmp_path: Path, use_tmp_cwd: bool) -> None:
    assert BENCH_SCRIPT.is_file()
    cwd = tmp_path if use_tmp_cwd else REPO_ROOT
    proc = subprocess.run(
        [sys.executable, str(BENCH_SCRIPT), "--help"],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    for flag in (
        "--output-dir",
        "--codes",
        "--shots",
        "--channel",
        "--p",
        "--state",
        "--warmup",
        "--repeats",
        "--seed",
        "--skip-eigenkit",
        "--json",
    ):
        assert flag in proc.stdout


@pytest.mark.parametrize(
    "bad_args",
    [
        ["--shots", "0"],
        ["--shots", "100", "-5"],
        ["--warmup", "-1"],
        ["--repeats", "0"],
        ["--p", "0.0"],
        ["--p", "-0.1"],
        ["--p", "1.5"],
        ["--p", "nan"],
        ["--codes", "unknown_code"],
        ["--channel", "unknown_channel"],
    ],
)
def test_run_benchmarks_invalid_cli_args_fail_cleanly(
    tmp_path: Path, bad_args: list[str]
) -> None:
    proc = subprocess.run(
        [sys.executable, str(BENCH_SCRIPT), "--output-dir", str(tmp_path), *bad_args],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc.returncode != 0
    assert not (tmp_path / "backend_benchmark.json").exists()


def test_run_benchmarks_small_shot_execution_schema_and_parity(tmp_path: Path) -> None:
    proc = subprocess.run(
        [
            sys.executable,
            str(BENCH_SCRIPT),
            "--codes",
            "bit_flip_3",
            "five_qubit",
            "--shots",
            "50",
            "200",
            "--warmup",
            "1",
            "--repeats",
            "2",
            "--output-dir",
            str(tmp_path),
            "--json",
        ],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr

    json_path = tmp_path / "backend_benchmark.json"
    md_path = tmp_path / "backend_benchmark.md"
    assert json_path.is_file() and json_path.stat().st_size > 100
    assert md_path.is_file() and md_path.stat().st_size > 100

    stdout_report = json.loads(proc.stdout)
    disk_report = json.loads(json_path.read_text(encoding="utf-8"))
    assert stdout_report == disk_report

    meta = disk_report["metadata"]
    assert meta["cuda_available"] is True
    assert meta["sm_120_supported"] is True
    assert meta["eigenkit_stochastic_supported"] is False
    assert meta["warmup_runs"] == 1
    assert meta["repeats"] == 2
    assert meta["codes"] == ["bit_flip_3", "five_qubit"]
    assert meta["shots_list"] == [50, 200]

    records = disk_report["records"]
    assert len(records) == 8  # 2 codes * 2 shot counts * 2 backends
    seen_cells = {(r["code"], r["shots"], r["backend"]) for r in records}
    assert seen_cells == {
        (c, s, b)
        for c in ("bit_flip_3", "five_qubit")
        for s in (50, 200)
        for b in ("numpy", "torch_cuda")
    }
    for rec in records:
        assert rec["min_seconds"] <= rec["median_seconds"] <= rec["max_seconds"]
        assert rec["shots_per_sec"] == pytest.approx(
            rec["shots"] / rec["median_seconds"], rel=1e-9
        )
        assert 0.0 <= rec["mean_fidelity"] <= 1.0
        assert rec["std_error"] >= 0.0

    comparisons = disk_report["comparisons"]
    assert len(comparisons) == 4
    for comp in comparisons:
        assert comp["cuda_speedup"] == pytest.approx(
            comp["numpy_median_seconds"] / comp["torch_cuda_median_seconds"], rel=1e-9
        )
        assert comp["fidelity_diff"] == pytest.approx(
            abs(comp["numpy_mean_fidelity"] - comp["torch_cuda_mean_fidelity"]), abs=1e-12
        )
        assert comp["parity_passed"] is True

    unitary = disk_report["unitary_parity"]
    assert unitary["backend"] == "eigenkit_unitary"
    assert unitary["stochastic_supported"] is False
    assert "eigenkit_evaluation.md" in unitary["evaluation_doc"]
    assert unitary["stochastic_error"]
    assert unitary["all_passed"] is True
    assert len(unitary["checks"]) == 4  # 2 codes * (p=0 + deterministic)
    for chk in unitary["checks"]:
        assert chk["numpy_mean_fidelity"] > 1.0 - 1e-10
        assert chk["torch_cuda_mean_fidelity"] > 1.0 - 1e-10
        assert chk["eigenkit_mean_fidelity"] > 1.0 - 1e-10
        assert chk["max_rho_diff"] < 1e-10
        assert chk["syndrome_counts_match"] is True
        assert chk["passed"] is True

    runtimes = _collect_floats_by_key(disk_report, TIMING_KEY_SUBSTRINGS)
    assert len(runtimes) >= 6
    assert all(math.isfinite(val) and val > 0.0 for val in runtimes)

    md_text = md_path.read_text(encoding="utf-8")
    for token in ("RTX 5080", "numpy", "torch_cuda", "eigenkit_unitary", "bit_flip_3", "five_qubit"):
        assert token in md_text


def test_run_benchmarks_skip_eigenkit_and_single_repeat_guardrails(tmp_path: Path) -> None:
    proc = subprocess.run(
        [
            sys.executable,
            str(BENCH_SCRIPT),
            "--codes",
            "dfs_2",
            "--shots",
            "64",
            "--warmup",
            "0",
            "--repeats",
            "1",
            "--skip-eigenkit",
            "--output-dir",
            str(tmp_path),
        ],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "dfs_2" in proc.stdout

    report = json.loads((tmp_path / "backend_benchmark.json").read_text(encoding="utf-8"))
    assert report["unitary_parity"] == {"skipped": True}
    assert report["metadata"]["channels_by_code"] == {"dfs_2": "collective_dephasing"}
    for rec in report["records"]:
        assert rec["rel_std_pct"] == 0.0
        assert rec["mean_fidelity"] == pytest.approx(1.0, abs=1e-12)

    runtimes = _collect_floats_by_key(report, TIMING_KEY_SUBSTRINGS)
    assert len(runtimes) >= 6
    assert all(math.isfinite(val) and val > 0.0 for val in runtimes)


def test_qecc_cli_benchmark_quick_and_json_forwarding(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    assert (
        cli_main(["benchmark", "--quick", "--shots", "100", "--codes", "bit_flip_3", "--json"])
        == 0
    )
    quick_out = json.loads(capsys.readouterr().out)
    assert [row["backend"] for row in quick_out] == ["numpy", "torch_cuda"]
    assert all(row["elapsed_seconds"] > 0.0 for row in quick_out)

    recorded_calls: list[tuple[list[str], dict[str, Any]]] = []

    class _FakeProc:
        def __init__(self, stdout: str, stderr: str = "", returncode: int = 0) -> None:
            self.stdout = stdout
            self.stderr = stderr
            self.returncode = returncode

    def _fake_run(cmd: list[str], **kwargs: Any) -> _FakeProc:
        recorded_calls.append((list(cmd), dict(kwargs)))
        if "--json" in cmd:
            return _FakeProc('{"status": "ok"}\n')
        return _FakeProc("Summary table\n")

    import qecc.cli as cli_mod

    monkeypatch.setattr(cli_mod.subprocess, "run", _fake_run)

    assert cli_main(["benchmark", "--json"]) == 0
    captured_json = capsys.readouterr()
    assert json.loads(captured_json.out) == {"status": "ok"}

    assert cli_main(["benchmark"]) == 0
    captured_text = capsys.readouterr()
    assert "Summary table" in captured_text.out

    assert cli_main(["benchmark", "--output-dir", "/tmp/bench_out", "--json"]) == 0
    captured_out_dir = capsys.readouterr()
    assert json.loads(captured_out_dir.out) == {"status": "ok"}

    assert len(recorded_calls) == 3
    cmd_json, kwargs_json = recorded_calls[0]
    assert Path(cmd_json[1]) == BENCH_SCRIPT
    assert cmd_json[2:] == ["--json"]
    assert kwargs_json.get("capture_output") is True
    assert kwargs_json.get("text") is True

    cmd_text, _ = recorded_calls[1]
    assert Path(cmd_text[1]) == BENCH_SCRIPT
    assert cmd_text[2:] == []

    cmd_outdir, _ = recorded_calls[2]
    assert Path(cmd_outdir[1]) == BENCH_SCRIPT
    assert cmd_outdir[2:] == ["--output-dir", "/tmp/bench_out", "--json"]


@pytest.mark.parametrize(
    ("state_label", "expected_f"),
    [
        ("0", 0.784),
        ("+", 1.0),
        ("+i", 0.784),
    ],
)
def test_run_benchmarks_state_specific_expected_fidelity_and_parity(
    tmp_path: Path, state_label: str, expected_f: float
) -> None:
    proc = subprocess.run(
        [
            sys.executable,
            str(BENCH_SCRIPT),
            "--codes",
            "bit_flip_3",
            "--shots",
            "2000",
            "--p",
            "0.3",
            "--state",
            state_label,
            "--warmup",
            "1",
            "--repeats",
            "1",
            "--skip-eigenkit",
            "--output-dir",
            str(tmp_path),
            "--json",
        ],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    report = json.loads(proc.stdout)
    comp = report["comparisons"][0]
    assert comp["expected_state_fidelity"] == pytest.approx(expected_f, abs=1e-12)
    assert comp["parity_passed"] is True
    assert comp["max_rho_diff"] >= 0.0
    assert 0.0 <= comp["syndrome_p_value"] <= 1.0


def test_probe_eigenkit_run_probe_all_passed() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json, sys; "
                f"sys.path.insert(0, {str(REPO_ROOT / 'scripts')!r}); "
                "import probe_eigenkit; "
                "print(json.dumps(probe_eigenkit.run_probe()))"
            ),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    res = json.loads(proc.stdout)
    assert res["all_passed"] is True
    assert res["failures"] == []
