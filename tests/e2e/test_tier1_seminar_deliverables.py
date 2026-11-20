"""Tier 1 E2E tests for Feature F9: Seminar manuscript, slides, 18-check rubric & verifier."""

from __future__ import annotations

import re
import subprocess

import pytest

pytestmark = pytest.mark.tier1

_RUBRIC_IDS = (
    "R-1.1",
    "R-1.2",
    "R-1.3",
    "R-2.1",
    "R-2.2",
    "R-2.3",
    "R-3.1",
    "R-3.2",
    "R-3.3",
    "R-3.4",
    "R-4.1",
    "R-4.2",
    "R-5.1",
    "R-6.1",
    "R-6.2",
    "R-6.3",
    "R-6.4",
    "R-7.1",
)


def test_seminar_manuscript_exists_and_covers_all_six_topics(docs_dir) -> None:
    manuscript = docs_dir / "seminar_qecc.md"
    assert manuscript.is_file()
    text = manuscript.read_text(encoding="utf-8")
    assert len(text) >= 10_000
    headings = re.findall(r"(?m)^#{1,3}\s+.+$", text)
    assert len(headings) >= 5

    low = text.lower()
    # Topic 1: Challenges vs classical ECC
    assert "no-cloning" in low or "no cloning" in low
    assert "wootters" in low or "zurek" in low or "inner product" in low
    assert "su(2)" in low or "bloch" in low or "continuous" in low
    assert "projective" in low or "collapse" in low

    # Topic 2: Pauli discretization & Knill-Laflamme
    assert "knill" in low and "laflamme" in low
    assert all(p in text for p in ("I", "X", "Y", "Z"))
    assert "discretization" in low or "syndrome" in low
    assert (
        re.search(
            r"E_\{?[abij]\}?\^\{\\dagger\}\s*E_\{?[abij]\}?|E_\{?[abij]\}?\^\\dagger\s*E_\{?[abij]\}?|(?:\\alpha|C)_\{[abij]{2}\}",
            text,
        )
        is not None
    )

    # Topic 3: 3-qubit codes, [[5,1,3]], Hamming bound, coherence & DFS
    assert "bit-flip" in low or "bit flip" in low
    assert "phase-flip" in low or "phase flip" in low
    assert "5,1,3" in text or "five-qubit" in low or "5-qubit" in low
    assert "hamming" in low
    assert "rho_{01}" in low or "off-diagonal" in low or "coherence" in low
    assert "decoherence-free" in low or "dfs" in low
    assert ("01" in text and "10" in text) and "collective" in low

    # Topic 4: Non-destructive ancilla syndrome extraction
    assert "ancilla" in low
    assert "stabilizer" in low
    assert "kickback" in low or "controlled" in low
    assert "feed-forward" in low or "conditional" in low or "recovery" in low

    # Topic 5: Recent developments
    assert "surface code" in low
    assert "threshold theorem" in low
    assert "willow" in low and ("2.14" in text or "0.143" in text)
    assert ("bluvstein" in low or "lukin" in low) and ("48" in text or "8,3,2" in text)
    assert ("ibm" in low or "qldpc" in low or "bivariate bicycle" in low) and (
        "288" in text or "0.7%" in text
    )
    assert "cat" in low and (
        "alice" in low or "aws" in low or "putterman" in low or "réglade" in low or "reglade" in low
    )

    # Topic 6: Gain thresholds & metric consistency
    assert "1 - p" in text or "1-p" in text
    assert "0.5" in text
    assert "0.1376" in text or "0.1311" in text
    assert "2/3" in text or r"\frac{2}{3}" in text or "entanglement" in low


def test_seminar_manuscript_five_qubit_stabilizers_and_16_syndrome_table(docs_dir) -> None:
    text = (docs_dir / "seminar_qecc.md").read_text(encoding="utf-8")
    for gen in ("XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"):
        assert gen in text

    assert ("XXXXX" in text or r"X^{\otimes 5}" in text or "X^{\\otimes 5}" in text)
    assert ("ZZZZZ" in text or r"Z^{\otimes 5}" in text or "Z^{\\otimes 5}" in text)

    for idx in range(16):
        assert f"{idx:04b}" in text


def test_seminar_slide_deck_structure_coverage_and_pacing(docs_dir, helpers) -> None:
    slides_path = docs_dir / "seminar_slides.md"
    assert slides_path.is_file()
    text = slides_path.read_text(encoding="utf-8")
    assert len(text.strip()) > 1000

    slides = [s for s in re.split(r"(?m)^---\s*$", text) if s.strip()]
    assert len(slides) >= 18

    low = text.lower()
    assert "no-cloning" in low or "no cloning" in low
    assert "knill" in low and "laflamme" in low
    assert ("bit-flip" in low or "bit flip" in low) and ("phase-flip" in low or "phase flip" in low)
    assert ("5,1,3" in text or "five-qubit" in low or "5-qubit" in low)
    assert "dfs" in low or "decoherence-free" in low
    assert "ancilla" in low
    assert "willow" in low and ("qldpc" in low or "288" in text) and "cat" in low
    assert "0.5" in text and "0.1376" in text

    img_targets = helpers.extract_markdown_image_targets(text)
    fig_links = [t for t in img_targets if "results/figures" in t or "figures/" in t]
    assert len(fig_links) >= 2


def test_review_rubric_covers_all_18_checks_and_passes(docs_dir, helpers) -> None:
    rubric_path = docs_dir / "review_rubric.md"
    assert rubric_path.is_file()
    text = rubric_path.read_text(encoding="utf-8")
    assert len(text.strip()) > 500
    lines = text.splitlines()

    for rid in _RUBRIC_IDS:
        matching_lines = [ln for ln in lines if rid in ln]
        assert len(matching_lines) >= 1, f"Missing rubric check ID {rid}"
        assert any("PASS" in ln for ln in matching_lines), f"Rubric check {rid} does not record PASS"

    assert not helpers.has_rubric_fail_verdict(text)


def test_verify_analytical_thresholds_script_executes_and_outputs_exact_values(
    repo_root, scripts_dir, python_executable
) -> None:
    script = scripts_dir / "verify_analytical_thresholds.py"
    assert script.is_file()
    proc = subprocess.run(
        [str(python_executable), str(script)],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout + "\n" + proc.stderr
    out_low = out.lower()
    assert "knill" in out_low or "kl" in out_low
    assert "1024" in out or "weight" in out_low
    assert "0.5" in out
    assert "0.1376" in out
    assert "0.1311" in out


def test_all_markdown_figure_links_resolve_to_nonempty_files(
    repo_root, docs_dir, results_dir, helpers
) -> None:
    manuscript_path = docs_dir / "seminar_qecc.md"
    slides_path = docs_dir / "seminar_slides.md"

    m_targets = helpers.extract_markdown_image_targets(manuscript_path.read_text(encoding="utf-8"))
    s_targets = helpers.extract_markdown_image_targets(slides_path.read_text(encoding="utf-8"))
    assert len(m_targets) >= 1
    assert len(s_targets) >= 1

    fig_dir = (results_dir / "figures").resolve()
    for md_path, targets in ((manuscript_path, m_targets), (slides_path, s_targets)):
        for tgt in targets:
            resolved = (md_path.parent / tgt).resolve()
            if not resolved.exists():
                resolved = (repo_root / tgt).resolve()
            assert resolved.is_file(), f"Missing figure referenced by {md_path.name}: {tgt}"
            assert fig_dir in resolved.parents, f"Figure {resolved} is not inside {fig_dir}"
            assert resolved.stat().st_size > 1024
            if resolved.suffix.lower() == ".png":
                with resolved.open("rb") as fh:
                    assert fh.read(8) == b"\x89PNG\r\n\x1a\n"
            elif resolved.suffix.lower() == ".pdf":
                with resolved.open("rb") as fh:
                    assert fh.read(5) == b"%PDF-"


def test_eigenkit_evaluation_and_readme_deliverables_present_and_complete(
    repo_root, docs_dir
) -> None:
    eval_path = docs_dir / "eigenkit_evaluation.md"
    readme_path = repo_root / "README.md"
    assert eval_path.is_file()
    assert readme_path.is_file()

    eval_text = eval_path.read_text(encoding="utf-8")
    assert len(eval_text.strip()) > 300
    eval_low = eval_text.lower()
    assert "eigenkit" in eval_low
    assert any(tok in eval_low for tok in ("rtx 5080", "sm_120", "cu126", "cu130"))
    assert any(tok in eval_low for tok in ("c_if", "measurement", "partial trace", "density matrix"))

    readme_text = readme_path.read_text(encoding="utf-8")
    assert len(readme_text.strip()) > 300
    readme_low = readme_text.lower()
    assert "uv" in readme_low or ".venv" in readme_low
    assert "pytest" in readme_low
    assert "run_sweeps.py" in readme_text
    assert "run_benchmarks.py" in readme_text
    assert "verify_analytical_thresholds.py" in readme_text
    assert "docs/" in readme_text


def test_seminar_manuscript_consistent_with_qecc_analysis_thresholds(
    docs_dir, qecc_analysis
) -> None:
    th_bf = float(qecc_analysis.analytical_gain_threshold("bit_flip_3", "bit_flip"))
    th_pf = float(qecc_analysis.analytical_gain_threshold("phase_flip_3", "phase_flip"))
    th_5q = float(qecc_analysis.analytical_gain_threshold("five_qubit", "depolarizing"))
    th_5q_xz = float(qecc_analysis.analytical_gain_threshold("five_qubit", "bit_flip"))

    assert th_bf == pytest.approx(0.5, abs=1e-9)
    assert th_pf == pytest.approx(0.5, abs=1e-9)
    assert th_5q == pytest.approx(0.1376275643, abs=1e-6)
    assert th_5q_xz == pytest.approx(0.1311231479, abs=1e-6)

    f_bf_half = float(qecc_analysis.analytical_fidelity("bit_flip_3", "bit_flip", 0.5))
    assert f_bf_half == pytest.approx(0.5, abs=1e-9)

    manuscript_text = (docs_dir / "seminar_qecc.md").read_text(encoding="utf-8")
    slides_text = (docs_dir / "seminar_slides.md").read_text(encoding="utf-8")
    for text in (manuscript_text, slides_text):
        assert f"{th_bf:.1f}" in text
        assert f"{th_5q:.4f}" in text or f"{th_5q * 100:.2f}" in text
