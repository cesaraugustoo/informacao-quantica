"""Shared pytest configuration, deferred-import fixtures, and math/parsing helpers for E2E tests.

All ``qecc.*`` imports are strictly deferred inside fixtures or test functions so
``pytest --collect-only tests/e2e`` succeeds cleanly even before ``src/qecc`` is built.
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
RESULTS_DIR = REPO_ROOT / "results"
DOCS_DIR = REPO_ROOT / "docs"
SCRIPTS_DIR = REPO_ROOT / "scripts"

_PAULI_MATRICES: dict[str, np.ndarray] = {
    "I": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.complex128),
    "X": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128),
    "Y": np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128),
    "Z": np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128),
}
_INT_TO_PAULI = ("I", "X", "Y", "Z")


def _ensure_src_on_sys_path() -> None:
    if SRC_DIR.is_dir():
        src_str = str(SRC_DIR)
        if src_str not in sys.path:
            sys.path.append(src_str)


def pytest_configure(config: pytest.Config) -> None:
    """Register custom E2E tier markers and ensure ``src/`` fallback on ``sys.path``."""
    config.addinivalue_line("markers", "tier1: Tier 1 core feature coverage E2E tests.")
    config.addinivalue_line("markers", "tier2: Tier 2 boundary and corner-case E2E tests.")
    config.addinivalue_line("markers", "tier3: Tier 3 cross-feature pairwise combination E2E tests.")
    config.addinivalue_line("markers", "tier4: Tier 4 end-to-end application scenario tests.")
    config.addinivalue_line("markers", "tier5: Tier 5 white-box adversarial E2E tests.")
    _ensure_src_on_sys_path()


# ---------------------------------------------------------------------------
# Pure linear-algebra, statistical, and markdown/artifact helpers (no qecc)
# ---------------------------------------------------------------------------


def ket_to_rho(psi: np.ndarray) -> np.ndarray:
    """Convert a single-qubit statevector (2,) into a normalized (2, 2) complex128 density matrix."""
    vec = np.asarray(psi, dtype=np.complex128).reshape(2)
    norm = np.linalg.norm(vec)
    if norm <= 0.0:
        raise ValueError("Zero-norm statevector cannot be converted to a density matrix.")
    vec = vec / norm
    return np.outer(vec, vec.conj())


def pauli_string_to_matrix(pauli_str: str) -> np.ndarray:
    """Build big-endian Kronecker matrix (qubit 0 is MSB) for a Pauli string over I/X/Y/Z."""
    if not pauli_str:
        raise ValueError("Pauli string must be non-empty.")
    mat = np.array([[1.0 + 0.0j]], dtype=np.complex128)
    for ch in pauli_str:
        if ch not in _PAULI_MATRICES:
            raise ValueError(f"Invalid Pauli symbol {ch!r} in {pauli_str!r}.")
        mat = np.kron(mat, _PAULI_MATRICES[ch])
    return mat


def purity(rho: np.ndarray) -> float:
    """Return Tr(rho^2) as a real float."""
    arr = np.asarray(rho, dtype=np.complex128)
    return float(np.real(np.trace(arr @ arr)))


def bloch_to_rho(bloch_vec: tuple[float, float, float] | np.ndarray) -> np.ndarray:
    """Construct single-qubit density matrix 0.5 * (I + r_x X + r_y Y + r_z Z)."""
    rx, ry, rz = (float(bloch_vec[0]), float(bloch_vec[1]), float(bloch_vec[2]))
    return 0.5 * (
        _PAULI_MATRICES["I"]
        + rx * _PAULI_MATRICES["X"]
        + ry * _PAULI_MATRICES["Y"]
        + rz * _PAULI_MATRICES["Z"]
    )


def jozsa_single_qubit_fidelity(
    r_vec: tuple[float, float, float] | np.ndarray,
    s_vec: tuple[float, float, float] | np.ndarray,
) -> float:
    """Exact squared Uhlmann-Jozsa fidelity for two single-qubit Bloch vectors."""
    r = np.asarray(r_vec, dtype=np.float64)
    s = np.asarray(s_vec, dtype=np.float64)
    r2 = float(np.dot(r, r))
    s2 = float(np.dot(s, s))
    return 0.5 * (1.0 + float(np.dot(r, s)) + math.sqrt(max(0.0, 1.0 - r2) * max(0.0, 1.0 - s2)))


def canonical_states() -> dict[str, np.ndarray]:
    """Return canonical 6 single-qubit cardinal statevectors."""
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    return {
        "|0>": np.array([1.0, 0.0], dtype=np.complex128),
        "|1>": np.array([0.0, 1.0], dtype=np.complex128),
        "|+>": np.array([inv_sqrt2, inv_sqrt2], dtype=np.complex128),
        "|->": np.array([inv_sqrt2, -inv_sqrt2], dtype=np.complex128),
        "|+i>": np.array([inv_sqrt2, 1.0j * inv_sqrt2], dtype=np.complex128),
        "|-i>": np.array([inv_sqrt2, -1.0j * inv_sqrt2], dtype=np.complex128),
    }


def normalize_pauli_samples(samples: Any, n_qubits: int) -> np.ndarray:
    """Normalize StochasticPauliChannel.sample output to (num_shots, n_qubits) str array."""
    arr = np.asarray(samples)
    if arr.ndim == 1:
        if arr.size == 0:
            return np.empty((0, n_qubits), dtype="U1")
        rows: list[list[str]] = []
        for item in arr.tolist():
            text = str(item).strip()
            if len(text) != n_qubits or any(ch not in _PAULI_MATRICES for ch in text):
                raise ValueError(f"Invalid sampled Pauli string {text!r} for n_qubits={n_qubits}.")
            rows.append(list(text))
        return np.asarray(rows, dtype="U1")
    if arr.ndim == 2 and arr.shape[1] == n_qubits:
        if arr.shape[0] == 0:
            return np.empty((0, n_qubits), dtype="U1")
        if np.issubdtype(arr.dtype, np.integer):
            out = np.empty(arr.shape, dtype="U1")
            for idx, sym in enumerate(_INT_TO_PAULI):
                out[arr == idx] = sym
            if np.any((arr < 0) | (arr > 3)):
                raise ValueError(f"Invalid integer Pauli code in samples: {arr}")
            return out
        out = arr.astype("U1")
        valid = np.isin(out, list(_PAULI_MATRICES.keys()))
        if not np.all(valid):
            raise ValueError("Invalid character in 2D Pauli sample array.")
        return out
    raise ValueError(f"Unexpected Pauli sample shape {arr.shape} for n_qubits={n_qubits}.")


def binom_sigma(p: float, n: int, z: float = 3.0) -> float:
    """Binomial z-sigma bound with a 1e-10 numerical floor at deterministic endpoints."""
    return max(float(z) * math.sqrt(max(p * (1.0 - p), 0.0) / max(n, 1)), 1e-10)


def binom_3sigma(p: float, n: int) -> float:
    """Binomial 3-sigma bound with a 1e-10 numerical floor at deterministic endpoints."""
    return binom_sigma(p, n, z=3.0)


def five_qubit_exact_pi(p: float) -> float:
    """Exact 1024-term entanglement fidelity P_I(p) of [[5,1,3]] under depolarizing noise."""
    q = 1.0 - p
    return (
        q**5
        + 5.0 * p * (q**4)
        + (20.0 / 9.0) * (p**3) * (q**2)
        + (5.0 / 3.0) * (p**4) * q
        + (5.0 / 27.0) * (p**5)
    )


def _iter_fenced_code_lines(text: str) -> tuple[list[str], bool]:
    """Return non-fenced lines and whether all fenced code blocks (``` / ~~~) are closed."""
    out_lines: list[str] = []
    fence_char: str | None = None
    fence_len = 0
    for line in text.splitlines(keepends=True):
        stripped = line.lstrip(" ")
        indent = len(line) - len(stripped)
        body = stripped.rstrip("\r\n")
        if fence_char is None:
            m = re.match(r"^(`{3,}|~{3,})(.*)$", body)
            if indent <= 3 and m:
                marker, info = m.group(1), m.group(2)
                if marker[0] == "`" and "`" in info:
                    out_lines.append(line)
                else:
                    fence_char = marker[0]
                    fence_len = len(marker)
            else:
                out_lines.append(line)
        else:
            m_close = re.match(r"^([`~]+)\s*$", body)
            if (
                indent <= 3
                and m_close
                and m_close.group(1)[0] == fence_char
                and len(m_close.group(1)) >= fence_len
            ):
                fence_char = None
                fence_len = 0
                out_lines.append("\n")
    return out_lines, (fence_char is None)


def has_balanced_fenced_code_blocks(text: str) -> bool:
    """Return True if all ``` and ~~~ fenced code blocks in Markdown text are properly closed."""
    _, balanced = _iter_fenced_code_lines(text)
    return balanced


def strip_fenced_code_blocks(text: str) -> str:
    """Remove ```...``` and ~~~...~~~ fenced code blocks from Markdown text."""
    out_lines, _ = _iter_fenced_code_lines(text)
    return "".join(out_lines)


def strip_inline_code_spans(text: str) -> str:
    """Remove Markdown inline code spans (`...`, ``...``, etc.) from text."""
    return re.sub(r"(`+)(?:(?!\1)[\s\S])*?\1", "", text)


def strip_escaped_dollars(text: str) -> str:
    """Remove escaped literal '\\$' while preserving '\\\\$' (escaped backslash before math '$')."""
    return re.sub(r"(?:\\\\)+|\\\$", lambda m: "" if m.group(0) == r"\$" else m.group(0), text)


def split_markdown_table_row(row: str) -> list[str]:
    """Split a GFM table row on unescaped '|' outside inline code spans and inline '$...$' math."""
    s = row.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|") and not s.endswith(r"\|"):
        s = s[:-1]
    masked = re.sub(r"(`+)(?:(?!\1)[^`])*\1", lambda m: " " * len(m.group(0)), s)
    masked = re.sub(r"(?<!\\)\$[^$\n]*(?<!\\)\$", lambda m: " " * len(m.group(0)), masked)
    masked = re.sub(r"(?:\\\\)+", lambda m: " " * len(m.group(0)), masked)
    return re.split(r"(?<!\\)\|", masked)


def slugify_heading(heading: str) -> str:
    """Convert a Markdown heading to a GitHub-style anchor slug."""
    cleaned = re.sub(r"[`*{}\[\]()#+!.$\\:&/,;'\"?<>|=~^%@]", "", heading.strip().lower())
    return re.sub(r"\s", "-", cleaned.strip())


def has_rubric_fail_verdict(text: str) -> bool:
    """Return True if rubric text contains a standalone FAIL verdict (ignoring 'PASS / FAIL' headers)."""
    cleaned = re.sub(r"\bPASS\s*/\s*FAIL\b", "", text)
    if re.search(r"\|\s*\*{0,2}FAIL\*{0,2}\s*\|", cleaned):
        return True
    return re.search(r"(?:^|\||\*\*|:|\[|\s)FAIL(?:\||\*\*|\]|\s|$)", cleaned) is not None


_EXCLUDED_METRIC_TOKENS: frozenset[str] = frozenset(
    {
        "gain",
        "diff",
        "delta",
        "advantage",
        "se",
        "ci",
        "err",
        "error",
        "stderr",
        "var",
        "variance",
        "sigma",
        "std",
        "stdev",
    }
)
_EXCLUDED_METRIC_SUBSTRINGS: tuple[str, ...] = (
    "gain",
    "diff",
    "delta",
    "advantage",
    "sigma",
    "stderr",
    "variance",
)


def is_fidelity_metric_name(name: str) -> bool:
    """Return True if key/column name denotes a fidelity value rather than error/diff/gain/uncertainty."""
    kl = str(name).strip().lower()
    if "fidelity" not in kl and not kl.startswith("f_"):
        return False
    if any(sub in kl for sub in _EXCLUDED_METRIC_SUBSTRINGS):
        return False
    tokens = set(re.split(r"[^a-z0-9]+", kl))
    return _EXCLUDED_METRIC_TOKENS.isdisjoint(tokens)


def extract_markdown_image_targets(text: str) -> list[str]:
    """Extract local image link targets from Markdown ![...](path) and <img src='path'>."""
    cleaned = strip_fenced_code_blocks(text)
    targets: list[str] = []
    for m in re.finditer(r"!\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)", cleaned):
        tgt = m.group(1).strip()
        if not re.match(r"^https?://", tgt, re.IGNORECASE):
            targets.append(tgt)
    for m in re.finditer(r"<img[^>]+src=[\"']([^\"']+)[\"']", cleaned, re.IGNORECASE):
        tgt = m.group(1).strip()
        if not re.match(r"^https?://", tgt, re.IGNORECASE):
            targets.append(tgt)
    return targets


def flatten_json_text(obj: Any) -> str:
    """Flatten nested JSON structures into lowercase searchable text."""
    if isinstance(obj, dict):
        return " ".join(f"{k} {flatten_json_text(v)}" for k, v in obj.items())
    if isinstance(obj, list):
        return " ".join(flatten_json_text(v) for v in obj)
    return str(obj)


# ---------------------------------------------------------------------------
# Pytest Fixtures (all qecc imports strictly deferred inside fixtures)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def results_dir() -> Path:
    return RESULTS_DIR


@pytest.fixture(scope="session")
def docs_dir() -> Path:
    return DOCS_DIR


@pytest.fixture(scope="session")
def scripts_dir() -> Path:
    return SCRIPTS_DIR


@pytest.fixture(scope="session")
def python_executable() -> Path:
    venv_py = REPO_ROOT / ".venv" / "bin" / "python"
    if venv_py.exists():
        return venv_py
    return Path(sys.executable)


@pytest.fixture(scope="session")
def helpers() -> SimpleNamespace:
    return SimpleNamespace(
        ket_to_rho=ket_to_rho,
        pauli_string_to_matrix=pauli_string_to_matrix,
        purity=purity,
        bloch_to_rho=bloch_to_rho,
        jozsa_single_qubit_fidelity=jozsa_single_qubit_fidelity,
        canonical_states=canonical_states,
        normalize_pauli_samples=normalize_pauli_samples,
        binom_sigma=binom_sigma,
        binom_3sigma=binom_3sigma,
        five_qubit_exact_pi=five_qubit_exact_pi,
        has_balanced_fenced_code_blocks=has_balanced_fenced_code_blocks,
        strip_fenced_code_blocks=strip_fenced_code_blocks,
        strip_inline_code_spans=strip_inline_code_spans,
        strip_escaped_dollars=strip_escaped_dollars,
        split_markdown_table_row=split_markdown_table_row,
        slugify_heading=slugify_heading,
        has_rubric_fail_verdict=has_rubric_fail_verdict,
        is_fidelity_metric_name=is_fidelity_metric_name,
        extract_markdown_image_targets=extract_markdown_image_targets,
        flatten_json_text=flatten_json_text,
    )


@pytest.fixture(scope="session")
def qecc_codes():
    _ensure_src_on_sys_path()
    import qecc.codes as mod

    return mod


@pytest.fixture(scope="session")
def qecc_backends():
    _ensure_src_on_sys_path()
    import qecc.backends as mod

    return mod


@pytest.fixture(scope="session")
def qecc_channels():
    _ensure_src_on_sys_path()
    import qecc.channels as mod

    return mod


@pytest.fixture(scope="session")
def qecc_metrics():
    _ensure_src_on_sys_path()
    import qecc.metrics as mod

    return mod


@pytest.fixture(scope="session")
def qecc_analysis():
    _ensure_src_on_sys_path()
    import qecc.analysis as mod

    return mod


@pytest.fixture(scope="session")
def numpy_backend(qecc_backends):
    return qecc_backends.get_backend("numpy")
