"""Stdlib ThreadingHTTPServer REST API and static asset server for ``qecc.web``."""

from __future__ import annotations

import json
import math
import mimetypes
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal, NoReturn
from urllib.parse import parse_qs, unquote, urlsplit

import numpy as np
from pydantic import (
    BeforeValidator,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    ValidationError,
    model_validator,
)
from pydantic import BaseModel

from qecc.backends import list_available_backends
from qecc.channels.pauli import DeterministicErrorInjector
from qecc.codes import list_available_codes
from qecc.web.analytics_service import get_metadata, get_theory, run_simulation, run_sweeps
from qecc.web.stepper import run_step_evolution

STATIC_DIR: Path = (Path(__file__).resolve().parent / "static").resolve()
MAX_BODY_BYTES: int = 1_048_576  # 1 MiB

_CODE_DIMS: dict[str, tuple[int, int]] = {
    "raw": (1, 0),
    "bit_flip_3": (3, 2),
    "phase_flip_3": (3, 2),
    "five_qubit": (5, 4),
    "dfs_2": (2, 0),
}

_MIME_TYPES: dict[str, str] = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".mjs": "application/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".ico": "image/x-icon",
}


def _validate_finite_real_input(v: Any) -> float:
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float)):
        raise ValueError("Expected a finite real number (not bool or str)")
    fv = float(v)
    if not math.isfinite(fv):
        raise ValueError("Value must be finite (NaN/Inf not allowed)")
    return fv


FiniteFloat = Annotated[float, BeforeValidator(_validate_finite_real_input)]


def _reject_nonfinite_json(token: str) -> NoReturn:
    raise ValueError(f"Non-finite JSON constant {token!r} is not allowed")


def _validate_custom_amplitudes(
    amplitudes: list[list[float]] | None,
) -> None:
    if amplitudes is None or len(amplitudes) != 2:
        raise ValueError("amplitudes must have length 2 when state_preset/state_mode == 'custom'")
    for idx, pair in enumerate(amplitudes):
        if len(pair) != 2:
            raise ValueError(f"amplitudes[{idx}] must be a 2-element [re, im] list")
    norm_sq = sum(pair[0] * pair[0] + pair[1] * pair[1] for pair in amplitudes)
    if abs(norm_sq - 1.0) > 1e-6:
        raise ValueError(
            f"Custom state amplitudes must be unit-normalized (||psi||^2 = {norm_sq})"
        )


class StepRequest(BaseModel):
    """Strict request schema for ``POST /api/step``."""

    model_config = ConfigDict(extra="forbid", strict=True)

    code: Literal["bit_flip_3", "phase_flip_3", "five_qubit", "dfs_2"]
    state_preset: Literal["0", "1", "+", "-", "+i", "-i", "bloch", "custom"] = "+i"
    theta: FiniteFloat | None = None
    phi_state: FiniteFloat | None = None
    amplitudes: list[list[FiniteFloat]] | None = None
    error_mode: Literal[
        "pauli_string",
        "single_qubit_pauli",
        "stochastic_sample",
        "su2_rotation",
        "coherent_collective",
    ]
    pauli_string: StrictStr | None = None
    error_qubit: StrictInt | None = None
    error_op: Literal["I", "X", "Y", "Z"] | None = None
    channel: Literal["bit_flip", "phase_flip", "depolarizing", "collective_dephasing"] | None = None
    p: FiniteFloat | None = Field(default=None, ge=0.0, le=1.0)
    seed: StrictInt | None = Field(default=None, ge=0)
    su2_qubit: StrictInt | None = None
    su2_theta: FiniteFloat | None = None
    su2_nx: FiniteFloat | None = None
    su2_ny: FiniteFloat | None = None
    su2_nz: FiniteFloat | None = None
    collective_phi: FiniteFloat | None = None
    selected_syndrome: StrictStr | None = None

    @model_validator(mode="after")
    def _validate_cross_fields(self) -> StepRequest:
        n_data, n_ancilla = _CODE_DIMS[self.code]

        if self.state_preset == "bloch":
            if self.theta is None:
                raise ValueError("theta is required when state_preset == 'bloch'")
            if self.phi_state is None:
                self.phi_state = 0.0
        elif self.state_preset == "custom":
            _validate_custom_amplitudes(self.amplitudes)

        if self.error_mode == "pauli_string":
            if self.pauli_string is None:
                raise ValueError("pauli_string is required for error_mode == 'pauli_string'")
            DeterministicErrorInjector.validate_pauli_string(self.pauli_string, n_data)
        elif self.error_mode == "single_qubit_pauli":
            if self.error_qubit is None or self.error_op is None:
                raise ValueError(
                    "error_qubit and error_op are required for error_mode == 'single_qubit_pauli'"
                )
            if not (0 <= self.error_qubit < n_data):
                raise ValueError(f"error_qubit {self.error_qubit} out of range [0, {n_data})")
        elif self.error_mode == "stochastic_sample":
            if self.channel is None or self.p is None:
                raise ValueError(
                    "channel and p are required for error_mode == 'stochastic_sample'"
                )
        elif self.error_mode == "su2_rotation":
            if (
                self.su2_qubit is None
                or self.su2_theta is None
                or self.su2_nx is None
                or self.su2_ny is None
                or self.su2_nz is None
            ):
                raise ValueError(
                    "su2_qubit, su2_theta, su2_nx, su2_ny, and su2_nz are required "
                    "for error_mode == 'su2_rotation'"
                )
            if not (0 <= self.su2_qubit < n_data):
                raise ValueError(f"su2_qubit {self.su2_qubit} out of range [0, {n_data})")
            axis_norm = math.hypot(self.su2_nx, self.su2_ny, self.su2_nz)
            if axis_norm <= 1e-12:
                raise ValueError(
                    "SU(2) rotation axis (su2_nx, su2_ny, su2_nz) must have non-zero norm"
                )
        elif self.error_mode == "coherent_collective":
            if self.collective_phi is None:
                raise ValueError(
                    "collective_phi is required for error_mode == 'coherent_collective'"
                )

        if self.selected_syndrome is not None:
            if n_ancilla == 0:
                if self.selected_syndrome != "":
                    raise ValueError(
                        f"selected_syndrome must be '' for zero-ancilla code {self.code!r}"
                    )
            else:
                if len(self.selected_syndrome) != n_ancilla or any(
                    ch not in {"0", "1"} for ch in self.selected_syndrome
                ):
                    raise ValueError(
                        f"selected_syndrome must be a {n_ancilla}-bit binary string for {self.code!r}"
                    )

        return self


class SimulateRequest(BaseModel):
    """Strict request schema for ``POST /api/simulate``."""

    model_config = ConfigDict(extra="forbid", strict=True)

    code: Literal["raw", "bit_flip_3", "phase_flip_3", "five_qubit", "dfs_2"]
    channel: Literal["bit_flip", "phase_flip", "depolarizing", "collective_dephasing"] | None = None
    p: FiniteFloat = Field(default=0.05, ge=0.0, le=1.0)
    num_shots: StrictInt = Field(default=1000, ge=1, le=200_000)
    state_mode: Literal[
        "0", "1", "+", "-", "+i", "-i", "cardinal_avg", "bloch", "custom"
    ] = "+i"
    theta: FiniteFloat | None = None
    phi_state: FiniteFloat | None = None
    amplitudes: list[list[FiniteFloat]] | None = None
    seed: StrictInt | None = Field(default=42, ge=0)
    backend: Literal["auto", "numpy", "torch_cuda", "eigenkit_unitary"] = "auto"
    deterministic_errors: StrictStr | list[StrictStr] | None = None

    @model_validator(mode="after")
    def _validate_cross_fields(self) -> SimulateRequest:
        n_data, _ = _CODE_DIMS[self.code]

        if self.state_mode == "bloch":
            if self.theta is None:
                raise ValueError("theta is required when state_mode == 'bloch'")
            if self.phi_state is None:
                self.phi_state = 0.0
        elif self.state_mode == "custom":
            _validate_custom_amplitudes(self.amplitudes)

        if self.deterministic_errors is not None:
            if "channel" in self.model_fields_set and self.channel is not None:
                raise ValueError("Cannot specify both channel and deterministic_errors")
            self.channel = None
            if "p" not in self.model_fields_set:
                self.p = 0.0
            elif self.p != 0.0:
                raise ValueError(
                    "Non-zero p cannot be combined with deterministic_errors; require p == 0.0"
                )
            if isinstance(self.deterministic_errors, str):
                DeterministicErrorInjector.validate_pauli_string(
                    self.deterministic_errors, n_data
                )
            else:
                if len(self.deterministic_errors) not in (1, self.num_shots):
                    raise ValueError(
                        f"deterministic_errors list length ({len(self.deterministic_errors)}) "
                        f"must be 1 or match num_shots ({self.num_shots})"
                    )
                DeterministicErrorInjector.batch_to_codes(self.deterministic_errors, n_data)
        else:
            if "channel" not in self.model_fields_set:
                self.channel = "depolarizing"
            elif self.channel is None and self.p > 0.0:
                raise ValueError(
                    "channel must be provided when p > 0 and deterministic_errors is None"
                )

        return self


class SweepsRequest(BaseModel):
    """Strict request schema for ``POST /api/sweeps``."""

    model_config = ConfigDict(extra="forbid", strict=True)

    codes: list[Literal["raw", "bit_flip_3", "phase_flip_3", "five_qubit", "dfs_2"]] = Field(
        default_factory=lambda: ["bit_flip_3", "phase_flip_3", "five_qubit", "dfs_2"],
        min_length=1,
    )
    channels: list[
        Literal["bit_flip", "phase_flip", "depolarizing", "collective_dephasing"]
    ] = Field(
        default_factory=lambda: [
            "bit_flip",
            "phase_flip",
            "depolarizing",
            "collective_dephasing",
        ],
        min_length=1,
    )
    metric: Literal["entanglement", "state"] = "entanglement"
    p_min: FiniteFloat = Field(default=0.0, ge=0.0, le=1.0)
    p_max: FiniteFloat = Field(default=0.5, ge=0.0, le=1.0)
    num_p: StrictInt = Field(default=21, ge=2, le=501)
    run_live_mc: StrictBool = False
    num_shots: StrictInt = Field(default=2000, ge=1, le=100_000)
    seed: StrictInt | None = Field(default=20261118, ge=0)
    backend: Literal["auto", "numpy", "torch_cuda", "eigenkit_unitary"] = "auto"

    @model_validator(mode="after")
    def _validate_range(self) -> SweepsRequest:
        if self.p_min >= self.p_max:
            raise ValueError(f"p_min ({self.p_min}) must be strictly less than p_max ({self.p_max})")
        return self


class QECCThreadingHTTPServer(ThreadingHTTPServer):
    """Multi-threaded HTTP server with fast socket reuse and daemon worker threads."""

    allow_reuse_address = True
    daemon_threads = True
    block_on_close = False


def _format_validation_error(exc: ValidationError) -> str:
    parts: list[str] = []
    for err in exc.errors():
        loc = ".".join(str(item) for item in err.get("loc", ()))
        msg = str(err.get("msg", "Invalid value"))
        parts.append(f"{loc}: {msg}" if loc else msg)
    return "; ".join(parts)


class QECCRequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler serving the QECC REST API and static frontend assets."""

    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return

    def _send_bytes(
        self,
        status: int,
        content_type: str,
        body: bytes,
        *,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-cache")
        if extra_headers:
            for k, v in extra_headers.items():
                self.send_header(k, v)
        if self.close_connection and (
            not extra_headers
            or all(k.lower() != "connection" for k in extra_headers)
        ):
            self.send_header("Connection", "close")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _send_json(
        self,
        status: int,
        payload: Any,
        *,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        raw = json.dumps(payload, allow_nan=False).encode("utf-8")
        self._send_bytes(
            status,
            "application/json; charset=utf-8",
            raw,
            extra_headers=extra_headers,
        )

    def _send_error_json(
        self,
        status: int,
        message: str,
        *,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self._send_json(status, {"error": message}, extra_headers=extra_headers)

    def _serve_static_file(self, target: Path) -> None:
        suffix = target.suffix.lower()
        ctype = _MIME_TYPES.get(suffix)
        if ctype is None:
            guessed, _ = mimetypes.guess_type(str(target))
            ctype = guessed or "application/octet-stream"
        data = target.read_bytes()
        self._send_bytes(HTTPStatus.OK, ctype, data)

    def _handle_static_rel(self, raw_rel: str) -> None:
        if "\x00" in raw_rel or "\\" in raw_rel or raw_rel.startswith("/"):
            self._send_error_json(HTTPStatus.FORBIDDEN, "Forbidden static path")
            return
        decoded = unquote(raw_rel)
        if not decoded or "\x00" in decoded or "\\" in decoded or decoded.startswith("/"):
            self._send_error_json(HTTPStatus.FORBIDDEN, "Forbidden static path")
            return
        raw_parts = PurePosixPath(raw_rel).parts
        dec_parts = PurePosixPath(decoded).parts
        if ".." in raw_parts or ".." in dec_parts:
            self._send_error_json(HTTPStatus.FORBIDDEN, "Forbidden static path")
            return

        static_root = STATIC_DIR.resolve()
        target = (static_root / decoded).resolve()
        if not target.is_relative_to(static_root) or target == static_root:
            self._send_error_json(HTTPStatus.FORBIDDEN, "Forbidden static path")
            return
        if not target.is_file():
            self._send_error_json(HTTPStatus.NOT_FOUND, f"Static asset not found: {decoded}")
            return
        self._serve_static_file(target)

    def do_GET(self) -> None:  # noqa: N802
        if self.headers.get("Content-Length") or self.headers.get("Transfer-Encoding"):
            self.close_connection = True
        parsed = urlsplit(self.path)
        path = parsed.path

        if path == "/":
            static_root = STATIC_DIR.resolve()
            index_path = (static_root / "index.html").resolve()
            if not index_path.is_relative_to(static_root) or not index_path.is_file():
                self._send_error_json(
                    HTTPStatus.NOT_FOUND,
                    "Frontend static asset not found: index.html",
                )
                return
            self._serve_static_file(index_path)
            return

        if path.startswith("/static/"):
            self._handle_static_rel(path[len("/static/") :])
            return

        if path == "/api/health":
            self._send_json(
                HTTPStatus.OK,
                {
                    "status": "ok",
                    "codes": list_available_codes(),
                    "backends": list_available_backends(),
                },
            )
            return

        if path == "/api/metadata":
            self._send_json(HTTPStatus.OK, get_metadata())
            return

        if path == "/api/theory":
            qs = parse_qs(parsed.query, keep_blank_values=True)
            if "code" not in qs:
                self._send_error_json(
                    HTTPStatus.BAD_REQUEST,
                    "Missing required query parameter 'code'",
                )
                return
            if len(qs["code"]) != 1:
                self._send_error_json(
                    HTTPStatus.BAD_REQUEST,
                    "Query parameter 'code' must be specified once",
                )
                return
            code_name = qs["code"][0]
            try:
                payload = get_theory(code_name)
            except ValueError as exc:
                self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
                return
            self._send_json(HTTPStatus.OK, payload)
            return

        if path in {"/api/step", "/api/simulate", "/api/sweeps"}:
            self._send_error_json(
                HTTPStatus.METHOD_NOT_ALLOWED,
                f"Method {self.command} not allowed for {path}; expected POST",
                extra_headers={"Allow": "POST"},
            )
            return

        self._send_error_json(HTTPStatus.NOT_FOUND, f"Unknown route: {self.command} {path}")

    def _handle_unsupported_method(self) -> None:
        if self.headers.get("Content-Length") or self.headers.get("Transfer-Encoding"):
            self.close_connection = True
        path = urlsplit(self.path).path
        if path in {"/api/health", "/api/metadata", "/api/theory", "/"} or path.startswith("/static/"):
            self._send_error_json(
                HTTPStatus.METHOD_NOT_ALLOWED,
                f"Method {self.command} not allowed for {path}; expected GET",
                extra_headers={"Allow": "GET, HEAD"},
            )
            return
        if path in {"/api/step", "/api/simulate", "/api/sweeps"}:
            self._send_error_json(
                HTTPStatus.METHOD_NOT_ALLOWED,
                f"Method {self.command} not allowed for {path}; expected POST",
                extra_headers={"Allow": "POST"},
            )
            return
        if path.startswith("/api/"):
            self._send_error_json(
                HTTPStatus.METHOD_NOT_ALLOWED,
                f"Method {self.command} not allowed for {path}",
            )
            return
        self._send_error_json(HTTPStatus.NOT_FOUND, f"Unknown route: {self.command} {path}")

    def do_HEAD(self) -> None:  # noqa: N802
        self.do_GET()

    def do_PUT(self) -> None:  # noqa: N802
        self._handle_unsupported_method()

    def do_DELETE(self) -> None:  # noqa: N802
        self._handle_unsupported_method()

    def do_PATCH(self) -> None:  # noqa: N802
        self._handle_unsupported_method()

    def do_OPTIONS(self) -> None:  # noqa: N802
        self._handle_unsupported_method()

    def _read_json_body(self) -> dict[str, Any] | None:
        has_te = bool(self.headers.get("Transfer-Encoding"))
        ctype_header = self.headers.get("Content-Type")
        if ctype_header is not None:
            media_type = ctype_header.split(";")[0].strip().lower()
            if media_type != "application/json":
                self.close_connection = True
                self._send_error_json(
                    HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                    f"Unsupported Content-Type {media_type!r}; expected 'application/json'",
                )
                return None

        cl_header = self.headers.get("Content-Length")
        if cl_header is None:
            if has_te:
                self.close_connection = True
            self._send_error_json(HTTPStatus.BAD_REQUEST, "Missing Content-Length header")
            return None
        try:
            content_length = int(cl_header)
        except ValueError:
            self.close_connection = True
            self._send_error_json(HTTPStatus.BAD_REQUEST, "Invalid Content-Length header")
            return None
        if content_length <= 0:
            if content_length < 0 or has_te:
                self.close_connection = True
            self._send_error_json(HTTPStatus.BAD_REQUEST, "Request body must not be empty")
            return None
        if content_length > MAX_BODY_BYTES:
            self.close_connection = True
            self._send_error_json(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                f"Request body exceeds maximum allowed size ({MAX_BODY_BYTES} bytes)",
            )
            return None
        if has_te:
            self.close_connection = True
            self._send_error_json(
                HTTPStatus.BAD_REQUEST,
                "Transfer-Encoding is not supported; send Content-Length",
            )
            return None

        raw_bytes = self.rfile.read(content_length)
        if len(raw_bytes) != content_length:
            self.close_connection = True
            self._send_error_json(HTTPStatus.BAD_REQUEST, "Truncated request body")
            return None

        try:
            decoded_text = raw_bytes.decode("utf-8")
            parsed = json.loads(decoded_text, parse_constant=_reject_nonfinite_json)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            self._send_error_json(HTTPStatus.BAD_REQUEST, f"Invalid JSON payload: {exc}")
            return None

        if not isinstance(parsed, dict):
            self._send_error_json(HTTPStatus.BAD_REQUEST, "JSON body must be an object")
            return None
        return parsed

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlsplit(self.path)
        path = parsed.path

        if path in {"/api/health", "/api/metadata", "/api/theory", "/"} or path.startswith("/static/"):
            if self.headers.get("Content-Length") or self.headers.get("Transfer-Encoding"):
                self.close_connection = True
            self._send_error_json(
                HTTPStatus.METHOD_NOT_ALLOWED,
                f"Method POST not allowed for {path}; expected GET",
                extra_headers={"Allow": "GET, HEAD"},
            )
            return

        if path not in {"/api/step", "/api/simulate", "/api/sweeps"}:
            if self.headers.get("Content-Length") or self.headers.get("Transfer-Encoding"):
                self.close_connection = True
            self._send_error_json(HTTPStatus.NOT_FOUND, f"Unknown route: POST {path}")
            return

        body = self._read_json_body()
        if body is None:
            return

        try:
            if path == "/api/step":
                req = StepRequest.model_validate(body)
                result = run_step_evolution(
                    code_name=req.code,
                    state_preset=req.state_preset,
                    theta=req.theta,
                    phi_state=req.phi_state,
                    amplitudes=req.amplitudes,
                    error_mode=req.error_mode,
                    pauli_string=req.pauli_string,
                    error_qubit=req.error_qubit,
                    error_op=req.error_op,
                    channel=req.channel,
                    p=req.p,
                    seed=req.seed,
                    su2_qubit=req.su2_qubit,
                    su2_theta=req.su2_theta,
                    su2_nx=req.su2_nx,
                    su2_ny=req.su2_ny,
                    su2_nz=req.su2_nz,
                    collective_phi=req.collective_phi,
                    selected_syndrome=req.selected_syndrome,
                )
            elif path == "/api/simulate":
                req = SimulateRequest.model_validate(body)
                result = run_simulation(
                    code=req.code,
                    channel=req.channel,
                    p=req.p,
                    num_shots=req.num_shots,
                    state_mode=req.state_mode,
                    theta=req.theta,
                    phi_state=req.phi_state,
                    amplitudes=req.amplitudes,
                    seed=req.seed,
                    backend=req.backend,
                    deterministic_errors=req.deterministic_errors,
                )
            else:
                req = SweepsRequest.model_validate(body)
                result = run_sweeps(
                    codes=req.codes,
                    channels=req.channels,
                    metric=req.metric,
                    p_min=req.p_min,
                    p_max=req.p_max,
                    num_p=req.num_p,
                    run_live_mc=req.run_live_mc,
                    num_shots=req.num_shots,
                    seed=req.seed,
                    backend=req.backend,
                )
        except ValidationError as exc:
            self._send_error_json(
                HTTPStatus.UNPROCESSABLE_ENTITY, _format_validation_error(exc)
            )
            return
        except NotImplementedError as exc:
            self._send_error_json(HTTPStatus.NOT_IMPLEMENTED, str(exc))
            return
        except (ValueError, RuntimeError) as exc:
            self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
            return

        self._send_json(HTTPStatus.OK, result)


def create_server(host: str = "127.0.0.1", port: int = 8765) -> QECCThreadingHTTPServer:
    """Create and bind a ``QECCThreadingHTTPServer`` (supports ``port=0``)."""
    if isinstance(port, bool) or not isinstance(port, int) or not (0 <= port <= 65535):
        raise ValueError(f"port must be an integer in [0, 65535], got {port!r}")
    return QECCThreadingHTTPServer((host, port), QECCRequestHandler)


def run_server(
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
) -> int:
    """Bind the local QECC web server, print its bound URL, and serve requests."""
    server = create_server(host=host, port=port)
    bound_host = str(server.server_address[0])
    bound_port = int(server.server_address[1])
    url = f"http://{bound_host}:{bound_port}"
    print(f"QECC web server running at {url}", flush=True)

    if open_browser:
        def _open_url() -> None:
            try:
                webbrowser.open(url)
            except Exception:
                pass

        timer = threading.Timer(0.25, _open_url)
        timer.daemon = True
        timer.start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0
