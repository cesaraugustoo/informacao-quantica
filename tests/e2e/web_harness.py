"""Opaque-box E2E test harness for the QECC & DFS interactive web application.

Provides:
- ``ServerHandle``: subprocess launcher for ``python -m qecc.web`` and ``qecc.cli serve``
- ``http_request_raw`` / ``http_request_json``: stdlib HTTP client preserving raw paths & 4xx/5xx bodies
- ``StdlibWS``: pure Python 3.12 stdlib RFC 6455 WebSocket client (masked client / fragmented & 64-bit server frames)
- ``BrowserSession`` / ``PageSession``: zero-dependency CDP driver for local ``chrome-headless-shell``
- Exact ``<= 1e-12`` mathematical oracle helpers (with deferred ``qecc.*`` imports)
- Bilingual & layout verification helpers

All ``qecc.*`` imports are strictly deferred inside functions so ``pytest --collect-only tests/e2e``
succeeds cleanly with zero top-level ``qecc`` dependencies.
"""

from __future__ import annotations

import base64
import glob
import hashlib
import http.client
import itertools
import json
import math
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"

_PAULI_MATS: dict[str, np.ndarray] = {
    "I": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.complex128),
    "X": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128),
    "Y": np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128),
    "Z": np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128),
}

_PORT_URL_RE = re.compile(r"https?://([0-9a-zA-Z.\-:]+):(\d+)")
_WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

REQUIRED_TESTIDS: tuple[str, ...] = (
    "code-select",
    "channel-select",
    "state-select",
    "p-input",
    "phi-input",
    "stage-list",
    "bloch-canvas",
    "coherence-panel",
    "curves-chart",
    "theory-panel",
    "seminar-panel",
)

STAGE_KEYS: tuple[str, ...] = (
    "initial",
    "encoded",
    "noisy",
    "syndrome_extracted",
    "recovered",
    "decoded",
)


def _ensure_src_on_sys_path() -> None:
    if SRC_DIR.is_dir():
        src_str = str(SRC_DIR)
        if src_str not in sys.path:
            sys.path.append(src_str)


def _python_executable() -> str:
    venv_py = REPO_ROOT / ".venv" / "bin" / "python"
    if venv_py.exists():
        return str(venv_py)
    return sys.executable


# ---------------------------------------------------------------------------
# HTTP Client Helpers (stdlib http.client — preserves raw paths & 4xx/5xx)
# ---------------------------------------------------------------------------


def _parse_base_url(base_url: str) -> tuple[str, int, str]:
    parsed = urllib.parse.urlsplit(base_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    prefix = parsed.path.rstrip("/")
    return host, port, prefix


def http_request_raw(
    base_url: str,
    path: str,
    *,
    method: str = "GET",
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 15.0,
) -> tuple[int, dict[str, str], bytes]:
    """Send an HTTP request preserving raw ``path`` (no client-side dot-segment normalization)."""
    host, port, prefix = _parse_base_url(base_url)
    req_path = f"{prefix}{path}" if prefix and path.startswith("/") else path
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        req_headers = dict(headers or {})
        conn.request(method.upper(), req_path, body=body, headers=req_headers)
        resp = conn.getresponse()
        raw_bytes = resp.read()
        resp_headers = {k.lower(): v for k, v in resp.getheaders()}
        return int(resp.status), resp_headers, raw_bytes
    finally:
        conn.close()


def http_request_json(
    base_url: str,
    path: str,
    *,
    method: str = "GET",
    payload: Any = None,
    raw_body: bytes | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 15.0,
) -> tuple[int, dict[str, str], Any]:
    """Send an HTTP request and parse JSON response if applicable (returns text on non-JSON)."""
    req_headers = {"Accept": "application/json"}
    if headers:
        req_headers.update(headers)

    body_bytes: bytes | None = raw_body
    if payload is not None and raw_body is None:
        body_bytes = json.dumps(payload).encode("utf-8")
        req_headers.setdefault("Content-Type", "application/json")
    elif raw_body is not None:
        req_headers.setdefault("Content-Type", "application/json")

    status, resp_headers, raw_bytes = http_request_raw(
        base_url,
        path,
        method=method,
        body=body_bytes,
        headers=req_headers,
        timeout=timeout,
    )
    text = raw_bytes.decode("utf-8", errors="replace")
    if not text.strip():
        return status, resp_headers, None
    try:
        return status, resp_headers, json.loads(text)
    except json.JSONDecodeError:
        return status, resp_headers, text


def assert_json_error(
    status: int,
    body: Any,
    *,
    allowed_statuses: tuple[int, ...] = (400, 422),
) -> str:
    """Assert HTTP status is in ``allowed_statuses`` and body is JSON containing ``error`` or ``detail``."""
    assert status in allowed_statuses, f"Expected status in {allowed_statuses}, got {status} with body={body!r}"
    assert isinstance(body, dict), f"Expected JSON dict error body, got {type(body).__name__}: {body!r}"
    msg = body.get("error") if "error" in body else body.get("detail")
    assert msg is not None and str(msg).strip() != "", f"Expected non-empty 'error' or 'detail' in {body!r}"
    return str(msg)


# ---------------------------------------------------------------------------
# Server Subprocess Lifecycle
# ---------------------------------------------------------------------------


class ServerHandle:
    """Isolated subprocess handle for ``python -m qecc.web`` or ``python -m qecc.cli serve``."""

    def __init__(
        self,
        *,
        use_cli: bool = False,
        host: str = "127.0.0.1",
        port: int = 0,
        extra_args: Sequence[str] = (),
        startup_timeout: float = 15.0,
    ) -> None:
        self.use_cli = use_cli
        self.host = host
        self.requested_port = port
        self.extra_args = list(extra_args)
        self.startup_timeout = startup_timeout
        self.proc: subprocess.Popen[str] | None = None
        self.base_url: str = ""
        self.logs: list[str] = []
        self._lock = threading.Lock()
        self._url_event = threading.Event()
        self._reader_thread: threading.Thread | None = None

    def start(self) -> "ServerHandle":
        py = _python_executable()
        if self.use_cli:
            cmd = [
                py,
                "-m",
                "qecc.cli",
                "serve",
                "--host",
                self.host,
                "--port",
                str(self.requested_port),
                "--no-browser",
                *self.extra_args,
            ]
        else:
            cmd = [
                py,
                "-m",
                "qecc.web",
                "--host",
                self.host,
                "--port",
                str(self.requested_port),
                "--no-browser",
                *self.extra_args,
            ]

        env = os.environ.copy()
        src_str = str(SRC_DIR)
        existing_pp = env.get("PYTHONPATH", "")
        if src_str not in existing_pp.split(os.pathsep):
            env["PYTHONPATH"] = f"{src_str}{os.pathsep}{existing_pp}" if existing_pp else src_str
        env.setdefault("PYTHONUNBUFFERED", "1")

        self.proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        def _read_output() -> None:
            assert self.proc is not None and self.proc.stdout is not None
            for line in self.proc.stdout:
                cleaned = line.rstrip("\r\n")
                with self._lock:
                    self.logs.append(cleaned)
                m = _PORT_URL_RE.search(cleaned)
                if m and not self.base_url:
                    bound_host = m.group(1)
                    if bound_host in ("0.0.0.0", "::", "[::]"):
                        bound_host = "127.0.0.1"
                    bound_port = int(m.group(2))
                    if bound_port > 0:
                        self.base_url = f"http://{bound_host}:{bound_port}"
                        self._url_event.set()

        self._reader_thread = threading.Thread(target=_read_output, daemon=True)
        self._reader_thread.start()

        deadline = time.monotonic() + self.startup_timeout
        if self.requested_port > 0 and not self.base_url:
            # Also allow fallback to requested_port if server logs minimally
            self._url_event.wait(timeout=min(2.0, self.startup_timeout))
            if not self.base_url:
                self.base_url = f"http://{self.host}:{self.requested_port}"
        else:
            while not self._url_event.is_set():
                if self.proc.poll() is not None:
                    self.stop()
                    raise RuntimeError(
                        f"Server exited prematurely with rc={self.proc.returncode}.\n"
                        f"Command: {' '.join(cmd)}\nLogs:\n{self.get_logs()}"
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self.stop()
                    raise TimeoutError(
                        f"Timed out waiting for server bound URL.\nCommand: {' '.join(cmd)}\nLogs:\n{self.get_logs()}"
                    )
                self._url_event.wait(timeout=min(0.1, remaining))

        # Poll /api/health until 200 OK
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                rc = self.proc.returncode
                self.stop()
                raise RuntimeError(
                    f"Server exited during health check (rc={rc}).\nLogs:\n{self.get_logs()}"
                )
            try:
                status, _, body = http_request_json(self.base_url, "/api/health", timeout=1.0)
                if status == 200 and isinstance(body, dict) and body.get("status") == "ok":
                    return self
            except Exception:
                pass
            time.sleep(0.05)

        self.stop()
        raise TimeoutError(
            f"Server at {self.base_url} failed /api/health check within {self.startup_timeout}s.\nLogs:\n{self.get_logs()}"
        )

    def get_logs(self) -> str:
        with self._lock:
            return "\n".join(self.logs)

    def stop(self, timeout: float = 5.0) -> int | None:
        if self.proc is None:
            return None
        rc = self.proc.poll()
        if rc is None:
            self.proc.terminate()
            try:
                rc = self.proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                rc = self.proc.wait(timeout=timeout)
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=1.0)
        return rc

    def __enter__(self) -> "ServerHandle":
        return self.start()

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.stop()


# ---------------------------------------------------------------------------
# Pure Python 3.12 Stdlib RFC 6455 WebSocket Client
# ---------------------------------------------------------------------------


class WebSocketClosed(EOFError):
    """Raised when the remote WebSocket peer closes the connection."""


class StdlibWS:
    """Minimal RFC 6455 WebSocket client built purely on Python stdlib sockets."""

    def __init__(self, ws_url: str, timeout: float = 15.0) -> None:
        parsed = urllib.parse.urlsplit(ws_url)
        if parsed.scheme != "ws":
            raise ValueError(f"Only ws:// URLs are supported, got {ws_url!r}")
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or 80
        path = parsed.path or "/"
        if parsed.query:
            path = f"{path}?{parsed.query}"

        self._sock = socket.create_connection((host, port), timeout=timeout)
        self._sock.settimeout(None)
        try:
            self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            pass
        self._send_lock = threading.Lock()
        self._rx_buf = bytearray()
        self._closed = False

        key_b64 = base64.b64encode(os.urandom(16)).decode("ascii")
        handshake = (
            f"GET {path} HTTP/1.1\r\n"
            f"Host: {host}:{port}\r\n"
            f"Upgrade: websocket\r\n"
            f"Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key_b64}\r\n"
            f"Sec-WebSocket-Version: 13\r\n"
            f"\r\n"
        ).encode("ascii")
        self._sock.sendall(handshake)

        while b"\r\n\r\n" not in self._rx_buf:
            chunk = self._sock.recv(4096)
            if not chunk:
                raise ConnectionError("EOF during WebSocket handshake")
            self._rx_buf.extend(chunk)

        hdr_end = self._rx_buf.index(b"\r\n\r\n") + 4
        raw_hdr = bytes(self._rx_buf[:hdr_end]).decode("iso-8859-1")
        del self._rx_buf[:hdr_end]

        lines = raw_hdr.split("\r\n")
        if "101" not in lines[0]:
            raise ConnectionError(f"WebSocket handshake failed: {lines[0]!r}")
        hdr_map: dict[str, str] = {}
        for ln in lines[1:]:
            if ":" in ln:
                k, v = ln.split(":", 1)
                hdr_map[k.strip().lower()] = v.strip()

        expected_accept = base64.b64encode(
            hashlib.sha1((key_b64 + _WS_GUID).encode("ascii")).digest()
        ).decode("ascii")
        if hdr_map.get("sec-websocket-accept") != expected_accept:
            raise ConnectionError("Invalid Sec-WebSocket-Accept from server")

    def _recv_exact(self, n: int) -> bytes:
        while len(self._rx_buf) < n:
            chunk = self._sock.recv(max(4096, n - len(self._rx_buf)))
            if not chunk:
                raise WebSocketClosed("WebSocket connection closed")
            self._rx_buf.extend(chunk)
        out = bytes(self._rx_buf[:n])
        del self._rx_buf[:n]
        return out

    def _send_frame(self, opcode: int, payload: bytes) -> None:
        header = bytearray()
        header.append(0x80 | (opcode & 0x0F))
        length = len(payload)
        if length < 126:
            header.append(0x80 | length)
        elif length < 65536:
            header.append(0x80 | 126)
            header.extend(length.to_bytes(2, "big"))
        else:
            header.append(0x80 | 127)
            header.extend(length.to_bytes(8, "big"))

        mask = os.urandom(4)
        header.extend(mask)
        if length:
            masked = bytes(b ^ mask[i & 3] for i, b in enumerate(payload))
        else:
            masked = b""
        with self._send_lock:
            if self._closed:
                raise WebSocketClosed("Cannot send on closed WebSocket")
            self._sock.sendall(bytes(header) + masked)

    def send_text(self, text: str) -> None:
        self._send_frame(0x1, text.encode("utf-8"))

    def recv_text(self) -> str:
        fragments: list[bytes] = []
        message_opcode: int | None = None

        while True:
            b1, b2 = self._recv_exact(2)
            fin = (b1 & 0x80) != 0
            opcode = b1 & 0x0F
            masked = (b2 & 0x80) != 0
            length = b2 & 0x7F
            if length == 126:
                length = int.from_bytes(self._recv_exact(2), "big")
            elif length == 127:
                length = int.from_bytes(self._recv_exact(8), "big")

            mask_key = self._recv_exact(4) if masked else b""
            payload = self._recv_exact(length) if length else b""
            if masked and payload:
                payload = bytes(b ^ mask_key[i & 3] for i, b in enumerate(payload))

            if opcode == 0x8:  # Close
                if not self._closed:
                    try:
                        self._send_frame(0x8, payload[:2])
                    except Exception:
                        pass
                    self._closed = True
                raise WebSocketClosed("Peer sent WebSocket close frame")
            if opcode == 0x9:  # Ping
                try:
                    self._send_frame(0xA, payload)
                except Exception:
                    pass
                continue
            if opcode == 0xA:  # Pong
                continue

            if opcode != 0x0:
                message_opcode = opcode
            fragments.append(payload)
            if fin:
                full = b"".join(fragments)
                if message_opcode in (0x1, None):
                    return full.decode("utf-8")
                # Binary not expected for CDP JSON, decode safely if received
                return full.decode("utf-8", errors="replace")

    def close(self) -> None:
        if not self._closed:
            try:
                self._send_frame(0x8, b"")
            except Exception:
                pass
            self._closed = True
        try:
            self._sock.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Headless Chrome CDP Driver
# ---------------------------------------------------------------------------


def find_chrome_binary() -> str:
    """Locate ``chrome-headless-shell`` or Chromium binary deterministically."""
    override = os.environ.get("QECC_CHROME_BIN", "").strip()
    if override:
        p = Path(override)
        if p.is_file():
            return str(p)
        raise FileNotFoundError(f"QECC_CHROME_BIN points to missing file: {override!r}")

    cache_root = Path.home() / ".cache" / "ms-playwright"
    if cache_root.is_dir():
        patterns = [
            str(cache_root / "chromium_headless_shell-*" / "chrome-headless-shell-linux64" / "chrome-headless-shell"),
            str(cache_root / "chromium-*" / "chrome-linux64" / "chrome"),
        ]
        for pat in patterns:
            matches = sorted(glob.glob(pat), reverse=True)
            for m in matches:
                if os.path.isfile(m) and os.access(m, os.X_OK):
                    return m

    for name in ("chrome-headless-shell", "google-chrome", "chromium", "chromium-browser"):
        found = shutil.which(name)
        if found:
            return found

    raise FileNotFoundError(
        "Could not locate chrome-headless-shell or Chromium binary. "
        "Set QECC_CHROME_BIN or install Playwright chromium_headless_shell."
    )


class CDPError(RuntimeError):
    """Raised when a CDP command returns a protocol error."""


class PageSession:
    """Multiplexed CDP page session with console/exception/network capture & DOM helpers."""

    def __init__(self, browser: "BrowserSession", target_id: str, session_id: str) -> None:
        self.browser = browser
        self.target_id = target_id
        self.session_id = session_id
        self.console_errors: list[str] = []
        self.console_warnings: list[str] = []
        self.console_all: list[tuple[str, str]] = []
        self.exceptions: list[str] = []
        self.log_errors: list[str] = []
        self._load_count = 0
        self._same_doc_count = 0
        self._requested_viewport_width: int | None = None
        self._event_cond = threading.Condition()
        self._closed = False

    def _on_event(self, method: str, params: dict[str, Any]) -> None:
        if method == "Runtime.consoleAPICalled":
            ctype = str(params.get("type", "log"))
            args = params.get("args", [])
            parts = []
            for arg in args:
                if "value" in arg:
                    parts.append(str(arg["value"]))
                elif "description" in arg:
                    parts.append(str(arg["description"]))
                else:
                    parts.append(str(arg.get("type", "")))
            text = " ".join(parts)
            self.console_all.append((ctype, text))
            if ctype == "error":
                self.console_errors.append(text)
            elif ctype == "warning":
                self.console_warnings.append(text)
        elif method == "Runtime.exceptionThrown":
            details = params.get("exceptionDetails", {})
            exc = details.get("exception", {})
            msg = details.get("text", "Uncaught exception")
            desc = exc.get("description") or exc.get("value") or ""
            full = f"{msg}: {desc}".strip(": ")
            self.exceptions.append(full)
        elif method == "Log.entryAdded":
            entry = params.get("entry", {})
            if entry.get("level") == "error":
                src = entry.get("source", "")
                txt = entry.get("text", "")
                self.log_errors.append(f"[{src}] {txt}".strip())
        elif method == "Page.loadEventFired":
            with self._event_cond:
                self._load_count += 1
                self._event_cond.notify_all()
        elif method == "Page.navigatedWithinDocument":
            with self._event_cond:
                self._same_doc_count += 1
                self._event_cond.notify_all()

    def cdp(self, method: str, params: dict[str, Any] | None = None, timeout: float = 15.0) -> dict[str, Any]:
        return self.browser.cdp(method, params=params, session_id=self.session_id, timeout=timeout)

    def clear_errors(self) -> None:
        self.console_errors.clear()
        self.console_warnings.clear()
        self.exceptions.clear()
        self.log_errors.clear()

    def assert_no_errors(self, *, allow_warnings: bool = True) -> None:
        problems: list[str] = []
        if self.console_errors:
            problems.append(f"console.error: {self.console_errors}")
        if self.exceptions:
            problems.append(f"uncaught exceptions: {self.exceptions}")
        if self.log_errors:
            problems.append(f"browser log errors: {self.log_errors}")
        if not allow_warnings and self.console_warnings:
            problems.append(f"console.warn: {self.console_warnings}")
        if problems:
            raise AssertionError("Browser runtime/console errors detected:\n" + "\n".join(problems))

    def goto(self, url: str, *, wait_load: bool = True, timeout: float = 15.0, settle_s: float = 0.2) -> None:
        with self._event_cond:
            prev_loads = self._load_count
            prev_same_doc = self._same_doc_count
        nav_res = self.cdp("Page.navigate", {"url": url}, timeout=timeout)
        if wait_load:
            deadline = time.monotonic() + timeout
            if "loaderId" not in nav_res:
                same_doc_deadline = min(deadline, time.monotonic() + 0.5)
                with self._event_cond:
                    while self._same_doc_count <= prev_same_doc:
                        rem = same_doc_deadline - time.monotonic()
                        if rem <= 0:
                            break
                        self._event_cond.wait(timeout=min(0.05, rem))
            else:
                with self._event_cond:
                    while self._load_count <= prev_loads:
                        rem = deadline - time.monotonic()
                        if rem <= 0:
                            break
                        self._event_cond.wait(timeout=min(0.2, rem))
            self.wait_for("document.readyState === 'complete'", timeout=max(1.0, deadline - time.monotonic()))
        if settle_s > 0:
            time.sleep(settle_s)

    def eval_js(self, expr: str, timeout: float = 10.0) -> Any:
        res = self.cdp(
            "Runtime.evaluate",
            {
                "expression": expr,
                "returnByValue": True,
                "awaitPromise": True,
                "userGesture": True,
            },
            timeout=timeout,
        )
        if "exceptionDetails" in res:
            details = res["exceptionDetails"]
            exc = details.get("exception", {})
            desc = exc.get("description") or details.get("text") or str(details)
            raise RuntimeError(f"JS evaluation error in {expr!r}: {desc}")
        return res.get("result", {}).get("value")

    def wait_for(self, js_predicate: str, *, timeout: float = 10.0, poll: float = 0.05) -> None:
        deadline = time.monotonic() + timeout
        while True:
            try:
                if self.eval_js(f"Boolean({js_predicate})", timeout=min(5.0, timeout)):
                    return
            except Exception:
                pass
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for JS predicate: {js_predicate}")
            time.sleep(poll)

    @staticmethod
    def _find_el_js(selector: str) -> str:
        sel_json = json.dumps(selector)
        # If plain identifier (no CSS punctuation), check [data-testid="..."] first, then querySelector
        if re.fullmatch(r"[A-Za-z0-9_-]+", selector):
            tid_json = json.dumps(f'[data-testid="{selector}"]')
            return f"(document.querySelector({tid_json}) || document.querySelector({sel_json}))"
        return f"document.querySelector({sel_json})"

    def has_element(self, selector: str) -> bool:
        return bool(self.eval_js(f"Boolean({self._find_el_js(selector)})"))

    def wait_for_testid(self, testid: str, timeout: float = 10.0) -> None:
        self.wait_for(f'document.querySelector(\'[data-testid="{testid}"]\') !== null', timeout=timeout)

    def select_option(self, selector: str, value: str, *, settle_s: float = 0.1) -> None:
        val_json = json.dumps(value)
        el_js = self._find_el_js(selector)
        ok = self.eval_js(
            f"""(() => {{
                const el = {el_js};
                if (!el) return {{ok: false, reason: 'element not found'}};
                const opts = Array.from(el.options || []).map(o => o.value);
                if (opts.length > 0 && !opts.includes({val_json})) {{
                    return {{ok: false, reason: 'option ' + {val_json} + ' not in ' + JSON.stringify(opts)}};
                }}
                el.value = {val_json};
                el.dispatchEvent(new Event('input', {{bubbles: true}}));
                el.dispatchEvent(new Event('change', {{bubbles: true}}));
                return {{ok: true, value: el.value}};
            }})()"""
        )
        if not isinstance(ok, dict) or not ok.get("ok"):
            raise RuntimeError(f"select_option({selector!r}, {value!r}) failed: {ok}")
        if settle_s > 0:
            time.sleep(settle_s)

    def fill_input(self, selector: str, value: str | float | int, *, settle_s: float = 0.1) -> None:
        val_json = json.dumps(str(value))
        el_js = self._find_el_js(selector)
        ok = self.eval_js(
            f"""(() => {{
                const el = {el_js};
                if (!el) return {{ok: false, reason: 'element not found'}};
                el.focus();
                el.value = {val_json};
                el.dispatchEvent(new Event('input', {{bubbles: true}}));
                el.dispatchEvent(new Event('change', {{bubbles: true}}));
                return {{ok: true, value: el.value}};
            }})()"""
        )
        if not isinstance(ok, dict) or not ok.get("ok"):
            raise RuntimeError(f"fill_input({selector!r}, {value!r}) failed: {ok}")
        if settle_s > 0:
            time.sleep(settle_s)

    def click(self, selector: str, *, trusted: bool = True, settle_s: float = 0.1) -> None:
        el_js = self._find_el_js(selector)
        info = self.eval_js(
            f"""(() => {{
                const el = {el_js};
                if (!el) return null;
                el.scrollIntoView({{block: 'center', inline: 'center'}});
                const r = el.getBoundingClientRect();
                return {{x: r.left + r.width / 2, y: r.top + r.height / 2, w: r.width, h: r.height}};
            }})()"""
        )
        if not info:
            raise RuntimeError(f"click({selector!r}) failed: element not found")
        if trusted and info["w"] > 0 and info["h"] > 0:
            x, y = float(info["x"]), float(info["y"])
            self.cdp("Input.dispatchMouseEvent", {"type": "mouseMoved", "x": x, "y": y})
            self.cdp(
                "Input.dispatchMouseEvent",
                {"type": "mousePressed", "x": x, "y": y, "button": "left", "clickCount": 1},
            )
            self.cdp(
                "Input.dispatchMouseEvent",
                {"type": "mouseReleased", "x": x, "y": y, "button": "left", "clickCount": 1},
            )
        else:
            self.eval_js(f"(() => {{ const el = {el_js}; if (el) el.click(); }})()")
        if settle_s > 0:
            time.sleep(settle_s)

    def ensure_visible(self, testid: str, *, settle_s: float = 0.15) -> dict[str, Any]:
        """Ensure a target ``data-testid`` panel is visible, activating any matching tab if hidden."""
        box = self.measure_box(testid)
        if box and box["width"] > 0 and box["height"] > 0:
            return box

        # Try clicking a tab/button targeting this section (e.g. curves, theory, seminar)
        prefix = testid.split("-")[0]
        clicked = self.eval_js(
            f"""(() => {{
                const candidates = [
                    document.querySelector('[data-testid="tab-{prefix}"]'),
                    document.querySelector('[data-testid="nav-{prefix}"]'),
                    document.querySelector('[data-tab="{prefix}"]'),
                    document.querySelector('[aria-controls="{testid}"]'),
                    ...Array.from(document.querySelectorAll('button, [role="tab"], a')).filter(el => {{
                        const tid = (el.getAttribute('data-testid') || '').toLowerCase();
                        const dt = (el.getAttribute('data-tab') || '').toLowerCase();
                        const href = (el.getAttribute('href') || '').toLowerCase();
                        return tid.includes('{prefix}') && !tid.includes('{testid}')
                            || dt.includes('{prefix}')
                            || href.includes('{prefix}');
                    }})
                ].filter(Boolean);
                for (const btn of candidates) {{
                    const r = btn.getBoundingClientRect();
                    if (r.width > 0 && r.height > 0) {{
                        btn.click();
                        return true;
                    }}
                }}
                return false;
            }})()"""
        )
        if clicked and settle_s > 0:
            time.sleep(settle_s)
        box = self.measure_box(testid)
        assert box is not None and box["width"] > 0 and box["height"] > 0, (
            f"Panel [data-testid='{testid}'] is missing or has zero dimensions: {box}"
        )
        return box

    def get_current_language(self) -> str:
        """Detect active UI language ('en' or 'pt-br') from toggle state, DOM text, and <html lang>."""
        res = self.eval_js(
            """(() => {
                const isActive = (el) => {
                    if (!el) return false;
                    return (
                        el.getAttribute('aria-pressed') === 'true'
                        || el.getAttribute('aria-selected') === 'true'
                        || el.getAttribute('aria-current') === 'true'
                        || el.classList.contains('active')
                        || el.classList.contains('selected')
                        || (el.hasAttribute('data-active') && el.getAttribute('data-active') !== 'false')
                    );
                };
                const btnPt = document.querySelector('[data-testid="lang-pt-br"]');
                const btnEn = document.querySelector('[data-testid="lang-en"]');
                const ptActive = isActive(btnPt);
                const enActive = isActive(btnEn);
                if (ptActive && !enActive) return 'pt-br';
                if (enActive && !ptActive) return 'en';

                const toggle = document.querySelector('[data-testid="lang-toggle"]');
                if (toggle) {
                    if (toggle.tagName === 'SELECT') {
                        const sv = (toggle.value || '').toLowerCase();
                        if (sv.startsWith('pt')) return 'pt-br';
                        if (sv.startsWith('en')) return 'en';
                    }
                    const curLang = (toggle.getAttribute('data-current-lang') || '').toLowerCase();
                    if (curLang.startsWith('pt')) return 'pt-br';
                    if (curLang.startsWith('en')) return 'en';
                }

                const txt = (document.body ? document.body.innerText : '').toLowerCase();
                const ptScore = [
                    'síndrome', 'estabilizador', 'estabilizadores', 'coerência', 'limiar',
                    'fidelidade', 'não-clonagem', 'correção', 'codificação', 'recuperação'
                ].filter(w => txt.includes(w)).length;
                const enScore = [
                    'syndrome', 'stabilizer', 'stabilizers', 'coherence', 'threshold',
                    'fidelity', 'no-cloning', 'correction', 'encoding', 'recovery'
                ].filter(w => txt.includes(w)).length;
                if (ptScore > enScore) return 'pt-br';
                if (enScore > ptScore) return 'en';

                if (toggle) {
                    const dl = (toggle.getAttribute('data-lang') || toggle.getAttribute('data-locale') || '').toLowerCase();
                    if (dl.startsWith('pt')) return 'pt-br';
                    if (dl.startsWith('en')) return 'en';
                }

                const htmlLang = (document.documentElement.lang || '').toLowerCase();
                if (htmlLang.startsWith('pt')) return 'pt-br';
                if (htmlLang.startsWith('en')) return 'en';
                return 'en';
            })()"""
        )
        return "pt-br" if str(res).lower().startswith("pt") else "en"

    def switch_language(self, lang: str, *, settle_s: float = 0.15) -> str:
        """Switch UI language to ``'en'`` or ``'pt-br'`` using dedicated buttons or ``lang-toggle``."""
        target = "pt-br" if lang.lower() in ("pt", "pt-br", "pt_br") else "en"
        direct_tid = f"lang-{target}"
        if self.has_element(direct_tid):
            self.click(direct_tid, settle_s=settle_s)
        elif self.has_element("lang-toggle"):
            toggle_tag = self.eval_js(
                'document.querySelector(\'[data-testid="lang-toggle"]\')?.tagName || ""'
            )
            if str(toggle_tag).upper() == "SELECT":
                opt_val = self.eval_js(
                    f"""(() => {{
                        const sel = document.querySelector('[data-testid="lang-toggle"]');
                        if (!sel) return null;
                        const prefix = {json.dumps("pt" if target == "pt-br" else "en")};
                        const opt = Array.from(sel.options || []).find(o =>
                            (o.value || '').toLowerCase().startsWith(prefix)
                            || (o.textContent || '').toLowerCase().includes(prefix)
                        );
                        return opt ? opt.value : null;
                    }})()"""
                )
                if opt_val is not None:
                    self.select_option("lang-toggle", str(opt_val), settle_s=settle_s)
            else:
                cur = self.get_current_language()
                if cur != target:
                    self.click("lang-toggle", settle_s=settle_s)
        else:
            raise RuntimeError(
                "No language switch control found ([data-testid='lang-toggle'] or 'lang-en'/'lang-pt-br')"
            )
        return self.get_current_language()

    def set_viewport(self, width: int, height: int, *, mobile: bool = False, scale: float = 1.0) -> None:
        req_w = int(width)
        self._requested_viewport_width = req_w if req_w > 0 else None
        self.cdp(
            "Emulation.setDeviceMetricsOverride",
            {
                "width": req_w,
                "height": int(height),
                "deviceScaleFactor": float(scale),
                "mobile": bool(mobile),
            },
        )
        time.sleep(0.08)

    def clear_viewport_override(self) -> None:
        self._requested_viewport_width = None
        self.cdp("Emulation.clearDeviceMetricsOverride")
        time.sleep(0.05)

    def viewport_dimensions(self) -> dict[str, int]:
        res = self.eval_js(
            "({"
            "scrollWidth: document.documentElement.scrollWidth, "
            "clientWidth: document.documentElement.clientWidth, "
            "bodyScrollWidth: document.body ? document.body.scrollWidth : 0, "
            "innerWidth: window.innerWidth, "
            "scrollHeight: document.documentElement.scrollHeight, "
            "clientHeight: document.documentElement.clientHeight, "
            "innerHeight: window.innerHeight"
            "})"
        )
        dims = {k: int(v) for k, v in res.items()}
        if self._requested_viewport_width is not None:
            dims["requestedWidth"] = self._requested_viewport_width
        return dims

    def check_no_horizontal_overflow(self) -> bool:
        dims = self.viewport_dimensions()
        client_w = dims["clientWidth"]
        if dims["scrollWidth"] > client_w:
            return False
        if dims["bodyScrollWidth"] > client_w:
            return False
        if self._requested_viewport_width is not None and client_w > self._requested_viewport_width:
            return False
        return True

    def measure_box(self, selector: str) -> dict[str, Any] | None:
        el_js = self._find_el_js(selector)
        return self.eval_js(
            f"""(() => {{
                const el = {el_js};
                if (!el) return null;
                const r = el.getBoundingClientRect();
                return {{
                    width: r.width,
                    height: r.height,
                    top: r.top,
                    left: r.left,
                    right: r.right,
                    bottom: r.bottom,
                    cw: typeof el.width === 'number' ? el.width : r.width,
                    ch: typeof el.height === 'number' ? el.height : r.height
                }};
            }})()"""
        )

    def body_text(self) -> str:
        return str(self.eval_js("document.body ? document.body.innerText : ''") or "")

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            try:
                self.browser.cdp("Target.closeTarget", {"targetId": self.target_id}, timeout=5.0)
            except Exception:
                pass
            self.browser._unregister_session(self.session_id)


class BrowserSession:
    """Manages a single ``chrome-headless-shell`` process and multiplexed CDP page sessions."""

    def __init__(self, *, chrome_bin: str | None = None, startup_timeout: float = 15.0) -> None:
        self.chrome_bin = chrome_bin or find_chrome_binary()
        self.startup_timeout = startup_timeout
        self._tmpdir: tempfile.TemporaryDirectory[str] | None = None
        self._proc: subprocess.Popen[str] | None = None
        self._ws: StdlibWS | None = None
        self._reader_thread: threading.Thread | None = None
        self._cond = threading.Condition()
        self._next_id = 1
        self._responses: dict[int, dict[str, Any]] = {}
        self._sessions: dict[str, PageSession] = {}
        self._reader_error: BaseException | None = None
        self._closing = False

    def start(self) -> "BrowserSession":
        self._tmpdir = tempfile.TemporaryDirectory(prefix="qecc_chrome_")
        cmd = [
            self.chrome_bin,
            "--headless",
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--remote-debugging-port=0",
            f"--user-data-dir={self._tmpdir.name}",
            "--no-first-run",
            "--no-default-browser-check",
            "about:blank",
        ]
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        ws_url: str | None = None
        stderr_lines: list[str] = []
        ws_event = threading.Event()

        def _read_stderr() -> None:
            nonlocal ws_url
            assert self._proc is not None and self._proc.stderr is not None
            for line in self._proc.stderr:
                stderr_lines.append(line.rstrip("\r\n"))
                m = re.search(r"DevTools listening on (ws://\S+)", line)
                if m and ws_url is None:
                    ws_url = m.group(1)
                    ws_event.set()

        t_err = threading.Thread(target=_read_stderr, daemon=True)
        t_err.start()

        if not ws_event.wait(timeout=self.startup_timeout) or ws_url is None:
            self.close()
            raise RuntimeError(
                f"chrome-headless-shell failed to report DevTools WS URL within {self.startup_timeout}s.\n"
                f"Stderr:\n" + "\n".join(stderr_lines)
            )

        self._ws = StdlibWS(ws_url, timeout=self.startup_timeout)
        self._reader_thread = threading.Thread(target=self._ws_reader_loop, daemon=True)
        self._reader_thread.start()
        return self

    def _ws_reader_loop(self) -> None:
        assert self._ws is not None
        try:
            while not self._closing:
                raw = self._ws.recv_text()
                msg = json.loads(raw)
                if "id" in msg:
                    with self._cond:
                        self._responses[int(msg["id"])] = msg
                        self._cond.notify_all()
                else:
                    sid = msg.get("sessionId")
                    if sid:
                        with self._cond:
                            sess = self._sessions.get(sid)
                        if sess is not None:
                            sess._on_event(msg.get("method", ""), msg.get("params", {}))
        except WebSocketClosed:
            with self._cond:
                if not self._closing:
                    self._reader_error = WebSocketClosed("CDP WebSocket closed unexpectedly")
                self._cond.notify_all()
        except BaseException as exc:
            with self._cond:
                self._reader_error = exc
                self._cond.notify_all()

    def _unregister_session(self, session_id: str) -> None:
        with self._cond:
            self._sessions.pop(session_id, None)

    def cdp(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        session_id: str | None = None,
        timeout: float = 15.0,
    ) -> dict[str, Any]:
        assert self._ws is not None
        with self._cond:
            cmd_id = self._next_id
            self._next_id += 1
            if self._reader_error is not None:
                raise RuntimeError(f"CDP connection failed: {self._reader_error}") from self._reader_error

        payload: dict[str, Any] = {"id": cmd_id, "method": method}
        if params:
            payload["params"] = params
        if session_id:
            payload["sessionId"] = session_id

        self._ws.send_text(json.dumps(payload))

        deadline = time.monotonic() + timeout
        with self._cond:
            while cmd_id not in self._responses:
                if self._reader_error is not None:
                    raise RuntimeError(f"CDP connection failed during {method}: {self._reader_error}")
                rem = deadline - time.monotonic()
                if rem <= 0:
                    raise TimeoutError(f"CDP call {method} timed out after {timeout}s")
                self._cond.wait(timeout=rem)
            resp = self._responses.pop(cmd_id)

        if "error" in resp:
            raise CDPError(f"CDP {method} error: {resp['error']}")
        return resp.get("result", {})

    def new_page(self) -> PageSession:
        res_target = self.cdp("Target.createTarget", {"url": "about:blank"})
        target_id = res_target["targetId"]
        res_attach = self.cdp("Target.attachToTarget", {"targetId": target_id, "flatten": True})
        session_id = res_attach["sessionId"]
        page = PageSession(self, target_id, session_id)
        with self._cond:
            self._sessions[session_id] = page
        page.cdp("Page.enable")
        page.cdp("Runtime.enable")
        page.cdp("Log.enable")
        return page

    def close(self) -> None:
        self._closing = True
        if self._ws is not None:
            self._ws.close()
        if self._proc is not None:
            if self._proc.poll() is None:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
                    self._proc.wait(timeout=5.0)
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=1.0)
        if self._tmpdir is not None:
            try:
                self._tmpdir.cleanup()
            except Exception:
                pass

    def __enter__(self) -> "BrowserSession":
        return self.start()

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Bilingual & Leak Checking Helpers
# ---------------------------------------------------------------------------

_LEAK_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("undefined", re.compile(r"\bundefined\b", re.IGNORECASE)),
    (
        "null",
        re.compile(
            r"\bnull\b(?!\s*-\s*(?:space|vector|syndrome|state|subspace|operator))"
            r"(?!\s+(?:space|spaces|vector|vectors|syndrome|syndromes|state|states|"
            r"subspace|subspaces|operator|operators|outcome|outcomes|hypothesis|"
            r"eigenvalue|eigenvalues|mode|modes)\b)",
            re.IGNORECASE,
        ),
    ),
    ("NaN", re.compile(r"\bNaN\b")),  # case-sensitive (avoid matching non-leak words)
    ("i18n.", re.compile(r"\bi18n\.[A-Za-z0-9_.-]+")),
    ("TODO", re.compile(r"\bTODO\b")),  # case-sensitive (Portuguese 'todo' means 'all')
)


def find_leak_tokens(text: str) -> list[str]:
    """Return any untranslated/debug leak tokens found in rendered UI text."""
    leaks: list[str] = []
    for label, pat in _LEAK_PATTERNS:
        for m in pat.finditer(text):
            leaks.append(f"{label}: {m.group(0)!r}")
    return leaks


def collect_seminar_panel_texts(page: PageSession, *, max_steps: int = 25) -> list[str]:
    """Collect visible prose across ``seminar-panel`` via Next/Prev, topic tabs, or slide <select>."""
    page.ensure_visible("seminar-panel")
    collected: list[str] = []
    seen: set[str] = set()

    def _record() -> str:
        txt = str(page.eval_js('document.querySelector(\'[data-testid="seminar-panel"]\')?.innerText || ""') or "")
        if txt and txt not in seen:
            seen.add(txt)
            collected.append(txt)
        return txt

    _record()

    # 1. Select dropdown inside seminar-panel (if present)
    select_opts = page.eval_js(
        """(() => {
            const p = document.querySelector('[data-testid="seminar-panel"]');
            if (!p) return [];
            const sel = p.querySelector('select');
            if (!sel) return [];
            return Array.from(sel.options || []).map(o => o.value);
        })()"""
    )
    if isinstance(select_opts, list) and len(select_opts) > 1:
        for val in select_opts[:max_steps]:
            page.eval_js(
                f"""(() => {{
                    const sel = document.querySelector('[data-testid="seminar-panel"] select');
                    if (!sel) return;
                    sel.value = {json.dumps(val)};
                    sel.dispatchEvent(new Event('input', {{bubbles: true}}));
                    sel.dispatchEvent(new Event('change', {{bubbles: true}}));
                }})()"""
            )
            time.sleep(0.04)
            _record()

    # 2. Topic/slide tabs or buttons inside seminar-panel (non-next/prev/try/apply)
    tab_count = int(
        page.eval_js(
            """(() => {
                const p = document.querySelector('[data-testid="seminar-panel"]');
                if (!p) return 0;
                const tabs = Array.from(p.querySelectorAll(
                    '[role="tab"], [data-slide], [data-topic], [data-testid*="topic"], [data-testid*="slide"]'
                )).filter(el => {
                    if (el.tagName === 'SELECT') return false;
                    const tid = (el.getAttribute('data-testid') || '').toLowerCase();
                    const txt = (el.innerText || '').toLowerCase();
                    if (tid.includes('next') || tid.includes('prev')) return false;
                    if (/^(next|prev|previous|próxim|anterior)$/i.test(txt.trim())) return false;
                    const r = el.getBoundingClientRect();
                    return r.width > 0 && r.height > 0;
                });
                return tabs.length;
            })()"""
        )
        or 0
    )
    if tab_count > 0:
        for idx in range(min(tab_count, max_steps)):
            page.eval_js(
                f"""(() => {{
                    const p = document.querySelector('[data-testid="seminar-panel"]');
                    if (!p) return;
                    const tabs = Array.from(p.querySelectorAll(
                        '[role="tab"], [data-slide], [data-topic], [data-testid*="topic"], [data-testid*="slide"]'
                    )).filter(el => {{
                        if (el.tagName === 'SELECT') return false;
                        const tid = (el.getAttribute('data-testid') || '').toLowerCase();
                        const txt = (el.innerText || '').toLowerCase();
                        if (tid.includes('next') || tid.includes('prev')) return false;
                        if (/^(next|prev|previous|próxim|anterior)$/i.test(txt.trim())) return false;
                        const r = el.getBoundingClientRect();
                        return r.width > 0 && r.height > 0;
                    }});
                    if (tabs[{idx}]) tabs[{idx}].click();
                }})()"""
            )
            time.sleep(0.04)
            _record()

    # 3. Next button stepping inside seminar-panel (if present)
    for _ in range(max_steps):
        prev_txt = str(page.eval_js('document.querySelector(\'[data-testid="seminar-panel"]\')?.innerText || ""') or "")
        moved = page.eval_js(
            """(() => {
                const p = document.querySelector('[data-testid="seminar-panel"]');
                if (!p) return false;
                const next = p.querySelector('[data-testid*="next"], button[aria-label*="Next" i], button[data-action="next"]')
                    || Array.from(p.querySelectorAll('button')).find(b => /next|próxim|avanç/i.test(b.innerText));
                if (next && !next.disabled) {
                    next.click();
                    return true;
                }
                return false;
            })()"""
        )
        if not moved:
            break
        time.sleep(0.04)
        cur_txt = _record()
        if cur_txt == prev_txt:
            break

    return collected


# ---------------------------------------------------------------------------
# Mathematical Oracle Helpers (deferred qecc.* imports, <= 1e-12 exact)
# ---------------------------------------------------------------------------


def decode_complex_scalar(val: Any) -> complex:
    if isinstance(val, (int, float)):
        return complex(float(val), 0.0)
    if isinstance(val, (list, tuple)) and len(val) == 2:
        return complex(float(val[0]), float(val[1]))
    if isinstance(val, dict) and "re" in val and "im" in val:
        return complex(float(val["re"]), float(val["im"]))
    raise ValueError(f"Cannot decode complex scalar from {val!r}")


def decode_complex_vec(vec: Any) -> np.ndarray:
    return np.array([decode_complex_scalar(x) for x in vec], dtype=np.complex128)


def decode_complex_mat(mat: Any) -> np.ndarray:
    return np.array([[decode_complex_scalar(cell) for cell in row] for row in mat], dtype=np.complex128)


def rho_to_bloch(rho: np.ndarray) -> np.ndarray:
    arr = np.asarray(rho, dtype=np.complex128)
    rx = float(2.0 * arr[0, 1].real)
    ry = float(-2.0 * arr[0, 1].imag)
    rz = float((arr[0, 0] - arr[1, 1]).real)
    return np.array([rx, ry, rz], dtype=np.float64)


def bloch_to_rho(bloch: Sequence[float] | np.ndarray) -> np.ndarray:
    rx, ry, rz = float(bloch[0]), float(bloch[1]), float(bloch[2])
    return 0.5 * (
        _PAULI_MATS["I"]
        + rx * _PAULI_MATS["X"]
        + ry * _PAULI_MATS["Y"]
        + rz * _PAULI_MATS["Z"]
    )


def abs_rho01(rho: np.ndarray) -> float:
    arr = np.asarray(rho, dtype=np.complex128)
    return float(abs(arr[0, 1]))


def su2_pauli_coefficients(theta: float, nx: float, ny: float, nz: float) -> dict[str, complex]:
    """Exact coefficients of U(theta, n) = c_I I + c_X X + c_Y Y + c_Z Z."""
    norm = math.sqrt(nx * nx + ny * ny + nz * nz)
    if norm <= 0.0:
        raise ValueError("SU(2) axis cannot have zero norm")
    ux, uy, uz = nx / norm, ny / norm, nz / norm
    half = 0.5 * theta
    c = math.cos(half)
    s = math.sin(half)
    return {
        "I": complex(c, 0.0),
        "X": complex(0.0, -s * ux),
        "Y": complex(0.0, -s * uy),
        "Z": complex(0.0, -s * uz),
    }


def su2_matrix_oracle(theta: float, nx: float, ny: float, nz: float) -> np.ndarray:
    coeffs = su2_pauli_coefficients(theta, nx, ny, nz)
    return (
        coeffs["I"] * _PAULI_MATS["I"]
        + coeffs["X"] * _PAULI_MATS["X"]
        + coeffs["Y"] * _PAULI_MATS["Y"]
        + coeffs["Z"] * _PAULI_MATS["Z"]
    )


def _single_gate_unitary_2x2(name: str, params: tuple[float, ...] = ()) -> np.ndarray:
    if name in _PAULI_MATS:
        return _PAULI_MATS[name]
    if name == "H":
        return np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128) / math.sqrt(2.0)
    if name == "S":
        return np.array([[1.0, 0.0], [0.0, 1.0j]], dtype=np.complex128)
    if name == "Sdg":
        return np.array([[1.0, 0.0], [0.0, -1.0j]], dtype=np.complex128)
    if name == "RX":
        return su2_matrix_oracle(float(params[0]), 1.0, 0.0, 0.0)
    if name == "RY":
        return su2_matrix_oracle(float(params[0]), 0.0, 1.0, 0.0)
    if name == "RZ":
        return su2_matrix_oracle(float(params[0]), 0.0, 0.0, 1.0)
    raise ValueError(f"Unsupported single-qubit gate {name!r}")


def apply_single_qubit_matrix_oracle(
    state: np.ndarray,
    u2: np.ndarray,
    target: int,
    n_total: int,
) -> np.ndarray:
    """Apply a 2x2 unitary to big-endian qubit ``target`` of a 2^n_total statevector."""
    t = state.reshape((2,) * n_total).copy()
    t = np.tensordot(u2, t, axes=([1], [target]))
    t = np.moveaxis(t, 0, target)
    return t.reshape(1 << n_total)


def apply_ops_oracle(state: np.ndarray, ops: Sequence[Any], n_total: int) -> np.ndarray:
    """Independent big-endian 2^n_total complex128 statevector oracle for GateOp sequences."""
    vec = np.asarray(state, dtype=np.complex128).copy().reshape(1 << n_total)
    for op in ops:
        name = op.name if hasattr(op, "name") else op["name"]
        targets = tuple(op.targets if hasattr(op, "targets") else op["targets"])
        controls = tuple(op.controls if hasattr(op, "controls") else op.get("controls", ()))
        params = tuple(op.params if hasattr(op, "params") else op.get("params", ()))

        if name == "I":
            continue
        if name in ("CNOT", "CZ"):
            ctrl = int(controls[0])
            tgt = int(targets[0])
            tview = vec.reshape((2,) * n_total)
            if name == "CNOT":
                s0 = [slice(None)] * n_total
                s0[ctrl] = 1
                s0[tgt] = 0
                s1 = [slice(None)] * n_total
                s1[ctrl] = 1
                s1[tgt] = 1
                tmp = tview[tuple(s0)].copy()
                tview[tuple(s0)] = tview[tuple(s1)]
                tview[tuple(s1)] = tmp
            else:
                s = [slice(None)] * n_total
                s[ctrl] = 1
                s[tgt] = 1
                tview[tuple(s)] *= -1.0
            vec = tview.reshape(1 << n_total)
        else:
            u2 = _single_gate_unitary_2x2(name, params)
            vec = apply_single_qubit_matrix_oracle(vec, u2, int(targets[0]), n_total)
    return vec


def logical_basis_states_oracle(code_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Return exact (|0_L>, |1_L>) on 2^n_data qubits constructed independently from encode_ops."""
    _ensure_src_on_sys_path()
    from qecc.codes import get_code

    code = get_code(code_name)
    dim = 1 << code.n_data
    ket0 = np.zeros(dim, dtype=np.complex128)
    ket1 = np.zeros(dim, dtype=np.complex128)
    ket0[0] = 1.0
    ket1[1 << (code.n_data - 1)] = 1.0
    ops = code.encode_ops()
    return (
        apply_ops_oracle(ket0, ops, code.n_data),
        apply_ops_oracle(ket1, ops, code.n_data),
    )


def project_to_logical_rho(
    vec: np.ndarray,
    n_data: int,
    n_ancilla: int,
    ket0_l: np.ndarray,
    ket1_l: np.ndarray,
) -> np.ndarray:
    """Trace ancillas and project onto {|0_L>, |1_L>} to obtain 2x2 logical_rho."""
    da = vec.reshape(1 << n_data, 1 << n_ancilla)
    rho_data = da @ da.conj().T
    proj = np.vstack([ket0_l.conj(), ket1_l.conj()])  # (2, 2^n_data)
    logical_rho = proj @ rho_data @ proj.conj().T
    return np.asarray(0.5 * (logical_rho + logical_rho.conj().T), dtype=np.complex128)


def compute_step_oracle(
    code_name: str,
    psi: Sequence[complex] | np.ndarray,
    *,
    error_mode: str,
    pauli_string: str | None = None,
    error_qubit: int | None = None,
    error_op: str | None = None,
    su2_qubit: int | None = None,
    su2_theta: float | None = None,
    su2_nx: float | None = None,
    su2_ny: float | None = None,
    su2_nz: float | None = None,
    collective_phi: float | None = None,
    selected_syndrome: str | None = None,
) -> dict[str, Any]:
    """Independent stage-by-stage exact statevector oracle matching Contract 3 (<= 1e-12)."""
    _ensure_src_on_sys_path()
    from qecc.channels import DeterministicErrorInjector, coherent_collective_dephasing_ops
    from qecc.codes import get_code
    from qecc.metrics import InitialStateSampler, partial_trace_to_qubit0, quantum_fidelity

    psi_vec = InitialStateSampler.normalize_statevector(np.asarray(psi, dtype=np.complex128))
    rho_in = np.outer(psi_vec, psi_vec.conj())
    code = get_code(code_name)
    n_data, n_ancilla, n_total = code.n_data, code.n_ancilla, code.n_total
    ket0_l, ket1_l = logical_basis_states_oracle(code_name)

    # 0. initial
    v0 = np.zeros(1 << n_total, dtype=np.complex128)
    v0[0] = psi_vec[0]
    v0[1 << (n_total - 1)] = psi_vec[1]
    rho0 = partial_trace_to_qubit0(v0, n_total)

    # 1. encoded
    v1 = apply_ops_oracle(v0, code.encode_ops(), n_total)
    rho1 = project_to_logical_rho(v1, n_data, n_ancilla, ket0_l, ket1_l)

    # 2. noisy
    if error_mode == "pauli_string":
        assert pauli_string is not None
        err_ops = DeterministicErrorInjector.to_gate_ops(pauli_string, n_data)
        v2 = apply_ops_oracle(v1, err_ops, n_total)
    elif error_mode == "single_qubit_pauli":
        assert error_qubit is not None and error_op is not None
        err_ops = DeterministicErrorInjector.to_gate_ops((error_qubit, error_op), n_data)
        v2 = apply_ops_oracle(v1, err_ops, n_total)
    elif error_mode == "su2_rotation":
        assert (
            su2_qubit is not None
            and su2_theta is not None
            and su2_nx is not None
            and su2_ny is not None
            and su2_nz is not None
        )
        u2 = su2_matrix_oracle(su2_theta, su2_nx, su2_ny, su2_nz)
        v2 = apply_single_qubit_matrix_oracle(v1, u2, int(su2_qubit), n_total)
    elif error_mode == "coherent_collective":
        assert collective_phi is not None
        err_ops = coherent_collective_dephasing_ops(n_data, float(collective_phi))
        v2 = apply_ops_oracle(v1, err_ops, n_total)
    else:
        raise ValueError(f"Unsupported deterministic oracle error_mode={error_mode!r}")
    rho2 = project_to_logical_rho(v2, n_data, n_ancilla, ket0_l, ket1_l)

    # 3. syndrome_extracted (compute probabilities, then project onto selected_syndrome)
    v3_pre = apply_ops_oracle(v2, code.syndrome_ops(), n_total)

    syndrome_probs: dict[str, float] = {}
    if n_ancilla == 0:
        chosen_synd = ""
        v3 = v3_pre
    else:
        da3 = v3_pre.reshape(1 << n_data, 1 << n_ancilla)
        probs_arr = np.sum(np.abs(da3) ** 2, axis=0)
        for m in range(1 << n_ancilla):
            s_key = format(m, f"0{n_ancilla}b")
            syndrome_probs[s_key] = float(probs_arr[m])
        if selected_syndrome is not None:
            chosen_synd = selected_syndrome
        else:
            best_m = int(np.argmax(probs_arr))
            chosen_synd = format(best_m, f"0{n_ancilla}b")
        m_idx = int(chosen_synd, 2)
        p_m = float(probs_arr[m_idx])
        if p_m <= 1e-15:
            raise ValueError(f"Selected syndrome {chosen_synd!r} has zero probability {p_m}")
        da_coll = np.zeros_like(da3)
        da_coll[:, m_idx] = da3[:, m_idx] / math.sqrt(p_m)
        v3 = da_coll.reshape(1 << n_total)
    rho3 = project_to_logical_rho(v3, n_data, n_ancilla, ket0_l, ket1_l)

    # 4. recovered
    rec_ops = code.recovery_ops(chosen_synd)
    v4 = apply_ops_oracle(v3, rec_ops, n_total)
    rho4 = project_to_logical_rho(v4, n_data, n_ancilla, ket0_l, ket1_l)

    # 5. decoded
    v5 = apply_ops_oracle(v4, code.decode_ops(), n_total)
    rho5 = partial_trace_to_qubit0(v5, n_total)

    vecs = [v0, v1, v2, v3, v4, v5]
    rhos = [rho0, rho1, rho2, rho3, rho4, rho5]
    stages: list[dict[str, Any]] = []
    for key, vec, rho in zip(STAGE_KEYS, vecs, rhos):
        tr = float(np.real(np.trace(rho)))
        if abs(tr - 1.0) <= 1e-6:
            fid = float(quantum_fidelity(rho_in, rho))
        else:
            fid = float(np.clip(np.real(np.vdot(psi_vec, rho @ psi_vec)), 0.0, 1.0))
        stages.append(
            {
                "key": key,
                "statevector": vec,
                "logical_rho": rho,
                "logical_bloch": rho_to_bloch(rho),
                "logical_abs_rho01": abs_rho01(rho),
                "fidelity_vs_input": fid,
            }
        )

    return {
        "code": code_name,
        "psi": psi_vec,
        "rho_in": rho_in,
        "syndrome_probabilities": syndrome_probs,
        "selected_syndrome": chosen_synd,
        "recovery_ops": rec_ops,
        "stages": stages,
    }


def _pauli_commutes(a: str, b: str) -> bool:
    anti = 0
    for ca, cb in zip(a, b, strict=True):
        if ca != "I" and cb != "I" and ca != cb:
            anti ^= 1
    return anti == 0


def _pauli_mul_unsigned(a: str, b: str) -> str:
    out: list[str] = []
    for ca, cb in zip(a, b, strict=True):
        if ca == "I":
            out.append(cb)
        elif cb == "I":
            out.append(ca)
        elif ca == cb:
            out.append("I")
        else:
            for cand in ("X", "Y", "Z"):
                if cand != ca and cand != cb:
                    out.append(cand)
                    break
    return "".join(out)


def five_qubit_single_axis_logical_cosets(axis: str, p: float) -> dict[str, float]:
    """Exact 32-term pure-X or pure-Z logical coset probabilities (P_I, P_X, P_Y, P_Z) for [[5,1,3]]."""
    _ensure_src_on_sys_path()
    from qecc.codes import get_code

    if axis not in ("X", "Z"):
        raise ValueError(f"axis must be 'X' or 'Z', got {axis!r}")
    code = get_code("five_qubit")
    stabs = code.stabilizers
    table = code.syndrome_table
    probs = {"I": 0.0, "X": 0.0, "Y": 0.0, "Z": 0.0}
    for bits in itertools.product(("I", axis), repeat=5):
        err = "".join(bits)
        w = sum(1 for ch in err if ch != "I")
        prob_e = (p**w) * ((1.0 - p) ** (5 - w))
        synd = "".join("0" if _pauli_commutes(err, s) else "1" for s in stabs)
        res = _pauli_mul_unsigned(table[synd], err)
        cx = _pauli_commutes(res, "XXXXX")
        cz = _pauli_commutes(res, "ZZZZZ")
        if cx and cz:
            probs["I"] += prob_e
        elif cx and not cz:
            probs["X"] += prob_e
        elif not cx and not cz:
            probs["Y"] += prob_e
        else:
            probs["Z"] += prob_e
    return probs


def analytical_coherence_oracle(code_name: str, channel: str, p: float) -> float:
    """Exact analytical |rho_01| for |+i> = (|0> + i|1>)/sqrt(2) across all (code, channel) pairs."""
    _ensure_src_on_sys_path()
    from qecc.analysis import analytical_fidelity

    if code_name == "raw":
        if channel == "depolarizing":
            return 0.5 * abs(1.0 - (4.0 / 3.0) * p)
        return 0.5 * abs(1.0 - 2.0 * p)

    if code_name == "dfs_2":
        if channel == "collective_dephasing":
            return 0.5
        if channel == "bit_flip":
            return 0.5 * abs(1.0 - 2.0 * p)
        if channel == "phase_flip":
            p_i = (1.0 - p) ** 2 + p**2
            return 0.5 * abs(2.0 * p_i - 1.0)
        if channel == "depolarizing":
            # Compute exact logical channel on span{|01>, |10>} for depolarizing
            p_x = p / 3.0
            p_y = p / 3.0
            p_z = p / 3.0
            p_i = 1.0 - p
            # Each 2-qubit Pauli acts on |01>, |10> -> |+i> coherence attenuated
            # X1 or X2 leaves DFS, while II, ZZ keep, Z1/Z2 swap phase:
            # Exact numerical mixture via partial trace on decoded state:
            from qecc.codes import get_code

            code = get_code("dfs_2")
            inv_s2 = 1.0 / math.sqrt(2.0)
            psi = np.array([inv_s2, 1.0j * inv_s2], dtype=np.complex128)
            v0 = np.zeros(4, dtype=np.complex128)
            v0[0], v0[2] = psi[0], psi[1]
            v1 = apply_ops_oracle(v0, code.encode_ops(), 2)
            rho_out = np.zeros((2, 2), dtype=np.complex128)
            single = [("I", p_i), ("X", p_x), ("Y", p_y), ("Z", p_z)]
            from qecc.channels import DeterministicErrorInjector
            from qecc.metrics import partial_trace_to_qubit0

            for c0, w0 in single:
                for c1, w1 in single:
                    w = w0 * w1
                    v2 = apply_ops_oracle(v1, DeterministicErrorInjector.to_gate_ops(c0 + c1, 2), 2)
                    v5 = apply_ops_oracle(v2, code.decode_ops(), 2)
                    rho_out += w * partial_trace_to_qubit0(v5, 2)
            return abs_rho01(rho_out)

    if code_name in ("bit_flip_3", "phase_flip_3"):
        if channel == "collective_dephasing":
            return 0.5 * abs(1.0 - 2.0 * p)
        if (code_name == "bit_flip_3" and channel == "bit_flip") or (
            code_name == "phase_flip_3" and channel == "phase_flip"
        ):
            f_e = float(analytical_fidelity(code_name, channel, p, metric="entanglement"))
            return 0.5 * abs(2.0 * f_e - 1.0)
        if (code_name == "bit_flip_3" and channel == "phase_flip") or (
            code_name == "phase_flip_3" and channel == "bit_flip"
        ):
            # Pure Z on bit_flip_3 or pure X on phase_flip_3 acts only in {I, Z_L} or {I, X_L}
            f_e = float(analytical_fidelity(code_name, channel, p, metric="entanglement"))
            return 0.5 * abs(2.0 * f_e - 1.0)
        if channel == "depolarizing":
            from qecc.channels import DeterministicErrorInjector
            from qecc.codes import get_code
            from qecc.metrics import partial_trace_to_qubit0

            code = get_code(code_name)
            inv_s2 = 1.0 / math.sqrt(2.0)
            psi = np.array([inv_s2, 1.0j * inv_s2], dtype=np.complex128)
            v0 = np.zeros(1 << code.n_total, dtype=np.complex128)
            v0[0], v0[1 << (code.n_total - 1)] = psi[0], psi[1]
            v1 = apply_ops_oracle(v0, code.encode_ops(), code.n_total)
            single = [("I", 1.0 - p), ("X", p / 3.0), ("Y", p / 3.0), ("Z", p / 3.0)]
            rho_out = np.zeros((2, 2), dtype=np.complex128)
            for c0, w0 in single:
                for c1, w1 in single:
                    for c2, w2 in single:
                        w = w0 * w1 * w2
                        v2 = apply_ops_oracle(v1, DeterministicErrorInjector.to_gate_ops(c0 + c1 + c2, 3), code.n_total)
                        v3 = apply_ops_oracle(v2, code.syndrome_ops(), code.n_total)
                        da3 = v3.reshape(1 << code.n_data, 1 << code.n_ancilla)
                        probs_m = np.sum(np.abs(da3) ** 2, axis=0)
                        for m, pm in enumerate(probs_m):
                            if pm <= 1e-15:
                                continue
                            s = format(m, f"0{code.n_ancilla}b")
                            da4 = np.zeros_like(da3)
                            da4[:, m] = da3[:, m] / math.sqrt(float(pm))
                            v4 = apply_ops_oracle(da4.reshape(1 << code.n_total), code.recovery_ops(s), code.n_total)
                            v5 = apply_ops_oracle(v4, code.decode_ops(), code.n_total)
                            rho_out += w * float(pm) * partial_trace_to_qubit0(v5, code.n_total)
            return abs_rho01(rho_out)

    if code_name == "five_qubit":
        if channel == "depolarizing":
            f_e = float(analytical_fidelity("five_qubit", "depolarizing", p, metric="entanglement"))
            return 0.5 * abs(1.0 - (4.0 / 3.0) * (1.0 - f_e))
        if channel in ("bit_flip", "phase_flip"):
            axis = "X" if channel == "bit_flip" else "Z"
            cosets = five_qubit_single_axis_logical_cosets(axis, p)
            return 0.5 * abs(cosets["I"] + cosets["Y"] - cosets["X"] - cosets["Z"])
        if channel == "collective_dephasing":
            # Compute via exact 2-term mixture (IIIII with 1-p, ZZZZZ with p) -> Z_L flips phase
            return 0.5 * abs(1.0 - 2.0 * p)

    raise ValueError(f"Unsupported analytical coherence combination: {code_name!r}, {channel!r}")


def get_code_oracle(code_name: str) -> Any:
    _ensure_src_on_sys_path()
    from qecc.codes import get_code

    return get_code(code_name)


def list_codes_oracle() -> list[str]:
    _ensure_src_on_sys_path()
    from qecc.codes import list_available_codes

    return list_available_codes()


def list_backends_oracle() -> list[str]:
    _ensure_src_on_sys_path()
    from qecc.backends import list_available_backends

    return list_available_backends()
