"""Module entrypoint for ``python -m qecc.web``."""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from qecc.web.app import run_server


def build_web_parser() -> argparse.ArgumentParser:
    """Build argument parser for ``python -m qecc.web``."""
    parser = argparse.ArgumentParser(
        prog="python -m qecc.web",
        description="Start the local interactive QECC & DFS visualization web server.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind host address (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="TCP port (0 for ephemeral port, default: 8765).",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not automatically open a browser window.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse CLI arguments and launch the QECC web server."""
    parser = build_web_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        return int(
            run_server(
                host=args.host,
                port=args.port,
                open_browser=not args.no_browser,
            )
        )
    except ValueError as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
