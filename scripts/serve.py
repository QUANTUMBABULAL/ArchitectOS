"""
Entry point for the ArchitectOS engine server.

Runs the existing engine with a WebSocket bridge attached so the desktop
interface can connect. The terminal keeps printing every log line exactly
as it does for the REPL: this process is the engine plus a socket, not a
different engine.

Usage:
    python -m scripts.serve
    python -m scripts.serve --port 8777

The desktop app connects to ws://127.0.0.1:<port>/ws by default.
"""

from __future__ import annotations

import argparse
import logging
import sys

import uvicorn

from src.api.ws_server import create_app
from src.config import get_settings
from src.constants import LOG_FORMAT, LOG_LEVEL
from src.logger import configure_logging, get_logger


def main() -> int:
    """
    Configure logging and run the engine server.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(
        description="Run the ArchitectOS engine with a desktop bridge."
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind address. Loopback by default: the engine drives an "
        "authenticated browser and must not be exposed to a network.",
    )
    parser.add_argument("--port", type=int, default=8777)
    args = parser.parse_args()

    configure_logging(
        level=getattr(logging, LOG_LEVEL),
        format_string=LOG_FORMAT,
    )
    settings = get_settings()
    level = getattr(logging, settings.log_level)
    root = logging.getLogger()
    root.setLevel(level)
    for handler in root.handlers:
        handler.setLevel(level)

    logger = get_logger(__name__)

    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        logger.warning(
            "Binding to %s exposes an authenticated browser session to "
            "the network. Use loopback unless you understand the risk.",
            args.host,
        )

    print("=" * 70)
    print("ARCHITECTOS ENGINE — desktop bridge")
    print("=" * 70)
    print(f"WebSocket : ws://{args.host}:{args.port}/ws")
    print(f"Health    : http://{args.host}:{args.port}/health")
    print("Logs continue in this terminal. Close with Ctrl+C.")
    print("=" * 70)

    try:
        uvicorn.run(
            create_app(),
            host=args.host,
            port=args.port,
            log_level=settings.log_level.lower(),
        )
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
