"""Command-line entry point: ``investing-engine-mcp`` or ``python -m investing_engine.mcp_server``.

stdio (default) is what desktop clients such as Claude Desktop launch.
Streamable HTTP is bound to loopback only, with DNS-rebinding protection,
until token authentication is configured for remote access.
"""

from __future__ import annotations

import argparse
import ipaddress
import logging
import sys

from mcp.server.transport_security import TransportSecuritySettings

from investing_engine.config import get_settings
from investing_engine.mcp_server.server import build_server
from investing_engine.services import MarketData


def _is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="investing-engine-mcp", description=__doc__)
    parser.add_argument("--transport", choices=("stdio", "streamable-http"), default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)

    # stdout carries the stdio protocol stream; logs must go to stderr.
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)

    options: dict[str, object] = {}
    if args.transport == "streamable-http":
        if not _is_loopback(args.host):
            parser.error("Remote binding requires authentication; bind to 127.0.0.1")
        authority = f"{args.host}:{args.port}"
        options = {
            "host": args.host,
            "port": args.port,
            "transport_security": TransportSecuritySettings(
                enable_dns_rebinding_protection=True,
                allowed_hosts=[authority, f"localhost:{args.port}"],
                allowed_origins=[f"http://{authority}", f"http://localhost:{args.port}"],
            ),
        }

    market = MarketData.from_settings(get_settings())
    try:
        build_server(market, **options).run(transport=args.transport)
    finally:
        market.close()


if __name__ == "__main__":
    main()
