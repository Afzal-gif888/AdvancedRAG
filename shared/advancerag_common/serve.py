"""Container entry point: run a service with uvicorn on $HOST:$PORT.

    python -m advancerag_common.serve app.main:app 8000      # the second argument is the default PORT

HOST=:: listens dual-stack: IPv6 *and* IPv4 on one socket. Railway's private network can be
IPv6-only, while health checks, Docker and public proxies may connect over IPv4. `uvicorn --host ::`
does not do this: asyncio sets IPV6_V6ONLY on IPv6 listening sockets, so IPv4 clients are refused
(found in the production rehearsal). Here the socket is created with IPV6_V6ONLY=0 and handed to
uvicorn. Otherwise the settings equal the previous `uvicorn ... --proxy-headers
--timeout-graceful-shutdown 30` command line; uvicorn stays PID 1 (the Dockerfile `exec`s this).
"""
import os
import socket
import sys
from typing import Optional, Sequence


def listen_socket(host: str, port: int) -> socket.socket:
    if ":" in host:  # IPv6 literal, e.g. "::" -> dual-stack
        sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
    else:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    sock.set_inheritable(True)
    return sock


def main(argv: Optional[Sequence[str]] = None) -> None:
    import uvicorn

    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2:
        raise SystemExit("usage: python -m advancerag_common.serve <module:app> <default-port>")
    app, default_port = args
    host = os.environ.get("HOST") or "0.0.0.0"
    port = int(os.environ.get("PORT") or default_port)
    config = uvicorn.Config(app, host=host, port=port, proxy_headers=True, timeout_graceful_shutdown=30)
    uvicorn.Server(config).run(sockets=[listen_socket(host, port)])


if __name__ == "__main__":
    main()
