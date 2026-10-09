"""Container entry point (shared/advancerag_common/serve.py): HOST=:: must accept IPv6 AND IPv4.
`uvicorn --host ::` does not (asyncio sets IPV6_V6ONLY), which made every IPv4 client - Docker
health checks, the published port - fail in the production rehearsal."""
import socket

import pytest

from advancerag_common.serve import listen_socket


def _accepts(server: socket.socket, family: int, address: str) -> bool:
    port = server.getsockname()[1]
    with socket.socket(family, socket.SOCK_STREAM) as client:
        client.settimeout(2)
        try:
            client.connect((address, port))
        except OSError:
            return False
    conn, _ = server.accept()
    conn.close()
    return True


@pytest.mark.skipif(not socket.has_ipv6, reason="no IPv6 on this host")
def test_double_colon_listens_dual_stack():
    server = listen_socket("::", 0)
    server.listen()
    try:
        assert server.getsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY) == 0
        assert _accepts(server, socket.AF_INET6, "::1")
        assert _accepts(server, socket.AF_INET, "127.0.0.1")
    finally:
        server.close()


def test_ipv4_host_is_plain_ipv4():
    server = listen_socket("127.0.0.1", 0)
    server.listen()
    try:
        assert server.family == socket.AF_INET and _accepts(server, socket.AF_INET, "127.0.0.1")
    finally:
        server.close()
