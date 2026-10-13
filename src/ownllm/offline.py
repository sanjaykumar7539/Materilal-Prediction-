"""Best-effort in-process offline enforcement.

For a hard offline guarantee also disable host/container egress at the OS layer
and pre-install all wheels into an offline image/wheelhouse.
"""
from __future__ import annotations

import os
import socket
import ipaddress


class OfflineViolation(RuntimeError):
    """Raised when code attempts a non-loopback network connection."""


_original_connect = socket.socket.connect
_enforced = False


def enforce_offline() -> None:
    """Block outbound sockets except loopback. Idempotent."""
    global _enforced
    if _enforced:
        return
    os.environ.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "NO_PROXY": "*",
            "no_proxy": "*",
        }
    )

    def blocked(self, address):  # type: ignore[no-untyped-def]
        host = address[0] if isinstance(address, tuple) else str(address)
        if host in {"127.0.0.1", "localhost", "::1"}:
            return _original_connect(self, address)
        allow_private = os.getenv("OWNLLM_OFFLINE_ALLOW_PRIVATE", "0") == "1"
        if allow_private:
            try:
                ip = ipaddress.ip_address(host)
                if ip.is_private or ip.is_loopback:
                    return _original_connect(self, address)
            except ValueError:
                pass
        raise OfflineViolation(f"Outbound network blocked (offline mode): {host}")

    socket.socket.connect = blocked  # type: ignore[method-assign]
    _enforced = True


def relax_offline() -> None:
    """Restore original socket.connect (mainly for tests)."""
    global _enforced
    socket.socket.connect = _original_connect  # type: ignore[method-assign]
    _enforced = False
