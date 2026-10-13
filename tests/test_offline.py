import socket
import pytest
from ownllm.offline import enforce_offline, OfflineViolation, relax_offline


def test_offline_blocks_external():
    enforce_offline()
    s = socket.socket()
    try:
        with pytest.raises(OfflineViolation):
            s.connect(("example.com", 80))
    finally:
        s.close()
        relax_offline()
