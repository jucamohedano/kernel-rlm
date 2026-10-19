import importlib.util
from pathlib import Path

import pytest

SOURCE = Path(__file__).parents[1] / "training/environments/kernel_rlm/dashboard/trackio_tunnel.py"
SPEC = importlib.util.spec_from_file_location("trackio_tunnel", SOURCE)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_tunnel_is_private_and_noninteractive():
    command = module.ssh_command("example.test", "riftuser", 60008, 7861, 7861)
    assert command[command.index("-L") + 1] == "127.0.0.1:7861:127.0.0.1:7861"
    assert "BatchMode=yes" in command
    assert "ExitOnForwardFailure=yes" in command
    assert "ServerAliveInterval=30" in command
    assert "-R" not in command
    assert command[-1] == "riftuser@example.test"


@pytest.mark.parametrize("port", [0, -1, 65536])
def test_invalid_ports_are_rejected(port):
    with pytest.raises(ValueError):
        module.ssh_command("example.test", "riftuser", port, 7861, 7861)
