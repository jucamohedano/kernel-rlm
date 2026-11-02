import asyncio
from types import SimpleNamespace

import pytest
from cute_rlm import verifier


@pytest.mark.asyncio
async def test_phase_deadline_is_distinct_from_unconfirmed_cleanup(monkeypatch):
    class Process:
        pid = 123456
        returncode = None

        async def communicate(self, value):
            await asyncio.Event().wait()

        async def wait(self):
            return 0

    async def create(*args, **kwargs):
        return Process()

    def kill(*args):
        pass

    async def cleanup_timeout(*args, **kwargs):
        args[0].close()
        raise TimeoutError("cleanup deadline")

    monkeypatch.setattr(verifier.asyncio, "create_subprocess_exec", create)
    monkeypatch.setattr(verifier.os, "killpg", kill)
    monkeypatch.setattr(verifier.asyncio, "wait_for", cleanup_timeout)
    with pytest.raises(RuntimeError, match="cleanup could not be confirmed"):
        await verifier.launch_phase(["fixture"], {}, SimpleNamespace(timeout_s=0.001))
