import asyncio
import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest


class CollectionCaptured(Exception):
    pass


def test_live_driver_forwards_collector_identity(monkeypatch, tmp_path):
    root = Path(__file__).parents[1] / "training/environments/kernel_rlm"
    monkeypatch.syspath_prepend(str(root))
    driver = importlib.import_module("kernel_trl.live_driver")
    commands = []

    class Ready:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        def get(self, url):
            return self

    async def capture(command, log, timeout):
        commands.append(command)
        raise CollectionCaptured

    async def stop(group):
        pass

    monkeypatch.setattr(driver.os, "getpgid", lambda group: group)
    monkeypatch.setattr(driver.aiohttp, "ClientSession", lambda **kwargs: Ready())
    monkeypatch.setattr(driver, "execute", capture)
    monkeypatch.setattr(driver, "stop_group", stop)
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text("{}")
    arguments = SimpleNamespace(
        output_dir=tmp_path,
        inference_process_group=123,
        run_id="execution-1",
        deadline=driver.time.time() + 1800,
        dataset=dataset,
        checkpoint=tmp_path / "checkpoint",
        python="python",
    )

    with pytest.raises(CollectionCaptured):
        asyncio.run(driver.run(arguments))

    command = commands[0]
    assert command[command.index("--execution-run-id") + 1] == "execution-1"
    assert command[command.index("--group-id") + 1] == "execution-1:group-0"
    assert command[command.index("--attempt-id") + 1] == "attempt-0"
