"""CPU-only subprocess transport regressions; no proxy or inference requests."""

import os
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from rlm_train.repl.subprocess import SubprocessReplBackend, WorkerProtocolError


@pytest.fixture(autouse=True)
def worker_import_path(monkeypatch):
    src = str(Path(__file__).parents[1] / "training/src")
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(filter(None, (src, os.getenv("PYTHONPATH")))))


@pytest.mark.asyncio
@pytest.mark.parametrize("stream_limit", [128, SubprocessReplBackend._STREAM_LIMIT])
async def test_full_json_response_larger_than_stream_limit(stream_limit):
    backend = SubprocessReplBackend()
    backend._STREAM_LIMIT = stream_limit
    size = 2 * stream_limit + 17
    await backend.start("http://unused", "transport")
    try:
        result = await backend.execute(f"print('x' * {size} + 'é終', end='')")
        assert result.stdout == "x" * size + "é終"
        assert (await backend.execute("print('next')")).stdout == "next\n"
    finally:
        await backend.stop()


@pytest.mark.asyncio
async def test_stderr_without_newline_does_not_stop_draining():
    backend = SubprocessReplBackend()
    backend._STREAM_LIMIT = 1024
    await backend.start("http://unused", "stderr")
    drain = backend._stderr_task
    try:
        result = await backend.execute("import os; os.write(2, b'e' * 65536); print('done')")
        assert result.stdout == "done\n"
        assert not drain.done()
        assert backend._stderr_text()
        assert set(backend._stderr_text()) == {"e"}
    finally:
        await backend.stop()
    assert drain.done()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["large_bad_json", "invalid_utf8", "partial_eof", "eof"])
async def test_protocol_failure_and_stop_reap_worker(failure):
    backend = SubprocessReplBackend()
    backend._STREAM_LIMIT = 1024
    await backend.start("http://unused", "bad-protocol")
    proc, drain = backend._proc, backend._stderr_task
    payload = {
        "large_bad_json": b"!" * 4096 + b"\n",
        "invalid_utf8": b"\xff\n",
        "partial_eof": b"!" * 4096,
        "eof": b"",
    }[failure]
    # Leave a second bad record for stop's shutdown read, or close stdout mid-record.
    code = f"import os; os.write(1, {payload!r})"
    code += "; os.close(1)" if "eof" in failure else f"; os.write(1, {payload!r})"
    try:
        with pytest.raises(WorkerProtocolError):
            await backend.execute(code)
    finally:
        await backend.stop()
    assert proc.returncode is not None
    assert backend._proc is None
    assert drain.done()


@pytest.mark.asyncio
async def test_stop_reaps_even_if_shutdown_read_raises_unexpected_error():
    backend = SubprocessReplBackend()
    await backend.start("http://unused", "stop-error")
    proc = backend._proc
    backend._read_line = AsyncMock(side_effect=RuntimeError("unexpected read failure"))
    with pytest.raises(RuntimeError, match="unexpected read failure"):
        await backend.stop()
    assert proc.returncode is not None
    assert backend._proc is None
