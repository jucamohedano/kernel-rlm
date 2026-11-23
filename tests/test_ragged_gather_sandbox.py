import asyncio
import os
from pathlib import Path

import pytest

from training.environments.kernel_rlm.research.ragged_gather_reduce import sandbox, task


def test_requires_the_pinned_worker(tmp_path):
    changed = tmp_path / "worker.py"
    changed.write_text("unrelated benign marker\n")
    with pytest.raises(ValueError, match="worker identity mismatch"):
        sandbox.SandboxedReplBackend(changed, tmp_path, max_sub_calls=0, wall_seconds=1)


@pytest.mark.parametrize(
    "url", ["http://localhost:1", "https://127.0.0.1:1", "http://127.0.0.1:1/alt"]
)
def test_gate_requires_canonical_trusted_origin(url):
    with pytest.raises(ValueError, match="loopback"):
        sandbox.ChildGate(url, "offline", 1)


def test_gate_rejects_other_routes_models_and_oversize_before_dispatch():
    async def check():
        gate = sandbox.ChildGate("http://127.0.0.1:1", "expected", 0)
        for path, body in (
            ("/rollout/other/llm_query", {"prompt": "x", "depth": 1}),
            ("/rollout/expected/unknown", {"prompt": "x", "depth": 1}),
            ("/rollout/expected/llm_query", {"prompt": "x", "depth": 1, "model": "not configured"}),
            ("/rollout/expected/llm_query", {"prompt": "x", "depth": True}),
            ("/rollout/expected/llm_query", {"prompt": "ø" * sandbox.PROMPT_BYTES, "depth": 1}),
        ):
            assert "invalid" in (await gate.forward({"path": path, "body": body}))["error"]
        assert (
            await gate.forward(
                {
                    "path": "/rollout/expected/llm_query",
                    "body": {"prompt": "x", "depth": 1},
                }
            )
        )["error"] == "sub-call budget exhausted"
        assert gate.counts["rejected_requests"] == 6
        assert gate.counts["reserved_prompts"] == 0

    asyncio.run(check())


@pytest.fixture
def native_paths():
    checkout = os.environ.get("HARNESS_CG_CHECKOUT")
    runtime = os.environ.get("RAGGED_PYTHON_RUNTIME")
    if not checkout or not runtime:
        pytest.skip(
            "set HARNESS_CG_CHECKOUT and RAGGED_PYTHON_RUNTIME for the rootless native integration"
        )
    return Path(checkout), Path(runtime)


def test_native_isolated_context_scratch_subcalls_and_exact_submit(
    native_paths, tmp_path, monkeypatch
):
    from rlm_train import proxy

    checkout, runtime = native_paths
    assert Path(proxy.__file__).resolve().is_relative_to(checkout.resolve())
    worker = checkout / "src/rlm_train/worker.py"
    marker = tmp_path / "host-only"
    marker.write_text("benign host marker\n")
    monkeypatch.setenv("RAGGED_DUMMY_HOST_ENV", "host-only")

    async def check():
        native_proxy = proxy.SubLLMProxy()
        records = []
        native_proxy.register(
            "offline",
            proxy.ClientHandle(
                client=None,
                model="configured-child-only",
                record_call=records.append,
                fake_query=lambda prompt, state: "local:" + prompt,
                fake_query_batched=lambda prompts, state: ["local:" + prompt for prompt in prompts],
            ),
        )
        await native_proxy.start()
        backend = sandbox.SandboxedReplBackend(worker, runtime, max_sub_calls=3, wall_seconds=15)
        blank = sandbox.SandboxedReplBackend(worker, runtime, max_sub_calls=0, wall_seconds=15)
        try:
            await backend.start(native_proxy.url, "offline")
            assert await backend.load_context(task.public_context()) == 0
            out = await backend.execute(
                f"""from pathlib import Path
import os
seed = 17
print('## contract.h' in context, Path({str(marker)!r}).exists(), 'RAGGED_DUMMY_HOST_ENV' in os.environ)
print(Path('/dev/nvidia0').exists(), os.access('/worker.py', os.W_OK))
print([line.split(':', 1)[0].strip() for line in Path('/proc/net/dev').read_text().splitlines()[2:]])
Path('/tmp/rg-marker').write_text('private tmpfs')
print(llm_query('test', model='ignored by native worker'))
"""
            )
            assert out.stdout.splitlines() == [
                "True False False",
                "False False",
                "['lo']",
                "local:test",
            ]
            assert not out.stderr and out.final_answer is None
            out = await backend.execute(
                """print(seed + 1, Path('/tmp/rg-marker').read_text())
print(llm_query_batched(['second', 'first']))
print('budget exhausted' in llm_query('not dispatched'))
"""
            )
            assert out.stdout.splitlines() == [
                "18 private tmpfs",
                "['local:second', 'local:first']",
                "True",
            ]
            assert out.final_answer is None and not out.stderr
            source = "// verbatim marker with trailing spaces  \n// λ\n\n"
            out = await backend.execute(f"answer['content'] = {source!r}; answer['ready'] = True")
            assert (
                task.source_from_state({"rlm_final_answer": out.final_answer}).source
                == source.encode()
            )
            assert backend.broker_evidence == dict(
                reserved_prompts=3,
                completed_prompts=3,
                completed_requests=2,
                rejected_requests=1,
                upstream_errors=0,
            )
            assert [(r["model"], r["prompt"]) for r in records] == [
                ("configured-child-only", prompt) for prompt in ("test", "second", "first")
            ]
            await backend.stop()
            await blank.start(native_proxy.url, "different")
            out = await blank.execute(
                """from pathlib import Path
print('seed' in SHOW_VARS(), Path('/tmp/rg-marker').exists(), 'context' in SHOW_VARS())
answer['content'] = 'not ready'
"""
            )
            assert out.stdout == "False False False\n" and out.final_answer is None
            out = await blank.execute("print(llm_query('not admitted'))")
            assert out.stdout.startswith("Error:") and out.final_answer is None
            assert blank.broker_evidence["reserved_prompts"] == 0
        finally:
            await backend.stop()
            await blank.stop()
            await native_proxy.stop()

    asyncio.run(check())


def test_startup_failure_never_falls_back(native_paths, monkeypatch):
    checkout, runtime = native_paths
    monkeypatch.setattr(sandbox, "BWRAP", Path("/usr/bin/false"))

    async def check():
        backend = sandbox.SandboxedReplBackend(
            checkout / "src/rlm_train/worker.py", runtime, max_sub_calls=0, wall_seconds=5
        )
        with pytest.raises(sandbox.SandboxError, match="failed to start"):
            await backend.start("http://127.0.0.1:1", "offline")
        assert backend.process is None and backend.temp is None

    asyncio.run(check())


def test_native_deadline_tears_down_process_and_socket(native_paths):
    checkout, runtime = native_paths
    worker = checkout / "src/rlm_train/worker.py"

    async def check():
        backend = sandbox.SandboxedReplBackend(worker, runtime, max_sub_calls=0, wall_seconds=1.5)
        with pytest.raises(sandbox.SandboxError, match="depth-1"):
            await backend.start("http://127.0.0.1:1", "offline", depth=2)
        await backend.start("http://127.0.0.1:1", "offline")
        process = backend.process
        temp = Path(backend.temp.name)
        try:
            with pytest.raises(sandbox.SandboxError, match="protocol failure or deadline"):
                await asyncio.wait_for(backend.execute("import time; time.sleep(15)"), timeout=5)
            assert process.returncode is not None and not temp.exists()
            with pytest.raises(sandbox.SandboxError, match="not running"):
                await backend.execute("print('fallback forbidden')")
        finally:
            await backend.stop()

    asyncio.run(check())
