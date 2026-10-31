"""Real recursive REPL/broker protocol; model and GPU launch results are scripted."""

import hashlib
import json
import os
from pathlib import Path

import pytest
from cute_rlm import feedback as feedback_module
from cute_rlm.contracts import VerificationConfig
from cute_rlm.dataset import task_context
from cute_rlm.docs import docs_bootstrap
from cute_rlm.prompts import CUTE_SYSTEM_PROMPT
from cute_rlm.resources import build_resources, resources_bootstrap
from rlm_train.native import NativeLimits, NativeRuntime
from rlm_train.proxy import SubLLMProxy
from rlm_train.repl.subprocess import SubprocessReplBackend

from tests.test_cute_compile_feedback import write_bundle
from tests.test_cute_eval_preparation import config
from tests.test_native_compile_feedback import ScriptedClient


@pytest.mark.asyncio
@pytest.mark.parametrize("repair_succeeds", [True, False])
async def test_documented_complete_model_repair_uses_child_repl_and_trusted_task(
    tmp_path, monkeypatch, repair_succeeds
):
    src = str(Path(__file__).parents[1] / "training/src")
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(filter(None, (src, os.getenv("PYTHONPATH")))))
    resources = build_resources(config())
    # Source is data only: neither host nor reasoning workers execute the GPU fixture.
    good_source = resources["examples"]["elementwise_add"]
    bad_source = good_source.replace("import cutlass.cute as cute", "import cute")
    reference = (
        "import torch\nclass Model(torch.nn.Module):\n def forward(self, a, b):\n  return a + b\n"
    )
    task = {
        "task_id": "root-addition-fixture",
        "reference_code": reference,
        "source": {"sha256": hashlib.sha256(reference.encode()).hexdigest()},
    }
    verification = VerificationConfig(**config().verification)
    requests = []

    async def launch(command, request, settings):
        requests.append(request)
        assert command == ["scripted-isolated-worker"]
        assert settings == verification
        assert request["task_id"] == task["task_id"]
        if request["role"] == "reference":
            assert request["source"] == reference and request["prepare_only"] is True
            write_bundle(Path(request["output_dir"]), settings)
            return {
                "status": "prepared",
                "source_sha256": request["source_sha256"],
                "training": False,
                "module_training": {"": False},
            }
        assert request["compile_feedback"] is True
        assert request["input_dir"] == str(service.input_directory)
        assert request["training"] is False
        assert request["source"] in (bad_source, good_source)
        failed_import = request["source"] == bad_source
        succeeded = not failed_import and repair_succeeds
        detail = "No module named 'cute'" if failed_import else "fixture-only next API blocker"
        return {
            "status": "compiled" if succeeded else "runtime_error",
            "source_sha256": request["source_sha256"],
            "detail": "one forward synchronized" if succeeded else detail,
            "error_type": None
            if succeeded
            else "ModuleNotFoundError"
            if failed_import
            else "TypeError",
            "traceback": "" if succeeded else "Traceback: " + detail,
            "stdout": "scripted worker stdout",
            "stderr": "" if succeeded else detail,
            "launcher_stderr": "scripted launcher diagnostic",
        }

    monkeypatch.setattr(feedback_module, "launch_phase", launch)
    service = feedback_module.CompileFeedback(
        task, verification, tmp_path, worker_command=["scripted-isolated-worker"]
    )
    root_turns = child_turns = 0

    def script(payload):
        nonlocal root_turns, child_turns
        assert payload.get("tools"), "A depth-1 child must be a REPL session, not a plain leaf"
        prompt = payload["messages"][1]["content"]
        if prompt == "root-repair":
            root_turns += 1
            if root_turns == 1:
                return (
                    f"best_source = {bad_source!r}\n"
                    "root_only = 'not inherited'\n"
                    "feedback = compile_candidate(best_source)\n"
                    "assert feedback['error_type'] == 'ModuleNotFoundError'\n"
                    "assert not answer['ready']"
                )
            assert root_turns == 2
            return resources["delegation_example"] + (
                "\nassert best_source == repair_request['candidate_source']\n"
                "assert not answer['ready'] and root_only == 'not inherited'\n"
                "assert 'child_only' not in SHOW_VARS()\n"
                "assert len(child_report['checks']) == 2\n"
                "assert child_report['checks'][0]['error_type'] == 'ModuleNotFoundError'\n"
                "if child_report['checks'][-1]['status'] == 'compiled':\n"
                "    best_source = child_source\n"
                "answer['content'] = best_source\n"
                "answer['ready'] = True"
            )
        delegated = json.loads(prompt.split("\nREQUEST_JSON:\n", 1)[1])
        assert set(delegated) == {"cute_dsl_task", "candidate_source", "last_feedback"}
        assert delegated["cute_dsl_task"] == reference
        assert delegated["candidate_source"] == bad_source
        assert delegated["last_feedback"]["error_type"] == "ModuleNotFoundError"
        assert "not inherited" not in prompt
        child_turns += 1
        if child_turns == 1:
            return (
                "import json\n"
                "assert isinstance(context, str) and 'root_only' not in SHOW_VARS()\n"
                "assert 'best_source' not in SHOW_VARS() and 'feedback' not in SHOW_VARS()\n"
                "assert callable(compile_candidate) and callable(search_docs) and callable(read_doc)\n"
                "assert resources['examples'] and resources['delegation_example']\n"
                "request = json.loads(context.split('\\nREQUEST_JSON:\\n', 1)[1])\n"
                "best_source = request['candidate_source']\n"
                "child_only = 'persistent child state'\n"
                "checks = [compile_candidate(best_source)]\n"
                "hit = search_docs('cutlass cute', limit=1)[0]\n"
                "assert 'import cutlass.cute as cute' in read_doc(hit['id'], start=hit['start'])['excerpt']\n"
                "try:\n"
                "    rlm_query('forbidden-grandchild')\n"
                "except RuntimeError as error:\n"
                "    assert 'maximum recursion depth' in str(error)\n"
                "else:\n"
                "    raise AssertionError('depth-1 child admitted a grandchild')\n"
                "assert not answer['ready']"
            )
        assert child_turns == 2
        return (
            "assert child_only == 'persistent child state' and len(checks) == 1\n"
            "best_source = best_source.replace('import cute', 'import cutlass.cute as cute')\n"
            "checks.append(compile_candidate(best_source))\n"
            "answer['content'] = json.dumps({'source': best_source, 'checks': checks})\n"
            "answer['ready'] = True"
        )

    docs = {
        "imports": {
            "title": "CuTe import fixture",
            "text": "import cutlass.cute as cute\nfrom cutlass.cute.runtime import from_dlpack\n",
            "sha256": "0" * 64,
        }
    }
    processes = []

    class RecordingBackend(SubprocessReplBackend):
        async def start(self, *args, **kwargs):
            await super().start(*args, **kwargs)
            processes.append(self._proc)

    proxy = SubLLMProxy(socket_path=tmp_path / "proxy.sock")
    await proxy.start()
    client = ScriptedClient(script)
    runtime = NativeRuntime(
        client,
        "scripted-model",
        proxy,
        limits=NativeLimits(
            max_depth=1,
            max_turns=3,
            max_children=2,
            max_model_calls=4,
            episode_timeout=15,
            request_timeout=5,
            max_prompt_chars=None,
        ),
        child_repl_at_limit=True,
        backend_factory=RecordingBackend,
        system_prompt=CUTE_SYSTEM_PROMPT,
        bootstrap_code=docs_bootstrap(docs) + resources_bootstrap(resources),
        compile_candidate=service,
        event_path=tmp_path / "events.jsonl",
    )
    try:
        record = await runtime.run("root-repair", task_context(task))
    finally:
        await runtime.close()
        await proxy.stop()

    assert record["status"] == "completed", record
    assert record["final_answer"] == (good_source if repair_succeeds else bad_source)
    assert root_turns == child_turns == 2
    assert runtime.calls == len(client.calls) == 4
    assert len(record["child_sessions"]) == 1
    child = record["child_sessions"][0]
    assert child["depth"] == 1 and child["parent_session_id"] == record["session_id"]
    assert child["status"] == "completed"
    report = json.loads(child["final_answer"])
    assert set(report) == {"source", "checks"}
    assert report["source"] == good_source
    assert [check["status"] for check in report["checks"]] == [
        "runtime_error",
        "compiled" if repair_succeeds else "runtime_error",
    ]
    assert report["checks"][0]["traceback"] == "Traceback: No module named 'cute'"
    assert all(check["session_id"] == child["session_id"] for check in report["checks"])
    assert [request["role"] for request in requests] == [
        "reference",
        "candidate",
        "candidate",
        "candidate",
    ]
    candidates = requests[1:]
    assert [request["session_id"] for request in candidates] == [
        record["session_id"],
        child["session_id"],
        child["session_id"],
    ]
    assert len({request["input_dir"] for request in candidates}) == 1
    assert len(record["compile_feedback"]) == 1 and len(child["compile_feedback"]) == 2
    assert len(processes) == 2 and all(process.returncode is not None for process in processes)
    assert all(session.backend._proc is None for session in runtime.sessions.values())
