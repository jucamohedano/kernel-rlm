"""CPU-only helper discovery and protected-name regressions."""

import json
import os
from pathlib import Path

import pytest
from rlm_train.repl.subprocess import SubprocessReplBackend
from rlm_train.worker import Worker


def test_inventory_exposes_injected_helpers_and_variables():
    worker = Worker("http://unused", "contract")
    worker.load_context({"reference_code": "reference"})
    worker.globals["resources"] = {"helper_help": {}}
    worker.globals["read_doc"] = lambda doc_id, start=0, chars=1200: {"excerpt": "docs"}
    worker.execute("best_source = 'candidate'")
    inventory = worker._show_vars()
    assert "compile_candidate(source)" in inventory
    assert "read_doc(doc_id, start=0, chars=1200)" in inventory
    assert "'resources': 'dict'" in inventory and "'best_source': 'str'" in inventory
    assert "'context': 'dict'" in inventory
    assert "no imports" in inventory


@pytest.mark.parametrize(
    "code",
    [
        "compile_candidate = None",
        "def compile_candidate(source):\n    return {'status': 'parsed'}",
        "from compile_candidate import compile_candidate",
        "from cute_rlm import resources",
        "read_doc = None",
        "del SHOW_VARS",
    ],
)
def test_protected_names_reject_rebinding_before_side_effects(monkeypatch, code):
    monkeypatch.setenv("RLM_TRAIN_READONLY_CONTEXT", "1")
    worker = Worker("http://unused", "contract")
    worker.globals["resources"] = {"marker": "original"}
    result = worker.execute("marker = 'executed'\n" + code)
    assert "read-only" in result["stderr"] and "do not import or redefine" in result["stderr"]
    assert "marker" not in worker.locals
    assert worker.globals["resources"] == {"marker": "original"}
    assert worker.globals["compile_candidate"].__self__ is worker


def test_direct_helper_call_and_submission_still_work(monkeypatch):
    monkeypatch.setenv("RLM_TRAIN_READONLY_CONTEXT", "1")
    worker = Worker("http://unused", "contract")
    calls = []

    def proxy(path, payload):
        calls.append((path, payload))
        return {"response": {"status": "compiled"}}

    monkeypatch.setattr(worker, "_proxy_post", proxy)
    result = worker.execute(
        "best_source = 'class ModelNew: pass'\n"
        "feedback = compile_candidate(best_source)\n"
        "answer['content'] = best_source\n"
        "answer['ready'] = True"
    )
    assert not result["stderr"]
    assert json.loads(result["stdout"]) == {"status": "compiled"}
    assert calls == [("compile_candidate", {"source": "class ModelNew: pass"})]
    assert result["final_answer"] == "class ModelNew: pass"


@pytest.mark.asyncio
async def test_subprocess_bootstrap_helpers_are_visible_and_protected(monkeypatch):
    src = str(Path(__file__).parents[1] / "training/src")
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(filter(None, (src, os.getenv("PYTHONPATH")))))
    monkeypatch.setenv("RLM_TRAIN_READONLY_CONTEXT", "1")
    backend = SubprocessReplBackend()
    await backend.start("http://unused", "contract")
    try:
        await backend.bootstrap(
            "resources = {'marker': 'original'}\n"
            "def read_doc(doc_id, start=0, chars=1200):\n"
            "    return {'excerpt': 'documentation'}\n"
        )
        inventory = await backend.execute("print(SHOW_VARS())")
        assert not inventory.stderr
        assert "compile_candidate(source)" in inventory.stdout
        assert "read_doc(doc_id, start=0, chars=1200)" in inventory.stdout
        rejected = await backend.execute("resources = {}")
        assert "read-only" in rejected.stderr
        observed = await backend.execute(
            "print(resources['marker'], read_doc('example')['excerpt'])"
        )
        assert observed.stdout == "original documentation\n" and not observed.stderr
    finally:
        await backend.stop()


def test_unprotected_worker_keeps_existing_rebinding_behavior(monkeypatch):
    monkeypatch.delenv("RLM_TRAIN_READONLY_CONTEXT", raising=False)
    worker = Worker("http://unused", "contract")
    result = worker.execute(
        "compile_candidate = lambda source: 'local'; print(compile_candidate('x'))"
    )
    assert result["stdout"] == "local\n" and not result["stderr"]
    assert worker.globals["compile_candidate"].__self__ is worker
