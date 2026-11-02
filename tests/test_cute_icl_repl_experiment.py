"""CPU-only freeze/launch sequencing plus a real scripted native root-child REPL."""

import asyncio
import base64
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from types import ModuleType, SimpleNamespace

import httpx
import pytest
from cute_rlm.icl import repl_experiment as experiment
from cute_rlm.icl.prompts import render_prompt as tool_free_prompt
from cute_rlm.icl.tasks import load_tasks
from rlm_train.native import NativeProtocolError, NativeRuntime
from rlm_train.proxy import SubLLMProxy
from rlm_train.repl.subprocess import SubprocessReplBackend
from rlm_train.token_budget import ContextBudgetPolicy, VllmPromptCounter


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    # A private source copy avoids coupling integrity tests to parallel implementation work.
    project = tmp_path / "project"
    for name in experiment.SOURCE_ROOTS:
        shutil.copytree(
            experiment.PROJECT / name, project / name, ignore=shutil.ignore_patterns("__pycache__")
        )
    for name in ("pyproject.toml", "uv.lock", "training/environments/cute_rlm/pyproject.toml"):
        target = project / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(experiment.PROJECT / name, target)
    monkeypatch.setattr(experiment, "PROJECT", project)
    run = tmp_path / "run"
    plan = experiment.prepare(run)
    return run, plan, project


def test_prepare_freezes_same_public_packets_and_24_independent_paired_roots(prepared, tmp_path):
    run, plan, project = prepared
    assert len(plan["entries"]) == 24
    assert Counter(entry["condition"] for entry in plan["entries"]) == {
        "api": 12,
        "api_example": 12,
    }
    tasks = {task["id"]: task for task in load_tasks()}
    seeds = {}
    for entry in plan["entries"]:
        prompt = json.loads((run / entry["prompt_path"]).read_text())
        resources = json.loads((run / entry["resources_path"]).read_text())
        original = tool_free_prompt(tasks[entry["task_id"]], entry["condition"])
        assert prompt["sources"] == original["sources"]
        assert resources["packets"] == {item["id"]: item["content"] for item in original["sources"]}
        assert resources["condition"] == entry["condition"]
        assert (
            "task specification" in prompt["system"] and "reference source" not in prompt["system"]
        )
        assert "you may use rlm_query" in prompt["system"]
        assert ("example" in [item["kind"] for item in resources["resource_index"]]) == (
            entry["condition"] == "api_example"
        )
        assert len(json.loads((run / entry["task_path"]).read_text())["cases"]) == 4
        seeds.setdefault(entry["task_id"], set()).add(entry["sampling_seed"])
    assert all(len(value) == 1 for value in seeds.values())
    assert experiment.load_plan(run)[0] == plan
    capture = json.loads((run / "source_capture.json").read_text())
    assert all(
        str(path.relative_to(project)) in capture
        for name in experiment.SOURCE_ROOTS
        for path in (project / name).rglob("*")
        if path.is_file()
    )
    lightning = experiment.ReplExperimentConfig(
        model_id="nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
        model_revision="a9904d24bcc1d289a1950fa9d2b978c47cf903b9",
    )
    second = experiment.prepare(tmp_path / "lightning", lightning)
    assert second["entries"] == plan["entries"]
    (run / plan["entries"][0]["resources_path"]).write_text("{}")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        experiment.load_plan(run)


def test_config_and_live_inventory_fail_closed(prepared):
    run, _, project = prepared
    for changes in (
        {"max_tokens": 8192},
        {"request_timeout": 600},
        {"max_depth": 2},
        {"seed": 1},
        {"model_revision": "a" * 40},
        {"temperature": 0},
    ):
        with pytest.raises(ValueError):
            experiment.ReplExperimentConfig(**changes)
    (project / "training/src/rlm_train/new.py").write_text("# newly exposed runtime source")
    with pytest.raises(ValueError, match="inventory changed"):
        experiment.load_plan(run)


def make_spec(prepared, tmp_path):
    run, _, project = prepared
    spec_path = tmp_path / "worker-spec.json"
    experiment.write_json(
        spec_path,
        {
            "bwrap": "/usr/bin/bwrap",
            "runtime_python": sys.executable,
            "readonly_bindings": [
                {
                    "source": str(project / "training/src/rlm_train"),
                    "target": "/runtime/python/rlm_train",
                },
                {
                    "source": str(project / "training/environments/cute_rlm/cute_rlm"),
                    "target": "/runtime/python/cute_rlm",
                },
            ],
            "gpu_devices": [{"source": "/dev/nvidia0", "target": "/dev/nvidia0"}],
        },
    )
    return run, spec_path


def test_worker_bindings_and_reasoning_spec_exclude_oracle_gpu_and_extra_docs(
    prepared, tmp_path, monkeypatch
):
    from cute_rlm.sandbox import SandboxSpec

    run, spec_path = make_spec(prepared, tmp_path)
    experiment.check_worker_sources(run, spec_path)
    monkeypatch.setattr(SandboxSpec, "validate_host", lambda self: None)
    wrapper = run / "repl-python"
    binding = experiment.make_repl_wrapper(run, spec_path, wrapper, sys.executable)
    repl_spec = json.loads(Path(binding["repl_spec"]).read_text())
    assert not repl_spec["gpu_devices"]
    assert [item["target"] for item in repl_spec["readonly_bindings"]] == [
        "/runtime/python/rlm_train"
    ]
    assert wrapper.stat().st_mode & 0o700 == 0o700
    experiment.check_launch_binding(binding)
    candidate = json.loads(Path(binding["candidate_spec"]).read_text())
    assert [item["target"] for item in candidate["readonly_bindings"]] == [
        "/runtime/python/cute_rlm"
    ]
    package = Path(binding["candidate_package"])
    assert set(binding["candidate_files"]) == set(experiment.CANDIDATE_FILES)
    assert not (package / "icl/tasks.py").exists() and not (package / "icl/assets").exists()
    (package / "icl/tasks.py").write_text("# disallowed oracle fixture")
    with pytest.raises(ValueError, match="frozen allowlist"):
        experiment.check_launch_binding(binding)
    (package / "icl/tasks.py").unlink()
    raw = json.loads(spec_path.read_text())
    raw["readonly_bindings"].append({"source": str(tmp_path), "target": "/runtime/python/docs"})
    experiment.write_json(spec_path, raw)
    with pytest.raises(ValueError, match="namespace|records"):
        experiment.check_worker_sources(run, spec_path)
    with pytest.raises(ValueError, match="binding changed"):
        experiment.check_launch_binding(binding)


def test_observed_namespace_rejects_extra_mount_sys_path_and_editable_pth(prepared, tmp_path):
    import copy

    _, spec_path = make_spec(prepared, tmp_path)
    spec = json.loads(spec_path.read_text())
    spec["readonly_bindings"].append({"source": "/usr", "target": "/usr"})
    experiment.write_json(spec_path, spec)
    # Synthetic layout only; this is not a claim about this host's namespace support.
    mounts = {
        "/": "tmpfs",
        "/proc": "proc",
        "/dev": "tmpfs",
        "/tmp": "tmpfs",
        "/dev/shm": "tmpfs",
        "/usr": "ext4",
    }
    layout = {
        "sys_path": ["/tmp", "/runtime/python", "/usr/lib/python3.12"],
        "pth_files": {"/usr/lib/python3.12/site-packages/_virtualenv.pth": "import _virtualenv\n"},
        "mountinfo": "\n".join(
            f"{index} 1 0:1 / {target} rw - {filesystem} none rw"
            for index, (target, filesystem) in enumerate(mounts.items(), 1)
        ),
    }
    experiment.validate_namespace_layout(layout, spec_path)
    for change in ("mount", "sys_path", "pth", "root"):
        invalid = copy.deepcopy(layout)
        if change == "mount":
            invalid["mountinfo"] += "\n99 1 0:1 / /home rw - ext4 none rw"
        elif change == "sys_path":
            invalid["sys_path"].append("/home/operator/project")
        elif change == "pth":
            invalid["pth_files"]["/usr/lib/python3.12/site-packages/editable.pth"] = (
                "import __editable___hidden_project"
            )
        else:
            invalid["mountinfo"] = invalid["mountinfo"].replace(
                "/ / rw - tmpfs", "/ / rw - ext4", 1
            )
        with pytest.raises(ValueError, match="Namespace|namespace|Unexpected"):
            experiment.validate_namespace_layout(invalid, spec_path)


@pytest.fixture
def fake_launch(prepared, tmp_path, monkeypatch):
    import cute_rlm.icl.repl_feedback
    import rlm_train.native
    import rlm_train.proxy

    run, plan, _ = prepared
    _, spec = make_spec(prepared, tmp_path)
    wrapper = run / "repl-python"
    binding = {}
    payloads = []
    instances = []
    feedback_instances = []

    original_make_wrapper = experiment.make_repl_wrapper

    def make_wrapper(run, spec_path, repl_python, host_python):
        from cute_rlm.sandbox import SandboxSpec

        monkeypatch.setattr(SandboxSpec, "validate_host", lambda self: None)
        binding.update(original_make_wrapper(run, spec_path, repl_python, host_python))
        return binding

    class Proxy:
        def __init__(self, **kwargs):
            self.url = "unix://fixture"

        async def start(self):
            pass

        async def stop(self):
            pass

    class Feedback:
        def __init__(self, task, directory, *, worker_command):
            self.fatal_error = None
            self.calls = []
            feedback_instances.append(self)

        async def __call__(self, owner, source):
            self.calls.append((owner, source))
            return {"status": "compile_error", "correct": None}

    class Runtime:
        def __init__(self, client, model, proxy, **kwargs):
            self.client, self.model, self.kwargs = client, model, kwargs
            self.root = SimpleNamespace(id=f"root-{len(instances)}")
            self.records = {}
            instances.append(self)

        async def run(self, prompt, context):
            index = len(instances) - 1
            assert context["condition"] in experiment.CONDITIONS
            payload = {
                "model": self.model,
                "max_tokens": self.kwargs["limits"].max_tokens,
                **self.kwargs["sampling_args"],
            }
            await self.client.post("chat/completions", json=payload)
            record = {
                "session_id": self.root.id,
                "status": "completed",
                "events": [],
                "final_answer": "class ModelNew: pass",
                "error_class": None,
                "error": None,
            }
            if index == 0:
                try:
                    await self.client.post("/tokenize", json={"model": "unapproved/child-model"})
                except NativeProtocolError as error:
                    self.records["wrong-model-child"] = {
                        "session_id": "wrong-model-child",
                        "parent_session_id": self.root.id,
                        "depth": 1,
                        "status": "failed",
                        "error_class": "protocol",
                        "error": str(error),
                        "failure": {
                            "stage": "prompt_count",
                            "exception_chain": [
                                {"type": type(error).__name__, "message": str(error)},
                            ],
                        },
                    }
                else:
                    pytest.fail("A child may not dispatch an unpinned model")
                record.update(
                    status="failed",
                    final_answer=None,
                    error_class="infrastructure",
                    error="model transport deadline",
                    failure={
                        "stage": "model_request",
                        "exception_chain": [{"type": "ReadTimeout", "message": "deadline"}],
                    },
                )
            elif index == 1:
                record["final_answer"] = {"not": "source string"}
            await self.kwargs["compile_candidate"](self.root.id, "class ModelNew: pass")
            self.records[self.root.id] = record
            self.kwargs["event_path"].write_text("CPU scripted journal\n")
            self.kwargs["checkpoint_path"].write_text("{}")
            return record

        async def close(self):
            pass

    async def preflight(*args):
        return {"fixture": True}

    def transport(request):
        if request.url.path.endswith("/models"):
            return httpx.Response(
                200,
                json={
                    "data": [
                        {"id": experiment.ReplExperimentConfig().model_id, "max_model_len": 262144}
                    ]
                },
            )
        payloads.append(json.loads(request.content))
        return httpx.Response(200, json={})

    monkeypatch.setattr(experiment, "make_repl_wrapper", make_wrapper)
    monkeypatch.setattr(experiment, "preflight_repl", preflight)
    monkeypatch.setattr(experiment, "preflight_candidate_namespace", preflight)
    monkeypatch.setattr(rlm_train.native, "NativeRuntime", Runtime)
    monkeypatch.setattr(rlm_train.proxy, "SubLLMProxy", Proxy)
    monkeypatch.setattr(cute_rlm.icl.repl_feedback, "ReplCompileFeedback", Feedback)
    receipt = tmp_path / "CPU_FIXTURE_READINESS.json"
    receipt.write_text("{}")
    readiness_module = ModuleType("cute_rlm.icl.repl_readiness")

    def check_readiness(run, plan, config, receipt_path, base_url):
        assert receipt_path == receipt
        return {
            "fixture": True,
            "server_identity": {"fixture": True},
            "receipt_sha256": experiment.sha256(receipt.read_bytes()),
            "sandbox_spec_sha256": experiment.sha256(spec.read_bytes()),
        }

    def check_model_stopped(run, config, started, stop_receipt_path):
        stop = json.loads(stop_receipt_path.read_text())
        if stop.get("generation_nonce") != started["nonce"]:
            raise ValueError("Stop receipt must bind this run")
        return {**stop, "receipt_sha256": experiment.sha256(stop_receipt_path.read_bytes())}

    async def check_server_identity(client, config, readiness):
        assert readiness["fixture"] is True

    monkeypatch.setattr("cute_rlm.server_identity.check_server_identity", check_server_identity)
    readiness_module.check_readiness = check_readiness
    readiness_module.check_model_stopped = check_model_stopped
    monkeypatch.setitem(sys.modules, readiness_module.__name__, readiness_module)
    return SimpleNamespace(
        run=run,
        plan=plan,
        spec=spec,
        wrapper=wrapper,
        binding=binding,
        receipt=receipt,
        payloads=payloads,
        instances=instances,
        feedback=feedback_instances,
        transport=httpx.MockTransport(transport),
    )


def test_generate_runs_all24_after_model_failure_and_never_validates_or_resumes(fake_launch):
    fixture = fake_launch
    result = asyncio.run(
        experiment.generate(
            fixture.run,
            "https://fixture.invalid/v1",
            sandbox_spec=fixture.spec,
            repl_python=fixture.wrapper,
            receipt=fixture.receipt,
            transport=fixture.transport,
        )
    )
    assert result["attempted"] == result["planned"] == len(fixture.payloads) == 24
    assert [item["status"] for item in result["results"][:3]] == [
        "session_failed",
        "invalid_source",
        "completed",
    ]
    for runtime in fixture.instances:
        kwargs = runtime.kwargs
        assert kwargs["child_repl_at_limit"] and kwargs["stream_responses"]
        assert kwargs["limits"].max_depth == 1
        assert kwargs["limits"].request_timeout == 800 and kwargs["limits"].episode_timeout == 1800
        assert kwargs["limits"].max_tokens == kwargs["limits"].child_max_tokens == 16384
        assert kwargs["budget_policy"].per_call_max_tokens == 16384
        assert kwargs["budget_policy"].max_generated_tokens == 1_000_000
        assert kwargs["sampling_args"]["temperature"] == 0.6
        assert kwargs["sampling_args"]["top_p"] == 0.95
    wrong_child = fixture.instances[0].records["wrong-model-child"]
    assert experiment.infrastructure_failure(wrong_child) is None
    assert all(
        payload["model"] == experiment.ReplExperimentConfig().model_id
        for payload in fixture.payloads
    )
    assert not (fixture.run / "validation").exists()
    assert (fixture.run / "GENERATION_FINISHED.json").exists()
    with pytest.raises(FileExistsError, match="no retries or resume"):
        asyncio.run(
            experiment.generate(
                fixture.run,
                "https://fixture.invalid/v1",
                sandbox_spec=fixture.spec,
                repl_python=fixture.wrapper,
                receipt=fixture.receipt,
                transport=fixture.transport,
            )
        )
    assert len(fixture.payloads) == 24


def test_generate_aborts_on_helper_infrastructure_even_if_broker_swallows_error(
    fake_launch, monkeypatch
):
    import cute_rlm.icl.repl_feedback

    fixture = fake_launch
    base = cute_rlm.icl.repl_feedback.ReplCompileFeedback

    class FatalFeedback(base):
        async def __call__(self, owner, source):
            self.fatal_error = OSError("CPU disk failure fixture")
            return {"status": "compile_error"}

    monkeypatch.setattr(cute_rlm.icl.repl_feedback, "ReplCompileFeedback", FatalFeedback)
    with pytest.raises(RuntimeError, match="helper infrastructure"):
        asyncio.run(
            experiment.generate(
                fixture.run,
                "https://fixture.invalid/v1",
                sandbox_spec=fixture.spec,
                repl_python=fixture.wrapper,
                receipt=fixture.receipt,
                transport=fixture.transport,
            )
        )
    assert len(fixture.payloads) == 1
    summary = json.loads((fixture.run / "generation_summary.json").read_text())
    assert summary["status"] == "aborted_infrastructure" and summary["attempted"] == 1
    assert not (fixture.run / "GENERATION_FINISHED.json").exists()


def test_validation_requires_stop_evidence_and_keeps_source_compile_numeric_separate(
    fake_launch, tmp_path, monkeypatch
):
    import cute_rlm.icl.validation
    from cute_rlm.sandbox import SandboxSpec

    fixture = fake_launch
    asyncio.run(
        experiment.generate(
            fixture.run,
            "https://fixture.invalid/v1",
            sandbox_spec=fixture.spec,
            repl_python=fixture.wrapper,
            receipt=fixture.receipt,
            transport=fixture.transport,
        )
    )
    started = json.loads((fixture.run / "GENERATION_STARTED.json").read_text())
    config = experiment.ReplExperimentConfig()
    receipt = tmp_path / "stopped.json"
    experiment.write_json(receipt, {"model_stopped": True, "generation_nonce": "wrong"})
    with pytest.raises(ValueError, match="bind this run"):
        asyncio.run(experiment.validate(fixture.run, fixture.spec, model_stopped_receipt=receipt))
    # Deliberately marked CPU fixture, never used as evidence for a real run.
    evidence = tmp_path / "CPU_FIXTURE_NOT_GPU.json"
    experiment.write_json(
        evidence, {"fixture": True, "owned_pids": [123], "port_closed": True, "gpu_idle": True}
    )
    experiment.write_json(
        receipt,
        {
            "model_stopped": True,
            "generation_nonce": started["nonce"],
            "run": str(fixture.run),
            "run_id": started["run_id"],
            "model_id": config.model_id,
            "model_revision": config.model_revision,
            "evidence": {"path": evidence.name, "sha256": experiment.sha256(evidence.read_bytes())},
        },
    )
    calls = []

    async def validator(task, source, directory, *, worker_command):
        assert len(task["cases"]) == 4
        assert source == "class ModelNew: pass\n"
        calls.append(task["id"])
        return {
            "status": "numerical_mismatch",
            "compile_forward": True,
            "cute_execution_observed": False,
            "numerical_correct": False,
            "valid_solution": None,
            "source_review_required": True,
        }

    monkeypatch.setattr(cute_rlm.icl.validation, "validate_candidate", validator)
    monkeypatch.setattr(SandboxSpec, "validate_host", lambda self: None)
    result = asyncio.run(
        experiment.validate(fixture.run, fixture.spec, model_stopped_receipt=receipt)
    )
    assert len(result["results"]) == 24 and len(calls) == 22
    checked = next(item for item in result["results"] if item["source_valid"])
    assert checked["compile_forward"] is True
    assert checked["cute_execution_observed"] is False
    assert checked["numerical_correct"] is False and checked["valid_solution"] is None
    with pytest.raises(FileExistsError, match="no retries or resume"):
        asyncio.run(experiment.validate(fixture.run, fixture.spec, model_stopped_receipt=receipt))


class SSE(httpx.AsyncByteStream):
    def __init__(self, body):
        self.body = body

    async def __aiter__(self):
        yield self.body


@pytest.mark.asyncio
async def test_scripted_native_child_has_own_repl_same_condition_resources_and_compile_helper(
    tmp_path,
):
    task = load_tasks()[0]
    prompt, resources = experiment.render_prompt(task, "api")
    requests, compile_calls = [], []

    def dispatch(request):
        payload = json.loads(request.content)
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"count": 20, "max_model_len": 262144})
        requests.append(payload)
        user = next(
            message["content"] for message in payload["messages"] if message["role"] == "user"
        )
        shared = (
            "assert resources['condition'] == 'api'; "
            "assert all(item['kind'] == 'api' for item in resources['resource_index']); "
            "assert compile_candidate('class ModelNew: pass')['status'] == 'compiled'; "
        )
        if user == "child":
            code = shared + (
                "assert 'root_only' not in str(SHOW_VARS()); "
                "answer['content'] = 'class ModelNew: pass'; answer['ready'] = True"
            )
        else:
            code = shared + (
                "root_only = 1; handle = rlm_query('child', blocking=False); "
                "best_source = rlm_wait(handle, timeout_s=10); "
                "answer['content'] = best_source; answer['ready'] = True"
            )
        delta = {
            "role": "assistant",
            "reasoning_content": "CPU scripted reasoning",
            "tool_calls": [
                {
                    "index": 0,
                    "id": f"tool-{len(requests)}",
                    "type": "function",
                    "function": {"name": "repl", "arguments": json.dumps({"code": code})},
                }
            ],
        }
        event = {
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "finish_reason": "tool_calls",
                    "logprobs": {"content": [{"token": "token_id:1", "logprob": -0.1}]},
                }
            ],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
        }
        body = b"data: " + json.dumps(event).encode() + b"\n\ndata: [DONE]\n\n"
        return httpx.Response(200, stream=SSE(body))

    async def compile_helper(owner, source):
        compile_calls.append((owner, source))
        return {"status": "compiled", "correct": None, "cute_verified": None}

    # The actual subprocess REPL is used here, not the GPU/namespace launcher.
    package_path = str(Path(__file__).resolve().parents[1] / "training/src")

    def factory(**kwargs):
        return SubprocessReplBackend(
            python=sys.executable,
            worker_env={
                "PATH": "/usr/bin:/bin",
                "PYTHONPATH": package_path,
                "RLM_TRAIN_READONLY_CONTEXT": "1",
            },
            **kwargs,
        )

    async with httpx.AsyncClient(
        base_url="https://fixture.invalid/v1/", transport=httpx.MockTransport(dispatch)
    ) as client:
        proxy = SubLLMProxy(socket_path=tmp_path / "broker.sock")
        await proxy.start()
        try:
            from rlm_train.native import NativeLimits

            runtime = NativeRuntime(
                client,
                experiment.ReplExperimentConfig().model_id,
                proxy,
                limits=NativeLimits(max_depth=1, max_tokens=16384, child_max_tokens=16384),
                budget_policy=ContextBudgetPolicy(per_call_max_tokens=16384),
                prompt_counter=VllmPromptCounter(client, 262144),
                backend_factory=factory,
                bootstrap_code=experiment.resources_bootstrap(resources),
                system_prompt=prompt["system"],
                compile_candidate=compile_helper,
                child_repl_at_limit=True,
                sampling_args={"temperature": 0.6, "top_p": 0.95, "seed": 20261101},
                stream_responses=True,
                event_path=tmp_path / "events.jsonl",
                checkpoint_path=tmp_path / "checkpoint.json",
            )
            record = await runtime.run(prompt["user"], {"task_spec": resources["task_spec"]})
        finally:
            await proxy.stop()
    assert record["status"] == "completed", record
    assert record["final_answer"] == "class ModelNew: pass"
    assert len(record["child_sessions"]) == 1 and record["child_sessions"][0]["depth"] == 1
    assert len(compile_calls) == 2 and len({owner for owner, _ in compile_calls}) == 2
    assert len(requests) == 2
    assert all(payload["max_tokens"] == 16384 for payload in requests)
    assert all(
        payload["temperature"] == 0.6 and payload["top_p"] == 0.95 and payload["seed"] == 20261101
        for payload in requests
    )
    assert (tmp_path / "checkpoint.json").exists()
    journal = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    chunks = [event for event in journal if event["type"] == "stream_chunk"]
    assert len(chunks) == 2
    assert all(b"[DONE]" in base64.b64decode(event["bytes_base64"]) for event in chunks)


@pytest.mark.asyncio
async def test_helper_infrastructure_cancels_actual_native_root_and_blocks_more_http(
    prepared, tmp_path, monkeypatch
):
    import cute_rlm.icl.repl_feedback
    from cute_rlm.sandbox import SandboxSpec

    run, plan, _ = prepared
    _, spec = make_spec(prepared, tmp_path)
    monkeypatch.setattr(SandboxSpec, "validate_host", lambda self: None)
    binding = experiment.make_repl_wrapper(run, spec, run / "repl-python", sys.executable)
    requests = []
    failures = []
    config = experiment.ReplExperimentConfig()

    class FatalFeedback:
        def __init__(self, task, directory, *, worker_command):
            assert worker_command[-1] == binding["candidate_spec"]
            self.fatal_error = None

        async def __call__(self, owner, source):
            self.fatal_error = OSError("CPU fixture host persistence failure")
            raise self.fatal_error

    monkeypatch.setattr(cute_rlm.icl.repl_feedback, "ReplCompileFeedback", FatalFeedback)

    def dispatch(request):
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"count": 20, "max_model_len": 262144})
        requests.append(request)
        code = (
            "try:\n    compile_candidate('class ModelNew: pass')\n"
            "except RuntimeError:\n    print('The model caught the broker error')\n"
            "answer['content'] = 'class ModelNew: pass'; answer['ready'] = True"
        )
        event = {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "fatal-helper",
                                "type": "function",
                                "function": {
                                    "name": "repl",
                                    "arguments": json.dumps({"code": code}),
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                    "logprobs": {"content": [{"token": "token_id:1", "logprob": -0.1}]},
                }
            ],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
        }
        return httpx.Response(
            200, stream=SSE(b"data: " + json.dumps(event).encode() + b"\n\ndata: [DONE]\n\n")
        )

    package_path = str(Path(__file__).resolve().parents[1] / "training/src")

    def factory(**kwargs):
        return SubprocessReplBackend(
            python=sys.executable,
            worker_env={
                "PATH": "/usr/bin:/bin",
                "PYTHONPATH": package_path,
                "RLM_TRAIN_READONLY_CONTEXT": "1",
            },
            **kwargs,
        )

    async with httpx.AsyncClient(
        base_url="https://fixture.invalid/v1/",
        transport=httpx.MockTransport(dispatch),
        event_hooks={"request": [experiment.request_guard(run, binding, config, failures)]},
    ) as client:
        with pytest.raises(RuntimeError, match="helper infrastructure"):
            await experiment._generate_entries(
                run,
                {**plan, "entries": plan["entries"][:1]},
                config,
                client=client,
                worker_command=[
                    sys.executable,
                    "-m",
                    "cute_rlm.sandbox",
                    "--spec",
                    binding["candidate_spec"],
                ],
                backend_factory=factory,
                binding=binding,
                host_failures=failures,
            )
        with pytest.raises(experiment.ExperimentInfrastructureError):
            await client.post(
                "chat/completions",
                json={
                    "model": config.model_id,
                    "max_tokens": 16384,
                    "temperature": 0.6,
                    "top_p": 0.95,
                },
            )
    assert len(requests) == 1 and failures
    summary = json.loads((run / "generation_summary.json").read_text())
    assert summary["status"] == "aborted_infrastructure" and summary["attempted"] == 1
    trajectory = json.loads(
        (run / "completions" / plan["entries"][0]["id"] / "trajectory.json").read_text()
    )
    assert trajectory["status"] == "failed" and not trajectory.get("cleanup_error")
    assert not (run / "GENERATION_FINISHED.json").exists()
