"""CPU protocol tests: real native roots/children, scripted SSE and worker, never GPU evidence."""

import asyncio
import hashlib
import json
import shutil
import sys
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
from cute_rlm import diagnose, feedback
from cute_rlm.config import DATASET_ID, DATASET_REVISION, EvalConfig
from cute_rlm.evaluate import isolated_backend_factory


@pytest.fixture
def prepared(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    repo = Path(__file__).parents[1]
    for relative in ("training/src/rlm_train", "training/environments/cute_rlm/cute_rlm"):
        shutil.copytree(
            repo / relative, project / relative, ignore=shutil.ignore_patterns("__pycache__")
        )
    for relative in (
        "training/environments/kernel_rlm/kernel_factory/__init__.py",
        "pyproject.toml",
        "uv.lock",
        "training/environments/cute_rlm/pyproject.toml",
    ):
        path = project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo / relative, path)
    tasks = []
    for index in range(2):
        source = f"# frozen specification {index}"
        tasks.append(
            {
                "task_id": f"task-{index}",
                "reference_code": source,
                "source": {
                    "repo": DATASET_ID,
                    "revision": DATASET_REVISION,
                    "file": "pool",
                    "row": index,
                    "sha256": hashlib.sha256(source.encode()).hexdigest(),
                },
            }
        )
    (project / "tasks.jsonl").write_text("".join(json.dumps(task) + "\n" for task in tasks))
    (project / "cute.md").write_text("CuTe documentation CPU fixture")
    (project / "docs.json").write_text(
        json.dumps(
            {
                "repo": "NVIDIA/cutlass",
                "revision": "a" * 40,
                "files": [
                    {
                        "id": "cute",
                        "path": "cute.md",
                        "sha256": hashlib.sha256((project / "cute.md").read_bytes()).hexdigest(),
                    }
                ],
            }
        )
    )
    config = EvalConfig(
        task_manifest="tasks.jsonl",
        docs_manifest="docs.json",
        seed=7,
        task_count=2,
        verification={
            "atol": 0,
            "rtol": 0,
            "seed": 7,
            "mode": "preserve",
            "correctness_runs": 1,
            "warmup_runs": 0,
            "timed_runs": 1,
            "timeout_s": 10,
        },
    )
    with (
        patch("kernel_factory.git_state", return_value={"commit": "cpu-fixture"}),
        patch("kernel_factory.subprocess.check_output", return_value=b""),
    ):
        readiness = diagnose.prepare_diagnostic(
            project, config, "cpu-fixture", tmp_path / "runs", rollouts_per_example=2
        )
    directory = tmp_path / "runs/cpu-fixture"
    # This is deliberately a CPU fixture, not a real readiness receipt.
    receipt = {
        "mode": diagnose.MODE,
        "config_sha256": config.sha256(),
        "manifest_sha256": readiness["manifest_sha256"],
        "observed_context_tokens": config.server_context_tokens,
        "hardware": {
            "fixture": True,
            "gpu_model": config.gpu_model,
            "gpu_compute_capability": config.gpu_compute_capability,
        },
        "software": {"fixture": True, "cuda_toolkit_version": config.cuda_toolkit_version},
        "server_identity": {"tensor_parallel_size": config.tensor_parallel_size},
        "checks": {
            key: {"status": "passed", "evidence": "CPU_FIXTURE_NOT_GPU"}
            for key in diagnose.diagnostic_checks(config)
        },
    }
    worker = tmp_path / "scripted-worker.py"
    worker.write_text(
        "import json, sys\nfrom pathlib import Path\nr=json.load(sys.stdin)\no=Path(r['output_dir']); o.mkdir()\n"
        "if r['role']=='reference':\n"
        " assert r['prepare_only'] is True\n"
        " for n in ['initial_state.npz','init_args.npz'] + [f'inputs-{i}.npz' for i in range(r['config']['correctness_runs'])]: (o/n).write_bytes(b'CPU fixture only')\n"
        " result={'status':'prepared','source_sha256':r['source_sha256'],'training':True,'module_training':{'':True}}\n"
        "else:\n"
        " assert r['compile_feedback'] is True\n"
        " assert r['profile_compile'] is True\n"
        " result={'status':'compile_error' if r['source']=='bad candidate' else 'compiled','source_sha256':r['source_sha256'],'stdout':'CPU scripted compiler','stderr':'CPU scripted warning'}\n"
        " (o/'compile-provenance.json').write_text(json.dumps({'fixture':True,'artifacts':[]}))\n"
        "print(json.dumps(result))\n"
    )
    return directory, config, tasks, receipt, [sys.executable, str(worker)]


def scripted_transport(payloads, *, broken=False, entered=None):
    async def dispatch(request):
        payload = json.loads(request.content)
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"count": 20, "max_model_len": 262144})
        payloads.append(payload)
        if entered is not None:
            entered.set()
            await asyncio.Future()
        if broken:
            raise httpx.ConnectError("CPU fixture disconnected", request=request)
        user = next(
            message["content"] for message in payload["messages"] if message["role"] == "user"
        )
        if user == "scripted-child":
            code = "assert isinstance(context,str); assert compile_candidate('child candidate')['status']=='compiled'; answer['content']='child-ok'; answer['ready']=True"
        else:
            code = "assert context['cute_dsl_task'].startswith('# frozen specification'); assert 'reference_code' not in context; assert 'seen' not in SHOW_VARS(); seen=True; assert read_doc('cute')['excerpt']; assert resources['limits']['turns_per_session']==64; assert compile_candidate('bad candidate')['status']=='compile_error'; h=rlm_query('scripted-child',blocking=False); assert rlm_wait(h,timeout_s=10)=='child-ok'; answer['content']='class ModelNew: pass'; answer['ready']=True"
        chunk = {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "reasoning_content": "CPU scripted",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "repl",
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

        class SSEBytes(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield ("data: " + json.dumps(chunk) + "\n\ndata: [DONE]\n\n").encode()

        return httpx.Response(200, stream=SSEBytes(), headers={"content-type": "text/event-stream"})

    return httpx.MockTransport(dispatch)


@pytest.mark.asyncio
async def test_independent_roots_real_children_feedback_and_host_final_smoke(prepared, monkeypatch):
    directory, config, tasks, receipt, command = prepared

    def forbidden(*args, **kwargs):
        pytest.fail(
            "diagnostics must never score, compare, time, observe, or re-render frozen inputs"
        )

    monkeypatch.setattr("cute_rlm.verifier.verify_submission", forbidden)
    monkeypatch.setattr("cute_rlm.observer.NsightObserver", forbidden)
    monkeypatch.setattr(diagnose, "first_user_message", forbidden)
    monkeypatch.setattr(diagnose, "build_resources", forbidden)
    payloads = []
    async with httpx.AsyncClient(
        base_url="http://cpu.invalid/v1/", transport=scripted_transport(payloads)
    ) as client:
        report = await diagnose.diagnose_prepared(
            directory,
            client=client,
            worker_command=command,
            backend_factory=isolated_backend_factory(config, sys.executable),
            receipt=receipt,
        )
    assert report["status"] == "completed"
    assert report["planned_rollouts"] == report["recorded_rollouts"] == 4
    assert report["selected_tasks"] == 2 and report["rollouts_per_example"] == 2
    assert report["final_smoke_statuses"] == {"compiled": 4}
    assert (
        report["correct"] is None and report["cute_verified"] is None and report["speedup"] is None
    )
    assert report["usage"]["root"]["completion_tokens"] == {"sum": 40, "recorded": 4, "total": 4}
    assert report["usage"]["child"]["completion_tokens"]["sum"] == 40
    assert len(payloads) == 8
    assert all(
        payload["max_tokens"] > 8192 and payload["logprobs"] is True and payload["stream"] is True
        for payload in payloads
    )
    records = [
        json.loads(path.read_text())
        for path in sorted(directory.glob("task-*/rollout-*/task.json"))
    ]
    assert len({record["session_id"] for record in records}) == 4
    assert [record["example_id"] for record in records] == [
        task["task_id"] for task in tasks for _ in range(2)
    ]
    for record in records:
        assert (
            record["status"] == "completed" and record["child_sessions"][0]["status"] == "completed"
        )
        assert record["host_final_smoke"]["model_action"] is False
        assert record["host_final_smoke"]["result"]["session_id"].startswith("host-final-smoke:")
    assert (
        len(list(directory.glob("task-*/rollout-*/compile-feedback/reference-*/request.json"))) == 4
    )
    assert len(list(directory.glob("task-*/rollout-*/compile-feedback/0*/response.json"))) == 12
    assert len(list(directory.glob("task-*/rollout-*/final-source.py"))) == 4
    assert list(
        directory.glob("task-*/rollout-*/compile-feedback/0*/output/compile-provenance.json")
    )
    assert (
        sum(
            json.loads(line)["event"] == "started"
            for line in (directory / "events.jsonl").read_text().splitlines()
        )
        == 1
    )
    async with httpx.AsyncClient(
        base_url="http://cpu.invalid/v1/", transport=scripted_transport([])
    ) as client:
        with pytest.raises(ValueError, match="already started"):
            await diagnose.diagnose_prepared(
                directory,
                client=client,
                worker_command=command,
                backend_factory=isolated_backend_factory(config, sys.executable),
                receipt=receipt,
            )


@pytest.mark.asyncio
async def test_stream_failure_retains_unknown_usage_and_no_fake_final_smoke(prepared):
    directory, config, _, receipt, command = prepared
    async with httpx.AsyncClient(
        base_url="http://cpu.invalid/v1/", transport=scripted_transport([], broken=True)
    ) as client:
        report = await diagnose.diagnose_prepared(
            directory,
            client=client,
            worker_command=command,
            backend_factory=isolated_backend_factory(config, sys.executable),
            receipt=receipt,
        )
    assert report["status"] == "incomplete"
    assert report["final_smoke_statuses"] == {"not_submitted": 4}
    assert report["usage"]["root"]["completion_tokens"] == {"sum": None, "recorded": 0, "total": 4}
    assert not list(directory.glob("task-*/rollout-*/final-source.py"))
    assert (directory / "report.json").is_file()


@pytest.mark.asyncio
async def test_cancelled_run_writes_partial_record_and_closes_runtime(prepared):
    directory, config, _, receipt, command = prepared
    entered = asyncio.Event()
    async with httpx.AsyncClient(
        base_url="http://cpu.invalid/v1/", transport=scripted_transport([], entered=entered)
    ) as client:
        attempt = asyncio.create_task(
            diagnose.diagnose_prepared(
                directory,
                client=client,
                worker_command=command,
                backend_factory=isolated_backend_factory(config, sys.executable),
                receipt=receipt,
            )
        )
        await asyncio.wait_for(entered.wait(), 10)
        attempt.cancel()
        with pytest.raises(asyncio.CancelledError):
            await attempt
    report = json.loads((directory / "report.json").read_text())
    assert report["status"] == "failed" and report["recorded_rollouts"] == 1
    assert report["failure"]["type"] == "CancelledError"
    record = json.loads(next(directory.glob("task-*/rollout-*/task.json")).read_text())
    assert record["diagnostic_error"]["type"] == "CancelledError"
    assert record["status"] != "running" and record.get("cleanup_error") is None


@pytest.mark.asyncio
async def test_frozen_count_or_source_tamper_rejected_before_dispatch(prepared):
    directory, config, _, receipt, command = prepared
    path = directory / "inputs/.eval-inputs/diagnostic-plan.json"
    plan = json.loads(path.read_text())
    plan["rollouts_per_example"] = 3
    path.write_text(json.dumps(plan))
    payloads = []
    async with httpx.AsyncClient(
        base_url="http://cpu.invalid/v1/", transport=scripted_transport(payloads)
    ) as client:
        with pytest.raises(ValueError, match="mode/count differs"):
            await diagnose.diagnose_prepared(
                directory,
                client=client,
                worker_command=command,
                backend_factory=isolated_backend_factory(config, sys.executable),
                receipt=receipt,
            )
    assert not payloads
    assert not list(directory.glob("task-*"))


def test_receipt_requires_real_bound_diagnostic_checks_not_scored_gates(prepared):
    directory, config, _, receipt, _ = prepared
    manifest = json.loads((directory / "provenance.json").read_text())
    assert (
        "reference_candidate_correctness" not in receipt["checks"]
        and "timing_protocol" not in receipt["checks"]
    )
    diagnose.check_diagnostic_receipt(receipt, config, manifest)
    receipt["checks"]["cute_compile_forward_smoke"]["status"] = "pending"
    with pytest.raises(ValueError, match="cute_compile_forward_smoke"):
        diagnose.check_diagnostic_receipt(receipt, config, manifest)
    for invalid in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="positive integer"):
            diagnose.diagnostic_plan(invalid)


@pytest.mark.asyncio
@pytest.mark.parametrize("capture", [False, True])
async def test_feedback_compiler_capture_is_explicit_opt_in(
    prepared, monkeypatch, tmp_path, capture
):
    _, config, tasks, _, _ = prepared
    calls = []

    async def launch(command, request, settings):
        calls.append(request)
        output = Path(request["output_dir"])
        output.mkdir()
        if request["role"] == "reference":
            for name in ("initial_state.npz", "init_args.npz", "inputs-0.npz"):
                (output / name).write_bytes(b"CPU fixture only")
            return {
                "status": "prepared",
                "source_sha256": request["source_sha256"],
                "training": True,
                "module_training": {"": True},
            }
        return {"status": "compiled", "source_sha256": request["source_sha256"]}

    monkeypatch.setattr(feedback, "launch_phase", launch)
    kwargs = {"capture_compiler_artifacts": True} if capture else {}
    service = feedback.CompileFeedback(
        tasks[0],
        config.verification,
        tmp_path / "service",
        worker_command=["CPU fixture"],
        **kwargs,
    )
    assert (await service("root", "candidate"))["status"] == "compiled"
    assert "profile_compile" not in calls[0]
    assert calls[1].get("profile_compile", False) is capture
