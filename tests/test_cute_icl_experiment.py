"""CPU-only protocol tests: fake task packets and native streaming transport."""

import asyncio
import json
import sys
import types
from dataclasses import asdict
from pathlib import Path

import httpx
import pytest
from cute_rlm.icl import experiment


class ByteStream(httpx.AsyncByteStream):
    def __init__(self, chunks, error=None):
        self.chunks = chunks
        self.error = error

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk
        if self.error is not None:
            raise self.error


def model_transport(handler):
    def dispatch(request):
        if request.url.path.endswith("/models"):
            return httpx.Response(
                200, json={"data": [{"id": experiment.MODEL_ID, "max_model_len": 262144}]}
            )
        return handler(request)

    return httpx.MockTransport(dispatch)


def event(value):
    return b"data: " + json.dumps(value).encode() + b"\n\n"


def response_chunks(content="class ModelNew:\n    pass", *, tool=False):
    message = {"role": "assistant", "content": content, "reasoning_content": "native reasoning"}
    if tool:
        message["tool_calls"] = [
            {
                "index": 0,
                "id": "call-1",
                "type": "function",
                "function": {"name": "never_run", "arguments": "{}"},
            }
        ]
    return [
        event(
            {
                "id": "native-id",
                "choices": [
                    {
                        "index": 0,
                        "delta": message,
                        "finish_reason": None,
                        "logprobs": {
                            "content": [{"token": "class", "logprob": -0.2, "bytes": [99]}]
                        },
                    }
                ],
            }
        ),
        event({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
        event(
            {
                "choices": [],
                "usage": {"prompt_tokens": 41, "completion_tokens": 9, "total_tokens": 50},
            }
        ),
        b"data: [DONE]\n\n",
        b": retained trailing native bytes\n\n",
    ]


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    tasks = types.ModuleType("cute_rlm.icl.tasks")
    tasks.load_tasks = lambda: [
        {
            "id": f"task_{i}",
            "family": "fixture",
            "cases": [{"shape": [2, 3]}],
            "description": "Public semantics",
        }
        for i in range(12)
    ]
    tasks.reference_source = lambda task: "# trusted oracle; never in prompt\n"
    prompts = types.ModuleType("cute_rlm.icl.prompts")
    prompts.render_prompt = lambda task, condition: {
        "system": "Return only Python code.",
        "user": task["description"] + "\n" + condition,
        "sources": []
        if condition == "task_only"
        else [
            {"id": "fixture", "content": "public API", "sha256": experiment.sha256(b"public API")}
        ],
    }
    monkeypatch.setitem(sys.modules, tasks.__name__, tasks)
    monkeypatch.setitem(sys.modules, prompts.__name__, prompts)
    run = tmp_path / "run"
    plan = experiment.prepare(run, experiment.ExperimentConfig())
    return run, plan


def test_prepare_freezes_36_independent_paired_requests(prepared):
    run, plan = prepared
    assert len(plan["entries"]) == 36
    assert {entry["condition"] for entry in plan["entries"]} == set(experiment.CONDITIONS)
    seeds = {}
    for entry in plan["entries"]:
        request = json.loads((run / entry["request_path"]).read_text())
        assert [message["role"] for message in request["messages"]] == ["system", "user"]
        assert "tools" not in request and "tool_choice" not in request
        assert request["stream"] and request["logprobs"]
        assert request["stream_options"] == {"include_usage": True}
        assert request["max_tokens"] == 8192
        assert request["temperature"] == 0.6 and request["top_p"] == 0.95
        assert request["n"] == 1
        assert "trusted oracle" not in json.dumps(request)
        seeds.setdefault(entry["task_id"], set()).add(request["seed"])
    assert all(len(pair_seeds) == 1 for pair_seeds in seeds.values())
    assert plan["entries"] != sorted(plan["entries"], key=lambda entry: entry["id"])
    loaded, config = experiment.load_plan(run)
    assert loaded == plan
    assert config.request_timeout == 600 and config.server_context_tokens == 262144
    assert config.requested_precision == "bf16" and config.seed == 20261101
    assert any(
        name.endswith("icl/experiment.py")
        for name in json.loads((run / "source_capture.json").read_text())
    )
    assert any(
        name.endswith("streaming.py")
        for name in json.loads((run / "source_capture.json").read_text())
    )


def test_prepare_explicit_model_preserves_pilot_requests(prepared, tmp_path):
    default_run, default_plan = prepared
    config = experiment.ExperimentConfig(
        model_id="nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
        model_revision="a9904d24bcc1d289a1950fa9d2b978c47cf903b9",
    )
    run = tmp_path / "lightning"
    plan = experiment.prepare(run, config)
    loaded, frozen_config = experiment.load_plan(run)
    assert loaded == plan and frozen_config == config
    assert plan["entries"] == default_plan["entries"]
    assert len(plan["entries"]) == 36
    for entry in plan["entries"]:
        payload = json.loads((run / entry["request_path"]).read_text())
        default_payload = json.loads((default_run / entry["request_path"]).read_text())
        assert payload["model"] == config.model_id
        assert default_payload["model"] == experiment.MODEL_ID
        default_payload["model"] = config.model_id
        assert payload == default_payload
    assert experiment.ExperimentConfig().model_revision == experiment.MODEL_REVISION


@pytest.mark.parametrize(
    "overrides",
    [
        {"model_id": ""},
        {"model_id": " "},
        {"model_revision": ""},
        {"model_revision": "main"},
        {"model_revision": "a" * 39},
        {"model_revision": "g" * 40},
    ],
)
def test_config_rejects_empty_model_or_unpinned_revision(overrides):
    with pytest.raises(ValueError):
        experiment.ExperimentConfig(**overrides)


def test_prepare_deterministic_order_and_integrity(prepared, tmp_path):
    run, plan = prepared
    second = experiment.prepare(tmp_path / "second", experiment.ExperimentConfig())
    assert second["entries"] == plan["entries"]
    entry = plan["entries"][0]
    (run / entry["request_path"]).write_text("{}")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        experiment.load_plan(run)


@pytest.mark.parametrize(
    "text", ["class ModelNew:\n    pass", "```python\nclass ModelNew:\n    pass\n```"]
)
def test_extract_source_accepts_only_code_without_executing(text, tmp_path):
    marker = tmp_path / "must-not-exist"
    source = experiment.extract_source(text + "\n")
    assert source.startswith("class ModelNew")
    experiment.extract_source(f"open({str(marker)!r}, 'w').write('no')")
    assert not marker.exists()


@pytest.mark.parametrize(
    "text",
    [
        "",
        "```python\nx = 1",
        "```python\nx=1\n```\n```python\ny=2\n```",
        "Here is code:\n```python\nx=1\n```",
        "not valid python!",
        "```\nx=1\n```",
    ],
)
def test_extract_source_rejects_ambiguous_or_invalid_code(text):
    with pytest.raises((ValueError, SyntaxError)):
        experiment.extract_source(text)


def test_generate_preserves_native_stream_metadata_and_refuses_resume(prepared):
    run, plan = prepared
    requests = []
    chunks = response_chunks()

    def handler(request):
        requests.append(json.loads(request.content))
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(200, stream=ByteStream(chunks))

    result = asyncio.run(
        experiment.generate(run, "https://fixture.invalid/v1", transport=model_transport(handler))
    )
    assert result["attempted"] == 36 and len(requests) == 36
    assert all(item["status"] == "completed" for item in result["results"])
    directory = run / "completions" / plan["entries"][0]["id"]
    assert (directory / "stream.bin").read_bytes() == b"".join(chunks)
    response = json.loads((directory / "response.json").read_text())
    assert response["usage"]["total_tokens"] == 50
    assert response["id"] == "native-id"
    assert response["choices"][0]["message"]["reasoning_content"] == "native reasoning"
    assert response["choices"][0]["logprobs"]["content"][0]["logprob"] == -0.2
    assert len((directory / "events.jsonl").read_text().splitlines()) == len(chunks)
    assert result["results"][0]["usage_status"] == "reported"
    with pytest.raises(FileExistsError):
        asyncio.run(
            experiment.generate(
                run, "https://fixture.invalid/v1", transport=model_transport(handler)
            )
        )
    assert len(requests) == 36


def test_generate_explicit_subset_in_order_in_fresh_run(prepared, tmp_path):
    original_run, plan = prepared
    original_bytes = {
        str(path.relative_to(original_run)): path.read_bytes()
        for path in original_run.rglob("*")
        if path.is_file()
    }
    run = tmp_path / "targeted"
    config = experiment.ExperimentConfig(max_tokens=16384, request_timeout=800)
    experiment.prepare(run, config)
    entry_ids = [entry["id"] for entry in reversed(plan["entries"][1:5])]
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, stream=ByteStream(response_chunks()))

    result = asyncio.run(
        experiment.generate(
            run,
            "https://fixture.invalid/v1",
            entry_ids=entry_ids,
            transport=model_transport(handler),
        )
    )
    assert result["planned"] == result["attempted"] == len(requests) == 4
    assert result["entry_ids"] == entry_ids
    assert [item["id"] for item in result["results"]] == entry_ids
    assert all(item["status"] == "completed" for item in result["results"])
    assert json.loads((run / "GENERATION_STARTED.json").read_text())["entry_ids"] == entry_ids
    assert json.loads((run / "generation_summary.json").read_text()) == result
    assert len(experiment.load_plan(run)[0]["entries"]) == 36
    assert {path.name for path in (run / "completions").iterdir()} == set(entry_ids)
    by_id = {entry["id"]: entry for entry in plan["entries"]}
    assert requests == [
        json.loads((run / by_id[entry_id]["request_path"]).read_text()) for entry_id in entry_ids
    ]
    assert all(request["max_tokens"] == 16384 for request in requests)
    assert all(item["completion_timeout_seconds"] == 800 for item in result["results"])
    assert original_bytes == {
        str(path.relative_to(original_run)): path.read_bytes()
        for path in original_run.rglob("*")
        if path.is_file()
    }
    with pytest.raises(FileExistsError):
        asyncio.run(
            experiment.generate(
                run,
                "https://fixture.invalid/v1",
                entry_ids=[plan["entries"][0]["id"]],
                transport=model_transport(handler),
            )
        )
    assert len(requests) == 4


@pytest.mark.parametrize("selection", ["empty", "duplicate", "unknown", "mixed"])
def test_generate_rejects_bad_selection_before_network_or_marker(prepared, selection):
    run, plan = prepared
    known = plan["entries"][0]["id"]
    entry_ids = {
        "empty": [],
        "duplicate": [known, known],
        "unknown": ["unknown-entry"],
        "mixed": [known, "unknown-entry"],
    }[selection]
    calls = []

    def handler(request):
        calls.append(request)
        raise AssertionError("Invalid selections must not contact the server")

    with pytest.raises(ValueError, match="nonempty and unique|Unknown entry IDs"):
        asyncio.run(
            experiment.generate(
                run,
                "https://fixture.invalid/v1",
                entry_ids=entry_ids,
                transport=httpx.MockTransport(handler),
            )
        )
    assert calls == []
    assert not (run / "GENERATION_STARTED.json").exists()
    assert not (run / "models.json").exists()
    assert not (run / "completions").exists()


def test_total_timeout_preserves_partial_and_moves_to_next_selected_entry(prepared, monkeypatch):
    run, plan = prepared
    selected = [plan["entries"][i]["id"] for i in (3, 1)]
    chunk = response_chunks()[0]
    calls = []
    timeout_calls = []
    original_timeout = asyncio.timeout

    def timeout(seconds):
        timeout_calls.append(seconds)
        return original_timeout(0 if len(timeout_calls) == 1 else seconds)

    monkeypatch.setattr(asyncio, "timeout", timeout)

    class HangingStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield chunk
            await asyncio.Event().wait()

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(200, stream=HangingStream())
        return httpx.Response(200, stream=ByteStream(response_chunks()))

    result = asyncio.run(
        experiment.generate(
            run,
            "https://fixture.invalid/v1",
            entry_ids=selected,
            transport=model_transport(handler),
        )
    )
    assert len(calls) == result["attempted"] == result["planned"] == 2
    assert timeout_calls == [600, 600]
    assert [item["id"] for item in result["results"]] == selected
    assert [item["status"] for item in result["results"]] == ["transport_error", "completed"]
    assert not result["stopped_on_transport_error"]
    first = result["results"][0]
    assert first["error"]["type"] == "TimeoutError"
    assert first["error"]["timeout_scope"] == "completion_total"
    assert first["completion_timeout_seconds"] == 600
    assert first["started_at"] < first["completion_deadline_at"]
    directory = run / "completions" / selected[0]
    assert (directory / "stream.bin").read_bytes() == chunk
    assert "ModelNew" in (directory / "partial_response.json").read_text()
    assert not (directory / "candidate.py").exists()
    assert json.loads((run / "generation_summary.json").read_text()) == result


def test_generate_cli_passes_repeated_entry_ids(monkeypatch, capsys):
    calls = []

    async def fake_generate(run, base_url, **kwargs):
        calls.append((run, base_url, kwargs))
        return {"attempted": 2, "stopped_on_transport_error": False}

    monkeypatch.setattr(experiment, "generate", fake_generate)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "experiment",
            "generate",
            "--run",
            "fresh-run",
            "--base-url",
            "https://fixture.invalid/v1",
            "--entry-id",
            "03-api",
            "--entry-id",
            "01-api_example",
        ],
    )
    experiment.main()
    assert calls == [
        (
            "fresh-run",
            "https://fixture.invalid/v1",
            {"api_key_env": None, "entry_ids": ["03-api", "01-api_example"]},
        )
    ]
    assert json.loads(capsys.readouterr().out)["attempted"] == 2


@pytest.mark.parametrize("error_type", [httpx.ReadError, httpx.ReadTimeout])
def test_transport_failure_preserves_partial_and_stops_without_retry(prepared, error_type):
    run, plan = prepared
    chunk = event(
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "partial code", "reasoning": "partial thought"},
                    "finish_reason": None,
                    "logprobs": {"content": [{"token": "partial", "logprob": -1.0}]},
                }
            ]
        }
    )
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, stream=ByteStream([chunk], error_type("stream disconnected")))

    result = asyncio.run(
        experiment.generate(run, "https://fixture.invalid/v1", transport=model_transport(handler))
    )
    assert len(calls) == result["attempted"] == 1
    assert result["stopped_on_transport_error"]
    item = result["results"][0]
    assert item["status"] == "transport_error"
    assert item["usage"] is None and item["usage_status"] == "unknown"
    assert error_type.__name__ in item["error"]["traceback"]
    if error_type is httpx.ReadTimeout:
        assert item["error"]["timeout_scope"] == "httpx_transport"
    directory = run / "completions" / plan["entries"][0]["id"]
    assert (directory / "stream.bin").read_bytes() == chunk
    partial = json.loads((directory / "partial_response.json").read_text())
    assert partial["choices"][0]["message"]["content"] == "partial code"
    assert partial["choices"][0]["logprobs"]["content"][0]["logprob"] == -1.0
    assert not (directory / "candidate.py").exists()


def test_response_error_continues_once_per_entry_and_tools_stay_inert(prepared):
    run, _ = prepared
    count = 0

    def handler(request):
        nonlocal count
        count += 1
        if count == 1:
            return httpx.Response(400, stream=ByteStream([b'{"error":"model rejected request"}']))
        if count == 2:
            return httpx.Response(200, stream=ByteStream(response_chunks(tool=True)))
        return httpx.Response(200, stream=ByteStream(response_chunks()))

    result = asyncio.run(
        experiment.generate(run, "https://fixture.invalid/v1", transport=model_transport(handler))
    )
    assert count == 36
    assert [item["status"] for item in result["results"][:3]] == [
        "response_error",
        "response_error",
        "completed",
    ]
    assert result["results"][0]["usage"] is None
    first_dir = run / "completions" / result["results"][0]["id"]
    assert (first_dir / "stream.bin").read_bytes() == b'{"error":"model rejected request"}'
    second_dir = run / "completions" / result["results"][1]["id"]
    assert "never_run" in (second_dir / "response.json").read_text()
    assert not (second_dir / "candidate.py").exists()


def test_malformed_stream_retains_later_chunks_and_partial_tool_fragments(tmp_path):
    directory = tmp_path / "stream"
    directory.mkdir()
    chunks = [
        event(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [{"index": 0, "function": {"arguments": "partial"}}]
                        },
                        "finish_reason": None,
                    }
                ]
            }
        ),
        b"data: broken JSON\n\n",
        b"all trailing native bytes",
    ]

    async def collect():
        async with httpx.AsyncClient(
            base_url="https://fixture.invalid/",
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, stream=ByteStream(chunks))
            ),
        ) as client:
            await experiment.collect_completion(client, {}, directory)

    with pytest.raises(json.JSONDecodeError):
        asyncio.run(collect())
    assert (directory / "stream.bin").read_bytes() == b"".join(chunks)
    assert "partial" in (directory / "partial_response.json").read_text()


def test_validate_dispatches_only_completed_sources_to_explicit_sandbox(
    prepared, tmp_path, monkeypatch
):
    run, plan = prepared
    entry = plan["entries"][0]
    directory = run / "completions" / entry["id"]
    source = "class ModelNew:\n    pass\n"
    experiment.write_json(
        directory / "result.json",
        {"status": "completed", "source_sha256": experiment.sha256(source.encode())},
    )
    (directory / "candidate.py").write_text(source)
    module = types.ModuleType("cute_rlm.icl.validation")
    calls = []

    async def fake_validate(task, candidate, output, *, worker_command):
        calls.append((task, candidate, output, worker_command))
        return {"numerical_correct": True, "cute_execution_observed": False, "valid_solution": None}

    module.validate_candidate = fake_validate
    monkeypatch.setitem(sys.modules, module.__name__, module)
    spec = tmp_path / "sandbox.json"
    spec.write_text(
        json.dumps(
            {
                "bwrap": "/usr/bin/bwrap",
                "runtime_python": "/venv/bin/python",
                "readonly_bindings": [
                    {
                        "source": str(Path(experiment.__file__).resolve().parent.parent),
                        "target": "/runtime/python/cute_rlm",
                    }
                ],
            }
        )
    )
    summary = asyncio.run(experiment.validate(run, spec, "/explicit/python"))
    assert len(calls) == 1
    assert calls[0][3] == ["/explicit/python", "-m", "cute_rlm.sandbox", "--spec", str(spec)]
    assert calls[0][1] == source
    assert summary["valid_solution"] is None and summary["source_review_required"]


def test_config_is_separate_explicit_new_pilot_defaults():
    config = asdict(experiment.ExperimentConfig())
    assert config["max_tokens"] == 8192 and config["request_timeout"] == 600
    with pytest.raises(ValueError):
        experiment.ExperimentConfig(server_context_tokens=8192)
    with pytest.raises(ValueError):
        experiment.ExperimentConfig(enable_thinking=False)
    assert experiment.ExperimentConfig(temperature=0).temperature == 0


@pytest.mark.parametrize(
    "metadata",
    [
        {"data": [{"id": "wrong-model", "max_model_len": 262144}]},
        {"data": [{"id": experiment.MODEL_ID, "max_model_len": 8192}]},
        None,
    ],
)
def test_models_identity_failure_stops_before_dispatch(prepared, metadata):
    run, _ = prepared
    calls = []

    def handler(request):
        calls.append(request.url.path)
        if metadata is None:
            raise httpx.ConnectError("metadata unavailable")
        return httpx.Response(200, json=metadata)

    with pytest.raises(ValueError, match="identity|differs"):
        asyncio.run(
            experiment.generate(
                run, "https://fixture.invalid/v1", transport=httpx.MockTransport(handler)
            )
        )
    assert calls == ["/v1/models"]
    assert (run / "models.json").exists()
    assert not (run / "GENERATION_STARTED.json").exists()
    assert not (run / "completions").exists()


def test_cancellation_preserves_partial_result_and_propagates(prepared):
    run, plan = prepared
    chunk = event(
        {"choices": [{"index": 0, "delta": {"content": "partial"}, "finish_reason": None}]}
    )
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(
            200, stream=ByteStream([chunk], asyncio.CancelledError("cancelled by operator"))
        )

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            experiment.generate(
                run, "https://fixture.invalid/v1", transport=model_transport(handler)
            )
        )
    assert len(calls) == 1
    directory = run / "completions" / plan["entries"][0]["id"]
    assert (directory / "stream.bin").read_bytes() == chunk
    assert "partial" in (directory / "partial_response.json").read_text()
    result = json.loads((directory / "result.json").read_text())
    assert result["status"] == "cancelled" and result["usage"] is None
    assert json.loads((run / "generation_summary.json").read_text())["attempted"] == 1


def test_changed_current_source_fails_closed(prepared, monkeypatch):
    run, _ = prepared
    original_read = Path.read_bytes
    source = experiment.__file__

    def changed_read(path):
        if str(path) == source:
            return b"changed validation semantics"
        return original_read(path)

    monkeypatch.setattr(Path, "read_bytes", changed_read)
    with pytest.raises(ValueError, match="Current source differs"):
        experiment.load_plan(run)


def test_validation_rejects_different_sandbox_worker_sources(prepared, tmp_path):
    run, _ = prepared
    package = tmp_path / "other-cute-rlm"
    package.mkdir()
    spec = tmp_path / "sandbox.json"
    spec.write_text(
        json.dumps(
            {
                "bwrap": "/usr/bin/bwrap",
                "runtime_python": "/venv/bin/python",
                "readonly_bindings": [
                    {
                        "source": str(package),
                        "target": "/runtime/python/cute_rlm",
                    }
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="Sandbox worker source differs"):
        asyncio.run(experiment.validate(run, spec))
    assert not (run / "validation").exists()
