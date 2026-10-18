import ast
import asyncio
import copy
import dataclasses
import importlib
import json
import os
import subprocess
import sys
from functools import wraps
from pathlib import Path
from types import SimpleNamespace

import pytest
from aiohttp import ClientSession

ROOT = Path(__file__).parents[1]
ENVIRONMENT = ROOT / "training/environments/kernel_rlm"


def run_async(function):
    @wraps(function)
    def run(*args, **kwargs):
        return asyncio.run(function(*args, **kwargs))

    return run


@pytest.fixture(autouse=True)
def packages(monkeypatch):
    monkeypatch.syspath_prepend(str(ENVIRONMENT))
    monkeypatch.syspath_prepend(str(ROOT / "training/src"))


def completion(content="```repl\nprint('hello')\n```", token_ids=None):
    tokens = token_ids or [101, 102]
    return {
        "id": "cpu-fixture",
        "object": "chat.completion",
        "created": 0,
        "model": "fixture-adapter",
        "prompt_token_ids": [11, 12],
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "token_ids": tokens,
                "message": {"role": "assistant", "content": content},
                "logprobs": {
                    "content": [
                        {"token": f"token_id:{token}", "logprob": -0.5, "top_logprobs": []}
                        for token in tokens
                    ]
                },
            }
        ],
        "usage": {
            "prompt_tokens": 2,
            "completion_tokens": len(tokens),
            "total_tokens": 2 + len(tokens),
        },
    }


@dataclasses.dataclass(frozen=True)
class FixtureID:
    instance_id: str
    repetition_id: int

    def to_string(self):
        return f"{self.instance_id}_{self.repetition_id}"


class FakeClient:
    def __init__(self):
        self.requests = []
        self.finished = []
        self.root_calls = 0

    async def chat_completion(self, payload):
        from kernel_rlm.subcalls import CHILD_SYSTEM

        body = copy.deepcopy(payload["json"])
        self.requests.append(body)
        if body["messages"][0]["content"] == CHILD_SYSTEM:
            return completion("Check the final partial block.", [701, 702])
        codes = [
            "advice = llm_query('Review indexing bounds. Return concise text.')\nprint(advice)",
            "source = 'CPU fixture candidate'\nprint(cuda_feedback(source))",
            "answer['content'] = source\nanswer['ready'] = True",
        ]
        code = codes[self.root_calls]
        self.root_calls += 1
        return completion(f"<think>private fixture reasoning</think>\n```repl\n{code}\n```")

    async def finish_session(self, session_id):
        self.finished.append(session_id)


@pytest.mark.parametrize("field", ["prompt_token_ids", "token_ids", "logprobs"])
def test_exact_tokens_fail_closed(field):
    from kernel_skyrl.contract import exact_sample

    response = completion()
    if field == "prompt_token_ids":
        response.pop(field)
    else:
        response["choices"][0].pop(field)
    with pytest.raises(ValueError, match="Missing exact"):
        exact_sample(response, 12288)


@pytest.mark.parametrize("corruption", ["alignment", "identity", "nan", "positive", "context"])
def test_exact_tokens_reject_corruption(corruption):
    from kernel_skyrl.contract import exact_sample

    response = completion()
    entries = response["choices"][0]["logprobs"]["content"]
    if corruption == "alignment":
        entries.pop()
    elif corruption == "identity":
        entries[0]["token"] = "decoded text is not an ID"
    elif corruption == "nan":
        entries[0]["logprob"] = float("nan")
    elif corruption == "positive":
        entries[0]["logprob"] = 0.1
    with pytest.raises(ValueError):
        exact_sample(response, 3 if corruption == "context" else 12288)


@pytest.mark.parametrize("speedup, expected", [(-1, 0.7), (0, 0.7), (1, 0.8), (10, 1.0)])
def test_terminal_speed_reward_matches_v1(speedup, expected):
    from kernel_skyrl.contract import terminal_reward

    source = (ENVIRONMENT / "kernel_rlm_v1/taskset.py").read_text()
    tree = ast.parse(source)
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "kernel_reward"
    )
    start = next(
        index
        for index, node in enumerate(method.body)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "reward" for target in node.targets)
    )
    namespace = {
        "result": {"compiled": True, "extension_used": True, "correct": True, "speedup": speedup},
        "self": SimpleNamespace(data=SimpleNamespace(speed_reward_weight=0.1)),
    }
    formula = ast.Module(body=method.body[start:-2], type_ignores=[])
    exec(compile(formula, "v1-reward", "exec"), namespace)
    assert terminal_reward(namespace["result"]) == pytest.approx(expected)
    assert terminal_reward(namespace["result"]) == pytest.approx(namespace["reward"])
    assert terminal_reward({"correct": False, "speedup": 100}) == 0.1


def test_filter_uses_exact_terminal_reward_including_speed():
    from kernel_skyrl.contract import has_signal

    assert not has_signal([0.1] * 4)
    assert not has_signal([0.8] * 4)
    assert has_signal([0.8, 0.8, 0.8, 0.800000001])
    with pytest.raises(ValueError):
        has_signal([float("nan"), 0.2])


@run_async
async def test_relay_preserves_small_recovery_cap_and_requires_auth(tmp_path):
    from kernel_skyrl.bridge import CompletionBridge

    client = FakeClient()
    bridge = CompletionBridge(client, "skyrl-lora", 12288, 4096, 7, tmp_path / "calls.jsonl")
    await bridge.start()
    try:
        async with ClientSession() as session:
            body = {
                "model": "base-model",
                "messages": [{"role": "system", "content": "root"}],
                "max_completion_tokens": 512,
            }
            response = await session.post(bridge.endpoint + "/chat/completions", json=body)
            assert response.status == 401
            response = await session.post(
                bridge.endpoint + "/chat/completions",
                json=body,
                headers={"Authorization": f"Bearer {bridge.api_key}"},
            )
            assert response.status == 200
        assert client.requests[0]["model"] == "skyrl-lora"
        assert client.requests[0]["max_tokens"] == 512
        assert client.requests[0]["return_token_ids"] is True
        assert client.requests[0]["cache_salt"] == "kernel-7"
        assert bridge.api_key not in (tmp_path / "calls.jsonl").read_text()
    finally:
        await bridge.stop()
    assert client.finished == [bridge.session_id]


@run_async
async def test_real_repl_program_with_child_and_cpu_feedback_fixture(tmp_path, monkeypatch):
    from kernel_skyrl.generator import KernelGenerator

    paths = [str(ENVIRONMENT), str(ROOT / "training/src"), str(ROOT)]
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(paths))
    monkeypatch.setenv("WANDB_API_KEY", "must-not-reach-program")
    script = (
        "import asyncio, json, os; from pathlib import Path; "
        "import kernel_rlm_v1.program as program; "
        "assert 'WANDB_API_KEY' not in os.environ; "
        "program.load_token_counter = lambda model: lambda text: len(text) // 8; "
        "program.build_local_feedback_bootstrap = lambda *args, **kwargs: "
        "\"def cuda_feedback(source):\\n    return {'compiled': True, 'fixture': 'CPU only'}\"; "
        "asyncio.run(program.run(json.loads(Path('task.json').read_text())))"
    )
    client = FakeClient()

    async def score(task, record):
        assert record["submission"] == "CPU fixture candidate"
        assert record["repl_calls"] == 3
        assert "CPU only" in json.dumps(record["messages"])
        return 0.8, {"fixture": "No CUDA compilation or GPU verification"}

    generator = KernelGenerator(
        client,
        "skyrl-lora",
        "no-tokenizer-download",
        tmp_path,
        program_command=[sys.executable, "-c", script],
        scorer=score,
        episode_timeout_s=30,
    )
    episode_dir = tmp_path / "episode"
    try:
        episode = await generator.run_episode(
            {"context": "Reference source fixture", "prompt": "Use the REPL", "name": "cpu"},
            episode_dir,
            2,
        )
    except Exception:
        log = episode_dir / "program.log"
        if log.exists():
            print(log.read_text())
        raise
    assert len(episode["samples"]) == 3
    assert all(sample.response_ids == [101, 102] for sample in episode["samples"])
    assert client.root_calls == 3
    assert len(client.requests) == 4
    assert "private fixture reasoning" not in json.dumps(client.requests[-1]["messages"])
    assert all(request["chat_template_kwargs"]["enable_thinking"] for request in client.requests)
    score_record = json.loads((episode_dir / "score.json").read_text())
    assert score_record["child_calls"] == 1
    assert len((episode_dir / "calls.jsonl").read_text().splitlines()) == 4
    assert episode["record"]["termination_reason"] == "submitted"
    assert len(client.finished) == 1


@run_async
async def test_stepwise_filter_and_oversampling_never_cut_a_trajectory(tmp_path):
    from kernel_skyrl.contract import RootSample
    from kernel_skyrl.generator import KernelGenerator

    task = {"name": "first", "context": "reference", "prompt": "solve"}
    replacement = {**task, "name": "second"}
    generator = KernelGenerator(FakeClient(), "adapter", "tokenizer", tmp_path, [replacement])

    async def episode(task, directory, version):
        repetition = int(directory.name.rsplit("-", 1)[1])
        sample = RootSample([11], [101], [-0.2], "stop")
        reward = 0.1 if task["name"] == "first" else 0.1 + repetition / 10
        return {"samples": [sample] * (repetition + 1), "reward": reward}

    generator.run_episode = episode
    inputs = {
        "prompts": [[{"role": "user", "content": "solve"}]] * 4,
        "trajectory_ids": [FixtureID("original", index) for index in range(4)],
        "env_extras": [task] * 4,
        "batch_metadata": SimpleNamespace(global_step=2, training_phase="train"),
    }
    output = await generator.generate(inputs)
    assert len(output["response_ids"]) == 10
    assert output["is_last_step"] == [
        True,
        False,
        True,
        False,
        False,
        True,
        False,
        False,
        False,
        True,
    ]
    assert output["rewards"] == pytest.approx([0.1, 0, 0.2, 0, 0, 0.3, 0, 0, 0, 0.4])
    assert all(item.instance_id == "original:replacement-1" for item in output["trajectory_ids"])
    assert output["rollout_metrics"]["kernel/zero_variance_groups"] == 1
    groups = json.loads(next(tmp_path.glob("*/groups.json")).read_text())
    assert [group["admitted"] for group in groups] == [False, True]
    assert all(mask == [1] for mask in output["loss_masks"])


@run_async
async def test_zero_signal_is_bounded_and_does_not_return_a_training_batch(tmp_path):
    from kernel_skyrl.contract import RootSample
    from kernel_skyrl.generator import KernelGenerator

    task = {"name": "zero", "context": "reference", "prompt": "solve"}
    generator = KernelGenerator(FakeClient(), "adapter", "tokenizer", tmp_path, [task])
    calls = []

    async def episode(task, directory, version):
        calls.append(directory)
        return {"samples": [RootSample([1], [2], [-0.2], "stop")], "reward": 0.1}

    generator.run_episode = episode
    with pytest.raises(RuntimeError, match="No learning signal"):
        await generator.generate(
            {
                "prompts": [[]] * 4,
                "env_extras": [task] * 4,
                "trajectory_ids": [FixtureID("zero", index) for index in range(4)],
                "batch_metadata": SimpleNamespace(global_step=1, training_phase="train"),
            }
        )
    assert len(calls) == 12
    assert len(json.loads(next(tmp_path.glob("*/groups.json")).read_text())) == 3


def test_preparation_pin_and_no_gpu_imports(tmp_path, monkeypatch):
    from kernel_skyrl import preflight

    source = tmp_path / "skyrl"
    for name in preflight.SOURCE_FILES:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture")
    monkeypatch.setattr(
        preflight,
        "git_value",
        lambda directory, *args: preflight.SKYRL_REVISION if args == ("rev-parse", "HEAD") else "",
    )
    manifest = tmp_path / "tasks.jsonl"
    manifest.write_text(json.dumps({"_task_id": "cpu", "code": "reference"}) + "\n")
    report = preflight.prepare(source, manifest)
    assert report["training_launch_enabled"] is False
    assert report["gpu_jobs_launched"] == 0
    assert report["dataset"]["examples"] == 1
    assert report["proposed_placement"]["gpu_count"] == 2
    assert report["optimizer"]["implemented"] is False
    assert report["rollout_contract"]["lora"]["rank"] == 8
    assert report["rollout_contract"]["native_dynamic_sampling"] is None
    assert report["code"]["source_hashes"]
    monkeypatch.setattr(
        preflight,
        "git_value",
        lambda directory, *args: preflight.SKYRL_REVISION
        if args == ("rev-parse", "HEAD")
        else " M changed.py",
    )
    with pytest.raises(ValueError, match="tracked source is modified"):
        preflight.prepare(source, manifest)
    monkeypatch.setattr(preflight, "git_value", lambda *args: "wrong pin")
    with pytest.raises(ValueError, match="does not match"):
        preflight.prepare(source, manifest)


def test_repl_program_no_longer_requires_verifiers_to_import():
    environment = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            [
                str(ENVIRONMENT),
                str(ROOT / "training/src"),
                str(ROOT),
            ]
        ),
    }
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import kernel_rlm_v1.program; "
            "import kernel_skyrl.preflight; assert 'verifiers' not in sys.modules; "
            "assert 'torch' not in sys.modules; assert 'ray' not in sys.modules",
        ],
        env=environment,
        text=True,
        capture_output=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    package = importlib.import_module("kernel_rlm_v1")
    with pytest.raises(AttributeError):
        assert package.MissingExport


@run_async
async def test_program_timeout_is_saved_and_session_is_closed(tmp_path):
    from kernel_skyrl.generator import KernelGenerator

    client = FakeClient()
    generator = KernelGenerator(
        client,
        "adapter",
        "tokenizer",
        tmp_path,
        episode_timeout_s=0.1,
        program_command=[sys.executable, "-c", "import time; time.sleep(60)"],
    )
    directory = tmp_path / "timed-out"
    with pytest.raises(TimeoutError):
        await generator.run_episode({"name": "timeout", "context": "reference"}, directory, 1)
    failure = json.loads((directory / "failure.json").read_text())
    assert failure["optimizer_update_allowed"] is False
    assert failure["error_type"] == "TimeoutError"
    assert not (directory / "score.json").exists()
    assert len(client.finished) == 1


@run_async
async def test_bad_token_transport_fails_closed_and_is_traced(tmp_path):
    from kernel_skyrl.bridge import CompletionBridge

    class BadClient(FakeClient):
        async def chat_completion(self, payload):
            response = completion()
            response.pop("prompt_token_ids")
            return response

    client = BadClient()
    bridge = CompletionBridge(client, "adapter", 12288, 4096, 1, tmp_path / "calls.jsonl")
    await bridge.start()
    try:
        async with ClientSession() as session:
            response = await session.post(
                bridge.endpoint + "/chat/completions",
                json={"messages": [{"role": "system", "content": "root"}], "max_tokens": 512},
                headers={"Authorization": f"Bearer {bridge.api_key}"},
            )
            assert response.status == 502
        assert bridge.failures
        assert bridge.samples == []
        assert "Missing exact" in (tmp_path / "calls.jsonl").read_text()
    finally:
        await bridge.stop()


def test_variable_turn_output_passes_pinned_skyrl_validator():
    from kernel_skyrl.contract import RootSample, stepwise_output

    source = Path("/tmp/kernel-skyrl-v040/skyrl/train/utils/trainer_utils.py")
    if not source.exists():
        pytest.skip("Optional pinned SkyRL source checkout is not present")
    tree = ast.parse(source.read_text())
    methods = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        in {
            "validate_generator_output",
            "_validate_step_wise_fields",
            "_validate_per_token_side_channels",
        }
    ]
    assert len(methods) == 3
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            *methods,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace = {
        "SAMPLE_SUPPORT_FIELD": "rollout_sample_support",
        "np": SimpleNamespace(
            concatenate=lambda rows: SimpleNamespace(sum=lambda: sum(map(sum, rows)))
        ),
    }
    exec(compile(module, str(source), "exec"), namespace)
    sample = RootSample([11, 12], [101, 102], [-0.2, -0.3], "stop")
    output = stepwise_output(
        [{"samples": [sample] * (index + 1), "reward": index / 10} for index in range(4)],
        [FixtureID("same-prompt", index) for index in range(4)],
    )
    namespace["validate_generator_output"](4, output, step_wise=True)
