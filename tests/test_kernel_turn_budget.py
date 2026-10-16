import asyncio
import copy
import importlib.util
import json
import tomllib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

ROOT = Path(__file__).parents[1] / "training/environments/kernel_rlm"


@pytest.fixture
def actions(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    from kernel_rlm import actions

    return actions


def test_submission_warning_arrives_before_the_deadline(actions):
    from kernel_rlm.subcalls import turn_budget

    assert "Only two turns" in turn_budget(3, 8, 0, 0, True)
    assert "NEXT TURN" in turn_budget(4, 8, 0, 0, True)
    due = turn_budget(5, 8, 0, 0, True)
    assert "SUBMISSION DEADLINE" in due
    assert "Feedback calls remaining: 0" in due
    assert "Child calls remaining: 0" in due
    assert "RECOVERY ONLY" in turn_budget(6, 8, 2, 0, True)
    assert "FINAL TURN" in turn_budget(7, 8, 2, 0, True)
    assert "FINAL TURN" in turn_budget(0, 1, 0, 0, False)


def test_recovery_requires_context_room_and_a_known_usage(actions):
    pending = [{"role": "user", "content": "Submit now."}]
    budget, estimate = actions.recovery_budget({"total_tokens": 9000}, pending, 12288, 2048, len)
    assert budget == 512
    assert estimate > 9000
    assert actions.recovery_budget({"total_tokens": 12200}, pending, 12288, 2048, len)[0] == 0
    assert actions.recovery_budget({}, pending, 12288, 2048, len) == (0, None)
    assert actions.recovery_budget({"total_tokens": 9000}, pending, 12288, 128, len)[0] == 128


@pytest.mark.parametrize(
    "code",
    [
        "answer['content'] = source\nanswer['ready'] = True",
        "answer = {'content': best_source, 'ready': True}",
    ],
)
def test_recovery_can_submit_existing_state(actions, code):
    assert actions.preparation_error(code, {"load_inline"}, recovery=True) is None


@pytest.mark.parametrize(
    "code",
    [
        "cuda_feedback(source)",
        "load_inline(name='candidate')",
        "llm_query('help')",
        "print(source)",
        "import torch",
        "source = 'replacement'",
    ],
)
def test_recovery_cannot_start_more_work(actions, code):
    assert actions.preparation_error(code, {"load_inline"}, recovery=True)


def test_build_guard_tracks_aliases_but_allows_candidate_strings(actions):
    names = {"load_inline"}
    assert (
        actions.preparation_error(
            "from torch.utils.cpp_extension import load_inline as build", names
        )
        is None
    )
    assert actions.preparation_error("build(name='candidate')", names)
    assert actions.preparation_error("torch.utils.cpp_extension.load_inline(name='x')", names)
    assert actions.preparation_error("source = \"load_inline(name='candidate')\"", names) is None


@pytest.fixture
def run_program(monkeypatch, tmp_path, actions):
    monkeypatch.syspath_prepend(str(ROOT.parents[1] / "src"))
    from rlm_train.repl.base import ExecResult

    spec = importlib.util.spec_from_file_location(
        "kernel_turn_program_test", ROOT / "kernel_rlm_v1/program.py"
    )
    program = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(program)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("KERNEL_INFERENCE_ADMIN_URL", raising=False)

    def run(codes, *, used_tokens=1020):
        monkeypatch.setenv("KERNEL_ENDPOINT", "http://mock/v1")
        monkeypatch.setenv("KERNEL_API_KEY", "mock")
        monkeypatch.setenv("KERNEL_MODEL", "mock-model")
        monkeypatch.setenv(
            "KERNEL_LIMITS",
            json.dumps({"max_turns": 8, "max_tokens": 2048, "max_context_tokens": 12288}),
        )
        requests = []
        remaining_codes = iter(codes)
        namespace = {"answer": {}}

        def create(**kwargs):
            requests.append(copy.deepcopy(kwargs))
            code = next(remaining_codes)
            message = {
                "content": "",
                "tool_calls": [
                    {
                        "id": "call",
                        "type": "function",
                        "function": {"name": "repl", "arguments": json.dumps({"code": code})},
                    }
                ],
            }
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content="", model_dump=lambda **kwargs: copy.deepcopy(message)
                        ),
                        finish_reason="tool_calls",
                    )
                ],
                usage=SimpleNamespace(model_dump=lambda: {"total_tokens": used_tokens}),
            )

        def execute(code):
            exec(code, namespace)
            answer = namespace["answer"]
            return ExecResult(final_answer=answer.get("content") if answer.get("ready") else None)

        backend = SimpleNamespace(
            start=AsyncMock(),
            load_context=AsyncMock(),
            bootstrap=AsyncMock(),
            execute=AsyncMock(side_effect=execute),
            stop=AsyncMock(),
        )
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(side_effect=create))),
            close=AsyncMock(),
        )
        monkeypatch.setattr(program, "AsyncOpenAI", lambda **kwargs: client)
        monkeypatch.setattr(program, "SubprocessReplBackend", lambda **kwargs: backend)
        monkeypatch.setattr(program, "load_token_counter", lambda model: len)
        asyncio.run(program.run({"prompt": "test", "context": "reference"}))
        return json.loads(Path("submission.json").read_text()), requests, backend

    return run


def test_missing_submission_can_recover_in_a_short_seventh_turn(run_program):
    record, requests, backend = run_program(
        ["source = 'candidate'"] * 6 + ["answer['content'] = source\nanswer['ready'] = True"]
    )
    assert record["submission"] == "candidate"
    assert record["turns"] == 7
    assert requests[-1]["max_tokens"] == 512
    assert record["model_calls"][-1]["phase"] == "recovery"
    assert "Only two turns" in str(requests[3]["messages"])
    assert "SUBMISSION DEADLINE" in str(requests[5]["messages"])
    assert backend.execute.await_count == 7


def test_no_recovery_request_when_context_is_full(run_program):
    record, requests, _backend = run_program(["source = 'candidate'"] * 6, used_tokens=12200)
    assert len(requests) == 6
    assert record["termination_reason"] == "recovery_no_headroom"
    assert record["recovery"][-1]["status"] == "skipped"


def test_recovery_feedback_is_rejected_before_execution(run_program):
    record, requests, backend = run_program(
        ["source = 'candidate'"] * 6
        + ["cuda_feedback(source)", "answer = {'content': source, 'ready': True}"]
    )
    assert len(requests) == 8
    assert record["submission"] == "candidate"
    assert backend.execute.await_count == 7
    assert record["action_rejections"][-1]["turn"] == 7


def test_direct_compilation_is_rejected_before_execution(run_program):
    record, _requests, backend = run_program(
        ["source = 'candidate'", "load_inline(name='candidate')"]
        + ["source = 'candidate'"] * 4
        + ["answer = {'content': source, 'ready': True}"]
    )
    assert record["submission"] == "candidate"
    assert backend.execute.await_count == 6
    assert "cuda_feedback" in record["action_rejections"][0]["error"]


def test_training_configs_match_the_recovery_context_limit():
    for filename in ("grpo_pilot.toml", "grpo_level1.toml"):
        config = tomllib.loads((ROOT / "configs" / filename).read_text())
        harness = config["orchestrator"]["train"]["source"][0]["env"]["agent"]["harness"]
        assert harness["max_turns"] == 8
        assert harness["max_context_tokens"] == config["seq_len"]
        assert harness["max_context_tokens"] == config["inference"]["vllm"]["max_model_len"]
