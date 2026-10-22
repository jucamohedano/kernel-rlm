import asyncio
import contextlib
import fcntl
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parents[1] / "training/environments/kernel_rlm"


@pytest.fixture(autouse=True)
def imports(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))


def sample():
    return {
        "prompt_token_ids": [1],
        "choices": [
            {
                "message": {"role": "assistant", "content": "advice"},
                "token_ids": [2],
                "finish_reason": "stop",
                "logprobs": {"content": [{"token": "token_id:2", "logprob": -0.1}]},
            }
        ],
    }


@pytest.mark.parametrize("child", [False, True])
def test_disconnect_cancels_backend_and_releases_bridge_owned_admission(
    tmp_path, monkeypatch, child
):
    import aiohttp
    from kernel_rlm import verifier_slot
    from kernel_rlm.subcalls import CHILD_SYSTEM
    from kernel_rollout.bridge import CompletionBridge

    gate = tmp_path / "gate"
    monkeypatch.setattr(verifier_slot, "LOCK_PATH", gate)
    monkeypatch.setenv("KERNEL_INFERENCE_ADMIN_URL", "http://127.0.0.1:1")

    async def check():
        started, cancelled = asyncio.Event(), asyncio.Event()

        class Client:
            async def chat_completion_with_progress(self, payload, progress):
                progress({"stage": "generating", "completion_tokens": 257})
                return await self.chat_completion(payload)

            async def chat_completion(self, payload):
                started.set()
                with gate.open("a") as lock:
                    with pytest.raises(BlockingIOError):
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                try:
                    await asyncio.Event().wait()
                finally:
                    with gate.open("a") as lock:
                        with pytest.raises(BlockingIOError):
                            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    await asyncio.sleep(0.01)
                    cancelled.set()

            async def finish_session(self, session):
                pass

        bridge = CompletionBridge(Client(), "adapter", 14336, 4096, 7, tmp_path / "calls")
        await bridge.start()
        try:
            async with aiohttp.ClientSession() as session:
                pending = asyncio.create_task(
                    session.post(
                        bridge.endpoint + "/chat/completions",
                        headers={"Authorization": "Bearer " + bridge.api_key},
                        json={
                            "messages": [
                                {"role": "system", "content": CHILD_SYSTEM if child else "root"}
                            ]
                        },
                    )
                )
                await asyncio.wait_for(started.wait(), 1)
                pending.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await pending
                await asyncio.wait_for(cancelled.wait(), 1)
                await asyncio.sleep(0.01)
            with gate.open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            assert not bridge.samples
            assert bridge.child_calls == 0
            assert bool(bridge.failures) is not child
            record = json.loads((tmp_path / "calls").read_text())
            assert record["outcome"] == "cancelled"
            assert "response" not in record
            if child:
                progress = [
                    json.loads(line)
                    for line in (tmp_path / "child-progress.jsonl").read_text().splitlines()
                ]
                assert progress[-1]["stage"] == "cancelled"
                assert progress[-1]["completion_tokens"] == 257
            else:
                assert not (tmp_path / "child-progress.jsonl").exists()
        finally:
            await bridge.stop()

    asyncio.run(check())


def test_drain_during_cpu_compile_allows_active_reader_to_finish_but_blocks_new_readers(tmp_path):
    from kernel_rlm.verifier_slot import inference_drain, inference_request_slot, verifier_slot

    gate = tmp_path / "gate"

    async def check():
        async with inference_request_slot(1, path=gate):
            with inference_drain(1, path=gate):
                with pytest.raises(TimeoutError):
                    async with inference_request_slot(0.01, path=gate):
                        pytest.fail("New inference entered during verifier drain")
                with pytest.raises(TimeoutError):
                    with verifier_slot(0.01, path=gate, intent_held=True):
                        pytest.fail("Verifier entered before active inference finished")
        with inference_drain(1, path=gate):
            with verifier_slot(1, path=gate, intent_held=True):
                pass
        async with inference_request_slot(1, path=gate):
            pass

    asyncio.run(check())


def test_failed_cpu_contract_never_compiles_or_acquires_gpu(monkeypatch):
    from kernel_rlm import verifier

    monkeypatch.setattr(
        verifier,
        "run_contract_preflight",
        lambda *args, **kwargs: {
            "status": "contract_failed",
            "error": "Missing linear.weight",
            "elapsed_seconds": 0.01,
        },
    )
    monkeypatch.setattr(
        verifier, "verify_cuda_execution", lambda *args, **kwargs: pytest.fail("CUDA")
    )
    result = verifier.verify_cuda("reference", "candidate")
    assert result["verification_status"] == "contract_failed"
    assert result["correct"] is None
    assert "linear.weight" in result["error"]


def test_feedback_drains_before_compilation_and_releases_intent_after_build_failure(
    tmp_path, monkeypatch
):
    from kernel_rlm import verifier, verifier_slot

    gate = tmp_path / "gate"
    monkeypatch.setenv("KERNEL_INFERENCE_ADMIN_URL", "http://127.0.0.1:1")
    monkeypatch.setattr(
        verifier,
        "run_contract_preflight",
        lambda *args, **kwargs: {"status": "unsupported", "elapsed_seconds": 0.0},
    )
    monkeypatch.setattr(
        verifier,
        "inference_drain",
        lambda timeout: verifier_slot.inference_drain(timeout, path=gate),
    )
    phases = []

    def compile_only(command, **kwargs):
        phases.append(command[-1])
        assert command[-1] == "compile"
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
        with gate.with_name("gate.writer").open("a") as intent:
            with pytest.raises(BlockingIOError):
                fcntl.flock(intent, fcntl.LOCK_SH | fcntl.LOCK_NB)
        return SimpleNamespace(
            returncode=0, stdout='{"compiled":false,"error":"fixture build failure"}', stderr=""
        )

    monkeypatch.setattr(verifier, "_run_verifier_process", compile_only)
    with gate.open("a") as active_reader:
        fcntl.flock(active_reader, fcntl.LOCK_SH)
        result = verifier.verify_cuda("reference", "candidate", benchmark=False)
    assert result["verification_status"] == "compile_failed"
    assert result["scheduling"]["drain_started_before_compile"]
    assert phases == ["compile"]
    with gate.with_name("gate.writer").open("a") as intent:
        fcntl.flock(intent, fcntl.LOCK_SH | fcntl.LOCK_NB)


@pytest.mark.parametrize("status", ["unsupported", "interface_passed"])
def test_preflight_never_substitutes_for_independent_cuda_verification(monkeypatch, status):
    from kernel_rlm import verifier

    monkeypatch.delenv("KERNEL_INFERENCE_ADMIN_URL", raising=False)
    monkeypatch.setattr(
        verifier,
        "run_contract_preflight",
        lambda *args, **kwargs: {"status": status, "elapsed_seconds": 0.01},
    )
    calls = []

    def execute(*args, **kwargs):
        calls.append(kwargs)
        return {"correct": True, "verification_status": "completed"}

    monkeypatch.setattr(verifier, "verify_cuda_execution", execute)
    assert verifier.verify_cuda("reference", "candidate")["correct"]
    assert len(calls) == 1
    assert calls[0]["benchmark"]


def test_all_lock_waits_are_subtracted_from_execution_budget(tmp_path, monkeypatch):
    from kernel_rlm import verifier

    clock, execution_budgets = [0.0], []
    monkeypatch.setattr(verifier.time, "monotonic", lambda: clock[0])
    monkeypatch.setenv("KERNEL_INFERENCE_ADMIN_URL", "http://127.0.0.1:1")

    @contextlib.contextmanager
    def slot(timeout, **kwargs):
        clock[0] += 20
        yield timeout - 20

    @contextlib.contextmanager
    def compile_slot(timeout, **kwargs):
        if clock[0] > 0:
            clock[0] += 10
        yield timeout

    @contextlib.contextmanager
    def pause(timeout):
        clock[0] += 5
        yield timeout - 5

    def run(command, **kwargs):
        if command[-1] == "compile":
            clock[0] += 54
        else:
            execution_budgets.append(kwargs["timeout_s"])
        return SimpleNamespace(returncode=0, stdout='{"compiled":true,"correct":true}', stderr="")

    monkeypatch.setattr(verifier, "verifier_slot", slot)
    monkeypatch.setattr(verifier, "compile_slot", compile_slot)
    monkeypatch.setattr(verifier, "paused_inference", pause)
    monkeypatch.setattr(verifier, "_run_verifier_process", run)
    result = verifier.verify_cuda_execution(
        "reference",
        "candidate",
        n_trials=1,
        atol=0.001,
        rtol=0.0001,
        timeout_s=120,
        python=None,
        cwd=None,
        benchmark=False,
        collect_phase_events=False,
        allow_feedback_overlap=False,
        drain_intent_held=True,
    )
    assert result["correct"]
    assert execution_budgets == [31]
    assert result["timings"]["verifier_queue_wait_seconds"] == 20
    assert result["timings"]["compile_cache_queue_wait_seconds"] == 10


def test_feedback_cleanup_and_children_fit_enclosing_timeouts():
    from kernel_rlm.actions import (
        BACKEND_TIMEOUT_S,
        FEEDBACK_CLEANUP_TIMEOUT_S,
        FEEDBACK_MAX_CALLS,
        FEEDBACK_TIMEOUT_S,
        REPL_TIMEOUT_S,
    )
    from kernel_rlm.subcalls import SUBCALL_MAX_CALLS, SUBCALL_TIMEOUT_S

    feedback = FEEDBACK_MAX_CALLS * (FEEDBACK_TIMEOUT_S + FEEDBACK_CLEANUP_TIMEOUT_S)
    children = SUBCALL_MAX_CALLS * SUBCALL_TIMEOUT_S
    assert feedback < REPL_TIMEOUT_S < BACKEND_TIMEOUT_S
    assert feedback + children < REPL_TIMEOUT_S + children < BACKEND_TIMEOUT_S + children
    assert REPL_TIMEOUT_S + children == 630
    assert BACKEND_TIMEOUT_S + children == 660


def test_generator_marks_bridge_admission_ownership_in_worker_environment(tmp_path):
    from kernel_skyrl.generator import KernelGenerator

    class Client:
        async def finish_session(self, session):
            pass

    generator = KernelGenerator(
        Client(),
        "adapter",
        "tokenizer",
        tmp_path,
        program_command=[
            sys.executable,
            "-c",
            (
                "import json,os; from pathlib import Path; "
                "assert json.loads(os.environ['KERNEL_LIMITS'])['bridge_owns_admission'] is True; "
                "Path('submission.json').write_text(json.dumps({'submission':''}))"
            ),
        ],
        defer_scoring=True,
    )
    result = asyncio.run(generator.run_episode({"context": "reference"}, tmp_path / "episode", 7))
    assert result["samples"] == []
    assert result["reward"] is None


def test_contract_rejection_does_not_add_a_new_reward_penalty():
    from kernel_rollout.contract import terminal_reward

    assert terminal_reward({"compiled": False, "extension_used": None, "correct": None}) == 0.1
    assert terminal_reward({"compiled": True, "extension_used": False, "correct": False}) == 0.1


def test_complex_cpp_signatures_fall_back_instead_of_guessing():
    from kernel_rlm.contract_preflight import UnsupportedContract, extension_signature

    assert (
        len(
            extension_signature(
                "torch::Tensor launch(torch::Tensor input, int scale);", "launch"
            ).parameters
        )
        == 2
    )
    with pytest.raises(UnsupportedContract):
        extension_signature("torch::Tensor launch(torch::Tensor input, int scale=2);", "launch")
    assert (
        len(
            extension_signature(
                "torch::Tensor launch(torch::Tensor input, std::array<int, 4> shape, std::vector<int> padding);",
                "launch",
            ).parameters
        )
        == 3
    )
    assert not extension_signature("int launch(void);", "launch").parameters
    with pytest.raises(UnsupportedContract):
        extension_signature("int launch(std::vector<int input);", "launch")


@pytest.mark.parametrize(
    "kind,expected",
    [
        ("valid", "interface_passed"),
        ("state", "contract_failed"),
        ("wrapper", "contract_failed"),
        ("binding", "contract_failed"),
        ("unsupported", "unsupported"),
    ],
)
def test_real_meta_contract_without_cuda_or_compilation(monkeypatch, kind, expected):
    torch = pytest.importorskip("torch")
    from kernel_rlm.contract_preflight import check_contract

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    reference = """
import torch
import torch.nn as nn
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 2)
def get_inputs():
    return [torch.randn(1, 3)]
"""
    candidate = """
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
extension = load_inline(name='example', cpp_sources='torch::Tensor launch(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);', functions=['launch'])
class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 2)
    def forward(self, inputs):
        return extension.launch(inputs, self.linear.weight, self.linear.bias)
"""
    if kind == "state":
        candidate = candidate.replace("self.linear", "self.other")
    elif kind == "wrapper":
        candidate = candidate.replace("forward(self, inputs)", "forward(self, inputs, missing)")
    elif kind == "binding":
        candidate = candidate.replace(
            "extension.launch(inputs, self.linear.weight, self.linear.bias)",
            "extension.launch(inputs)",
        )
    elif kind == "unsupported":
        candidate = "import os\n" + candidate
    result = check_contract(reference, candidate)
    assert result["status"] == expected, result
    assert not torch.cuda.is_initialized()
