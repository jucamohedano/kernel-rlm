import asyncio
import contextlib
import hashlib
import io
import json
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).parents[1] / "training/environments/kernel_rlm"


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    from kernel_rlm import feedback, feedback_overlap, gpu_sharing, verifier, verifier_slot

    monkeypatch.setattr(
        verifier,
        "run_contract_preflight",
        lambda *args, **kwargs: {"status": "unsupported", "elapsed_seconds": 0.0},
    )

    return SimpleNamespace(
        feedback=feedback,
        policy=feedback_overlap,
        sharing=gpu_sharing,
        verifier=verifier,
        slot=verifier_slot,
    )


@pytest.fixture
def sources():
    return (
        (ROOT / "kernel_rlm/tasks/vector_add.py").read_text(),
        (ROOT / "examples/vector_add_submission.py").read_text(),
    )


def test_only_measured_references_and_supported_allocations_overlap(modules, sources):
    reference, candidate = sources
    limit, reason = modules.policy.overlap_budget(reference, candidate)
    assert limit == 512 * 1024**2
    assert reason == "qualified_reference"
    assert modules.policy.overlap_budget(reference.replace("4096", "8192"), candidate)[0] is None
    assert modules.policy.overlap_budget(reference, "broken(")[0] is None
    assert modules.policy.overlap_budget(reference, candidate + "\n# cudaMalloc\n")[0] is None
    curriculum = ROOT.parents[3] / "notes/curriculum/lvl1_seed42.jsonl"
    rows = [json.loads(line) for line in curriculum.read_text().splitlines()]
    softplus = next(row for row in rows if row["_task_id"] == "lvl1-seed42-0011")
    assert modules.policy.overlap_budget(softplus["code"], candidate)[0] == 4 * 1024**3


def test_submission_compilation_caches_are_source_isolated(modules, monkeypatch, tmp_path):
    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", str(tmp_path))
    first = modules.verifier.submission_extension_root("source one")
    repeated = modules.verifier.submission_extension_root("source one")
    second = modules.verifier.submission_extension_root("source two")
    assert first == repeated
    assert first != second


def test_verifier_oom_preserves_unknown_correctness(modules, sources, monkeypatch, tmp_path):
    monkeypatch.delenv("KERNEL_INFERENCE_ADMIN_URL", raising=False)
    monkeypatch.setattr(
        modules.verifier, "verifier_slot", lambda timeout: contextlib.nullcontext(timeout)
    )

    def run(command, **kwargs):
        result = {"compiled": True}
        if command[-1] == "verify":
            result.update(correct=False, max_diff=0.0, error="OutOfMemoryError: no memory")
        return SimpleNamespace(returncode=0, stdout=json.dumps(result), stderr="")

    monkeypatch.setattr(modules.verifier, "_run_verifier_process", run)
    result = modules.verifier.verify_cuda(*sources)
    assert result["verification_status"] == "infrastructure_error"
    assert result["failure_kind"] == "verifier_oom"
    assert result["correct"] is None
    assert result["max_diff"] is None


def test_runner_assigns_distinct_build_directories_for_same_extension_name(
    modules, monkeypatch, tmp_path
):
    from kernel_rlm.verifier import _RUNNER

    torch = ModuleType("torch")
    utils = ModuleType("torch.utils")
    extension = ModuleType("torch.utils.cpp_extension")
    utils.cpp_extension = extension
    torch.utils = utils
    build_directories = []

    def load_inline(*args, **kwargs):
        build_directories.append(Path(kwargs["build_directory"]))
        return SimpleNamespace()

    extension.load_inline = load_inline
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.utils", utils)
    monkeypatch.setitem(sys.modules, "torch.utils.cpp_extension", extension)
    monkeypatch.setenv("KERNEL_RLM_EXTENSION_BUILD_ROOT", str(tmp_path / "extensions"))
    reference = tmp_path / "reference.py"
    reference.write_text("reference")
    outputs = []
    for code in ("one", "two", "one"):
        submission = tmp_path / f"submission-{code}.py"
        submission.write_text(
            "import torch.utils.cpp_extension\n"
            "CPP_SOURCE='cpp'\n"
            f"CUDA_SOURCE={code!r}\n"
            "module=torch.utils.cpp_extension.load_inline("
            "name='kernel_rlm_candidate', cpp_sources=CPP_SOURCE, cuda_sources=CUDA_SOURCE)\n"
        )
        source_hash = hashlib.sha256(submission.read_bytes()).hexdigest()
        monkeypatch.setenv(
            "KERNEL_RLM_EXTENSION_BUILD_ROOT", str(tmp_path / "extensions" / source_hash)
        )
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "runner",
                str(reference),
                str(submission),
                "1",
                "0.001",
                "0.0001",
                "0",
                str(tmp_path / f"{code}.json"),
                "compile",
            ],
        )
        output = io.StringIO()
        with contextlib.redirect_stdout(output), pytest.raises(SystemExit):
            exec(compile(_RUNNER, "runner", "exec"), {})
        outputs.append(json.loads(output.getvalue().splitlines()[-1])["compiled"])
    assert outputs == [True, True, True]
    assert build_directories[0] == build_directories[2]
    assert build_directories[0] != build_directories[1]


def test_feedback_shares_inference_but_serializes_verification_and_reload(
    modules, tmp_path, monkeypatch
):
    path = tmp_path / "gpu.lock"
    monkeypatch.setattr(modules.sharing, "LOCK_PATH", path)
    monkeypatch.setenv("KERNEL_INFERENCE_ADMIN_URL", "http://127.0.0.1:8000")
    updates = []

    @modules.sharing.exclusive_weight_update
    async def update():
        updates.append(True)

    async def check():
        with modules.slot.feedback_slot(1, path=path):
            async with modules.slot.inference_request_slot(1, path=path):
                pass
            with pytest.raises(TimeoutError):
                with modules.slot.feedback_slot(0.01, path=path):
                    pytest.fail("Two feedback checks must not share memory")
            with pytest.raises(TimeoutError):
                with modules.slot.verifier_slot(0.01, path=path):
                    pytest.fail("Final scoring must stay exclusive")
            pending = asyncio.create_task(update())
            await asyncio.sleep(0.02)
            assert not updates
        await asyncio.wait_for(pending, 1)
        assert updates == [True]
        with pytest.raises(ValueError):
            with modules.slot.feedback_slot(1, path=path):
                raise ValueError("failed check")
        with modules.slot.verifier_slot(1, path=path):
            pass

    asyncio.run(check())


@pytest.mark.parametrize("defer", [False, "insufficient_free_memory", "allocator_limit_or_oom"])
def test_guarded_verifier_and_bounded_exclusive_fallback(
    modules, sources, monkeypatch, tmp_path, defer
):
    reference, candidate = sources
    monkeypatch.setenv("KERNEL_INFERENCE_ADMIN_URL", "http://127.0.0.1:8000")
    monkeypatch.setattr(
        modules.verifier,
        "feedback_slot",
        lambda timeout: modules.slot.feedback_slot(timeout, path=tmp_path / "gate"),
    )
    monkeypatch.setattr(
        modules.verifier,
        "verifier_slot",
        lambda timeout, **kwargs: modules.slot.verifier_slot(
            timeout, path=tmp_path / "gate", **kwargs
        ),
    )
    pauses = []

    @contextlib.contextmanager
    def pause(timeout):
        pauses.append(timeout)
        yield timeout

    monkeypatch.setattr(modules.verifier, "paused_inference", pause)
    calls = []

    def run(command, *, timeout_s, cwd, env):
        calls.append((command, timeout_s))
        Path(command[-2]).write_text('{"phase":"complete"}')
        result = {"compiled": True, "correct": True}
        if command[-1] == "verify" and len(command) == 11:
            guard = json.loads(command[-3])
            assert guard["allocator_limit_bytes"] == 512 * 1024**2
            if defer:
                result.update(correct=None, overlap_deferred=defer)
                time.sleep(0.02)
        return SimpleNamespace(returncode=0, stdout=json.dumps(result), stderr="")

    monkeypatch.setattr(modules.verifier, "_run_verifier_process", run)
    result = modules.verifier.verify_cuda(
        reference, candidate, timeout_s=1, benchmark=False, allow_feedback_overlap=True
    )
    assert result["correct"]
    assert result["scheduling"]["mode"] == ("exclusive" if defer else "overlap")
    assert len(pauses) == int(bool(defer))
    assert len(calls) == (3 if defer else 2)
    if defer:
        assert result["scheduling"]["fallback_reason"] == defer
        assert calls[-1][1] < calls[-2][1]
    else:
        assert result["timings"]["inference_paused_seconds"] == 0


@pytest.mark.parametrize("benchmark", [True, False])
def test_final_benchmark_and_unprofiled_tasks_stay_exclusive(
    modules, sources, monkeypatch, tmp_path, benchmark
):
    monkeypatch.setenv("KERNEL_INFERENCE_ADMIN_URL", "http://127.0.0.1:8000")
    monkeypatch.setattr(modules.verifier, "feedback_slot", lambda timeout: pytest.fail("overlap"))
    monkeypatch.setattr(modules.verifier, "paused_inference", contextlib.nullcontext)
    monkeypatch.setattr(
        modules.verifier,
        "verifier_slot",
        lambda timeout, **kwargs: modules.slot.verifier_slot(
            timeout, path=tmp_path / "gate", **kwargs
        ),
    )

    def run(command, **kwargs):
        Path(command[-2]).write_text('{"phase":"complete"}')
        return SimpleNamespace(returncode=0, stdout='{"compiled":true,"correct":true}', stderr="")

    monkeypatch.setattr(modules.verifier, "_run_verifier_process", run)
    reference, candidate = sources
    if not benchmark:
        reference = reference.replace("4096", "8192")
    result = modules.verifier.verify_cuda(
        reference, candidate, benchmark=benchmark, allow_feedback_overlap=True
    )
    assert result["scheduling"] == {
        "mode": "exclusive",
        "reason": "final_benchmark" if benchmark else "unprofiled_reference",
        "drain_started_before_compile": not benchmark,
    }


@pytest.mark.parametrize("low_memory", [True, False])
def test_runner_checks_free_memory_and_enforces_allocator_cap(
    modules, monkeypatch, tmp_path, low_memory
):
    torch = ModuleType("torch")
    utils = ModuleType("torch.utils")
    extension = ModuleType("torch.utils.cpp_extension")
    utils.cpp_extension = extension
    torch.utils = utils
    torch.OutOfMemoryError = type("OutOfMemoryError", (RuntimeError,), {})
    fractions = []
    torch.cuda = SimpleNamespace(
        is_available=lambda: True,
        mem_get_info=lambda: ((1 if low_memory else 10) * 1024**3, 80 * 1024**3),
        get_device_name=lambda: "NVIDIA A100",
        reset_peak_memory_stats=lambda: None,
        set_per_process_memory_fraction=lambda fraction, device: fractions.append(fraction),
    )
    extension.load_inline = lambda *args, **kwargs: None
    for name, module in (
        ("torch", torch),
        ("torch.utils", utils),
        ("torch.utils.cpp_extension", extension),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    reference = tmp_path / "reference.py"
    reference.write_text("import torch\nraise torch.OutOfMemoryError('test allocator')\n")
    guard = {"allocator_limit_bytes": 4 * 1024**3, "reserve_bytes": 2 * 1024**3}
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "runner",
            str(reference),
            "unused",
            "1",
            "0.001",
            "0.001",
            "0",
            json.dumps(guard),
            str(tmp_path / "phase.json"),
            "verify",
        ],
    )
    output = io.StringIO()
    with contextlib.redirect_stdout(output), pytest.raises(SystemExit):
        exec(compile(modules.verifier._RUNNER, "runner", "exec"), {})
    result = json.loads(output.getvalue().strip().splitlines()[-1])
    assert result["correct"] is None
    assert result["overlap_deferred"] == (
        "insufficient_free_memory" if low_memory else "allocator_limit_or_oom"
    )
    assert fractions == ([] if low_memory else [0.05])


def test_repl_helper_requests_guard_without_awarding_reward(modules, monkeypatch):
    calls = []

    def verify(*args, **kwargs):
        calls.append(kwargs)
        return {"compiled": True, "correct": True}

    monkeypatch.setattr(modules.verifier, "verify_cuda", verify)
    namespace = {}
    exec(modules.feedback.build_local_feedback_bootstrap("reference", max_calls=1), namespace)
    result = namespace["cuda_feedback"]("source")
    assert calls[0]["allow_feedback_overlap"] is True
    assert calls[0]["benchmark"] is False
    assert "reward" not in result
    namespace["cuda_feedback"]("source")
    assert len(calls) == 1
