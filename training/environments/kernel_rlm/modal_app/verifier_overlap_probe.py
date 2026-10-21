"""Deterministic verifier measurements and same-server inference comparisons."""

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import modal

from .attention_lora import MODEL_ID, REVISION, models
from .frozen_probe import inference_cache
from .gpu_telemetry import sample_gpu_stats_to_file
from .inference_runtime import cache_environment, wait_for_inference
from .rollout_smoke import traces
from .shared_gpu_probe import image

app = modal.App("kernel-rlm-verifier-overlap-probe")


def softplus_submission() -> str:
    cpp = "#include <torch/extension.h>\ntorch::Tensor softplus_cuda(torch::Tensor input);"
    cuda = r"""#include <torch/extension.h>
#include <cuda_runtime.h>
__global__ void softplus_kernel(const float* input, float* output, int64_t count) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        const float value = input[index];
        output[index] = value > 20.0f ? value : log1pf(expf(value));
    }
}
torch::Tensor softplus_cuda(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous(), "CUDA contiguous input required");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "float32 required");
    auto output = torch::empty_like(input);
    softplus_kernel<<<(input.numel() + 255) / 256, 256>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), input.numel());
    return output;
}
"""
    return (
        "from torch.utils.cpp_extension import load_inline\n"
        f"CPP_SOURCE = {cpp!r}\nCUDA_SOURCE = {cuda!r}\n"
        "extension = load_inline(name='overlap_probe_softplus', cpp_sources=CPP_SOURCE, "
        "cuda_sources=CUDA_SOURCE, functions=['softplus_cuda'], extra_cuda_cflags=['-O3'])\n"
        "def cuda_forward(input):\n    return extension.softplus_cuda(input)\n"
    )


def setup_extension_cache() -> None:
    from importlib.metadata import version

    cache = Path("/inference-cache/verifier-extensions") / (
        f"torch-{version('torch')}-py{sys.version_info.major}{sys.version_info.minor}-sm80"
    )
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_EXTENSIONS_DIR"] = str(cache)
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"
    os.environ["MAX_JOBS"] = "2"


def compile_fixtures() -> list[dict]:
    import tempfile

    from kernel_rlm.verifier import _RUNNER, _run_verifier_process

    setup_extension_cache()
    results = []
    for source in (Path("/root/known_cuda.py").read_text(), softplus_submission()):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "runner.py").write_text(_RUNNER)
            (directory / "submission.py").write_text(source)
            process = _run_verifier_process(
                [
                    sys.executable,
                    str(directory / "runner.py"),
                    "unused_reference.py",
                    str(directory / "submission.py"),
                    "1",
                    "0.001",
                    "0.001",
                    "0",
                    str(directory / "phase.json"),
                    "compile",
                ],
                timeout_s=120,
                cwd=None,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
            )
            if process.returncode:
                raise RuntimeError(process.stderr[-2000:])
            result = json.loads(process.stdout.strip().splitlines()[-1])
            if not result.get("compiled"):
                raise RuntimeError(result)
            results.append(result)
    inference_cache.commit()
    return results


@app.function(
    image=image,
    cpu=12,
    memory=16384,
    timeout=300,
    retries=0,
    volumes={"/inference-cache": inference_cache},
)
def compile_preflight() -> list[dict]:
    return compile_fixtures()


async def verify_case(directory: Path, case: dict, *, paused: bool, benchmark: bool) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "case.json").write_text(json.dumps(case))
    code = (
        "import json,sys; from pathlib import Path; from kernel_rlm.verifier import verify_cuda; "
        "root=Path(sys.argv[1]); case=json.loads((root/'case.json').read_text()); "
        "result=verify_cuda(case['reference'],case['submission'],n_trials=1,timeout_s=120,"
        f"benchmark={benchmark!r},collect_phase_events=True); "
        "(root/'result.json').write_text(json.dumps(result,indent=2))"
    )
    environment = {**os.environ, "TORCH_CUDA_ARCH_LIST": "8.0", "MAX_JOBS": "2"}
    environment.pop("KERNEL_INFERENCE_ADMIN_URL", None)
    if paused:
        environment["KERNEL_INFERENCE_ADMIN_URL"] = "http://127.0.0.1:8000"
    started = time.monotonic()
    with (directory / "worker.log").open("w") as log:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            code,
            str(directory),
            env=environment,
            stdout=log,
            stderr=log,
            start_new_session=True,
        )
        try:
            await asyncio.wait_for(process.wait(), 135)
        finally:
            if process.returncode is None:
                os.killpg(process.pid, signal.SIGKILL)
                await process.wait()
    if process.returncode:
        raise RuntimeError(f"Verifier worker exited {process.returncode}: {directory}")
    result = json.loads((directory / "result.json").read_text())
    result["wall_seconds"] = time.monotonic() - started
    result["case"] = case["name"]
    print(json.dumps({"event": "verification", **result}), flush=True)
    return result


async def inference_load(client, directory: Path, started_event: asyncio.Event, *, gated: bool):
    from kernel_rlm.verifier_slot import inference_request_slot

    async def worker(worker_id):
        records = []
        for request_id in range(4):
            started = time.monotonic()
            tokens = 0
            content_chunk_times = []
            async with inference_request_slot(150, enabled=gated) as admission_wait:
                stream = await client.chat.completions.create(
                    model=MODEL_ID,
                    messages=[
                        {"role": "user", "content": "Explain CUDA vector addition in detail."}
                    ],
                    temperature=0,
                    max_tokens=256,
                    stream=True,
                    stream_options={"include_usage": True},
                    extra_body={
                        "ignore_eos": True,
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                )
                async for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        content_chunk_times.append(time.monotonic())
                        started_event.set()
                    if chunk.usage:
                        tokens = chunk.usage.completion_tokens
            record = {
                "worker": worker_id,
                "request": request_id,
                "completion_tokens": tokens,
                "seconds": time.monotonic() - started,
                "admission_wait_seconds": admission_wait,
                "content_chunk_times": content_chunk_times,
            }
            records.append(record)
            with (directory / "requests.jsonl").open("a") as output:
                output.write(json.dumps(record) + "\n")
        return records

    started = time.monotonic()
    async with asyncio.TaskGroup() as group:
        workers = [group.create_task(worker(worker_id)) for worker_id in range(2)]
    records = [record for worker_task in workers for record in worker_task.result()]
    return {
        "requests": records,
        "seconds": time.monotonic() - started,
        "completion_tokens": sum(record["completion_tokens"] for record in records),
    }


async def check_guarded_feedback(client, reference: str, directory: Path) -> list[dict]:
    from kernel_rlm.feedback import build_local_feedback_bootstrap
    from rlm_train.repl.subprocess import SubprocessReplBackend

    source = Path("/root/known_cuda.py").read_text()
    oversized = source.replace(
        "    return _module.vector_add(left, right)",
        "    scratch = torch.empty(256 * 1024 * 1024, device='cuda')\n"
        "    return _module.vector_add(left, right)",
    )
    if oversized == source:
        raise RuntimeError("Allocation fixture was not inserted")
    checks = []
    for name, candidate, expected in (
        ("qualified", source, "overlap"),
        ("allocation_fallback", oversized, "exclusive"),
    ):
        stage = directory / name
        stage.mkdir(parents=True, exist_ok=True)
        backend = SubprocessReplBackend(request_timeout=160)
        ready = asyncio.Event()
        pressure = None
        try:
            await backend.start(proxy_url="http://127.0.0.1:1", rollout_id=name)
            await backend.load_context(reference)
            await backend.bootstrap(
                build_local_feedback_bootstrap(
                    reference, max_calls=1, details_path=str(stage / "feedback.jsonl")
                )
            )
            pressure = asyncio.create_task(inference_load(client, stage, ready, gated=True))
            await asyncio.wait_for(ready.wait(), 45)
            code = f"cuda_feedback({candidate!r})"
            execution = await backend.execute(code)
            (stage / "repl.json").write_text(
                json.dumps(
                    {
                        "code": code,
                        "stdout": execution.stdout,
                        "stderr": execution.stderr,
                        "final_answer": execution.final_answer,
                    },
                    indent=2,
                )
            )
            if execution.stderr:
                raise RuntimeError(execution.stderr)
            result = json.loads(
                next(
                    line.removeprefix("CUDA feedback: ")
                    for line in execution.stdout.splitlines()
                    if line.startswith("CUDA feedback: ")
                )
            )
            if not result.get("correct") or result["scheduling"]["mode"] != expected:
                raise RuntimeError(f"Guarded feedback check failed: {result}")
            if name == "allocation_fallback":
                assert result["scheduling"]["fallback_reason"] == "allocator_limit_or_oom"
            assert result["feedback_calls_used"] == 1
            assert "reward" not in result and execution.final_answer is None
            load_result = await asyncio.wait_for(pressure, 150)
            checks.append({"name": name, "result": result, "inference": load_result})
            print(
                json.dumps(
                    {
                        "event": "guarded_repl_check",
                        "name": name,
                        "mode": expected,
                        "correct": result["correct"],
                    }
                ),
                flush=True,
            )
        finally:
            if pressure is not None:
                if not pressure.done():
                    pressure.cancel()
                await asyncio.gather(pressure, return_exceptions=True)
            await backend.stop()
    return checks


async def compare(client, directory: Path, case: dict, mode: str) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    started_event = asyncio.Event()
    started = time.monotonic()
    generation = asyncio.create_task(
        inference_load(client, directory, started_event, gated=mode == "paused")
    )
    try:
        await asyncio.wait_for(started_event.wait(), 45)
        verification = None
        if mode != "inference_only":
            verification = await verify_case(
                directory / "verifier",
                case,
                paused=mode == "paused",
                benchmark=False,
            )
        generation_done_at_verifier_end = generation.done()
        inference = await asyncio.wait_for(generation, 150)
        elapsed = time.monotonic() - started
        result = {
            "mode": mode,
            "case": case["name"],
            "wall_seconds": elapsed,
            "inference": inference,
            "verification": verification,
            "generation_done_at_verifier_end": generation_done_at_verifier_end,
            "combined_tokens_per_second": inference["completion_tokens"] / elapsed,
        }
        (directory / "comparison.json").write_text(json.dumps(result, indent=2))
        print(
            json.dumps(
                {
                    "event": "comparison",
                    "mode": mode,
                    "case": case["name"],
                    "wall_seconds": elapsed,
                    "combined_tokens_per_second": result["combined_tokens_per_second"],
                    "correct": verification.get("correct") if verification else None,
                }
            ),
            flush=True,
        )
        return result
    finally:
        if not generation.done():
            generation.cancel()
        await asyncio.gather(generation, return_exceptions=True)


@app.function(
    image=image,
    gpu="A100-80GB",
    cpu=12,
    memory=98304,
    volumes={"/models": models, "/traces": traces, "/inference-cache": inference_cache},
    timeout=1800,
    startup_timeout=120,
    retries=0,
    max_containers=1,
)
async def probe(run_id: str, live: bool = False) -> dict:
    from importlib.metadata import version

    from huggingface_hub import snapshot_download
    from openai import AsyncOpenAI

    directory = Path("/traces/verifier-overlap-probe") / run_id
    directory.mkdir(parents=True, exist_ok=True)
    setup_extension_cache()
    started = time.monotonic()
    report = {
        "run_id": run_id,
        "status": "running",
        "gpu_count": 1,
        "optimizer_steps": 0,
        "standalone": [],
        "comparisons": [],
        "model_id": MODEL_ID,
        "model_revision": REVISION,
        "memory_utilization": 0.86,
        "max_model_len": 8192,
    }
    stop = asyncio.Event()
    sampler = asyncio.create_task(sample_gpu_stats_to_file(directory / "gpu_stats.jsonl", stop))
    server = None

    def save():
        (directory / "report.json").write_text(json.dumps(report, indent=2))

    async def progress():
        while True:
            await asyncio.sleep(20)
            log = directory / "inference.log"
            tail = log.read_text(errors="replace").splitlines()[-1:] if log.exists() else []
            print(
                json.dumps(
                    {
                        "event": "progress",
                        "elapsed_seconds": time.monotonic() - started,
                        "inference_log_tail": tail,
                    }
                ),
                flush=True,
            )

    heartbeat = asyncio.create_task(progress())
    try:
        async with asyncio.timeout(1740):
            rows = [
                json.loads(line)
                for line in Path("/root/kernel_rlm/tasks/lvl1_seed42.jsonl")
                .read_text()
                .splitlines()
            ]
            softplus = next(row for row in rows if row["_task_id"] == "lvl1-seed42-0011")
            cases = [
                {
                    "name": "vector_add_4096",
                    "reference": Path("/root/kernel_rlm/tasks/vector_add.py").read_text(),
                    "submission": Path("/root/known_cuda.py").read_text(),
                },
                {
                    "name": softplus["_task_id"],
                    "reference": softplus["code"],
                    "submission": softplus_submission(),
                },
            ]
            for case in cases:
                case["reference"] += "\nimport torch\ntorch.manual_seed(42)\n"
                for repeat in range(2):
                    result = await verify_case(
                        directory / "standalone" / case["name"] / str(repeat),
                        case,
                        paused=False,
                        benchmark=True,
                    )
                    report["standalone"].append(result)
                    save()
                    if not result.get("correct") or result.get("speedup") is None:
                        raise RuntimeError(f"Standalone verifier failed: {case['name']}")
            await traces.commit.aio()
            if live:
                snapshot = snapshot_download(MODEL_ID, revision=REVISION, local_files_only=True)
                config = {
                    "router": None,
                    "server": {"host": "127.0.0.1", "port": 8000},
                    "vllm": {
                        "model": snapshot,
                        "served_model_name": [MODEL_ID],
                        "safetensors_load_strategy": "prefetch",
                        "gpu_memory_utilization": 0.86,
                        "max_model_len": 8192,
                        "max_num_seqs": 8,
                        "max_num_batched_tokens": 1024,
                        "enable_lora": True,
                        "max_lora_rank": 8,
                        "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                        "reasoning_parser": "poolside_v1",
                        "tool_call_parser": "poolside_v1",
                    },
                }
                config_path = directory / "inference.json"
                config_path.write_text(json.dumps(config))
                identity = {
                    "model_revision": REVISION,
                    "gpu": "A100-80GB-sm80",
                    **{name: version(name) for name in ("vllm", "torch", "triton")},
                }
                cache = cache_environment(Path("/inference-cache"), identity)
                command = [
                    sys.executable,
                    "-c",
                    "import sys; from pathlib import Path; "
                    "from prime_rl.configs.inference import InferenceConfig; "
                    "from prime_rl.inference.vllm.server import server; "
                    "server(InferenceConfig.model_validate_json(Path(sys.argv[1]).read_text()))",
                    str(config_path),
                ]
                with (directory / "inference.log").open("w") as log:
                    server = subprocess.Popen(
                        command,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        env={**os.environ, **cache, "CUDA_VISIBLE_DEVICES": "0"},
                        start_new_session=True,
                    )
                report["startup_seconds"] = await wait_for_inference(server, timeout_s=900)
                save()
                async with AsyncOpenAI(
                    base_url="http://127.0.0.1:8000/v1", api_key="local", max_retries=0, timeout=150
                ) as client:
                    await client.chat.completions.create(
                        model=MODEL_ID,
                        messages=[{"role": "user", "content": "Hello"}],
                        max_tokens=1,
                        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                    )
                    for repeat, modes in enumerate(
                        (
                            ("inference_only", "paused", "overlap"),
                            ("overlap", "paused", "inference_only"),
                        )
                    ):
                        for mode in modes:
                            for case in cases if mode != "inference_only" else cases[:1]:
                                result = await compare(
                                    client,
                                    directory / f"round-{repeat}" / mode / case["name"],
                                    case,
                                    mode,
                                )
                                report["comparisons"].append(result)
                                save()
                                if result["verification"] and not result["verification"].get(
                                    "correct"
                                ):
                                    raise RuntimeError("Shared verifier correctness check failed")
            report["status"] = "completed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        if server is not None and server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
            try:
                await asyncio.to_thread(server.wait, timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                await asyncio.to_thread(server.wait, timeout=10)
        heartbeat.cancel()
        await asyncio.gather(heartbeat, return_exceptions=True)
        stop.set()
        await sampler
        report["elapsed_seconds"] = time.monotonic() - started
        save()
        await traces.commit.aio()
        await inference_cache.commit.aio()
    return report


@app.local_entrypoint()
def main(live: bool = False):
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    print(f"Probe {run_id}: live={live}", flush=True)
    print(json.dumps({"cpu_compile_preflight": compile_preflight.remote()}), flush=True)
    result = probe.remote(run_id, live)
    directory = Path("logs/verifier-overlap-probe") / run_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "report.json").write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {
                "run_id": run_id,
                "status": result["status"],
                "elapsed_seconds": result["elapsed_seconds"],
                "comparisons": len(result["comparisons"]),
                "report": str(directory / "report.json"),
            },
            indent=2,
        ),
        flush=True,
    )
    if result["status"] != "completed":
        raise RuntimeError(result.get("error", "Probe failed"))
