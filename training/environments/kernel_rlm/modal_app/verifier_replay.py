"""Replay saved CUDA candidates before spending on another training update."""

import ast
import asyncio
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

import modal

from .attention_lora import MODEL_ID, REVISION, models
from .frozen_probe import inference_cache
from .gpu_telemetry import sample_gpu_stats_to_file
from .inference_runtime import cache_environment, wait_for_inference
from .rollout_smoke import traces
from .shared_gpu_probe import image
from .verifier_overlap_probe import (
    check_guarded_feedback,
    inference_load,
    setup_extension_cache,
    softplus_submission,
)

app = modal.App("kernel-rlm-verifier-replay")


def literal_string(node: ast.AST, values: dict[str, str]) -> str:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return values[node.id]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return literal_string(node.left, values) + literal_string(node.right, values)
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "repr"
        and len(node.args) == 1
        and not node.keywords
    ):
        return repr(literal_string(node.args[0], values))
    raise ValueError("Candidate recovery accepts only literal string assignments")


def saved_cases(path: Path) -> list[dict]:
    episodes = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    cases = []
    for episode_index, action_index in ((3, 2), (3, 3)):
        episode = episodes[episode_index]
        messages = [
            message
            for message in episode["traces"][0]["info"]["repl_transcript"]
            if message.get("tool_calls")
        ]
        code = json.loads(messages[action_index]["tool_calls"][0]["function"]["arguments"])["code"]
        values = {}
        for node in ast.parse(code).body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                if isinstance(node.targets[0], ast.Name):
                    try:
                        values[node.targets[0].id] = literal_string(node.value, values)
                    except (ValueError, KeyError):
                        continue
        source = values["source"]
        ast.parse(source)
        cases.append(
            {
                "name": f"saved-maxpool-{action_index}",
                "reference": episode["task"]["data"]["context"],
                "submission": source,
                "expected_correct": True,
            }
        )
    submitted = episodes[5]
    cases.append(
        {
            "name": "saved-partial-submission",
            "reference": submitted["task"]["data"]["context"],
            "submission": submitted["traces"][0]["info"]["kernel_submission"],
            "expected_correct": False,
        }
    )
    return cases


def compile_case(case: dict) -> dict:
    from kernel_rlm.verifier import (
        _RUNNER,
        _run_verifier_process,
        submission_compile_lock,
        submission_extension_root,
    )
    from kernel_rlm.verifier_slot import compile_slot

    started = time.monotonic()
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        (directory / "runner.py").write_text(_RUNNER)
        (directory / "submission.py").write_text(case["submission"])
        with compile_slot(90, path=submission_compile_lock(case["submission"])) as remaining:
            result = _run_verifier_process(
                [
                    sys.executable,
                    str(directory / "runner.py"),
                    "unused_reference.py",
                    str(directory / "submission.py"),
                    "1",
                    "0.001",
                    "0.0001",
                    "0",
                    str(directory / "phase.json"),
                    "compile",
                ],
                timeout_s=remaining,
                cwd=None,
                env={
                    **os.environ,
                    "CUDA_VISIBLE_DEVICES": "",
                    "MAX_JOBS": "1",
                    "KERNEL_RLM_EXTENSION_BUILD_ROOT": str(
                        submission_extension_root(case["submission"])
                    ),
                },
            )
        if result.returncode:
            raise RuntimeError(result.stderr[-2000:])
        record = json.loads(result.stdout.strip().splitlines()[-1])
        record.update(name=case["name"], seconds=time.monotonic() - started)
        if not record.get("compiled"):
            raise RuntimeError(record)
        return record


@app.function(
    image=image,
    cpu=12,
    memory=24576,
    timeout=360,
    retries=0,
    volumes={"/inference-cache": inference_cache, "/traces": traces},
)
def prepare(run_id: str, cases: list[dict]) -> dict:
    setup_extension_cache()
    fixture_cases = [
        {"name": "vector-add", "submission": Path("/root/known_cuda.py").read_text()},
        {"name": "softplus", "submission": softplus_submission()},
    ]
    directory = Path("/traces/verifier-replay") / run_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "cases.json").write_text(json.dumps(cases, indent=2))
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(compile_case, cases + fixture_cases))
    inference_cache.commit()
    report = {"run_id": run_id, "status": "passed", "cpu_compilation": results}
    (directory / "compile-report.json").write_text(json.dumps(report, indent=2))
    traces.commit()
    return report


@app.function(
    image=image,
    gpu="A100-80GB",
    cpu=12,
    memory=98304,
    volumes={"/models": models, "/traces": traces, "/inference-cache": inference_cache},
    timeout=1260,
    startup_timeout=120,
    retries=0,
    max_containers=1,
)
async def replay(run_id: str, cases: list[dict]) -> dict:
    from huggingface_hub import snapshot_download
    from kernel_rlm.verifier import verify_cuda
    from openai import AsyncOpenAI

    setup_extension_cache()
    directory = Path("/traces/verifier-replay") / run_id
    directory.mkdir(parents=True, exist_ok=True)
    report = {"run_id": run_id, "status": "running", "standalone": [], "shared": []}
    started = time.monotonic()
    server = None
    stop = asyncio.Event()
    sampler = asyncio.create_task(sample_gpu_stats_to_file(directory / "gpu_stats.jsonl", stop))

    def save():
        (directory / "report.json").write_text(json.dumps(report, indent=2))

    async def check(case: dict, shared: bool) -> dict:
        result = await asyncio.to_thread(
            verify_cuda,
            case["reference"],
            case["submission"],
            n_trials=1,
            timeout_s=120,
            benchmark=not shared,
            collect_phase_events=True,
            allow_feedback_overlap=shared,
        )
        result["name"] = case["name"]
        print(json.dumps({"event": "replay", "shared": shared, **result}), flush=True)
        if result.get("compiled") is not True:
            raise RuntimeError(f"Replay compilation failed: {result}")
        if result.get("correct") is not case["expected_correct"]:
            raise RuntimeError(f"Unexpected replay correctness: {result}")
        if (
            "ImportError" in result.get("error", "")
            or result.get("verification_status") == "timeout"
        ):
            raise RuntimeError(f"Replay infrastructure failure: {result}")
        return result

    try:
        async with asyncio.timeout(1200):
            for case in cases:
                report["standalone"].append(await check(case, False))
                save()
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
            cache = cache_environment(
                Path("/inference-cache"),
                {
                    "model_revision": REVISION,
                    "gpu": "A100-80GB-sm80",
                    **{name: version(name) for name in ("vllm", "torch", "triton")},
                },
            )
            with (directory / "inference.log").open("w") as log:
                server = subprocess.Popen(
                    [
                        sys.executable,
                        "-c",
                        "import sys; from pathlib import Path; "
                        "from prime_rl.configs.inference import InferenceConfig; "
                        "from prime_rl.inference.vllm.server import server; "
                        "server(InferenceConfig.model_validate_json(Path(sys.argv[1]).read_text()))",
                        str(config_path),
                    ],
                    stdout=log,
                    stderr=log,
                    env={**os.environ, **cache},
                    start_new_session=True,
                )
            report["startup_seconds"] = await wait_for_inference(server, timeout_s=600)
            os.environ["KERNEL_INFERENCE_ADMIN_URL"] = "http://127.0.0.1:8000"
            async with AsyncOpenAI(
                base_url="http://127.0.0.1:8000/v1", api_key="local", max_retries=0, timeout=150
            ) as client:
                reference = Path("/root/kernel_rlm/tasks/vector_add.py").read_text()
                os.environ["RLM_TRAIN_EXEC_TIMEOUT_S"] = "150"
                report["guarded_feedback"] = await check_guarded_feedback(
                    client, reference, directory / "guarded-repl"
                )
                for case in cases:
                    stage = directory / "shared" / case["name"]
                    stage.mkdir(parents=True, exist_ok=True)
                    ready = asyncio.Event()
                    pressure = asyncio.create_task(inference_load(client, stage, ready, gated=True))
                    try:
                        await asyncio.wait_for(ready.wait(), 45)
                        report["shared"].append(await check(case, True))
                        await asyncio.wait_for(pressure, 150)
                        save()
                    finally:
                        pressure.cancel()
                        await asyncio.gather(pressure, return_exceptions=True)
            report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
    finally:
        if server is not None and server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
            try:
                await asyncio.to_thread(server.wait, timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                await asyncio.to_thread(server.wait, timeout=10)
        stop.set()
        await sampler
        report["elapsed_seconds"] = time.monotonic() - started
        save()
        await traces.commit.aio()
        await inference_cache.commit.aio()
    return report


@app.local_entrypoint()
def main(trace_path: str):
    cases = saved_cases(Path(trace_path))
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    directory = Path("logs/verifier-replay") / run_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "cases.json").write_text(json.dumps(cases, indent=2))
    print(f"Replay {run_id}: {len(cases)} saved candidates", flush=True)
    compiled = prepare.remote(run_id, cases)
    (directory / "compile-report.json").write_text(json.dumps(compiled, indent=2))
    print(json.dumps(compiled), flush=True)
    report = replay.remote(run_id, cases)
    (directory / "report.json").write_text(json.dumps(report, indent=2))
    print(
        json.dumps({key: value for key, value in report.items() if key != "guarded_feedback"}),
        flush=True,
    )
    if report["status"] != "passed":
        raise RuntimeError(report.get("error", "Verifier replay failed"))
