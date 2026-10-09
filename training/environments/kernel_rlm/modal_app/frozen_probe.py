"""Bounded frozen-policy probe through the real V1 training client and harness."""

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import modal

from .attention_lora import MODEL_ID, REVISION, models
from .grpo_live_smoke import live_image
from .inference_runtime import (
    PROBE_TIMEOUT_S,
    STARTUP_TIMEOUT_S,
    cache_environment,
    wait_for_inference,
)
from .rollout_smoke import traces

app = modal.App("kernel-rlm-frozen-probe")
inference_cache = modal.Volume.from_name("kernel-rlm-inference-cache", create_if_missing=True)


@app.function(image=live_image, timeout=120, cpu=1, memory=4096)
def preflight() -> dict:
    from renderers import LagunaXS2RendererConfig
    from verifiers.v1.clients import ModelContext
    from verifiers.v1.configs.client import TrainClientConfig
    from verifiers.v1.envs.single_agent.env import SingleAgentEnvConfig

    config = SingleAgentEnvConfig.model_validate(
        {
            "taskset": {"id": "kernel_rlm_v1", "manifest_path": "/tmp/probe.jsonl"},
            "agent": {
                "harness": {
                    "id": "kernel_rlm_v1",
                    "max_turns": 5,
                    "max_tokens": 2048,
                    "model_timeout_s": 90,
                },
                "runtime": {"type": "subprocess"},
                "sampling": {"temperature": 1.0, "max_tokens": 2048},
                "timeout": {"rollout": 480, "scoring": 260},
            },
        }
    )
    context = ModelContext(
        model=MODEL_ID,
        client=TrainClientConfig(
            base_url="http://127.0.0.1:8000/v1",
            api_key_var="PROBE_API_KEY",
            renderer_model_name=MODEL_ID,
            renderer=LagunaXS2RendererConfig(),
        ),
    )
    return {"config": config.model_dump(mode="json"), "model": context.model}


@app.function(
    image=live_image,
    volumes={"/models": models, "/traces": traces, "/inference-cache": inference_cache},
    gpu="A100-80GB:2",
    cpu=12,
    memory=98304,
    timeout=1260,
    startup_timeout=180,
    retries=0,
    max_containers=1,
    scaledown_window=2,
)
async def probe(run_id: str, rollouts: int = 1, curriculum: bool = False) -> dict:
    if rollouts not in (1, 4):
        raise ValueError("Probe supports one trajectory or one four-rollout group")
    import asyncio
    import os
    import signal
    import subprocess
    import time
    import traceback
    from importlib.metadata import version

    import verifiers.v1 as vf
    from huggingface_hub import snapshot_download
    from renderers import LagunaXS2RendererConfig
    from verifiers.v1.clients import ModelContext
    from verifiers.v1.configs.client import TrainClientConfig
    from verifiers.v1.envs.single_agent.env import SingleAgentEnvConfig

    directory = Path("/traces/frozen-probe") / run_id
    directory.mkdir(parents=True, exist_ok=True)
    report = {
        "run_id": run_id,
        "model_id": MODEL_ID,
        "status": "running",
        "groups": [],
        "gate_passed": False,
        "startup_timeout_seconds": STARTUP_TIMEOUT_S,
        "probe_timeout_seconds": PROBE_TIMEOUT_S,
    }
    started = time.monotonic()
    server = None
    try:
        async with asyncio.timeout(PROBE_TIMEOUT_S):
            snapshot = snapshot_download(MODEL_ID, revision=REVISION, local_files_only=True)
            cache_identity = {
                "model_revision": REVISION,
                "gpu": "A100-80GB-sm80",
                **{package: version(package) for package in ("vllm", "torch", "triton")},
            }
            cache_env = cache_environment(Path("/inference-cache"), cache_identity)
            report["inference_cache"] = {"identity": cache_identity, "environment": cache_env}
            server_env = {**os.environ, **cache_env, "CUDA_VISIBLE_DEVICES": "0"}
            command = [
                "python",
                "-m",
                "vllm.entrypoints.openai.api_server",
                "--model",
                snapshot,
                "--served-model-name",
                MODEL_ID,
                "--host",
                "127.0.0.1",
                "--port",
                "8000",
                "--max-model-len",
                "8192",
                "--max-num-seqs",
                "2",
                "--max-num-batched-tokens",
                "1024",
                "--gpu-memory-utilization",
                "0.95",
                "--enable-lora",
                "--max-lora-rank",
                "8",
                "--max-loras",
                "1",
                "--reasoning-parser",
                "poolside_v1",
                "--tool-call-parser",
                "poolside_v1",
            ]
            report["inference_command"] = command
            report["scope"] = (
                f"Frozen {'softplus + sigmoid curriculum' if curriculum else 'vector-add'} "
                f"probe: {rollouts} rollouts per task; no optimizer"
            )
            report["title"] = report["scope"]
            with (directory / "inference.log").open("w") as logfile:
                server = subprocess.Popen(
                    command,
                    env=server_env,
                    stdout=logfile,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            os.environ["CUDA_VISIBLE_DEVICES"] = "1"
            os.environ["PROBE_API_KEY"] = "unused-local-endpoint"
            os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"
            report["inference_startup_seconds"] = await wait_for_inference(server)
            (directory / "report.json").write_text(json.dumps(report, indent=2))
            await inference_cache.commit.aio()
            selected = [
                {
                    "_task_id": "vector_add",
                    "code": Path("/root/kernel_rlm/tasks/vector_add.py").read_text(),
                },
            ]
            if curriculum:
                task_ids = {"lvl1-seed42-0011", "lvl1-seed42-0049"}
                selected = [
                    row
                    for line in Path("/root/kernel_rlm/tasks/lvl1_seed42.jsonl").read_text().splitlines()
                    if (row := json.loads(line))["_task_id"] in task_ids
                ]
                if len(selected) != len(task_ids):
                    raise ValueError("Missing or duplicated curriculum probe tasks")
            report["task_ids"] = [row["_task_id"] for row in selected]
            manifest = directory / "manifest.jsonl"
            manifest.write_text("\n".join(json.dumps(row) for row in selected))
            config = SingleAgentEnvConfig.model_validate(
                {
                    "taskset": {"id": "kernel_rlm_v1", "manifest_path": str(manifest)},
                    "agent": {
                        "harness": {
                            "id": "kernel_rlm_v1",
                            "max_turns": 5,
                            "max_tokens": 2048,
                            "model_timeout_s": 90,
                        },
                        "runtime": {"type": "subprocess"},
                        "sampling": {"temperature": 1.0, "max_tokens": 2048},
                        "timeout": {"rollout": 480, "scoring": 260},
                    },
                }
            )
            context = ModelContext(
                model=MODEL_ID,
                client=TrainClientConfig(
                    base_url="http://127.0.0.1:8000/v1",
                    api_key_var="PROBE_API_KEY",
                    renderer_model_name=snapshot,
                    renderer=LagunaXS2RendererConfig(),
                ),
            )
            (directory / "config.json").write_text(config.model_dump_json(indent=2))
            environment = vf.load_environment(config)
            async with environment.serving():
                for task in vf.load_taskset(config.taskset):
                    group = {
                        "task": task.data.name,
                        "rewards": [],
                        "submitted": 0,
                        "feedback_calls": 0,
                        "correct": 0,
                        "quality_rewards": [],
                    }
                    report["groups"].append(group)
                    for _rollout in range(rollouts):
                        episode = await environment.run_episode(task, context)
                        with (directory / "traces.jsonl").open("a") as output:
                            output.write(episode.model_dump_json() + "\n")
                        if not episode.ok:
                            errors = [
                                str(error) for trace in episode.traces for error in trace.errors
                            ]
                            report["episode_errors"] = errors
                            raise RuntimeError(f"Episode failed: {errors}")
                        trace = episode.traces[0]
                        group["rewards"].append(trace.reward)
                        group["submitted"] += bool(trace.info.get("kernel_submission"))
                        group["feedback_calls"] += trace.info.get("feedback_calls", 0)
                        group["correct"] += bool(trace.info.get("kernel_verify", {}).get("correct"))
                        verification = trace.info.get("kernel_verify", {})
                        quality_reward = 0.0
                        if trace.info.get("kernel_static", {}).get("valid"):
                            quality_reward += 0.1
                            if verification.get("compiled") and verification.get("extension_used"):
                                quality_reward += 0.2
                            if verification.get("correct"):
                                quality_reward += 0.4
                        group["quality_rewards"].append(quality_reward)
                        print(json.dumps(group), flush=True)
                        (directory / "report.json").write_text(json.dumps(report, indent=2))
                        await traces.commit.aio()
            report["trajectory_passed"] = all(
                group["correct"] > 0 and group["feedback_calls"] > 0 for group in report["groups"]
            )
            report["gate_passed"] = (
                rollouts == 4
                and report["trajectory_passed"]
                and all(
                    len(group["rewards"]) == rollouts and len(set(group["rewards"])) > 1
                    for group in report["groups"]
                )
            )
            report["status"] = (
                "trajectory_passed" if report["trajectory_passed"] else "trajectory_failed"
            )
            if rollouts == 4:
                report["status"] = "group_passed" if report["gate_passed"] else "group_gate_failed"
    except Exception:
        report["status"] = "failed"
        report["error"] = traceback.format_exc()
    finally:
        if server is not None:
            try:
                os.killpg(server.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                await asyncio.to_thread(server.wait, timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(server.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                await asyncio.to_thread(server.wait, timeout=5)
        report["elapsed_seconds"] = time.monotonic() - started
        (directory / "report.json").write_text(json.dumps(report, indent=2))
        await traces.commit.aio()
        await inference_cache.commit.aio()
    return report


@app.local_entrypoint()
def main(rollouts: int = 1, curriculum: bool = False) -> None:
    if rollouts not in (1, 4):
        raise ValueError("Use --rollouts 1 or --rollouts 4")
    print(json.dumps(preflight.remote(), indent=2), flush=True)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    report = probe.remote(run_id, rollouts, curriculum)
    destination = Path("logs/frozen-probe") / run_id
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
