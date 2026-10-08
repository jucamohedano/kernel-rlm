"""Bounded frozen-policy probe through the real V1 training client and harness."""

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import modal

from .attention_lora import MODEL_ID, REVISION, models
from .grpo_live_smoke import live_image
from .rollout_smoke import traces

app = modal.App("kernel-rlm-frozen-probe")


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
                "harness": {"id": "kernel_rlm_v1", "max_turns": 5, "max_tokens": 2048},
                "runtime": {"type": "subprocess"},
                "sampling": {"temperature": 1.0, "max_tokens": 2048},
                "timeout": {"rollout": 360, "scoring": 180},
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
    volumes={"/models": models, "/traces": traces},
    gpu="A100-80GB:2",
    cpu=12,
    memory=98304,
    timeout=960,
    startup_timeout=180,
    retries=0,
    max_containers=1,
    scaledown_window=2,
)
async def probe(run_id: str) -> dict:
    import asyncio
    import os
    import signal
    import subprocess
    import time
    import traceback
    import urllib.request

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
    }
    started = time.monotonic()
    server = None
    try:
        async with asyncio.timeout(900):
            snapshot = snapshot_download(MODEL_ID, revision=REVISION, local_files_only=True)
            server_env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0"}
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
                "--enforce-eager",
                "--enable-lora",
                "--trust-remote-code",
            ]
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
            for _attempt in range(225):
                if server.poll() is not None:
                    raise RuntimeError("Inference server exited; inspect inference.log")
                try:
                    await asyncio.to_thread(
                        urllib.request.urlopen, "http://127.0.0.1:8000/health", timeout=2
                    )
                    break
                except OSError:
                    await asyncio.sleep(2)
            else:
                raise TimeoutError("Inference startup exceeded 450 seconds")
            rows = [
                json.loads(line)
                for line in Path("/root/kernel_rlm/tasks/lvl1_seed42.jsonl")
                .read_text()
                .splitlines()
            ]
            selected = [
                {
                    "_task_id": "vector_add",
                    "code": Path("/root/kernel_rlm/tasks/vector_add.py").read_text(),
                },
                next(row for row in rows if row["_task_id"] == "lvl1-seed42-0049"),
            ]
            manifest = directory / "manifest.jsonl"
            manifest.write_text("\n".join(json.dumps(row) for row in selected))
            config = SingleAgentEnvConfig.model_validate(
                {
                    "taskset": {"id": "kernel_rlm_v1", "manifest_path": str(manifest)},
                    "agent": {
                        "harness": {"id": "kernel_rlm_v1", "max_turns": 5, "max_tokens": 2048},
                        "runtime": {"type": "subprocess"},
                        "sampling": {"temperature": 1.0, "max_tokens": 2048},
                        "timeout": {"rollout": 360, "scoring": 180},
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
                    }
                    report["groups"].append(group)
                    for _rollout in range(4):
                        episode = await environment.run_episode(task, context)
                        with (directory / "traces.jsonl").open("a") as output:
                            output.write(episode.model_dump_json() + "\n")
                        if not episode.ok:
                            raise RuntimeError(f"Episode failed: {episode.errors}")
                        trace = episode.traces[0]
                        group["rewards"].append(trace.reward)
                        group["submitted"] += bool(trace.info.get("kernel_submission"))
                        group["feedback_calls"] += trace.info.get("feedback_calls", 0)
                        group["correct"] += bool(trace.info.get("kernel_verify", {}).get("correct"))
                        print(json.dumps(group), flush=True)
                        (directory / "report.json").write_text(json.dumps(report, indent=2))
                        await traces.commit.aio()
            report["gate_passed"] = all(
                len(set(group["rewards"])) > 1
                and group["correct"] > 0
                and group["feedback_calls"] > 0
                for group in report["groups"]
            )
            report["status"] = "passed" if report["gate_passed"] else "no_training_signal"
    except Exception:
        report["status"] = "failed"
        report["error"] = traceback.format_exc()
    finally:
        if server is not None:
            try:
                os.killpg(server.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        report["elapsed_seconds"] = time.monotonic() - started
        (directory / "report.json").write_text(json.dumps(report, indent=2))
        await traces.commit.aio()
    return report


@app.local_entrypoint()
def main() -> None:
    print(json.dumps(preflight.remote(), indent=2), flush=True)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    report = probe.remote(run_id)
    destination = Path("logs/frozen-probe") / run_id
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
