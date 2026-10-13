"""One-GPU resource gate followed by one thinking-enabled V1 trajectory."""

import json
from datetime import UTC, datetime
from pathlib import Path

import modal

from .attention_lora import MODEL_ID, REVISION, models
from .frozen_probe import inference_cache
from .grpo_live_smoke import APP_ROOT, live_image
from .inference_runtime import cache_environment, wait_for_inference
from .rollout_smoke import traces

app = modal.App("kernel-rlm-shared-gpu-probe")
image = live_image.add_local_file(
    APP_ROOT / "examples/vector_add_submission.py", "/root/known_cuda.py", copy=True
)


@app.function(image=image, timeout=180, cpu=2, memory=8192)
def preflight():
    import tomllib

    from prime_rl.configs.rl import RLConfig
    from renderers import LagunaXS2RendererConfig

    from .grpo_live_smoke import patch_prime_laguna_config

    patch_prime_laguna_config()
    config = tomllib.loads(Path("/root/configs/grpo_level1.toml").read_text())
    inference = RLConfig.model_validate(config).inference
    thinking = LagunaXS2RendererConfig(enable_thinking=True)
    return {"inference": inference.model_dump(mode="json"), "thinking": thinking.model_dump()}


@app.function(
    image=image,
    volumes={"/models": models, "/traces": traces, "/inference-cache": inference_cache},
    gpu="A100-80GB",
    cpu=12,
    memory=98304,
    timeout=720,
    startup_timeout=120,
    retries=0,
    max_containers=1,
)
async def probe(run_id: str):
    import asyncio
    import os
    import signal
    import subprocess
    import time
    from importlib.metadata import version

    import verifiers.v1 as vf
    from huggingface_hub import snapshot_download
    from kernel_rlm.gpu_sharing import paused_inference
    from kernel_rlm.verifier import verify_cuda
    from kernel_rlm.verifier_slot import verifier_slot
    from openai import AsyncOpenAI
    from renderers import LagunaXS2RendererConfig
    from verifiers.v1.clients import ModelContext
    from verifiers.v1.configs.client import TrainClientConfig
    from verifiers.v1.envs.single_agent.env import SingleAgentEnvConfig

    directory = Path("/traces/shared-gpu-probe") / run_id
    directory.mkdir(parents=True, exist_ok=True)
    report = {"run_id": run_id, "status": "running", "gpu_count": 1, "optimizer_steps": 0}
    started = time.monotonic()
    server = None
    client = AsyncOpenAI(base_url="http://127.0.0.1:8000/v1", api_key="local", max_retries=0)
    try:
        async with asyncio.timeout(660):
            snapshot = snapshot_download(MODEL_ID, revision=REVISION, local_files_only=True)
            config = {
                "router": None,
                "server": {"host": "127.0.0.1", "port": 8000},
                "vllm": {
                    "model": snapshot,
                    "served_model_name": [MODEL_ID],
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
            (directory / "inference.json").write_text(json.dumps(config))
            identity = {
                "model_revision": REVISION,
                "gpu": "A100-80GB-sm80",
                **{name: version(name) for name in ("vllm", "torch", "triton")},
            }
            cache = cache_environment(Path("/inference-cache"), identity)
            command = [
                "python",
                "-c",
                "import sys; from pathlib import Path; "
                "from prime_rl.configs.inference import InferenceConfig; "
                "from prime_rl.inference.vllm.server import server; "
                "server(InferenceConfig.model_validate_json(Path(sys.argv[1]).read_text()))",
                str(directory / "inference.json"),
            ]
            with (directory / "inference.log").open("w") as output:
                server = subprocess.Popen(
                    command,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    env={**os.environ, **cache, "CUDA_VISIBLE_DEVICES": "0"},
                    start_new_session=True,
                )
            report["startup_seconds"] = await wait_for_inference(server, timeout_s=420)
            os.environ["KERNEL_INFERENCE_ADMIN_URL"] = "http://127.0.0.1:8000"
            os.environ["PROBE_API_KEY"] = "local"
            os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"
            with verifier_slot(20) as remaining:
                with paused_inference(remaining):
                    pending = asyncio.create_task(
                        client.chat.completions.create(
                            model=MODEL_ID,
                            messages=[{"role": "user", "content": "Reply OK."}],
                            max_tokens=16,
                            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                        )
                    )
                    await asyncio.sleep(1)
                    assert not pending.done(), "Generation did not pause"
            await asyncio.wait_for(pending, 30)
            report["pause_resume_passed"] = True
            reference = Path("/root/kernel_rlm/tasks/vector_add.py").read_text()
            submission = Path("/root/known_cuda.py").read_text()
            result = await asyncio.to_thread(
                verify_cuda,
                reference.replace("4096", "536870912"),
                submission,
                n_trials=1,
                timeout_s=120,
            )
            report["large_verification"] = result
            if not result["correct"] or not result["speedup"]:
                raise RuntimeError("Shared GPU verification failed")
            (directory / "report.json").write_text(json.dumps(report, indent=2))
            traces.commit()
            manifest = directory / "manifest.jsonl"
            manifest.write_text(json.dumps({"_task_id": "thinking-vector-add", "code": reference}))
            env_config = SingleAgentEnvConfig.model_validate(
                {
                    "taskset": {"id": "kernel_rlm_v1", "manifest_path": str(manifest)},
                    "agent": {
                        "harness": {
                            "id": "kernel_rlm_v1",
                            "max_turns": 5,
                            "max_tokens": 2048,
                            "enable_subcall": True,
                            "subcall_probe": False,
                        },
                        "runtime": {"type": "subprocess"},
                        "sampling": {"temperature": 1.0, "max_tokens": 2048},
                        "timeout": {"rollout": 180, "scoring": 60},
                    },
                }
            )
            context = ModelContext(
                model=MODEL_ID,
                client=TrainClientConfig(
                    base_url="http://127.0.0.1:8000/v1",
                    api_key_var="PROBE_API_KEY",
                    renderer_model_name=snapshot,
                    renderer=LagunaXS2RendererConfig(enable_thinking=True),
                ),
            )
            environment = vf.load_environment(env_config)
            async with environment.serving():
                task = next(iter(vf.load_taskset(env_config.taskset)))
                episode = await environment.run_episode(task, context)
                (directory / "traces.jsonl").write_text(episode.model_dump_json() + "\n")
                report["episode_ok"] = episode.ok
                report["reward"] = episode.traces[0].reward
                report["termination"] = episode.traces[0].info.get("kernel_termination")
            report["status"] = "completed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        await client.close()
        if server is not None and server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
            try:
                await asyncio.to_thread(server.wait, timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                await asyncio.to_thread(server.wait, timeout=10)
        report["elapsed_seconds"] = time.monotonic() - started
        (directory / "report.json").write_text(json.dumps(report, indent=2))
        traces.commit()
        inference_cache.commit()
    return report


@app.local_entrypoint()
def main(cpu_only: bool = False):
    print(json.dumps(preflight.remote(), indent=2))
    if not cpu_only:
        print(json.dumps(probe.remote(datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")), indent=2))
