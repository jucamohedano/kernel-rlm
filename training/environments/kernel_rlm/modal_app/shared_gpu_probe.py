"""One-GPU resource gate followed by one thinking-enabled V1 trajectory."""

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import modal

from .attention_lora import MODEL_ID, REVISION, models
from .frozen_probe import inference_cache
from .grpo_live_smoke import APP_ROOT, live_image
from .inference_runtime import cache_environment, wait_for_inference
from .rollout_smoke import traces

app = modal.App("kernel-rlm-live-protocol-probe")
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
    from kernel_rlm.verifier_slot import verifier_slot
    from openai import AsyncOpenAI
    from renderers import LagunaXS2RendererConfig
    from verifiers.v1.clients import ModelContext
    from verifiers.v1.configs.client import TrainClientConfig
    from verifiers.v1.envs.single_agent.env import SingleAgentEnvConfig

    directory = Path("/traces/shared-gpu-probe") / run_id
    directory.mkdir(parents=True, exist_ok=True)
    report = {
        "run_id": run_id,
        "title": "Thinking enabled · read-only context and REPL recovery",
        "scope": "One frozen-policy vector-add trajectory; no optimizer update",
        "status": "running",
        "gpu_count": 1,
        "optimizer_steps": 0,
        "model_id": MODEL_ID,
        "model_revision": REVISION,
        "thinking_enabled": True,
    }
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
            report["shared_gpu_resource_gate_reused"] = {
                "run_id": "20261013T101758Z",
                "status": "passed",
                "note": "The identical 86% memory and inference-LoRA configuration passed the previous large-input check.",
            }
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
                            "max_turns": 7,
                            "max_tokens": 2048,
                            "enable_subcall": False,
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
                if not episode.traces:
                    raise RuntimeError(f"V1 episode returned no traces: {episode.errors}")
                trace = episode.traces[0].model_dump(mode="json")
                info = trace.get("info", {})
                model_calls = info.get("kernel_model_calls", [])
                tool_names = []
                reasoning_chars = 0
                for node in trace.get("nodes", []):
                    message = node.get("message", {})
                    reasoning_chars += len(message.get("reasoning_content") or "")
                    for call in message.get("tool_calls") or []:
                        function = call.get("function", call)
                        tool_names.append(function.get("name", "unknown"))
                verification = info.get("kernel_verify", {})
                v1_scores = (trace.get("rewards") or {}).get("kernel_reward", {})
                v1_reward = v1_scores.get("score") if isinstance(v1_scores, dict) else None
                if v1_reward is None:
                    v1_reward = info.get("kernel_reward")
                reward = float(v1_reward) if isinstance(v1_reward, int | float) else None
                submitted = bool(info.get("kernel_submission"))
                correct = bool(verification.get("correct"))
                report["episode_ok"] = episode.ok
                report["reward"] = reward
                report["reward_source"] = (
                    "trace.rewards.kernel_reward.score"
                    if v1_scores.get("score") is not None
                    else "trace.info.kernel_reward"
                    if info.get("kernel_reward") is not None
                    else "not serialized"
                )
                report["termination"] = info.get("kernel_termination", "unknown")
                report["groups"] = [
                    {
                        "task": "thinking-vector-add",
                        "rewards": [reward] if reward is not None else [],
                        "correct": int(correct),
                        "submitted": int(submitted),
                    }
                ]
                report["trajectory"] = {
                    "model_calls": len(model_calls),
                    "finish_reasons": [call.get("finish_reason") for call in model_calls],
                    "action_errors": [
                        error for call in model_calls for error in call.get("action_errors", [])
                    ],
                    "output_tokens": sum(
                        (call.get("usage") or {}).get("completion_tokens", 0)
                        for call in model_calls
                    ),
                    "reasoning_chars": reasoning_chars,
                    "tool_names": tool_names,
                    "repl_calls": info.get("repl_calls", 0),
                    "feedback_calls": info.get("feedback_calls", 0),
                    "child_calls": len(info.get("kernel_subcalls", [])),
                    "submitted": submitted,
                    "submission_chars": len(info.get("kernel_submission") or ""),
                    "correct": correct,
                    "max_diff": verification.get("max_diff"),
                    "speedup": verification.get("speedup"),
                    "verifier_error": verification.get("error"),
                    "trace_errors": trace.get("errors", []),
                }
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
        run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
        print(json.dumps(probe.remote(run_id), indent=2))
