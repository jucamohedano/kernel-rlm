"""One-GPU shared-verifier test with concurrent frozen-policy rollouts."""

import json
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import modal

from .attention_lora import MODEL_ID, REVISION, models
from .frozen_probe import inference_cache
from .gpu_telemetry import sample_gpu_stats_to_file
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
    cpu=12,
    memory=16384,
    timeout=300,
    retries=0,
    volumes={"/inference-cache": inference_cache},
)
def prepare_guarded_feedback():
    from .verifier_overlap_probe import compile_fixtures

    return compile_fixtures()


@app.function(
    image=image,
    volumes={"/models": models, "/traces": traces, "/inference-cache": inference_cache},
    gpu="A100-80GB",
    cpu=12,
    memory=98304,
    timeout=1260,
    startup_timeout=120,
    retries=0,
    max_containers=1,
)
async def probe(run_id: str, rollouts: int = 4, guarded_feedback: bool = False):
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
    if guarded_feedback:
        from .verifier_overlap_probe import setup_extension_cache

        setup_extension_cache()
    report = {
        "run_id": run_id,
        "title": f"Shared-A100 guarded feedback · {rollouts} frozen vector-add rollouts",
        "scope": f"{rollouts} concurrent frozen-policy vector-add rollouts; no optimizer update",
        "status": "running",
        "gpu_count": 1,
        "rollout_count": rollouts,
        "optimizer_steps": 0,
        "model_id": MODEL_ID,
        "model_revision": REVISION,
        "thinking_enabled": True,
        "guarded_feedback": guarded_feedback,
    }
    started = time.monotonic()
    server = None
    gpu_sampler_stop = asyncio.Event()
    gpu_sampler = asyncio.create_task(
        sample_gpu_stats_to_file(directory / "gpu_stats.jsonl", gpu_sampler_stop)
    )

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
    client = AsyncOpenAI(base_url="http://127.0.0.1:8000/v1", api_key="local", max_retries=0)
    try:
        async with asyncio.timeout(1200):
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
            report["startup_seconds"] = await wait_for_inference(server, timeout_s=700)
            os.environ["KERNEL_INFERENCE_ADMIN_URL"] = "http://127.0.0.1:8000"
            os.environ["PROBE_API_KEY"] = "local"
            os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"
            os.environ["KERNEL_FEEDBACK_LOG_DIR"] = str(directory / "feedback")
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
            if guarded_feedback:
                from .verifier_overlap_probe import check_guarded_feedback

                os.environ["RLM_TRAIN_EXEC_TIMEOUT_S"] = "150"
                report["guarded_feedback_checks"] = await check_guarded_feedback(
                    client, reference, directory / "guarded-repl"
                )
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
                            "enable_thinking": True,
                            "enable_subcall": False,
                            "subcall_probe": False,
                        },
                        "runtime": {"type": "subprocess"},
                        "sampling": {"temperature": 1.0, "max_tokens": 2048},
                        "timeout": {"rollout": 900, "scoring": 300},
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
                episodes = await asyncio.gather(
                    *(environment.run_episode(deepcopy(task), context) for _ in range(rollouts)),
                    return_exceptions=True,
                )
                (directory / "traces.jsonl").write_text(
                    "\n".join(
                        episode.model_dump_json()
                        for episode in episodes
                        if not isinstance(episode, Exception)
                    )
                    + "\n",
                    encoding="utf-8",
                )
                rollout_results = []
                for episode in episodes:
                    if isinstance(episode, Exception):
                        rollout_results.append(
                            {"episode_ok": False, "error": f"{type(episode).__name__}: {episode}"}
                        )
                        continue
                    if not episode.traces:
                        rollout_results.append(
                            {"episode_ok": False, "trace_errors": episode.errors}
                        )
                        continue
                    trace = episode.traces[0].model_dump(mode="json")
                    info = trace.get("info", {})
                    calls = info.get("kernel_model_calls", [])
                    verification = info.get("kernel_verify", {})
                    reward_record = (trace.get("rewards") or {}).get("kernel_reward", {})
                    reward = reward_record.get("score") if isinstance(reward_record, dict) else None
                    if reward is None:
                        reward = info.get("kernel_reward")
                    admission_waits = [
                        float(call.get("request_admission_wait_seconds") or 0) for call in calls
                    ]
                    feedback_results = []
                    for message in info.get("repl_transcript", []):
                        content = message.get("content", "") if isinstance(message, dict) else ""
                        marker = "CUDA feedback: "
                        if marker in content:
                            try:
                                feedback_results.append(
                                    json.loads(content.split(marker, 1)[1].split("\n", 1)[0])
                                )
                            except json.JSONDecodeError:
                                pass
                    verifications = [*feedback_results, verification]
                    rollout_results.append(
                        {
                            "episode_ok": episode.ok,
                            "reward": reward,
                            "termination": info.get("kernel_termination", "unknown"),
                            "model_calls": len(calls),
                            "model_request_timeouts": sum(
                                call.get("status") == "timeout" for call in calls
                            ),
                            "model_admission_timeouts": sum(
                                call.get("status") == "admission_timeout" for call in calls
                            ),
                            "model_request_errors": sum(
                                call.get("status") == "error" for call in calls
                            ),
                            "admission_wait_seconds": admission_waits,
                            "verifier_statuses": [
                                result.get("verification_status") for result in verifications
                            ],
                            "verifier_phases": [
                                result.get("verifier_phase") or result.get("timeout_phase")
                                for result in verifications
                            ],
                            "verifier_paused_seconds": sum(
                                float(
                                    (result.get("timings") or {}).get("inference_paused_seconds")
                                    or 0
                                )
                                for result in verifications
                            ),
                            "compile_seconds": sum(
                                float(
                                    (result.get("timings") or {}).get("compile_elapsed_seconds")
                                    or 0
                                )
                                for result in verifications
                            ),
                            "submitted": bool(info.get("kernel_submission")),
                            "correct": verification.get("correct"),
                            "final_scheduling": verification.get("scheduling"),
                            "max_diff": verification.get("max_diff"),
                            "speedup": verification.get("speedup"),
                            "trace_errors": trace.get("errors", []),
                        }
                    )
                rewards = [
                    result["reward"]
                    for result in rollout_results
                    if result.get("reward") is not None
                ]
                all_waits = [
                    wait
                    for result in rollout_results
                    for wait in result.get("admission_wait_seconds", [])
                ]
                report["rollouts"] = rollout_results
                report["groups"] = [
                    {
                        "task": "thinking-vector-add",
                        "rewards": rewards,
                        "submitted": sum(
                            result.get("submitted", False) for result in rollout_results
                        ),
                        "correct": sum(result.get("correct") is True for result in rollout_results),
                    }
                ]
                report["request_admission_wait_seconds_max"] = max(all_waits, default=0.0)
                report["model_request_timeouts"] = sum(
                    result.get("model_request_timeouts", 0) for result in rollout_results
                )
                report["model_admission_timeouts"] = sum(
                    result.get("model_admission_timeouts", 0) for result in rollout_results
                )
                report["model_request_errors"] = sum(
                    result.get("model_request_errors", 0) for result in rollout_results
                )
                report["total_verifier_paused_seconds"] = sum(
                    result.get("verifier_paused_seconds", 0) for result in rollout_results
                )
                report["total_compile_seconds"] = sum(
                    result.get("compile_seconds", 0) for result in rollout_results
                )
                report["gpu_stats_path"] = str(directory / "gpu_stats.jsonl")
                report["feedback_log_dir"] = str(directory / "feedback")
                feedback_records = [
                    json.loads(line)
                    for path in (directory / "feedback").glob("*.jsonl")
                    for line in path.read_text().splitlines()
                    if line.strip()
                ]
                report["live_feedback_modes"] = [
                    record.get("scheduling", {}).get("mode", "not_started")
                    for record in feedback_records
                ]
                report["gate_passed"] = (
                    len(rollout_results) == rollouts
                    and all(result.get("episode_ok") for result in rollout_results)
                    and report["model_request_timeouts"] == 0
                    and report["model_admission_timeouts"] == 0
                    and report["model_request_errors"] == 0
                )
            report["status"] = "completed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        heartbeat.cancel()
        await asyncio.gather(heartbeat, return_exceptions=True)
        await client.close()
        if server is not None and server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
            try:
                await asyncio.to_thread(server.wait, timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                await asyncio.to_thread(server.wait, timeout=10)
        gpu_sampler_stop.set()
        await gpu_sampler
        report["elapsed_seconds"] = time.monotonic() - started
        (directory / "report.json").write_text(json.dumps(report, indent=2))
        traces.commit()
        inference_cache.commit()
    return report


@app.local_entrypoint()
def main(rollouts: int = 4, cpu_only: bool = False, guarded_feedback: bool = False):
    if rollouts not in (2, 4):
        raise ValueError("Use --rollouts 2 or --rollouts 4 for the shared-A100 gate test")
    print(json.dumps(preflight.remote(), indent=2))
    if not cpu_only:
        if guarded_feedback:
            print(
                json.dumps({"cpu_compile_preflight": prepare_guarded_feedback.remote()}), flush=True
            )
        run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
        result = probe.remote(run_id, rollouts, guarded_feedback)
        directory = Path("logs/shared-gpu-probe") / run_id
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "report.json").write_text(json.dumps(result, indent=2))
        summary = {key: value for key, value in result.items() if key != "guarded_feedback_checks"}
        summary["guarded_feedback_checks"] = [
            {"name": check["name"], "result": check["result"]}
            for check in result.get("guarded_feedback_checks", [])
        ]
        print(json.dumps(summary, indent=2))
        if result["status"] != "completed":
            raise RuntimeError(result.get("error", "Probe failed"))
