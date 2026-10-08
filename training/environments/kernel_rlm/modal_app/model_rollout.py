"""One bounded, frozen-model RLM episode with durable traces."""

from __future__ import annotations

import json
import os
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import modal

from .attention_lora import MODEL_ID, REVISION, models
from .kernel_smoke import image as verifier_image
from .rollout_smoke import rollout_image, traces

app = modal.App("kernel-rlm-model-rollout")
MAX_TURNS = 3
MAX_TOKENS = 2048
MAX_FEEDBACK_CALLS = 2
INFERENCE_SECONDS = 600
VERIFY_SECONDS = 180
VLLM_VERSION = "0.26.0"
inference_image = (
    modal.Image.from_registry(
        f"vllm/vllm-openai:v{VLLM_VERSION}",
        setup_dockerfile_commands=["RUN ln -s /usr/bin/python3 /usr/local/bin/python"],
    )
    .entrypoint([])
    .pip_install("vllm==0.26.0", "cloudpickle==3.1.2", "importlib-metadata==8.7.1")
    .env({"HF_HOME": "/models/hf", "HF_HUB_OFFLINE": "1", "VLLM_WORKER_MULTIPROC_METHOD": "spawn"})
)


@app.function(
    image=inference_image,
    volumes={"/models": models},
    env={"HF_HUB_OFFLINE": "0"},
    cpu=2,
    memory=2048,
    timeout=120,
    retries=0,
)
def preflight() -> dict[str, Any]:
    import importlib.util

    from huggingface_hub import hf_hub_download, snapshot_download
    from transformers import AutoTokenizer

    snapshot = Path(
        snapshot_download(
            MODEL_ID,
            revision=REVISION,
            local_files_only=True,
            allow_patterns=["*.json", "*.py", "*.safetensors"],
        )
    )
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    missing = [name for name in set(index.values()) if not (snapshot / name).is_file()]
    assert not missing, f"Missing cached shards: {missing}"
    assert importlib.util.find_spec("vllm.model_executor.models.laguna")
    hf_hub_download(MODEL_ID, "chat_template.jinja", revision=REVISION)
    models.commit()
    tokenizer = AutoTokenizer.from_pretrained(snapshot, trust_remote_code=True)
    rendered = tokenizer.apply_chat_template(
        [{"role": "system", "content": "Use a Python REPL."}, {"role": "user", "content": "Hello"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return {
        "status": "passed",
        "snapshot": str(snapshot),
        "shards": len(set(index.values())),
        "template_probe": rendered,
        "vllm_version": VLLM_VERSION,
    }


@app.function(
    image=inference_image,
    volumes={"/models": models},
    gpu="A100-80GB",
    cpu=4,
    memory=65536,
    timeout=INFERENCE_SECONDS,
    startup_timeout=120,
    retries=0,
    max_containers=1,
    scaledown_window=2,
)
def infer(requests: modal.Queue, responses: modal.Queue, snapshot: str) -> None:
    from vllm import LLM, SamplingParams

    started = time.monotonic()
    try:
        model = LLM(
            model=snapshot,
            tokenizer=snapshot,
            trust_remote_code=True,
            dtype="bfloat16",
            max_model_len=8192,
            max_num_seqs=1,
            max_num_batched_tokens=1024,
            gpu_memory_utilization=0.92,
            enforce_eager=True,
            seed=42,
        )
        responses.put({"ready": True, "load_seconds": time.monotonic() - started})
        for _ in range(MAX_TURNS):
            messages = requests.get(timeout=60)
            if messages is None:
                return
            outputs = model.chat(
                messages,
                SamplingParams(max_tokens=MAX_TOKENS, temperature=0.6, seed=42),
                use_tqdm=False,
            )
            output = outputs[0]
            candidate = output.outputs[0]
            responses.put(
                {
                    "content": candidate.text,
                    "finish_reason": candidate.finish_reason,
                    "prompt_tokens": len(output.prompt_token_ids),
                    "completion_tokens": len(candidate.token_ids),
                }
            )
    except Exception:
        responses.put({"error": traceback.format_exc()})
        raise


@app.function(
    image=verifier_image,
    gpu="A100-80GB",
    cpu=2,
    memory=8192,
    timeout=VERIFY_SECONDS,
    startup_timeout=60,
    retries=0,
    max_containers=1,
    scaledown_window=2,
    block_network=True,
    restrict_modal_access=True,
)
def verify(reference: str, submission: str, benchmark: bool = True) -> dict[str, Any]:
    from kernel_rlm.verifier import verify_cuda

    started = time.monotonic()
    result = verify_cuda(reference, submission, timeout_s=150, benchmark=benchmark)
    result["elapsed_seconds"] = time.monotonic() - started
    return result


@app.function(
    image=rollout_image,
    volumes={"/traces": traces},
    cpu=1,
    memory=2048,
    timeout=900,
    retries=0,
    max_containers=1,
    scaledown_window=2,
)
async def episode(run_id: str, snapshot: str) -> dict[str, Any]:
    from unittest.mock import patch

    from aiohttp import web
    from kernel_rlm import env as kernel_env
    from pydantic_core import to_json
    from verifiers.types import ClientConfig

    trace_path = Path("/traces/model-rollout") / run_id / "trace.json"
    trace: dict[str, Any] = {
        "run_id": run_id,
        "status": "running",
        "model_id": MODEL_ID,
        "model_revision": REVISION,
        "vllm_version": VLLM_VERSION,
        "max_turns": MAX_TURNS,
        "max_tokens_per_call": MAX_TOKENS,
        "model_requests": [],
        "model_responses": [],
        "feedback_verifier_calls": 0,
        "feedback_results": [],
        "verifier_calls": 0,
        "training_updates": 0,
        "trace_path": str(trace_path),
    }
    started = time.monotonic()

    async def save() -> None:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_bytes(to_json(trace, indent=2))
        await traces.commit.aio()

    await save()
    async with modal.Queue.ephemeral() as requests, modal.Queue.ephemeral() as responses:
        worker = await infer.spawn.aio(requests, responses, snapshot)
        environment = None
        runner = None
        try:
            ready = await responses.get.aio(timeout=420)
            if "error" in ready:
                raise RuntimeError(ready["error"])
            trace["inference_startup"] = ready
            await save()

            async def respond(request: web.Request) -> web.Response:
                if len(trace["model_requests"]) >= MAX_TURNS:
                    raise web.HTTPBadRequest(text="Model call budget exhausted")
                body = await request.json()
                trace["model_requests"].append(body)
                await save()
                await requests.put.aio(body["messages"])
                result = await responses.get.aio(timeout=150)
                trace["model_responses"].append(result)
                await save()
                if "error" in result:
                    raise web.HTTPBadGateway(text=result["error"])
                return web.json_response(
                    {
                        "id": f"{run_id}-{len(trace['model_responses'])}",
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": MODEL_ID,
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": result["content"]},
                                "finish_reason": result["finish_reason"],
                            }
                        ],
                        "usage": {
                            "prompt_tokens": result["prompt_tokens"],
                            "completion_tokens": result["completion_tokens"],
                            "total_tokens": result["prompt_tokens"] + result["completion_tokens"],
                        },
                    }
                )

            bridge = web.Application()
            bridge.router.add_post("/v1/chat/completions", respond)

            async def feedback(request: web.Request) -> web.Response:
                if trace["feedback_verifier_calls"] >= MAX_FEEDBACK_CALLS:
                    return web.json_response(
                        {"error": "CUDA feedback call budget exhausted"}, status=429
                    )
                body = await request.json()
                source = str(body.get("source") or "")
                trace["feedback_verifier_calls"] += 1
                result = verify.remote(reference, source, False)
                trace["feedback_results"].append(result)
                await save()
                return web.json_response(result)

            bridge.router.add_post("/kernel-feedback", feedback)
            reference = json.loads(
                Path("/opt/kernel_rlm/vector_add.jsonl").read_text(encoding="utf-8").splitlines()[0]
            )["code"]
            runner = web.AppRunner(bridge)
            await runner.setup()
            site = web.TCPSite(runner, "127.0.0.1", 0)
            await site.start()
            os.environ["KERNEL_SMOKE_KEY"] = "local-bridge-no-provider-key"
            environment = kernel_env.load_environment(
                manifest_path="/opt/kernel_rlm/vector_add.jsonl",
                max_iterations=MAX_TURNS,
                feedback_url=f"http://127.0.0.1:{runner.addresses[0][1]}/kernel-feedback",
                feedback_max_calls=MAX_FEEDBACK_CALLS,
            )
            environment.timeout_seconds = 500

            def score_once(reference: str, submission: str) -> dict[str, Any]:
                if trace["verifier_calls"]:
                    raise RuntimeError("Verifier call budget exhausted")
                trace["verifier_calls"] += 1
                trace["submitted_source"] = submission
                trace_path.write_bytes(to_json(trace, indent=2))
                traces.commit()
                worker.cancel()
                result = verify.remote(reference, submission)
                trace["verification"] = result
                return result

            with patch.object(kernel_env, "verify_cuda", side_effect=score_once):
                output = await environment.run_rollout(
                    input=environment.dataset[0],
                    client=ClientConfig(
                        api_base_url=f"http://127.0.0.1:{runner.addresses[0][1]}/v1",
                        api_key_var="KERNEL_SMOKE_KEY",
                        max_retries=0,
                        timeout=170,
                    ),
                    model=MODEL_ID,
                    sampling_args={"max_tokens": MAX_TOKENS},
                    max_retries=0,
                    state_columns=[
                        "rlm_final_answer",
                        "rlm_iterations",
                        "rlm_repl_calls",
                        "rlm_sub_llm_calls",
                        "kernel_verify",
                        "rlm_history",
                        "rlm_final_repl_outputs",
                        "rlm_backend",
                    ],
                )
            trace["output"] = output
            verification = output.get("kernel_verify") or {}
            trace["status"] = (
                "passed"
                if verification.get("correct")
                and verification.get("extension_used")
                and output.get("rlm_repl_calls", 0) > 0
                else "failed"
            )
        except Exception:
            trace.update(status="failed", error=traceback.format_exc())
        finally:
            await worker.cancel.aio()
            if environment is not None:
                await environment.teardown_rlm()
            if runner is not None:
                await runner.cleanup()
            trace["elapsed_seconds"] = time.monotonic() - started
            await save()
    return json.loads(to_json(trace))


@app.local_entrypoint()
def main(gpu: bool = False) -> None:
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:12]
    directory = Path("logs/model-rollout") / run_id
    directory.mkdir(parents=True, exist_ok=True)
    checked = preflight.remote()
    (directory / "preflight.json").write_text(json.dumps(checked, indent=2))
    print(json.dumps(checked, indent=2))
    if gpu:
        print(f"Trace: kernel-rlm-traces:model-rollout/{run_id}/trace.json", flush=True)
        result = episode.remote(run_id, checked["snapshot"])
        (directory / "trace.json").write_text(json.dumps(result, indent=2))
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "trace": str(directory / "trace.json"),
                    "verifier_calls": result["verifier_calls"],
                    "error": result.get("error"),
                    "verification": result.get("verification"),
                },
                indent=2,
            )
        )
