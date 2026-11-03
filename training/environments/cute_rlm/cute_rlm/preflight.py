"""GPU preflight helpers. Import and CPU tests do not start a server or inference."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from pathlib import Path

from rlm_train.token_budget import VllmPromptCounter


def kv_cache_evidence(server_log: str) -> dict:
    matches = re.findall(r"GPU KV cache size:\s*([\d,]+)\s*tokens", server_log)
    if not matches:
        raise ValueError("Server log has no observed GPU KV-cache capacity")
    return {
        "capacity_tokens": int(matches[-1].replace(",", "")),
        "log_sha256": hashlib.sha256(server_log.encode()).hexdigest(),
        "scope": "Reported capacity; concurrent long requests still require a live check",
    }


async def probe_two_long_requests(client, config, messages, evidence_dir: Path) -> dict:
    """Execute only when explicitly invoked on the later GPU host.

    ``messages`` supplies two representative near-window prompts; no benchmark
    task is silently used. Each request must occupy at least 90% of the configured
    window. This checks the server's actual behavior, not startup arguments alone.
    """
    if len(messages) != 2:
        raise ValueError("Provide exactly two long probe conversations")
    counter = VllmPromptCounter(client, config.server_context_tokens)
    payloads = []
    for conversation in messages:
        payload = {
            "model": config.model_id,
            "messages": conversation,
            "temperature": 0,
            "max_tokens": 256,
            "logprobs": True,
            "top_logprobs": 0,
            "return_tokens_as_token_ids": True,
            "chat_template_kwargs": {"enable_thinking": True},
        }
        count = await counter(payload)
        if (
            count < int(config.server_context_tokens * 0.9)
            or count + 256 > config.server_context_tokens
        ):
            raise ValueError("Probe prompts must fit near the full configured window")
        payloads.append((payload, count))
    evidence_dir.mkdir(parents=True, exist_ok=False)

    async def one(index, payload, count):
        started = time.monotonic()
        try:
            response = await client.post("chat/completions", json=payload)
            response.raise_for_status()
            raw = response.json()
        except Exception as error:
            record = {
                "request": payload,
                "rendered_prompt_tokens": count,
                "elapsed_s": time.monotonic() - started,
                "exception_type": type(error).__name__,
                "exception": str(error),
                "http_status": getattr(getattr(error, "response", None), "status_code", None),
                "response_body": getattr(getattr(error, "response", None), "text", None),
                "scope": "failed capacity probe; no inference-OOM classification inferred",
            }
            (evidence_dir / f"long-request-{index}-failed.json").write_text(json.dumps(record))
            raise
        if raw.get("usage", {}).get("prompt_tokens") != count:
            raise ValueError("Probe tokenizer count differs from actual prompt usage")
        choices = raw.get("choices") or []
        if not choices or not (choices[0].get("logprobs") or {}).get("content"):
            raise ValueError("Probe did not return selected-token log probabilities")
        record = {
            "rendered_prompt_tokens": count,
            "request": payload,
            "response": raw,
            "elapsed_s": time.monotonic() - started,
        }
        path = evidence_dir / f"long-request-{index}.json"
        path.write_text(json.dumps(record))
        return {
            "prompt_tokens": count,
            "evidence": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    results = await asyncio.gather(
        *(one(i, payload, count) for i, (payload, count) in enumerate(payloads))
    )
    return {
        "status": "passed",
        "simultaneous_submissions": 2,
        "execution_overlap": "requires scheduler/metrics evidence; response success alone is insufficient",
        "requests": results,
        "scope": "Observed simultaneous long-request submission; inspect scheduler metrics for execution overlap. Correctness and isolation are separate checks",
    }


def inference_concurrency_after_preflight(result: dict, *, requested: int = 2) -> dict:
    """Select capacity before freezing a scored run; never replay scored requests.

    The caller supplies a structured outcome from an actual inference preflight,
    not a guessed error substring. Compilation failures and timeouts are not OOM.
    """
    if requested not in (1, 2):
        raise ValueError("this preset supports one or two concurrent inference requests")
    if result.get("status") == "passed":
        effective, reason = requested, "preflight_passed"
    elif (
        result.get("failure_stage") == "model_inference"
        and result.get("error_type") == "cuda_out_of_memory"
    ):
        effective, reason = 1, "inference_preflight_cuda_oom"
    else:
        raise ValueError("preflight did not establish inference capacity")
    return {
        "requested_concurrency": requested,
        "effective_concurrency": effective,
        "reason": reason,
        "selection_stage": "before_frozen_scored_dispatch",
        "requires_serial_recheck": reason == "inference_preflight_cuda_oom",
        "automatic_scored_retry": False,
    }
