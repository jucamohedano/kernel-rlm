"""Bounded child-only streaming with exact tokens and content-free progress events."""

import json
import time

from kernel_rollout.contract import exact_sample


async def read_child_stream(response, prompt_ids, max_tokens, context_tokens, progress):
    started = time.monotonic()
    result = {
        "object": "chat.completion",
        "prompt_token_ids": None,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": ""},
                "token_ids": [],
                "logprobs": {"content": []},
                "finish_reason": None,
            }
        ],
    }
    choice = result["choices"][0]
    last_reported = 0
    done = False
    first_token_seconds = None
    async for raw_line in response.content:
        line = raw_line.decode("utf-8").strip()
        if not line or line.startswith(":"):
            continue
        if not line.startswith("data: "):
            raise ValueError("Invalid child SSE event")
        payload = line[6:]
        if payload == "[DONE]":
            done = True
            break
        chunk = json.loads(payload)
        if "error" in chunk:
            raise RuntimeError(f"Child stream failed: {chunk['error']}")
        for key in ("id", "created", "model"):
            if key in chunk:
                result[key] = chunk[key]
        if chunk.get("prompt_token_ids") is not None:
            if chunk["prompt_token_ids"] != prompt_ids:
                raise ValueError("Child stream prompt IDs differ from tokenization")
            result["prompt_token_ids"] = chunk["prompt_token_ids"]
        if chunk.get("usage") is not None:
            result["usage"] = chunk["usage"]
        choices = chunk.get("choices", [])
        if len(choices) > 1 or (choices and choices[0].get("index") != 0):
            raise ValueError("Expected one child stream choice")
        if not choices:
            continue
        update = choices[0]
        delta = update.get("delta", {})
        if delta.get("tool_calls"):
            raise ValueError("Children cannot request tools")
        for key in ("content", "reasoning", "reasoning_content"):
            if delta.get(key) is not None:
                choice["message"][key] = choice["message"].get(key, "") + delta[key]
        tokens = update.get("token_ids") or []
        entries = (update.get("logprobs") or {}).get("content") or []
        if len(tokens) != len(entries):
            raise ValueError("Child stream token/logprob alignment violated")
        choice["token_ids"].extend(tokens)
        choice["logprobs"]["content"].extend(entries)
        count = len(choice["token_ids"])
        if count > max_tokens:
            raise ValueError("Child stream exceeded its output cap")
        if tokens and (first_token_seconds is None or count - last_reported >= 256):
            elapsed = time.monotonic() - started
            if first_token_seconds is None:
                first_token_seconds = elapsed
            progress(
                {
                    "stage": "generating",
                    "completion_tokens": count,
                    "first_token_seconds": first_token_seconds,
                    "backend_elapsed_seconds": elapsed,
                }
            )
            last_reported = count
        if update.get("finish_reason") is not None:
            choice["finish_reason"] = update["finish_reason"]
    if not done:
        raise ValueError("Child stream ended without DONE; partial advice is unusable")
    sample = exact_sample(result, context_tokens)
    usage = result.get("usage", {})
    if usage.get("completion_tokens") != len(sample.response_ids) or usage.get(
        "prompt_tokens"
    ) != len(sample.prompt_ids):
        raise ValueError("Child stream usage differs from exact token counts")
    progress(
        {
            "stage": "generated",
            "completion_tokens": len(sample.response_ids),
            "first_token_seconds": first_token_seconds,
            "backend_elapsed_seconds": time.monotonic() - started,
        }
    )
    return result
