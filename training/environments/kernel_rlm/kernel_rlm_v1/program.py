"""Bounded RLM loop; root and optional child requests use V1 interception."""

import asyncio
import json
import os
import sys
import time
import uuid
from pathlib import Path

from kernel_rlm.actions import (
    BACKEND_TIMEOUT_S,
    FEEDBACK_MAX_CALLS,
    FEEDBACK_TIMEOUT_S,
    REPL_TIMEOUT_S,
    REPL_TOOL,
    load_token_counter,
    preparation_error,
    recovery_budget,
    repl_actions,
    submission_turn,
)
from kernel_rlm.capacity import context_exhaustion_details
from kernel_rlm.child_tools import (
    TOOL_CHILD_TIMEOUT_S,
    TOOL_SUBCALL_INSTRUCTION,
    IterativeSubcalls,
    enclosing_timeouts,
)
from kernel_rlm.cuda_docs import DOCS_INSTRUCTION, docs_manifest, hardware_instruction
from kernel_rlm.feedback import build_local_feedback_bootstrap
from kernel_rlm.subcalls import (
    KERNEL_SYSTEM,
    PROBE_INSTRUCTION,
    SUBCALL_INSTRUCTION,
    SUBCALL_TIMEOUT_S,
    BoundedSubcalls,
    turn_budget,
)
from kernel_rlm.thinking import visible_response
from kernel_rlm.verifier_slot import INFERENCE_ADMISSION_TIMEOUT_S, inference_request_slot
from openai import AsyncOpenAI
from rlm_train.proxy import ClientHandle, SubLLMProxy
from rlm_train.repl.base import ExecResult
from rlm_train.repl.subprocess import SubprocessReplBackend

from rlm.utils.prompts import (
    QueryMetadata,
    build_rlm_system_prompt,
    build_user_prompt,
)


async def run(task: dict) -> None:
    limits = json.loads(os.environ["KERNEL_LIMITS"])
    deadline_turn = submission_turn(limits["max_turns"])
    client = AsyncOpenAI(
        base_url=os.environ["KERNEL_ENDPOINT"],
        api_key=os.environ.pop("KERNEL_API_KEY"),
        max_retries=0,
        timeout=limits.get("model_timeout_s", 150),
    )
    max_child_calls = int(limits.get("max_subcalls", 1))
    child_timeout = TOOL_CHILD_TIMEOUT_S if limits.get("child_tools") else SUBCALL_TIMEOUT_S
    subcall_timeout = child_timeout * max_child_calls if limits.get("enable_subcall") else 0
    subcall_instruction = SUBCALL_INSTRUCTION.format(max_calls=max_child_calls)
    child_prompt_tokens = limits.get("child_prompt_tokens", 2048)
    if child_prompt_tokens == 4096:
        subcall_instruction = subcall_instruction.replace(
            "prompt limit is 2,048", "prompt limit is 4,096"
        )
    if limits.get("child_tools"):
        subcall_instruction = TOOL_SUBCALL_INSTRUCTION.format(max_calls=max_child_calls)
    repl_timeout = REPL_TIMEOUT_S + subcall_timeout
    backend_timeout = BACKEND_TIMEOUT_S + subcall_timeout
    if limits.get("child_tools"):
        repl_timeout, backend_timeout = enclosing_timeouts(
            max_child_calls,
            repl_base=REPL_TIMEOUT_S,
            backend_base=BACKEND_TIMEOUT_S,
            feedback_timeout=FEEDBACK_TIMEOUT_S,
            feedback_calls=FEEDBACK_MAX_CALLS,
        )
    backend = SubprocessReplBackend(request_timeout=backend_timeout)
    proxy = None
    task_prompt = task["prompt"]
    if limits.get("enable_subcall"):
        task_prompt = task_prompt.replace(
            "Do not delegate to sub-LLMs.",
            "Sub-LLM calls are available in the REPL; follow the system's bounded delegation instructions.",
        )
        if limits.get("subcall_probe"):
            task_prompt += "\n" + PROBE_INSTRUCTION
    messages = build_rlm_system_prompt(
        KERNEL_SYSTEM,
        QueryMetadata(task["context"]),
        root_prompt=task_prompt,
        orchestrator=False,
    )
    for message in messages:
        message["content"] = message["content"].replace(
            "Each sub-LLM call can handle roughly ~100k tokens at once.", ""
        )
    if not limits.get("enable_subcall"):
        messages[0]["content"] += " Subcalls are disabled for this run."
    record = {
        "submission": "",
        "repl_calls": 0,
        "feedback_calls": 0,
        "turns": 0,
        "messages": messages,
        "termination_reason": "turn_limit",
        "model_calls": [],
        "subcalls": [],
        "recovery": [],
        "action_rejections": [],
        "reasoning_enabled": bool(limits.get("enable_thinking")),
    }
    docs_path = limits.get("cuda_docs")
    hardware_context = ""
    if docs_path:
        record["cuda_docs"] = docs_manifest(Path(docs_path), limits["cuda_docs_sha256"])
        hardware_context = hardware_instruction(
            record["cuda_docs"], os.environ.get("TORCH_CUDA_ARCH_LIST")
        )
        messages[0]["content"] += "\n" + DOCS_INSTRUCTION + "\n" + hardware_context
    last_usage = {}
    last_usage_message_count = 0
    compiler_names = {"load_inline"}
    count_tokens = None
    feedback_log_path = None
    feedback_log_dir = os.environ.get("KERNEL_FEEDBACK_LOG_DIR")
    if feedback_log_dir:
        feedback_log_path = str(Path(feedback_log_dir) / f"feedback-{uuid.uuid4().hex}.jsonl")
        record["feedback_log_path"] = feedback_log_path
    try:
        os.environ["RLM_TRAIN_EXEC_TIMEOUT_S"] = str(repl_timeout)
        os.environ["RLM_TRAIN_READONLY_CONTEXT"] = "1"
        proxy_url = "http://127.0.0.1:1"
        if limits.get("enable_subcall"):
            count_tokens = load_token_counter(os.environ["KERNEL_MODEL"])
            subcalls = BoundedSubcalls(
                client,
                os.environ["KERNEL_MODEL"],
                count_tokens,
                record["subcalls"],
                max_calls=max_child_calls,
                enable_thinking=bool(limits.get("enable_thinking")),
                admission_enabled=not limits.get("bridge_owns_admission", False),
                prompt_tokens=child_prompt_tokens,
                hardware_context=hardware_context,
            )
            if limits.get("child_tools"):
                subcalls = IterativeSubcalls(
                    client,
                    os.environ["KERNEL_MODEL"],
                    count_tokens,
                    record["subcalls"],
                    reference=task["context"],
                    directory=Path(feedback_log_dir) / "child-tools",
                    max_calls=max_child_calls,
                    hardware_context=hardware_context,
                )
            proxy = SubLLMProxy()
            proxy.register(
                "bounded-v1",
                ClientHandle(
                    client=None,
                    model=os.environ["KERNEL_MODEL"],
                    fake_query=subcalls.query,
                    max_concurrent=1,
                ),
            )
            await proxy.start()
            proxy_url = proxy.url
            messages[0]["content"] += (
                "\nDeployment limits override general examples: " + subcall_instruction
            )
        await backend.start(proxy_url=proxy_url, rollout_id="bounded-v1")
        await backend.load_context(task["context"])
        if docs_path:
            docs_log = (
                str(Path(feedback_log_dir) / "docs-retrieval.jsonl") if feedback_log_dir else None
            )
            await backend.bootstrap(
                "from kernel_rlm.cuda_docs import load_docs as _kernel_load_docs\n"
                f"cuda_docs, search_cuda_docs, read_cuda_doc = _kernel_load_docs({docs_path!r}, "
                f"{limits['cuda_docs_sha256']!r}, {docs_log!r})"
            )
        await backend.bootstrap(
            build_local_feedback_bootstrap(
                task["context"],
                max_calls=FEEDBACK_MAX_CALLS,
                timeout_s=FEEDBACK_TIMEOUT_S,
                details_path=feedback_log_path,
            )
        )
        for turn in range(limits["max_turns"]):
            turn_prompt = build_user_prompt(
                root_prompt=task_prompt,
                iteration=turn,
                context_count=1,
                history_count=0,
                max_iterations=limits["max_turns"],
            )
            turn_prompt["content"] += turn_budget(
                turn,
                limits["max_turns"],
                record["feedback_calls"],
                len(record["subcalls"]),
                limits.get("enable_subcall", False),
                max_child_calls,
            )
            if messages[-1]["role"] == "user":
                messages[-1]["content"] += "\n\n" + turn_prompt["content"]
            else:
                messages.append(turn_prompt)
            recovery = turn >= deadline_turn
            requested_tokens = limits["max_tokens"]
            if recovery:
                if count_tokens is None:
                    count_tokens = load_token_counter(os.environ["KERNEL_MODEL"])
                requested_tokens, estimated_prompt = recovery_budget(
                    last_usage,
                    messages[last_usage_message_count:],
                    limits.get("max_context_tokens", 8192),
                    limits["max_tokens"],
                    count_tokens,
                )
                record["recovery"].append(
                    {
                        "turn": turn + 1,
                        "status": "admitted" if requested_tokens else "skipped",
                        "estimated_prompt_tokens": estimated_prompt,
                        "max_tokens": requested_tokens,
                        "reason": "missing_usage"
                        if estimated_prompt is None
                        else "headroom_available"
                        if requested_tokens
                        else "insufficient_context",
                    }
                )
                if not requested_tokens:
                    record["termination_reason"] = (
                        "recovery_no_headroom"
                        if estimated_prompt is not None
                        else "recovery_usage_unavailable"
                    )
                    break
            phase = (
                "recovery"
                if recovery
                else "submission"
                if turn + 1 == deadline_turn
                else "iteration"
            )
            request_args = {
                "model": os.environ["KERNEL_MODEL"],
                "messages": messages,
                "max_tokens": requested_tokens,
                "tools": [REPL_TOOL],
                "tool_choice": "auto",
            }
            if limits.get("enable_thinking"):
                request_args["extra_body"] = {"chat_template_kwargs": {"enable_thinking": True}}
            admission_started = time.monotonic()
            request_started = None
            admission_wait = None
            prompt_chars = len(json.dumps(messages, ensure_ascii=False))
            try:
                async with inference_request_slot(
                    INFERENCE_ADMISSION_TIMEOUT_S,
                    enabled=bool(os.environ.get("KERNEL_INFERENCE_ADMIN_URL"))
                    and not limits.get("bridge_owns_admission", False),
                ) as admission_wait:
                    request_started = time.monotonic()
                    response = await client.chat.completions.create(**request_args)
            except Exception as error:
                exhaustion = context_exhaustion_details(error)
                elapsed_seconds = time.monotonic() - (
                    request_started if request_started is not None else admission_started
                )
                record["model_calls"].append(
                    {
                        "turn": turn + 1,
                        "phase": phase,
                        "requested_max_tokens": requested_tokens,
                        "status": "context_exhausted"
                        if exhaustion is not None
                        else "admission_timeout"
                        if request_started is None and isinstance(error, TimeoutError)
                        else "timeout"
                        if "timeout" in type(error).__name__.lower()
                        else "error",
                        "elapsed_seconds": elapsed_seconds,
                        "request_admission_wait_seconds": admission_wait
                        if admission_wait is not None
                        else elapsed_seconds,
                        "prompt_message_count": len(messages),
                        "prompt_chars": prompt_chars,
                        "error_type": type(error).__name__,
                        "error": str(error)[:500],
                    }
                )
                if exhaustion is not None:
                    record["termination_reason"] = "context_exhausted"
                    record["context_exhaustion"] = exhaustion
                    break
                raise
            choice = response.choices[0]
            raw_message = choice.message.model_dump(exclude_none=True)
            codes, action_errors = repl_actions(raw_message, choice.finish_reason)
            content = visible_response(choice.message.content or "")
            assistant_message = {"role": "assistant", "content": content}
            structured_calls = raw_message.get("tool_calls") or []
            repl_calls = []
            if structured_calls and codes:
                expected_codes = iter(codes)
                for call in structured_calls:
                    function = call.get("function", call)
                    if function.get("name") != "repl":
                        continue
                    arguments = function.get("arguments", "{}")
                    try:
                        arguments = (
                            json.loads(arguments) if isinstance(arguments, str) else arguments
                        )
                    except json.JSONDecodeError:
                        continue
                    code = arguments.get("code") if isinstance(arguments, dict) else None
                    if not isinstance(code, str):
                        continue
                    try:
                        expected_code = next(expected_codes)
                    except StopIteration:
                        break
                    if code.strip() != expected_code:
                        continue
                    call_id = call.get("id") or f"repl_{record['turns']}_{len(repl_calls)}"
                    repl_calls.append(
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {"name": "repl", "arguments": json.dumps({"code": code})},
                        }
                    )
            structured_action = bool(repl_calls) and len(repl_calls) == len(codes)
            if structured_action:
                assistant_message["tool_calls"] = repl_calls
            messages.append(assistant_message)
            usage = response.usage.model_dump() if getattr(response, "usage", None) else {}
            last_usage = usage
            last_usage_message_count = len(messages)
            completion_details = usage.get("completion_tokens_details") or {}
            record["model_calls"].append(
                {
                    "turn": turn + 1,
                    "phase": phase,
                    "requested_max_tokens": requested_tokens,
                    "status": "ok",
                    "elapsed_seconds": time.monotonic() - request_started,
                    "request_admission_wait_seconds": admission_wait,
                    "prompt_message_count": len(messages) - 1,
                    "prompt_chars": prompt_chars,
                    "finish_reason": choice.finish_reason,
                    "action_errors": action_errors,
                    "reasoning_tokens": completion_details.get("reasoning_tokens", 0),
                    "reasoning_characters": len(
                        raw_message.get("reasoning_content") or raw_message.get("reasoning") or ""
                    ),
                    "usage": usage,
                }
            )
            record["turns"] += 1
            for code_index, code in enumerate(codes):
                rejection = preparation_error(code, compiler_names, recovery=recovery)
                if rejection:
                    record["action_rejections"].append({"turn": turn + 1, "error": rejection})
                    result = ExecResult(stderr=rejection)
                else:
                    result = await backend.execute(code)
                record["repl_calls"] += 1
                observation = "REPL output:\n" + (result.stdout + result.stderr)[:20000]
                for line in result.stdout.splitlines():
                    if line.startswith("CUDA feedback: "):
                        feedback = json.loads(line.removeprefix("CUDA feedback: "))
                        record["feedback_calls"] = feedback["feedback_calls_used"]
                if structured_action:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": repl_calls[code_index]["id"],
                            "content": observation,
                        }
                    )
                elif messages[-1]["role"] == "user":
                    messages[-1]["content"] += "\n\n" + observation
                else:
                    messages.append({"role": "user", "content": observation})
                if result.final_answer is not None:
                    record["submission"] = result.final_answer
                    break
            if action_errors:
                messages.append({"role": "user", "content": "\n".join(action_errors)})
            if record["submission"]:
                record["termination_reason"] = "submitted"
                break
        if (
            not record["submission"]
            and record["termination_reason"] == "turn_limit"
            and record["model_calls"][-1]["finish_reason"] == "length"
        ):
            record["termination_reason"] = "truncated"
    except Exception as error:
        record["termination_reason"] = "error"
        record["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        Path("submission.json").write_text(json.dumps(record))
        await backend.stop()
        if proxy is not None:
            await proxy.stop()
        await client.close()


if __name__ == "__main__":
    asyncio.run(run(json.loads(Path(sys.argv[1]).read_text())))
