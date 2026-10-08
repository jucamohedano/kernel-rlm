"""Bounded root-only RLM loop; model requests pass through V1 interception."""

import asyncio
import json
import os
import sys
from pathlib import Path

from kernel_rlm.actions import (
    BACKEND_TIMEOUT_S,
    FEEDBACK_MAX_CALLS,
    FEEDBACK_TIMEOUT_S,
    REPL_TIMEOUT_S,
    repl_actions,
)
from kernel_rlm.feedback import build_local_feedback_bootstrap
from openai import AsyncOpenAI
from rlm.utils.prompts import (
    RLM_SYSTEM_PROMPT,
    QueryMetadata,
    build_rlm_system_prompt,
    build_user_prompt,
)

from rlm_train.repl.subprocess import SubprocessReplBackend


async def run(task: dict) -> None:
    limits = json.loads(os.environ["KERNEL_LIMITS"])
    client = AsyncOpenAI(
        base_url=os.environ["KERNEL_ENDPOINT"],
        api_key=os.environ.pop("KERNEL_API_KEY"),
        max_retries=0,
        timeout=150,
    )
    backend = SubprocessReplBackend(request_timeout=BACKEND_TIMEOUT_S)
    messages = build_rlm_system_prompt(
        RLM_SYSTEM_PROMPT,
        QueryMetadata(task["context"]),
        root_prompt=task["prompt"],
        orchestrator=False,
    )
    record = {
        "submission": "",
        "repl_calls": 0,
        "feedback_calls": 0,
        "turns": 0,
        "messages": messages,
        "termination_reason": "turn_limit",
        "model_calls": [],
    }
    try:
        os.environ["RLM_TRAIN_EXEC_TIMEOUT_S"] = str(REPL_TIMEOUT_S)
        await backend.start(proxy_url="http://127.0.0.1:1", rollout_id="bounded-v1")
        await backend.load_context(task["context"])
        await backend.bootstrap(
            build_local_feedback_bootstrap(
                task["context"], max_calls=FEEDBACK_MAX_CALLS, timeout_s=FEEDBACK_TIMEOUT_S
            )
        )
        for turn in range(limits["max_turns"]):
            turn_prompt = build_user_prompt(
                root_prompt=task["prompt"],
                iteration=turn,
                context_count=1,
                history_count=0,
                max_iterations=limits["max_turns"],
            )
            if messages[-1]["role"] == "user":
                messages[-1]["content"] += "\n\n" + turn_prompt["content"]
            else:
                messages.append(turn_prompt)
            response = await client.chat.completions.create(
                model=os.environ["KERNEL_MODEL"],
                messages=messages,
                max_tokens=limits["max_tokens"],
            )
            choice = response.choices[0]
            raw_message = choice.message.model_dump(exclude_none=True)
            codes, action_errors = repl_actions(raw_message, choice.finish_reason)
            content = choice.message.content or ""
            if raw_message.get("tool_calls"):
                content += "\nStructured REPL actions:\n" + json.dumps(raw_message["tool_calls"])
            messages.append({"role": "assistant", "content": content})
            record["model_calls"].append(
                {"finish_reason": choice.finish_reason, "action_errors": action_errors}
            )
            record["turns"] += 1
            if action_errors:
                messages.append({"role": "user", "content": "\n".join(action_errors)})
            for code in codes:
                result = await backend.execute(code)
                record["repl_calls"] += 1
                observation = "REPL output:\n" + (result.stdout + result.stderr)[:20000]
                for line in result.stdout.splitlines():
                    if line.startswith("CUDA feedback: "):
                        feedback = json.loads(line.removeprefix("CUDA feedback: "))
                        record["feedback_calls"] = feedback["feedback_calls_used"]
                if messages[-1]["role"] == "user":
                    messages[-1]["content"] += "\n\n" + observation
                else:
                    messages.append({"role": "user", "content": observation})
                if result.final_answer is not None:
                    record["submission"] = result.final_answer
                    break
            if record["submission"]:
                record["termination_reason"] = "submitted"
                break
        if not record["submission"] and record["model_calls"][-1]["finish_reason"] == "length":
            record["termination_reason"] = "truncated"
    except Exception as error:
        record["termination_reason"] = "error"
        record["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        Path("submission.json").write_text(json.dumps(record))
        await backend.stop()
        await client.close()


if __name__ == "__main__":
    asyncio.run(run(json.loads(Path(sys.argv[1]).read_text())))
