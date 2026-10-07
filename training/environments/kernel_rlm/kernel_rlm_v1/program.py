"""Bounded root-only RLM loop; model requests pass through V1 interception."""

import asyncio
import json
import os
import sys
from pathlib import Path

from openai import AsyncOpenAI
from rlm.utils.parsing import find_code_blocks
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
    backend = SubprocessReplBackend(request_timeout=35)
    messages = build_rlm_system_prompt(
        RLM_SYSTEM_PROMPT,
        QueryMetadata(task["context"]),
        root_prompt=task["prompt"],
        orchestrator=False,
    )
    record = {"submission": "", "repl_calls": 0, "turns": 0, "messages": messages}
    try:
        await backend.start(proxy_url="http://127.0.0.1:1", rollout_id="bounded-v1")
        await backend.load_context(task["context"])
        for turn in range(limits["max_turns"]):
            messages.append(
                build_user_prompt(
                    root_prompt=task["prompt"],
                    iteration=turn,
                    context_count=1,
                    history_count=0,
                    max_iterations=limits["max_turns"],
                )
            )
            response = await client.chat.completions.create(
                model=os.environ["KERNEL_MODEL"],
                messages=messages,
                max_tokens=limits["max_tokens"],
            )
            content = response.choices[0].message.content or ""
            messages.append({"role": "assistant", "content": content})
            record["turns"] += 1
            for code in find_code_blocks(content):
                result = await backend.execute(code)
                record["repl_calls"] += 1
                messages.append(
                    {
                        "role": "user",
                        "content": "REPL output:\n" + (result.stdout + result.stderr)[:20000],
                    }
                )
                if result.final_answer is not None:
                    record["submission"] = result.final_answer
                    break
            if record["submission"]:
                break
    finally:
        Path("submission.json").write_text(json.dumps(record))
        await backend.stop()
        await client.close()


if __name__ == "__main__":
    asyncio.run(run(json.loads(Path(sys.argv[1]).read_text())))
