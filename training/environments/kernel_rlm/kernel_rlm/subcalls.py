"""One-shot child completions through the same intercepted V1 client."""

import asyncio
import time

CHILD_SYSTEM = (
    "You are a CUDA programming assistant. Answer the requested subtask directly. "
    "You have no REPL, tools, or child models. Do not claim to execute or test code."
)
SUBCALL_TIMEOUT_S = 90
SUBCALL_INSTRUCTION = (
    "You may call llm_query(prompt) inside the REPL once per trajectory. "
    "The child is the same model with a fresh context, no REPL, and no further recursion. "
    "Its prompt limit is 2,048 tokens; its output limit is 2,048 tokens. "
    "Send only the relevant reference or compiler error. Store and inspect the returned "
    "text, then implement and verify the kernel yourself. Calling a child earns no reward. "
    "Batched calls share the same single-call budget; rlm_query is only a one-shot alias."
)
PROBE_INSTRUCTION = (
    "For this subcall wiring probe, use llm_query exactly once from a REPL block before "
    "building the kernel. Ask it for a CUDA implementation approach using context, "
    "store its reply in child_advice, and print(child_advice). Then complete the task."
)


class BoundedSubcalls:
    def __init__(self, client, model, count_tokens, records, timeout_s=SUBCALL_TIMEOUT_S):
        self.client = client
        self.model = model
        self.count_tokens = count_tokens
        self.records = records
        self.timeout_s = timeout_s
        self.used = False

    async def query(self, prompt, state=None):
        if self.used:
            raise ValueError("Subcall budget exhausted: one child call per trajectory")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Subcall prompt must be a non-empty string")
        if len(prompt.encode("utf-8")) > 32768 or self.count_tokens(prompt) > 2048:
            raise ValueError("Subcall prompt exceeds the 2,048-token limit; shorten it")
        self.used = True
        entry = {"prompt": prompt, "model": self.model, "status": "running"}
        self.records.append(entry)
        started = time.monotonic()
        try:
            async with asyncio.timeout(self.timeout_s):
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": CHILD_SYSTEM},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=2048,
                )
            choice = response.choices[0]
            content = choice.message.content or ""
            entry.update(
                status="completed",
                response=content,
                finish_reason=choice.finish_reason,
                usage=response.usage.model_dump() if response.usage else None,
            )
            if choice.finish_reason == "length":
                return "Child output was truncated; treat it as incomplete advice.\n" + content
            return content
        except BaseException as error:
            entry.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            entry["elapsed_seconds"] = time.monotonic() - started
