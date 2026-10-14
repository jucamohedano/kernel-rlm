"""One-shot child completions through the same intercepted V1 client."""

import asyncio
import time

from kernel_rlm.thinking import visible_response

CHILD_SYSTEM = (
    "You are a CUDA programming assistant. Answer the requested subtask directly. "
    "You have no REPL, tools, or child models. Do not claim to execute or test code."
)
SUBCALL_TIMEOUT_S = 90
SUBCALL_INSTRUCTION = (
    "You may call llm_query(prompt) inside the REPL once per trajectory. "
    "The child is the same model with a fresh context, no REPL, and no further recursion. "
    "Its prompt limit is 2,048 tokens; its output limit is 2,048 tokens. "
    "Delegate a specific subproblem, not the entire kernel: indexing, synchronization, "
    "a compiler error, or a memory-access choice. Give an explicit question, the relevant "
    "code, and the expected answer format. Never send bare context without a question. "
    "For example: llm_query('Explain this indexing error; return a corrected expression "
    "and a short explanation. Fragment: ' + fragment). Store and inspect the returned "
    "text, then implement and verify the kernel yourself. Calling a child earns no reward. "
    "Batched calls share the same single-call budget; rlm_query is only a one-shot alias."
)
PROBE_INSTRUCTION = (
    "For this subcall wiring probe, use llm_query exactly once from a REPL block before "
    "building the kernel. Ask a specific question about indexing or bounds using context, "
    "store its reply in child_advice, and print(child_advice). Then complete the task."
)

KERNEL_SYSTEM = (
    "You solve CUDA tasks through a persistent Python REPL. The only available tool is "
    "repl, which accepts one JSON string argument named code. Use it for Python actions. "
    "Never call python, bash, or another tool name. You may instead put Python inside a "
    "fenced repl block. Do not mix the two action formats in one response. The "
    "read-only variable context is a string containing reference Python source, "
    "not a module, Model instance, or callable; Model and get_inputs are not REPL "
    "objects. Never use import context, context.forward, Model(), call get_inputs(), "
    "or execute the reference. Inspect it with print(context) or ast.parse(context), "
    "and infer the signature, inputs, and outputs from the source text. Treat context "
    "and context_0 as read-only; never assign to either name. "
    "Preserve the reference's input-mutation and output behavior. "
    "Observations arrive in "
    "the next user message. Do not invent tool calls or results. "
    "Prepare kernel source as a string and use cuda_feedback(source) for compiler and "
    "correctness feedback. The REPL disables exec, eval, and compile; use ast.parse "
    "for syntax checks. Submit with answer['content'] = source and answer['ready'] = True "
    "inside the REPL. Do not submit just a variable name or prose. "
    "Preserve the best source and reserve the final turn for submission."
)


def turn_budget(turn, max_turns, feedback_used, child_used, enabled):
    remaining = max(0, 2 - feedback_used)
    message = (
        f"\nTurn {turn + 1}/{max_turns}. Feedback calls remaining: {remaining}. "
        f"Child calls remaining: {int(enabled and not child_used)}."
    )
    if remaining == 0:
        message += " Do not call cuda_feedback again."
    if turn + 1 == max_turns:
        message += (
            " FINAL TURN: make any final repair and submit your best complete source now "
            "with answer['content'] = source; answer['ready'] = True. "
            "Do not request more feedback or delegation."
        )
    return message


class BoundedSubcalls:
    def __init__(
        self,
        client,
        model,
        count_tokens,
        records,
        timeout_s=SUBCALL_TIMEOUT_S,
        enable_thinking=False,
    ):
        self.client = client
        self.model = model
        self.count_tokens = count_tokens
        self.records = records
        self.timeout_s = timeout_s
        self.enable_thinking = enable_thinking
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
                request_args = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": CHILD_SYSTEM},
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": 2048,
                }
                if self.enable_thinking:
                    request_args["extra_body"] = {"chat_template_kwargs": {"enable_thinking": True}}
                response = await self.client.chat.completions.create(**request_args)
            choice = response.choices[0]
            content = visible_response(choice.message.content or "")
            if not content.strip() or getattr(choice.message, "tool_calls", None):
                raise ValueError(
                    "Child must return text advice, not an empty response or tool call"
                )
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
