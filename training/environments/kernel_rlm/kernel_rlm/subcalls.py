"""One-shot child completions through the same intercepted V1 client."""

import asyncio
import os
import time

from kernel_rlm.actions import submission_turn
from kernel_rlm.thinking import visible_response
from kernel_rlm.verifier_slot import INFERENCE_ADMISSION_TIMEOUT_S, inference_request_slot

CHILD_SYSTEM = (
    "You are a CUDA programming assistant. Answer the requested subtask directly. "
    "You have no REPL, tools, or child models. Do not claim to execute or test code."
)
SUBCALL_TIMEOUT_S = 90
SUBCALL_MAX_CALLS = 2
SUBCALL_REPLY_TOKENS = 4096
SUBCALL_INSTRUCTION = (
    "You may call llm_query(prompt) inside the REPL up to {max_calls} times per trajectory. "
    "The child is the same model with a fresh context, no REPL, and no further recursion. "
    "Its prompt limit is 2,048 tokens; its output limit is 4,096 tokens including reasoning. "
    "Solve simple single-output elementwise tasks directly. If the reference has registered "
    "parameters or buffers, non-tensor constructor arguments, structured outputs, or a "
    "nontrivial reduction or scan, make one focused child call before coding. Ask it to check "
    "the runtime-versus-constructor inputs, output structure, state-dict names and shapes, "
    "or one specific indexing/reduction detail. If the configured call budget has a second "
    "call available, use it only for a distinct compiler or numerical issue before the deadline. "
    "Delegate a specific subproblem, not the entire kernel: reference semantics such as "
    "padding or reduction axes; indexing (mapping threads to tensor elements, strides, "
    "and boundary checks); reduction or scan design; numerical stability; synchronization; "
    "a compiler error; or memory-access efficiency. Give an explicit question, the relevant "
    "code, and the expected answer format. For a contract review, ask for a short checklist, "
    "not a kernel. The child may return a code fragment with a brief explanation or concise "
    "text. Never send bare context without a question. Store the reply "
    "in a REPL variable and inspect only what you need; do not print long child responses into "
    "the root conversation. You remain responsible for assembling and verifying the complete "
    "kernel. A child's advice is untested: integrate it into the complete source and use "
    "cuda_feedback(source) for compilation and correctness; final verification measures speed. "
    "Calling a child earns no reward. Batched calls share this {max_calls}-call budget; "
    "rlm_query is only a one-shot alias.\n"
    "A useful child checks a separable part of your plan while you retain responsibility "
    "for the complete implementation. Specify inputs, outputs, assumptions, your proposed "
    "approach, and the exact deliverable. Ask for a derivation, boundary cases, a state "
    "contract review, or a minimal repair to one issue, not a replacement full kernel. "
    "When searchable CUDA documentation is available, pass only a short relevant excerpt. "
    "Example of delegating an indexing review (illustrative, not a task solution):\n"
    "```repl\n"
    "child_advice = llm_query(\n"
    "    'Review only the indexing for pixel-unshuffle on contiguous NCHW tensors. '\n"
    "    'Input [1,3,8,8], factor 2, output [1,12,4,4]. '\n"
    "    'My plan assigns one thread to each output element. '\n"
    "    'Derive output coordinates to input offset; check channel ordering and bounds. '\n"
    "    'Return a compact formula and three boundary test cases, not a full kernel. '\n"
    "    'Do not claim to run code; I will test and integrate your advice.'\n"
    ")\n"
    "print(child_advice[:1200])\n"
    "```\n"
    "Keep the full reply in child_advice; inspect another short slice only if needed. "
    "Do not call a child on the submission deadline or during recovery."
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
    "Preserve the best source and submit when ready, within the current run's turn budget. "
    "Keep the submission action short and reuse the stored source. "
    "Any additional turns are for submission recovery, not more kernel development."
)


def turn_budget(turn, max_turns, feedback_used, child_calls_used, enabled, max_child_calls=1):
    remaining = max(0, 2 - feedback_used)
    child_calls_remaining = max(0, max_child_calls - child_calls_used) if enabled else 0
    deadline = submission_turn(max_turns)
    current = turn + 1
    if current >= deadline:
        remaining = 0
        child_calls_remaining = 0
    message = (
        f"\nTurn {current}/{max_turns}. "
        f"Feedback calls remaining: {remaining}. Child calls remaining: {child_calls_remaining}."
    )
    if remaining == 0:
        message += " Do not call cuda_feedback again."
    if max_turns < 10:
        message += f" Submit by turn {deadline} at the latest."
    if max_turns < 10 and current == deadline - 2:
        message += " Only two turns remain before submission is due. Store your best source now."
    if max_turns < 10 and current == deadline - 1:
        message += (
            " NEXT TURN is the submission deadline. Reuse the stored source; avoid a full rewrite."
        )
    if current == deadline:
        message += (
            " SUBMISSION DEADLINE: submit your best stored complete source now with "
            "answer['content'] = source; answer['ready'] = True. "
            "Do not request more feedback or delegation. Do not just print the source."
        )
    if current > deadline:
        message += (
            " RECOVERY ONLY: make a short submission using an existing source variable. "
            "Only answer assignments are allowed; no imports, feedback, delegation, "
            "or rebuilding the kernel."
        )
    if current == max_turns:
        message += " FINAL TURN: set answer['content'] = source and answer['ready'] = True now."
    return message


class BoundedSubcalls:
    def __init__(
        self,
        client,
        model,
        count_tokens,
        records,
        timeout_s=SUBCALL_TIMEOUT_S,
        max_calls=SUBCALL_MAX_CALLS,
        enable_thinking=False,
        admission_enabled=True,
        prompt_tokens=2048,
        hardware_context="",
    ):
        if prompt_tokens not in {2048, 4096}:
            raise ValueError("Child prompt allowance must be 2048 or 4096 tokens")
        self.client = client
        self.model = model
        self.count_tokens = count_tokens
        self.records = records
        self.timeout_s = timeout_s
        self.max_calls = max_calls
        self.enable_thinking = enable_thinking
        self.admission_enabled = admission_enabled
        self.prompt_tokens = prompt_tokens
        self.hardware_context = hardware_context
        self.calls_used = 0

    async def query(self, prompt, state=None):
        if self.calls_used >= self.max_calls:
            raise ValueError("Subcall budget exhausted for this trajectory")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Subcall prompt must be a non-empty string")
        if self.hardware_context:
            prompt = self.hardware_context + "\nAssignment:\n" + prompt
        if (
            len(prompt.encode("utf-8")) > self.prompt_tokens * 16
            or self.count_tokens(prompt) > self.prompt_tokens
        ):
            raise ValueError(
                f"Subcall prompt exceeds the {self.prompt_tokens:,}-token limit; shorten it"
            )
        self.calls_used += 1
        entry = {"prompt": prompt, "model": self.model, "status": "running"}
        entry["call_index"] = self.calls_used
        self.records.append(entry)
        started = time.monotonic()
        try:
            async with inference_request_slot(
                INFERENCE_ADMISSION_TIMEOUT_S,
                enabled=self.admission_enabled
                and bool(os.environ.get("KERNEL_INFERENCE_ADMIN_URL")),
            ) as admission_wait:
                entry["admission_wait_seconds"] = admission_wait
                async with asyncio.timeout(self.timeout_s):
                    request_args = {
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": CHILD_SYSTEM},
                            {"role": "user", "content": prompt},
                        ],
                        "max_tokens": SUBCALL_REPLY_TOKENS,
                    }
                    if self.enable_thinking:
                        request_args["extra_body"] = {
                            "chat_template_kwargs": {"enable_thinking": True}
                        }
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
