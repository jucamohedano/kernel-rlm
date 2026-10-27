"""Opt-in child compile/repair sessions with aggregate token and tool budgets."""

import asyncio
import ctypes
import json
import os
import re
import signal
import sys
import time
import uuid
from pathlib import Path

from kernel_rlm.submission_contract import validate_submission_source
from kernel_rlm.thinking import visible_response

TOOL_CHILD_SYSTEM = (
    "You are a bounded CUDA research child with compiler/correctness feedback. "
    "No shell, Python REPL, recursion or reference edits. Do not JSON-escape source code. "
    "To test, write CHECK on its own first line, then one fenced python block containing "
    "the complete candidate Python source, using real newlines and normal triple quotes. "
    "The controller compiles and checks against the immutable reference; inspect feedback and repair. "
    "At most two checks, then write FINISH on its own first line, concise findings, "
    "and one fenced python block containing your best complete candidate. "
    "Resource faults/timeouts are infrastructure, not numerical failure. Never invent measurements. "
    "Final grading is independent. You have CHILD_TOTAL_TOKENS generated tokens TOTAL including reasoning "
    "across this entire session, and CHILD_REPLY_TOKENS per model reply."
)
TOOL_CHILD_TIMEOUT_S = 480
TOOL_SUBCALL_INSTRUCTION = (
    "You may call llm_query(assignment) inside the REPL up to {max_calls} times per trajectory. "
    "Each child has an isolated conversation, no shell or Python REPL, and no recursion. "
    "It may compile/check/repair a complete CUDA candidate against the unchanged reference, "
    "at most two checks. The reference and hardware guidance are attached automatically; "
    "do not duplicate context in the assignment. Reference plus hardware plus assignment must "
    "fit 4096 input tokens. Include the current complete source only when it fits; otherwise "
    "ask a focused contract/indexing question, not a silently truncated repair. Each child has "
    "4096 generated tokens TOTAL including reasoning and a 480-second session deadline. "
    "The return value is a JSON string: parse it with json.loads. Fields are advice, source, "
    "verified_by_child_check and final_grading_required. Keep the full result in a variable; "
    "print only a short advice slice. A child check is not independent final grading.\n"
    "Choose one focused use early: review constructor arguments, state-dict names/shapes, "
    "runtime inputs and output structure before coding; repair one exact compiler/numerical "
    "failure; or improve one bottleneck after correctness. These are ALTERNATIVE examples, "
    "not a sequence to execute: all uses share the same {max_calls}-call budget.\n"
    "Before implementation, request a contract checklist, not a full kernel:\n"
    "```repl\n"
    "import json\n"
    "contract_review = json.loads(llm_query(\n"
    "    'Review the attached reference before implementation. List constructor versus runtime '\n"
    "    'arguments, exact state-dict names/shapes, and output structure. Return findings only; '\n"
    "    'finish with empty source and do not compile.'\n"
    "))\n"
    "print(contract_review['advice'][:1000])\n"
    "```\n"
    "After a failed cuda_feedback(source), reuse its saved feedback and pass the exact error:\n"
    "```repl\n"
    "import json\n"
    "child_result = json.loads(llm_query(\n"
    "    'Repair only this failure; preserve the constructor, state keys and output contract. '\n"
    "    'Check the repair and report remaining issues. Candidate:\\n' + source +\n"
    "    '\\nExact feedback:\\n' + str(feedback)\n"
    "))\n"
    "print(child_result['advice'][:1000])\n"
    "```\n"
    "After correctness, request one bounded performance change, not an unrestricted rewrite:\n"
    "```repl\n"
    "import json\n"
    "child_result = json.loads(llm_query(\n"
    "    'Improve memory-access efficiency in this correct candidate. Preserve its exact '\n"
    "    'interface, state and numerical behavior; check the change and report uncertainty. '\n"
    "    'Candidate:\\n' + source\n"
    "))\n"
    "print(child_result['advice'][:1000])\n"
    "```\n"
    "For either source-changing example, inspect the returned source before integrating it. "
    "Only if a root feedback call remains, independently check it and retain the old source "
    "unless that check passes:\n"
    "```repl\n"
    "if child_result['source'] and feedback.get('feedback_calls_remaining', 0) > 0:\n"
    "    proposed_source = child_result['source']\n"
    "    independent_feedback = cuda_feedback(proposed_source)\n"
    "    print(str(independent_feedback)[:1000])\n"
    "    if independent_feedback.get('correct') is True:\n"
    "        source = proposed_source\n"
    "```\n"
    "Do not treat verified_by_child_check as a replacement for independent verification. "
    "If no root feedback calls remain, retain your known-good source rather than replacing "
    "it solely on child advice. Never invent speed measurements. Calling children earns no "
    "reward; do not delegate on the submission deadline or during recovery."
)


def enclosing_timeouts(max_calls, *, repl_base, backend_base, feedback_timeout, feedback_calls):
    if max_calls not in {1, 2}:
        raise ValueError("Use at most two bounded child sessions")
    child_seconds = TOOL_CHILD_TIMEOUT_S * max_calls
    repl_seconds = max(repl_base, feedback_timeout * feedback_calls + 30) + child_seconds
    return repl_seconds, max(backend_base + child_seconds, repl_seconds + 30)


def parse_action(content):
    text = content.strip()
    if text.startswith(("CHECK\n", "FINISH\n")):
        match = re.fullmatch(r"(CHECK|FINISH)\n(.*?)```python\n(.*)\n```", text, re.S)
        if match is None or text.count("```") != 2:
            raise ValueError("Expected one complete fenced Python candidate")
        return {
            "action": "check" if match[1] == "CHECK" else "finish",
            "advice": match[2].strip(),
            "source": match[3],
        }
    if text.startswith("```json\n") and text.endswith("\n```"):
        text = text[8:-4].strip()
    action = json.loads(text)
    if not isinstance(action, dict):
        raise ValueError("Child must return one JSON action object")
    return action


async def stop_verifier(process):
    if process.returncode is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        await asyncio.wait_for(process.wait(), 12)
    except TimeoutError:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        await process.wait()


async def check_candidate(reference, source, directory, timeout_s):
    directory.mkdir(parents=True, exist_ok=False)
    request, output = directory / "request.json", directory / "result.json"
    request.write_text(
        json.dumps({"reference": reference, "source": source, "timeout_s": timeout_s})
    )
    with (directory / "worker.log").open("wb") as log:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "kernel_rlm.child_tools",
            "--verify",
            str(request),
            str(output),
            stdout=log,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
            env={**os.environ, "KERNEL_CHILD_PARENT_PID": str(os.getpid())},
        )
    try:
        status = await asyncio.wait_for(process.wait(), timeout_s + 12)
        if status:
            raise RuntimeError(f"Child verifier exited {status}; see {directory}")
        return json.loads(output.read_text())
    finally:
        await stop_verifier(process)


class IterativeSubcalls:
    def __init__(
        self,
        client,
        model,
        count_tokens,
        records,
        *,
        reference,
        directory,
        max_calls=2,
        timeout_s=TOOL_CHILD_TIMEOUT_S,
        checker=check_candidate,
        hardware_context="",
        child_reply_tokens=4096,
        child_total_tokens=4096,
        child_prompt_tokens=4096,
        context_tokens=8192,
    ):
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in (child_reply_tokens, child_total_tokens, child_prompt_tokens)
        ):
            raise ValueError("Child token caps must be positive integers")
        if max_calls not in {1, 2} or not 0 < timeout_s <= TOOL_CHILD_TIMEOUT_S:
            raise ValueError("Use at most two children with a bounded session deadline")
        self.client, self.model, self.count_tokens = client, model, count_tokens
        self.records, self.reference, self.directory = records, reference, Path(directory)
        self.max_calls, self.timeout_s, self.checker = max_calls, timeout_s, checker
        self.calls_used = 0
        self.hardware_context = hardware_context
        self.child_reply_tokens = child_reply_tokens
        self.child_total_tokens = child_total_tokens
        self.child_prompt_tokens = child_prompt_tokens
        self.context_tokens = context_tokens

    async def query(self, prompt, state=None):
        if self.calls_used >= self.max_calls:
            raise ValueError("Subcall budget exhausted for this trajectory")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Child assignment must be nonempty text")
        assignment = "Immutable reference source:\n" + self.reference + "\nAssignment:\n" + prompt
        if self.hardware_context:
            assignment = self.hardware_context + "\n" + assignment
        if (
            len(assignment.encode()) > 65536
            or self.count_tokens(assignment) > self.child_prompt_tokens
        ):
            raise ValueError(
                f"Child reference plus assignment exceeds {self.child_prompt_tokens} input tokens"
            )
        dispatch_context = dict(state or {})
        self.calls_used += 1
        record = {
            "child_invocation_id": uuid.uuid4().hex,
            "requests": [],
            "parent_turn_id": dispatch_context.get("parent_turn_id"),
            "parent_tool_call_id": dispatch_context.get("parent_tool_call_id"),
            "parent_tool_call_id_generated": dispatch_context.get("parent_tool_call_id_generated"),
            "parent_action_id": dispatch_context.get("parent_action_id"),
            "call_index": self.calls_used,
            "mode": "compile_repair",
            "prompt": prompt,
            "status": "running",
            "checks": [],
            "child_reply_tokens": self.child_reply_tokens,
            "child_total_tokens": self.child_total_tokens,
            "child_prompt_tokens": self.child_prompt_tokens,
            "model": self.model,
            "usage_accounting": "unknown",
            "completion_tokens": 0,
            "requests_dispatched": 0,
            "turns": [],
            "child_tokens_in_root_loss": 0,
        }
        self.records.append(record)
        directory = self.directory / f"child-{self.calls_used}"
        directory.mkdir(parents=True, exist_ok=False)
        started = time.monotonic()
        messages = [
            {
                "role": "system",
                "content": TOOL_CHILD_SYSTEM.replace(
                    "CHILD_TOTAL_TOKENS", str(self.child_total_tokens)
                ).replace("CHILD_REPLY_TOKENS", str(self.child_reply_tokens)),
            },
            {"role": "user", "content": assignment},
        ]
        try:
            async with asyncio.timeout(self.timeout_s):
                for step in range(4):
                    remaining = self.child_total_tokens - (record["completion_tokens"] or 0)
                    prompt_tokens = self.count_tokens(json.dumps(messages))
                    context_allowance = min(8192, self.context_tokens) - prompt_tokens
                    request_cap = min(self.child_reply_tokens, remaining, context_allowance)
                    if context_allowance < 128:
                        record["termination_reason"] = "context_exhausted"
                        raise ValueError("Child request has insufficient active context headroom")
                    if remaining < 128:
                        record["termination_reason"] = "aggregate_token_budget"
                        raise ValueError("Child aggregate token budget exhausted")
                    request_id = uuid.uuid4().hex
                    request_record = {
                        "call_id": request_id,
                        "request_id": request_id,
                        "model_turn_id": uuid.uuid4().hex,
                        "step": step + 1,
                        "requested_max_tokens": request_cap,
                        "context_output_allowance": context_allowance,
                    }
                    request_record["prompt_messages"] = json.loads(json.dumps(messages))
                    request_record.update(
                        parent_turn_id=record["parent_turn_id"],
                        parent_tool_call_id=record["parent_tool_call_id"],
                        parent_tool_call_id_generated=record["parent_tool_call_id_generated"],
                        parent_action_id=record.get("parent_action_id"),
                    )
                    record["requests"].append(request_record)
                    record["requests_dispatched"] += 1
                    turn_record = {
                        **request_record,
                        "status": "pending",
                        "action_id": uuid.uuid4().hex,
                    }
                    record["turns"].append(turn_record)
                    response = await self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        max_tokens=request_cap,
                        extra_headers={
                            "X-Kernel-Call-Id": request_id,
                            "X-Kernel-Parent-Turn-Id": record["parent_turn_id"] or "",
                            "X-Kernel-Parent-Tool-Call-Id": record["parent_tool_call_id"] or "",
                            "X-Kernel-Parent-Tool-Call-Id-Generated": str(
                                bool(record["parent_tool_call_id_generated"])
                            ).lower(),
                            "X-Kernel-Parent-Action-Id": record.get("parent_action_id") or "",
                        },
                        extra_body={"chat_template_kwargs": {"enable_thinking": True}},
                    )
                    choice = response.choices[0]
                    raw_reply = (
                        choice.message.model_dump(exclude_none=True)
                        if hasattr(choice.message, "model_dump")
                        else vars(choice.message)
                    )
                    usage_dump = (
                        response.usage.model_dump()
                        if hasattr(response.usage, "model_dump")
                        else vars(response.usage)
                        if response.usage
                        else None
                    )
                    usage = response.usage.completion_tokens if response.usage else None
                    turn_record.update(
                        status="response_received",
                        raw_reply=raw_reply,
                        finish_reason=choice.finish_reason,
                        usage=usage_dump,
                    )
                    if not isinstance(usage, int) or isinstance(usage, bool) or usage <= 0:
                        record["usage_accounting"] = "incomplete"
                        record["observed_completion_tokens"] = record["completion_tokens"]
                        record["completion_tokens"] = None
                        turn_record["observed_completion_tokens"] = usage
                        turn_record["outcome"] = "usage_unavailable"
                        record["termination_reason"] = "usage_unavailable"
                        raise ValueError(
                            "Child usage unavailable; aggregate token budget cannot be verified"
                        )
                    record["completion_tokens"] += usage
                    record["observed_completion_tokens"] = record["completion_tokens"]
                    record["usage_accounting"] = "complete"
                    turn_record["observed_completion_tokens"] = usage
                    turn_record["status"] = "usage_recorded"
                    content = visible_response(choice.message.content or "")
                    turn_record["visible_reply"] = content
                    if choice.finish_reason == "length":
                        record["termination_reason"] = "length"
                        raise ValueError("Child reply ended at the per-request length cap")
                    if getattr(choice.message, "tool_calls", None):
                        record["termination_reason"] = "unsupported_native_tool_call"
                        raise ValueError("Child emitted unsupported native tool calls")
                    if usage > request_cap:
                        record["usage_accounting"] = "incomplete"
                        record["observed_completion_tokens"] = record["completion_tokens"]
                        record["completion_tokens"] = None
                        turn_record["outcome"] = "request_token_cap_exceeded"
                        record["termination_reason"] = "request_token_cap_exceeded"
                        raise ValueError("Provider usage exceeded child per-request token cap")
                    if record["completion_tokens"] > self.child_total_tokens:
                        record["usage_accounting"] = "incomplete"
                        record["observed_completion_tokens"] = record["completion_tokens"]
                        record["completion_tokens"] = None
                        turn_record["outcome"] = "aggregate_token_budget"
                        record["termination_reason"] = "aggregate_token_budget"
                        raise ValueError("Provider usage exceeded child aggregate token budget")
                    try:
                        action = parse_action(content)
                    except ValueError as error:
                        record.setdefault("format_errors", []).append(str(error))
                        turn_record["format_error"] = str(error)
                        turn_record["outcome"] = "malformed_action"
                        if step == 3:
                            raise
                        messages.append({"role": "assistant", "content": content})
                        messages.append(
                            {
                                "role": "user",
                                "content": "Action format error; no check was executed. Return CHECK on its own first "
                                "line followed by one fenced python block with complete source using real "
                                "newlines, or FINISH then findings and one python block. Do not JSON-escape code.",
                            }
                        )
                        continue
                    source = action.get("source", "")
                    if not isinstance(source, str) or len(source.encode()) > 65536:
                        raise ValueError("Invalid child candidate source")
                    turn_record["parsed_action"] = action
                    if action.get("action") == "finish":
                        if (
                            not isinstance(action.get("advice"), str)
                            or not action["advice"].strip()
                        ):
                            raise ValueError("Child must return explicit findings")
                        result = {
                            "advice": action["advice"],
                            "source": source,
                            "verified_by_child_check": any(
                                check["source"] == source and check["result"].get("correct") is True
                                for check in record["checks"]
                            ),
                            "final_grading_required": True,
                        }
                        turn_record["outcome"] = "completed"
                        record.update(
                            status="completed",
                            response=json.dumps(result),
                            termination_reason="finish",
                        )
                        return record["response"]
                    if action.get("action") != "check" or len(record["checks"]) >= 2 or step == 3:
                        raise ValueError("Child has at most two checks followed by a final answer")
                    valid, error = validate_submission_source(source)
                    result = (
                        await self.checker(
                            self.reference, source, directory / f"check-{len(record['checks'])}", 90
                        )
                        if valid
                        else {
                            "compiled": False,
                            "correct": None,
                            "verification_status": "static_rejected",
                            "error": error,
                        }
                    )
                    action_id = turn_record["action_id"]
                    observation_id = uuid.uuid4().hex
                    record["checks"].append(
                        {
                            "action_id": action_id,
                            "observation_id": observation_id,
                            "source": source,
                            "result": result,
                        }
                    )
                    turn_record.update(
                        outcome="check_executed" if valid else "static_rejected",
                        observation_id=observation_id,
                        observation=result,
                    )
                    messages.append({"role": "assistant", "content": content})
                    feedback = {
                        key: result.get(key)
                        for key in (
                            "compiled",
                            "correct",
                            "max_diff",
                            "verification_status",
                            "failure_kind",
                        )
                    }
                    feedback["error"] = str(result.get("error", ""))[:2000]
                    messages.append(
                        {
                            "role": "user",
                            "content": "Actual check feedback: "
                            + json.dumps(feedback)
                            + f". Checks remaining: {2 - len(record['checks'])}. Repair or finish; do not invent timing.",
                        }
                    )
                    (directory / "session.json").write_text(json.dumps(record, indent=2))
        except BaseException as error:
            if "termination_reason" not in record:
                record["termination_reason"] = (
                    "cancelled" if isinstance(error, asyncio.CancelledError) else "error"
                )
            pending_turn = next(
                (turn for turn in reversed(record["turns"]) if turn["status"] == "pending"),
                None,
            )
            if pending_turn is not None:
                record["usage_accounting"] = "incomplete"
                record["observed_completion_tokens"] = record["completion_tokens"] or 0
                record["completion_tokens"] = None
                pending_turn["status"] = "request_failed_or_cancelled"
            record.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            record["elapsed_seconds"] = time.monotonic() - started
            (directory / "session.json").write_text(json.dumps(record, indent=2))


def install_worker_cancellation(expected_parent: int):
    def cancelled(signum, frame):
        raise KeyboardInterrupt("Child verifier cancelled; unwind nested compiler/runner groups")

    signal.signal(signal.SIGTERM, cancelled)
    library = ctypes.CDLL(None, use_errno=True)
    if library.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "Cannot bind child verifier lifetime to its parent")
    if os.getppid() != expected_parent:
        raise KeyboardInterrupt("Child verifier parent exited before cancellation was installed")


def verify_worker(request_path, output_path):
    install_worker_cancellation(int(os.environ["KERNEL_CHILD_PARENT_PID"]))
    from kernel_rlm.verifier import verify_cuda

    request = json.loads(Path(request_path).read_text())
    result = verify_cuda(
        request["reference"],
        request["source"],
        timeout_s=request["timeout_s"],
        n_trials=1,
        benchmark=False,
        collect_phase_events=True,
    )
    Path(output_path).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] != "--verify":
        raise SystemExit("Expected --verify REQUEST RESULT")
    verify_worker(sys.argv[2], sys.argv[3])
