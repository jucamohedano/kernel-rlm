"""Opt-in child compile/repair sessions with aggregate token and tool budgets."""

import asyncio
import json
import os
import signal
import sys
import time
from pathlib import Path

from kernel_rlm.submission_contract import validate_submission_source
from kernel_rlm.thinking import visible_response

TOOL_CHILD_SYSTEM = (
    "You are a bounded CUDA research child with compiler/correctness feedback. "
    "No shell, Python REPL, recursion or reference edits. Return one JSON object only, no Markdown fences. "
    'To test return {"action":"check","source":"complete candidate Python source"}. '
    "The controller compiles and checks against the immutable reference; inspect feedback and repair. "
    "At most two checks, then finish with "
    '{"action":"finish","advice":"concise findings","source":"best candidate or empty string"}. '
    "Resource faults/timeouts are infrastructure, not numerical failure. Never invent measurements. "
    "Final grading is independent. You have 4096 generated tokens TOTAL including reasoning "
    "across this entire session, not 4096 per reply."
)
TOOL_CHILD_TIMEOUT_S = 480


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
    ):
        if max_calls not in {1, 2} or not 0 < timeout_s <= TOOL_CHILD_TIMEOUT_S:
            raise ValueError("Use at most two children with a bounded session deadline")
        self.client, self.model, self.count_tokens = client, model, count_tokens
        self.records, self.reference, self.directory = records, reference, Path(directory)
        self.max_calls, self.timeout_s, self.checker = max_calls, timeout_s, checker
        self.calls_used = 0

    async def query(self, prompt, state=None):
        if self.calls_used >= self.max_calls:
            raise ValueError("Subcall budget exhausted for this trajectory")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Child assignment must be nonempty text")
        assignment = "Immutable reference source:\n" + self.reference + "\nAssignment:\n" + prompt
        if len(assignment.encode()) > 65536 or self.count_tokens(assignment) > 4096:
            raise ValueError("Child reference plus assignment exceeds 4096 input tokens")
        self.calls_used += 1
        record = {
            "call_index": self.calls_used,
            "mode": "compile_repair",
            "prompt": prompt,
            "status": "running",
            "checks": [],
            "completion_tokens": 0,
            "child_tokens_in_root_loss": 0,
        }
        self.records.append(record)
        directory = self.directory / f"child-{self.calls_used}"
        directory.mkdir(parents=True, exist_ok=False)
        started = time.monotonic()
        messages = [
            {"role": "system", "content": TOOL_CHILD_SYSTEM},
            {"role": "user", "content": assignment},
        ]
        try:
            async with asyncio.timeout(self.timeout_s):
                for step in range(3):
                    remaining = 4096 - record["completion_tokens"]
                    if remaining < 128 or self.count_tokens(json.dumps(messages)) > 8192:
                        raise ValueError("Child aggregate token or history budget exhausted")
                    response = await self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        max_tokens=remaining,
                        extra_body={"chat_template_kwargs": {"enable_thinking": True}},
                    )
                    choice = response.choices[0]
                    if choice.finish_reason == "length" or getattr(
                        choice.message, "tool_calls", None
                    ):
                        raise ValueError(
                            "Incomplete child action or unsupported native tool request"
                        )
                    usage = response.usage.completion_tokens if response.usage else None
                    if type(usage) is not int or not 0 < usage <= remaining:
                        raise ValueError("Child usage must account for every generated token")
                    record["completion_tokens"] += usage
                    content = visible_response(choice.message.content or "")
                    action = json.loads(content)
                    if not isinstance(action, dict):
                        raise ValueError("Child must return one JSON action object")
                    source = action.get("source", "")
                    if not isinstance(source, str) or len(source.encode()) > 65536:
                        raise ValueError("Invalid child candidate source")
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
                        record.update(status="completed", response=json.dumps(result))
                        return record["response"]
                    if action.get("action") != "check" or step == 2:
                        raise ValueError("Child has at most two checks followed by a final answer")
                    valid, error = validate_submission_source(source)
                    result = (
                        await self.checker(self.reference, source, directory / f"check-{step}", 90)
                        if valid
                        else {
                            "compiled": False,
                            "correct": None,
                            "verification_status": "static_rejected",
                            "error": error,
                        }
                    )
                    record["checks"].append({"source": source, "result": result})
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
                            + f". Checks remaining: {1 - step}. Repair or finish; do not invent timing.",
                        }
                    )
                    (directory / "session.json").write_text(json.dumps(record, indent=2))
        except BaseException as error:
            record.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            record["elapsed_seconds"] = time.monotonic() - started
            (directory / "session.json").write_text(json.dumps(record, indent=2))


def verify_worker(request_path, output_path):
    from kernel_rlm.verifier import verify_cuda

    def cancelled(signum, frame):
        raise KeyboardInterrupt("Child verifier cancelled; unwind nested compiler/runner groups")

    signal.signal(signal.SIGTERM, cancelled)
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
