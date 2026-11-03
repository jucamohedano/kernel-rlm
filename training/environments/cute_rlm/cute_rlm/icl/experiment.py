"""Frozen, tool-free CuTe ICL pilot. Importing this module starts no workload.

These are new pilot defaults, not changes to the existing evaluation protocol.
``prepare`` needs only CPU dependencies. ``generate`` makes at most one request
per frozen entry, with no retries or resume. ``validate`` runs only the explicit
sandbox command; generated Python is never imported by this host process.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import copy
import hashlib
import json
import math
import os
import random
import re
import sys
import traceback
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from rlm_train.streaming import (
    ChatCompletionAccumulator,
    SSEDecoder,
    StreamInterruptedError,
    StreamProtocolError,
)

from cute_rlm.config import MODEL_ID, MODEL_REVISION

CONDITIONS = ("task_only", "api", "api_example")
MODE = "plain-cute-icl"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json_bytes(value))


def now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class ExperimentConfig:
    """Explicit proposed pilot settings, frozen before any model request.

    BF16 and the revision are requested deployment properties, not attestations
    about an arbitrary OpenAI-compatible server. Raw /models metadata is saved.
    """

    model_id: str = MODEL_ID
    model_revision: str = MODEL_REVISION
    requested_precision: str = "bf16"
    server_context_tokens: int = 262144
    max_tokens: int = 8192
    request_timeout: float = 600
    temperature: float = 0.6
    top_p: float = 0.95
    seed: int = 20261101
    enable_thinking: bool = True
    logprobs: bool = True
    tokenize: bool = False

    def __post_init__(self) -> None:
        if (
            not isinstance(self.model_id, str)
            or not self.model_id
            or any(character.isspace() for character in self.model_id)
        ):
            raise ValueError("model_id must be a nonempty model identifier without whitespace")
        if not isinstance(self.model_revision, str) or not re.fullmatch(
            r"[0-9a-fA-F]{40}", self.model_revision
        ):
            raise ValueError("model_revision must be a pinned full 40-character hexadecimal SHA")
        if self.requested_precision != "bf16" or self.server_context_tokens != 262144:
            raise ValueError("Freeze the requested BF16 deployment and 262144-token context")
        if type(self.max_tokens) is not int or not 0 < self.max_tokens < self.server_context_tokens:
            raise ValueError("max_tokens must fit inside the frozen context")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be a nonnegative 32-bit integer")
        for name in ("request_timeout", "temperature", "top_p"):
            value = getattr(self, name)
            lower_ok = value >= 0 if name == "temperature" else value > 0
            if isinstance(value, bool) or not math.isfinite(value) or not lower_ok:
                raise ValueError(f"Invalid {name}")
        if self.temperature > 2 or self.top_p > 1:
            raise ValueError("temperature must be <=2 and top_p <=1")
        if self.enable_thinking is not True or self.logprobs is not True:
            raise ValueError("Thinking and selected token logprobs remain enabled")
        if type(self.tokenize) is not bool:
            raise ValueError("tokenize must be a boolean")


def request_payload(task: dict, prompt: dict, config: ExperimentConfig) -> dict:
    # A task receives the same sampling seed in all three independent contexts.
    pair_seed = (config.seed + int(sha256(task["id"].encode())[:8], 16)) % 2**32
    return {
        "model": config.model_id,
        "messages": [
            {"role": "system", "content": prompt["system"]},
            {"role": "user", "content": prompt["user"]},
        ],
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "seed": pair_seed,
        "n": 1,
        "stream": True,
        "stream_options": {"include_usage": True},
        "logprobs": config.logprobs,
        "chat_template_kwargs": {"enable_thinking": config.enable_thinking},
    }


def capture_sources(output: Path) -> dict[str, str]:
    package = Path(__file__).resolve().parent
    environment = package.parent
    training = environment.parents[2]
    paths = [p for p in package.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    paths.extend(
        environment / name
        for name in (
            "__init__.py",
            "sandbox.py",
            "verifier.py",
            "contracts.py",
            "config.py",
            "outputs.py",
            "verify_worker.py",
            "compile_trace.py",
        )
    )
    paths.append(training / "src/rlm_train/streaming.py")
    records = {}
    for path in sorted(paths):
        relative = path.relative_to(training)
        destination = output / "source_capture" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        content = path.read_bytes()
        destination.write_bytes(content)
        records[str(relative)] = sha256(content)
    write_json(output / "source_capture.json", records)
    return records


def prepare(output: str | Path, config: ExperimentConfig | None = None) -> dict:
    """Freeze all 36 prompts, task cases, protocol settings and source bytes."""
    from .prompts import render_prompt
    from .tasks import load_tasks, reference_source

    config = config or ExperimentConfig()
    tasks = load_tasks()
    if len(tasks) != 12 or len({task["id"] for task in tasks}) != 12:
        raise ValueError("The pilot requires exactly 12 unique tasks")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", asdict(config))
    write_json(
        output / "protocol.json",
        {
            "mode": MODE,
            "task_count": 12,
            "conditions": list(CONDITIONS),
            "planned_completions": 36,
            "independent_contexts": True,
            "tools": False,
            "retries": 0,
            "resume": False,
            "defaults_scope": "new proposed pilot; existing evaluation settings unchanged",
            "deployment_properties": "requested, not server-attested",
            "source_extraction": "plain Python or exactly one complete python fence; AST parse only",
            "valid_solution": None,
            "source_review_required": True,
        },
    )
    entries = []
    for task_index, task in enumerate(tasks):
        task_path = f"tasks/{task_index:02d}/task.json"
        write_json(output / task_path, task)
        (output / task_path).with_name("reference.py").write_text(reference_source(task))
        for condition in CONDITIONS:
            entry_id = f"{task_index:02d}-{condition}"
            directory = output / "prompts" / entry_id
            prompt = render_prompt(task, condition)
            if not isinstance(prompt["system"], str) or not isinstance(prompt["user"], str):
                raise ValueError("Prompts must be plain system and user strings")
            write_json(directory / "prompt.json", prompt)
            (directory / "system.txt").write_bytes(prompt["system"].encode())
            (directory / "user.txt").write_bytes(prompt["user"].encode())
            write_json(directory / "sources.json", prompt["sources"])
            payload = request_payload(task, prompt, config)
            request_path = f"prompts/{entry_id}/request.json"
            write_json(output / request_path, payload)
            entries.append(
                {
                    "id": entry_id,
                    "task_id": task["id"],
                    "condition": condition,
                    "task_path": task_path,
                    "request_path": request_path,
                    "sampling_seed": payload["seed"],
                }
            )
    random.Random(config.seed).shuffle(entries)
    capture_sources(output)
    artifacts = {
        str(path.relative_to(output)): sha256(path.read_bytes())
        for path in sorted(output.rglob("*"))
        if path.is_file()
    }
    plan = {"mode": MODE, "created_at": now(), "entries": entries, "artifacts": artifacts}
    write_json(output / "plan.json", plan)
    (output / "plan.sha256").write_text(sha256((output / "plan.json").read_bytes()) + "\n")
    return plan


def load_plan(run: Path) -> tuple[dict, ExperimentConfig]:
    data = (run / "plan.json").read_bytes()
    if sha256(data) != (run / "plan.sha256").read_text().strip():
        raise ValueError("Frozen plan SHA256 mismatch")
    plan = json.loads(data)
    if plan["mode"] != MODE or len(plan["entries"]) != 36:
        raise ValueError("Expected the 36-entry plain-cute-icl plan")
    for name, expected in plan["artifacts"].items():
        path = (run / name).resolve()
        if not path.is_relative_to(run.resolve()) or sha256(path.read_bytes()) != expected:
            raise ValueError(f"Frozen artifact SHA256 mismatch: {name}")
    training = Path(__file__).resolve().parents[4]
    captured = json.loads((run / "source_capture.json").read_text())
    for name, expected in captured.items():
        current = training / name
        if not current.is_file() or sha256(current.read_bytes()) != expected:
            raise ValueError(f"Current source differs from frozen capture: {name}")
    return plan, ExperimentConfig(**json.loads((run / "config.json").read_text()))


def extract_source(text: str) -> str:
    """Accept only plain code or one complete Python fence; never execute it."""
    text = text.strip()
    if "```" in text:
        lines = text.splitlines()
        if (
            len(lines) < 3
            or lines[0] != "```python"
            or lines[-1] != "```"
            or text.count("```") != 2
        ):
            raise ValueError("Expected plain Python or one complete python fence")
        text = "\n".join(lines[1:-1]).strip()
    if not text:
        raise ValueError("Empty generated source")
    ast.parse(text)
    return text + "\n"


def partial_response(accumulator: ChatCompletionAccumulator) -> dict:
    result = copy.deepcopy(accumulator.response)
    result["choices"] = []
    for index, choice in sorted(accumulator.choices.items()):
        choice = copy.deepcopy(choice)
        if index in accumulator.tools:
            choice["message"]["tool_calls"] = [
                copy.deepcopy(tool) for _, tool in sorted(accumulator.tools[index].items())
            ]
        result["choices"].append(choice)
    return result


async def collect_completion(client: httpx.AsyncClient, payload: dict, directory: Path) -> dict:
    """Journal native bytes before parsing, including partial/invalid responses.

    Read through EOF rather than returning at [DONE], so trailing native bytes
    are also retained. Tool fragments are inert evidence and are never executed.
    """
    decoder = SSEDecoder()
    accumulator = ChatCompletionAccumulator()
    state: dict[str, Any] = {"done": False, "chunks": 0, "bytes": 0}
    parse_error: Exception | None = None
    with (
        (directory / "stream.bin").open("wb") as raw,
        (directory / "events.jsonl").open("wb") as events,
    ):
        try:
            async with client.stream(
                "POST", "chat/completions", json=payload, headers={"Accept-Encoding": "identity"}
            ) as response:
                state.update(http_status=response.status_code, headers=dict(response.headers))
                encoding = response.headers.get("content-encoding", "identity").strip().lower()
                if encoding not in {"", "identity"}:
                    parse_error = StreamProtocolError(
                        f"Unsupported stream Content-Encoding: {encoding}"
                    )
                async for chunk in response.aiter_raw():
                    offset = state["bytes"]
                    raw.write(chunk)
                    raw.flush()
                    state["chunks"] += 1
                    state["bytes"] += len(chunk)
                    events.write(
                        json.dumps(
                            {"offset": offset, "length": len(chunk), "sha256": sha256(chunk)}
                        ).encode()
                        + b"\n"
                    )
                    events.flush()
                    if not response.is_success or parse_error is not None:
                        continue
                    try:
                        for event in decoder.feed(chunk):
                            if state["done"]:
                                raise StreamProtocolError("SSE event received after [DONE]")
                            if event == "[DONE]":
                                accumulator.complete()
                                state["done"] = True
                                continue
                            data = json.loads(event)
                            if not isinstance(data, dict):
                                raise StreamProtocolError("SSE data must be an object")
                            accumulator.add(data)
                    except Exception as error:
                        parse_error = error
                response.raise_for_status()
            if parse_error is not None:
                raise parse_error
            if not state["done"]:
                raise StreamInterruptedError("Stream ended before [DONE]")
            return accumulator.complete()
        finally:
            write_json(directory / "stream_state.json", state)
            write_json(directory / "partial_response.json", partial_response(accumulator))


async def capture_metadata(
    client: httpx.AsyncClient,
    method: str,
    endpoint: str,
    destination: Path,
    payload: dict | None = None,
) -> dict:
    record: dict[str, Any] = {"endpoint": endpoint}
    try:
        response = await client.request(method, endpoint, json=payload)
        destination.with_suffix(".body").write_bytes(response.content)
        record.update(http_status=response.status_code, headers=dict(response.headers))
        response.raise_for_status()
        record["response"] = response.json()
    except (Exception, asyncio.CancelledError) as error:
        record["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        write_json(destination, record)
        if isinstance(error, asyncio.CancelledError):
            raise
    write_json(destination, record)
    return record


async def generate(
    run: str | Path,
    base_url: str,
    *,
    api_key_env: str | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> dict:
    """Issue one completion per entry. A started run can never be resumed."""
    run = Path(run).resolve()
    plan, config = load_plan(run)
    if api_key_env is not None and not os.environ.get(api_key_env):
        raise ValueError(f"Missing API key environment variable: {api_key_env}")
    headers = {"Authorization": f"Bearer {os.environ[api_key_env]}"} if api_key_env else {}
    marker = {
        "started_at": now(),
        "base_url": base_url,
        "api_key_env": api_key_env,
        "config_sha256": plan["artifacts"]["config.json"],
    }
    if (run / "GENERATION_STARTED.json").exists():
        raise FileExistsError("Generation already started; no retries or resume")
    summary: dict[str, Any] = {
        "planned": 36,
        "attempted": 0,
        "stopped_on_transport_error": False,
        "results": [],
    }
    async with httpx.AsyncClient(
        base_url=base_url.rstrip("/") + "/",
        headers=headers,
        timeout=config.request_timeout,
        transport=transport,
    ) as client:
        metadata = await capture_metadata(client, "GET", "models", run / "models.json")
        if "error" in metadata:
            raise ValueError("Cannot establish server identity; see models.json")
        cards = metadata["response"].get("data", [])
        card = next((item for item in cards if item.get("id") == config.model_id), None)
        if card is None or card.get("max_model_len") != config.server_context_tokens:
            raise ValueError("Server model ID/context differs from frozen request; see models.json")
        with (run / "GENERATION_STARTED.json").open("x") as started:
            json.dump(marker, started, indent=2)
        for entry in plan["entries"]:
            directory = run / "completions" / entry["id"]
            directory.mkdir(parents=True, exist_ok=False)
            payload = json.loads((run / entry["request_path"]).read_text())
            write_json(directory / "request.json", payload)
            result: dict[str, Any] = {
                **entry,
                "started_at": now(),
                "config_sha256": plan["artifacts"]["config.json"],
                "request_sha256": sha256(json_bytes(payload)),
                "usage": None,
                "usage_status": "unknown",
                "status": "started",
            }
            summary["attempted"] += 1
            cancellation = None
            try:
                if config.tokenize:
                    await capture_metadata(
                        client,
                        "POST",
                        "tokenize",
                        directory / "tokenize.json",
                        {
                            "model": config.model_id,
                            "messages": payload["messages"],
                            "add_generation_prompt": True,
                            "chat_template_kwargs": payload["chat_template_kwargs"],
                        },
                    )
                async with asyncio.timeout(config.request_timeout):
                    response = await collect_completion(client, payload, directory)
                write_json(directory / "response.json", response)
                result["usage"] = response.get("usage")
                result["usage_status"] = (
                    "reported" if response.get("usage") is not None else "unknown"
                )
                choices = response["choices"]
                if len(choices) != 1 or choices[0]["index"] != 0:
                    raise ValueError("Expected exactly one completion")
                message = choices[0]["message"]
                if (
                    message.get("tool_calls")
                    or message.get("function_call")
                    or message.get("refusal")
                ):
                    raise ValueError("Expected tool-free, non-refusal source output")
                if choices[0]["finish_reason"] != "stop":
                    raise ValueError(f"Incomplete source response: {choices[0]['finish_reason']}")
                source = extract_source(message.get("content") or "")
                (directory / "candidate.py").write_bytes(source.encode())
                result.update(status="completed", source_sha256=sha256(source.encode()))
            except (Exception, asyncio.CancelledError) as error:
                cancellation = error if isinstance(error, asyncio.CancelledError) else None
                result["error"] = {
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                }
                transport_failure = isinstance(error, (httpx.TransportError, TimeoutError))
                result["status"] = (
                    "cancelled"
                    if cancellation
                    else ("transport_error" if transport_failure else "response_error")
                )
                if (directory / "partial_response.json").exists() and result["usage"] is None:
                    partial = json.loads((directory / "partial_response.json").read_text())
                    if partial.get("usage") is not None:
                        result.update(usage=partial["usage"], usage_status="reported_partial")
                summary["stopped_on_transport_error"] = transport_failure
            result["finished_at"] = now()
            write_json(directory / "result.json", result)
            summary["results"].append(result)
            write_json(run / "generation_summary.json", summary)
            if cancellation is not None:
                raise cancellation
            if summary["stopped_on_transport_error"]:
                break
    summary["finished_at"] = now()
    write_json(run / "generation_summary.json", summary)
    return summary


def check_worker_sources(run: Path, spec_path: Path) -> None:
    """Bind the isolated worker package to the same captured grading sources."""
    from cute_rlm.sandbox import SandboxSpec

    spec = SandboxSpec.read(spec_path)
    bindings = [
        binding
        for binding in spec.readonly_bindings
        if binding["target"] == "/runtime/python/cute_rlm"
    ]
    if len(bindings) != 1:
        raise ValueError("Sandbox needs one explicit frozen cute_rlm package binding")
    package = Path(bindings[0]["source"]).resolve()
    prefix = "environments/cute_rlm/cute_rlm/"
    captured = json.loads((run / "source_capture.json").read_text())
    for name, expected in captured.items():
        if not name.startswith(prefix):
            continue
        path = package / name.removeprefix(prefix)
        if not path.is_file() or sha256(path.read_bytes()) != expected:
            raise ValueError(f"Sandbox worker source differs from frozen capture: {name}")


async def validate(
    run: str | Path, sandbox_spec: str | Path, host_python: str = sys.executable
) -> dict:
    from .validation import validate_candidate

    run = Path(run).resolve()
    plan, _ = load_plan(run)
    spec = Path(sandbox_spec).resolve(strict=True)
    check_worker_sources(run, spec)
    worker_command = [host_python, "-m", "cute_rlm.sandbox", "--spec", str(spec)]
    results = []
    for entry in plan["entries"]:
        directory = run / "completions" / entry["id"]
        if not (directory / "result.json").exists():
            continue
        generation = json.loads((directory / "result.json").read_text())
        if generation["status"] != "completed":
            continue
        source = (directory / "candidate.py").read_text()
        if sha256(source.encode()) != generation["source_sha256"]:
            raise ValueError(f"Candidate source SHA256 mismatch: {entry['id']}")
        task = json.loads((run / entry["task_path"]).read_text())
        result = await validate_candidate(
            task, source, run / "validation" / entry["id"], worker_command=worker_command
        )
        results.append({"id": entry["id"], **result})
    summary = {"results": results, "valid_solution": None, "source_review_required": True}
    write_json(run / "validation_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser("prepare")
    preparation.add_argument("--output", required=True)
    preparation.add_argument("--config", help="JSON overrides for the new pilot defaults")
    generation = commands.add_parser("generate")
    generation.add_argument("--run", required=True)
    generation.add_argument("--base-url", required=True)
    generation.add_argument("--api-key-env", help="Environment variable name, never the key itself")
    validation = commands.add_parser("validate")
    validation.add_argument("--run", required=True)
    validation.add_argument("--sandbox-spec", required=True)
    validation.add_argument("--host-python", default=sys.executable)
    args = parser.parse_args()
    if args.command == "prepare":
        overrides = json.loads(Path(args.config).read_text()) if args.config else {}
        plan = prepare(args.output, ExperimentConfig(**overrides))
        print(json.dumps({"mode": plan["mode"], "planned_completions": len(plan["entries"])}))
    elif args.command == "generate":
        summary = asyncio.run(generate(args.run, args.base_url, api_key_env=args.api_key_env))
        print(
            json.dumps(
                {
                    "attempted": summary["attempted"],
                    "stopped_on_transport_error": summary["stopped_on_transport_error"],
                }
            )
        )
    else:
        summary = asyncio.run(validate(args.run, args.sandbox_spec, args.host_python))
        print(json.dumps({"validated": len(summary["results"]), "source_review_required": True}))


if __name__ == "__main__":
    main()
