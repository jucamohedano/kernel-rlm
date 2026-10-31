"""Explicitly resume one interrupted root request; never retry a failed run automatically."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import httpx
from rlm_train.checkpoint import read_checkpoint
from rlm_train.native import NativeLimits, NativeRuntime
from rlm_train.proxy import SubLLMProxy
from rlm_train.token_budget import ContextBudgetPolicy, VllmPromptCounter

from cute_rlm.config import EvalConfig, check_gpu_receipt
from cute_rlm.dataset import load_tasks, task_context
from cute_rlm.docs import docs_bootstrap
from cute_rlm.evaluate import cute_turn_notice, isolated_backend_factory
from cute_rlm.feedback import CompileFeedback
from cute_rlm.launch import check_evidence_files, write_repl_wrapper
from cute_rlm.observer import NsightObserver
from cute_rlm.resources import resources_bootstrap
from cute_rlm.sandbox import SandboxSpec
from cute_rlm.server_identity import check_server_identity
from cute_rlm.verifier import verify_submission


def check_checkpoint_binding(
    saved: dict, original: Path, manifest: dict, config: EvalConfig
) -> dict:
    task_id = saved["session"]["context"].get("task_id")
    task = next(
        task
        for task in load_tasks(original / "inputs" / config.task_manifest)
        if task["task_id"] == task_id
    )
    if saved["session"]["context"] != task_context(task):
        raise ValueError("checkpoint task context differs from original frozen specification")
    limits = NativeLimits(
        max_depth=config.max_depth,
        max_turns=config.max_turns,
        max_children=config.max_children,
        max_live_children=config.max_live_children,
        max_concurrent_requests=config.max_concurrent_requests,
        max_model_calls=config.max_model_calls,
        request_timeout=config.request_timeout,
        episode_timeout=config.episode_timeout,
        max_prompt_chars=config.max_prompt_chars,
    )
    root = original / "inputs/.eval-inputs"
    policy = ContextBudgetPolicy(
        server_context_tokens=config.server_context_tokens,
        output_fraction=config.output_fraction,
        max_generated_tokens=config.max_generated_tokens,
        logprobs=config.logprobs,
    )
    expected = {
        "model": config.model_id,
        "limits": asdict(limits),
        "budget_policy": asdict(policy),
        "bootstrap_code": docs_bootstrap(json.loads((root / "docs-corpus.json").read_text()))
        + resources_bootstrap(json.loads((root / "resources.json").read_text())),
        "system_prompt": (root / "system-prompt.txt").read_text(),
        "enable_thinking": config.enable_thinking,
        "stream_responses": True,
        "child_repl_at_limit": True,
    }
    user_prompts = json.loads((root / "user-prompts.json").read_text())
    if saved["config"] != expected or saved["session"]["prompt"] != user_prompts[task_id]:
        raise ValueError("checkpoint prompt/model/budget differs from the original frozen run")
    return task


async def resume_prepared(
    original: Path,
    checkpoint: Path,
    output: Path,
    *,
    base_url: str,
    receipt_path: Path,
    sandbox_spec: Path,
    observer_root: Path,
    additional_seconds: float,
    charge_uncertain_reservation: bool = False,
    host_python: str = sys.executable,
    api_key: str | None = None,
) -> dict:
    import math

    if (
        isinstance(additional_seconds, bool)
        or not math.isfinite(additional_seconds)
        or additional_seconds <= 0
    ):
        raise ValueError("additional_seconds must be finite and positive")
    original, checkpoint, output = original.resolve(), checkpoint.resolve(), output.resolve()
    if output == original or output.is_relative_to(original):
        raise ValueError("resume output must be a new directory outside the failed attempt")
    saved = read_checkpoint(checkpoint)
    if not saved.get("recoverable") or not saved.get("cleanup_confirmed"):
        raise ValueError("checkpoint is not eligible for manual infrastructure recovery")
    manifest = json.loads((original / "provenance.json").read_text())
    config = EvalConfig(**manifest["config"])
    for item in manifest["inputs"]:
        if (
            hashlib.sha256((original / "inputs" / item["path"]).read_bytes()).hexdigest()
            != item["sha256"]
        ):
            raise ValueError("original frozen input changed")
    receipt = json.loads(receipt_path.read_text())
    check_gpu_receipt(receipt, config, manifest["manifest_sha256"])
    check_evidence_files(receipt, receipt_path.parent)
    if hashlib.sha256(sandbox_spec.read_bytes()).hexdigest() != receipt.get("sandbox_spec_sha256"):
        raise ValueError("readiness receipt does not bind this sandbox")
    spec = SandboxSpec.read(sandbox_spec)
    spec.validate_host()
    if spec.observer_root is None or Path(spec.observer_root).resolve() != observer_root.resolve():
        raise ValueError("observer root differs from sandbox")
    task = check_checkpoint_binding(saved, original, manifest, config)
    task_id = task["task_id"]
    limits = NativeLimits(**saved["config"]["limits"])
    output.mkdir(parents=True, exist_ok=False)
    wrapper = output / "repl-python"
    write_repl_wrapper(wrapper, sandbox_spec, host_python)
    command = [host_python, "-m", "cute_rlm.sandbox", "--spec", str(sandbox_spec)]
    import tempfile

    with tempfile.TemporaryDirectory(prefix="cute-resume-broker-") as temporary:
        proxy = SubLLMProxy(
            socket_path=Path(temporary) / "broker.sock",
            max_request_bytes=config.broker_request_bytes,
        )
        await proxy.start()
        try:
            async with httpx.AsyncClient(
                base_url=base_url.rstrip("/") + "/",
                timeout=config.request_timeout,
                headers={"Authorization": "Bearer " + (api_key or "EMPTY")},
            ) as client:
                await check_server_identity(client, config, receipt)
                feedback = CompileFeedback(
                    task, config.verification, output / "compile-feedback", worker_command=command
                )
                policy = saved["config"]["budget_policy"]
                runtime = NativeRuntime(
                    client,
                    config.model_id,
                    proxy,
                    limits=limits,
                    backend_factory=isolated_backend_factory(config, str(wrapper)),
                    budget_policy=ContextBudgetPolicy(**policy) if policy else None,
                    prompt_counter=VllmPromptCounter(client, config.server_context_tokens),
                    bootstrap_code=saved["config"]["bootstrap_code"],
                    system_prompt=saved["config"]["system_prompt"],
                    enable_thinking=saved["config"]["enable_thinking"],
                    stream_responses=saved["config"]["stream_responses"],
                    child_repl_at_limit=saved["config"]["child_repl_at_limit"],
                    checkpoint_path=output / "checkpoint.json",
                    event_path=output / "native-events.jsonl",
                    compile_candidate=feedback,
                    turn_notice=cute_turn_notice,
                )
                try:
                    record = await runtime.resume(
                        checkpoint,
                        additional_seconds=additional_seconds,
                        charge_uncertain_reservation=charge_uncertain_reservation,
                    )
                finally:
                    if runtime.root is not None:
                        partial = runtime.records[runtime.root.id]
                        partial["example_id"] = task_id
                        (output / "task.json").write_text(json.dumps(partial, indent=2))
                record["example_id"] = task_id
                (output / "task.json").write_text(json.dumps(record, indent=2))
                if record.get("final_answer") is not None:
                    record["verification"] = await verify_submission(
                        task,
                        record["final_answer"],
                        config.verification,
                        output / "verification",
                        worker_command=command,
                        cute_observer=NsightObserver(observer_root.resolve()),
                    )
                    (output / "task.json").write_text(json.dumps(record, indent=2))
                report = {
                    "original_run_id": manifest["run_id"],
                    "task_id": task_id,
                    "resume_of": record.get("resume_of"),
                    "additional_seconds": additional_seconds,
                    "status": record["status"],
                    "scope": "manual linked recovery attempt, not an independent task or silent retry",
                }
                (output / "resume-report.json").write_text(json.dumps(report, indent=2))
                return report
        finally:
            await proxy.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--sandbox-spec", type=Path, required=True)
    parser.add_argument("--observer-root", type=Path, required=True)
    parser.add_argument("--additional-seconds", type=float, required=True)
    parser.add_argument("--charge-uncertain-reservation", action="store_true")
    parser.add_argument("--host-python", default=sys.executable)
    args = parser.parse_args()
    print(
        json.dumps(
            asyncio.run(
                resume_prepared(
                    args.original,
                    args.checkpoint,
                    args.output,
                    base_url=args.base_url,
                    receipt_path=args.receipt,
                    sandbox_spec=args.sandbox_spec,
                    observer_root=args.observer_root,
                    additional_seconds=args.additional_seconds,
                    charge_uncertain_reservation=args.charge_uncertain_reservation,
                    host_python=args.host_python,
                    api_key=os.environ.get("OPENAI_API_KEY"),
                )
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
