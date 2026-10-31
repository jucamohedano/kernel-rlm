"""Independent compile/one-forward diagnostics, never a scored evaluation."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

import httpx

from cute_rlm.config import EvalConfig, required_gpu_checks, serving_arguments
from cute_rlm.dataset import load_tasks, task_context
from cute_rlm.docs import docs_bootstrap, load_docs
from cute_rlm.evaluate import cute_turn_notice, isolated_backend_factory
from cute_rlm.feedback import CompileFeedback
from cute_rlm.launch import check_evidence_files, write_repl_wrapper
from cute_rlm.prompts import CUTE_SYSTEM_PROMPT, first_user_message
from cute_rlm.resources import build_resources, resources_bootstrap
from cute_rlm.sandbox import SandboxSpec
from cute_rlm.server_identity import check_server_identity

MODE = "compile-forward-diagnostic"


def diagnostic_plan(rollouts_per_example: int) -> dict:
    if type(rollouts_per_example) is not int or rollouts_per_example <= 0:
        raise ValueError("rollouts_per_example must be a positive integer")
    return {
        "mode": MODE,
        "rollouts_per_example": rollouts_per_example,
        "scheduling": "sequential independent roots",
        "final_smoke": "one explicit host compile/forward call for each submitted source",
        "compiler_artifacts": "observational PTX/CUBIN capture; absence is not a failure",
        "numerical_correctness": "not_evaluated",
        "timing": "not_evaluated",
        "cute_provenance": "not_evaluated",
        "automatic_retry_or_resume": False,
    }


def diagnostic_checks(config: EvalConfig) -> tuple[str, ...]:
    return tuple(
        "cute_compile_forward_smoke" if name == "cute_compile_launch" else name
        for name in required_gpu_checks(config)
        if name not in {"reference_candidate_correctness", "timing_protocol"}
    )


def prepare_diagnostic(
    project: Path, config: EvalConfig, run_id: str, output_root: Path, *, rollouts_per_example: int
) -> dict:
    from kernel_factory import prepare_evaluation

    plan = diagnostic_plan(rollouts_per_example)
    tasks = load_tasks(project / config.task_manifest)
    if len(tasks) != config.task_count:
        raise ValueError("Freeze the selected task manifest before preparing the run")
    for task in tasks:
        source = task["source"]
        if (
            source.get("repo") != config.dataset_id
            or source.get("revision") != config.dataset_revision
        ):
            raise ValueError("Task source does not match the pinned cudaLLM dataset")
    docs = load_docs(project / config.docs_manifest)
    if not docs:
        raise ValueError("Provide the pinned CuTe documentation corpus")
    code_paths = [
        str(path.relative_to(project))
        for root in (
            project / "training/src/rlm_train",
            project / "training/environments/cute_rlm/cute_rlm",
        )
        for path in sorted(root.rglob("*.py"))
    ]
    code_paths.extend(
        [
            "training/environments/kernel_rlm/kernel_factory/__init__.py",
            "pyproject.toml",
            "uv.lock",
            "training/environments/cute_rlm/pyproject.toml",
            "training/environments/cute_rlm/cute_rlm/examples/elementwise_add.py.txt",
            "training/environments/cute_rlm/cute_rlm/examples/delegation.py.txt",
        ]
    )
    docs_path = project / config.docs_manifest
    docs_manifest = json.loads(docs_path.read_text())
    doc_paths = [
        str((docs_path.parent / item["path"]).resolve().relative_to(project.resolve()))
        for item in docs_manifest["files"]
    ]
    captured = {
        "docs-corpus.json": json.dumps(docs, sort_keys=True, ensure_ascii=False).encode(),
        "resources.json": json.dumps(build_resources(config), sort_keys=True).encode(),
        "resolved-config.json": json.dumps(config.payload(), sort_keys=True).encode(),
        "system-prompt.txt": CUTE_SYSTEM_PROMPT.encode(),
        "user-prompts.json": json.dumps(
            {task["task_id"]: first_user_message(task, config) for task in tasks}, sort_keys=True
        ).encode(),
        "diagnostic-plan.json": json.dumps(plan, sort_keys=True).encode(),
    }
    manifest = prepare_evaluation(
        project,
        "laguna-xs-2-1-cute-compile-diagnostic",
        run_id,
        output_root,
        {"id": config.model_id, "revision": config.model_revision, "dtype": "bfloat16"},
        {"id": config.dataset_id, "revision": config.dataset_revision},
        config.payload(),
        config.task_manifest,
        code_paths,
        docs_paths=[config.docs_manifest, *doc_paths],
        input_bytes={".eval-inputs/" + name: content for name, content in captured.items()},
        runtime={"status": "not_run", "mode": MODE},
        prompt_contract={"diagnostic": plan, "task_data": "REPL context via task_context"},
    )
    readiness = {
        "status": "CPU_prepared_GPU_pending",
        "mode": MODE,
        "config_sha256": config.sha256(),
        "manifest_sha256": manifest["manifest_sha256"],
        "selected_tasks": len(tasks),
        "rollouts_per_example": rollouts_per_example,
        "planned_rollouts": len(tasks) * rollouts_per_example,
        "checks": {
            key: {"status": "pending", "evidence": None} for key in diagnostic_checks(config)
        },
        "serving_arguments": serving_arguments("/path/to/pinned/model", config),
        "scope": "Compile/forward only; no numerical comparison, timing, Nsight, or score",
    }
    (output_root / run_id / "diagnostic-readiness.json").write_text(json.dumps(readiness, indent=2))
    return readiness


def check_diagnostic_receipt(receipt: dict, config: EvalConfig, manifest: dict) -> None:
    if receipt.get("mode") != MODE:
        raise ValueError("An explicit diagnostic readiness receipt is required")
    if receipt.get("config_sha256") != config.sha256():
        raise ValueError("Diagnostic receipt does not match this configuration")
    if receipt.get("manifest_sha256") != manifest["manifest_sha256"]:
        raise ValueError("Diagnostic receipt does not match frozen inputs and rollout count")
    for name in diagnostic_checks(config):
        check = receipt.get("checks", {}).get(name, {})
        if check.get("status") != "passed" or not check.get("evidence"):
            raise ValueError(f"Diagnostic GPU validation still required: {name}")
    if receipt.get("observed_context_tokens") != config.server_context_tokens:
        raise ValueError("Observed server window differs from frozen context")
    identity = receipt.get("server_identity", {})
    if (
        type(identity.get("tensor_parallel_size")) is not int
        or identity["tensor_parallel_size"] != config.tensor_parallel_size
    ):
        raise ValueError("Observed tensor parallel size differs from frozen configuration")
    hardware = receipt.get("hardware", {})
    if (
        hardware.get("gpu_model") != config.gpu_model
        or hardware.get("gpu_compute_capability") != config.gpu_compute_capability
    ):
        raise ValueError("Observed GPU differs from frozen target")
    if receipt.get("software", {}).get("cuda_toolkit_version") != config.cuda_toolkit_version:
        raise ValueError("Observed CUDA toolkit differs from frozen target")


def token_usage(records: list[dict]) -> dict:
    events = {
        event["call_id"]: event
        for record in records
        for session in [record, *record.get("child_sessions", [])]
        for event in session.get("events", [])
    }
    result = {}
    for branch in ("root", "child", "plain"):
        calls = [event for event in events.values() if event.get("branch") == branch]
        result[branch] = {}
        for metric in ("prompt_tokens", "completion_tokens", "total_tokens"):
            values = [(event.get("response", {}).get("usage") or {}).get(metric) for event in calls]
            known = [value for value in values if type(value) is int and value >= 0]
            result[branch][metric] = {
                "sum": sum(known) if known else 0 if not calls else None,
                "recorded": len(known),
                "total": len(calls),
            }
    return result


async def diagnose_prepared(
    directory: Path, *, client, worker_command: list[str], backend_factory, receipt: dict
) -> dict:
    from kernel_factory import start_evaluation
    from rlm_train.native import NativeLimits, NativeRuntime
    from rlm_train.proxy import SubLLMProxy
    from rlm_train.token_budget import ContextBudgetPolicy, VllmPromptCounter

    directory = directory.resolve()
    manifest = json.loads((directory / "provenance.json").read_text())
    config = EvalConfig(**manifest["config"])
    check_diagnostic_receipt(receipt, config, manifest)
    inputs = directory / "inputs/.eval-inputs"
    plan = json.loads((inputs / "diagnostic-plan.json").read_text())
    if (
        plan != diagnostic_plan(plan["rollouts_per_example"])
        or manifest["prompt_contract"].get("diagnostic") != plan
    ):
        raise ValueError("Frozen diagnostic mode/count differs from the manifest")
    tasks = load_tasks(directory / "inputs" / config.task_manifest)
    if len(tasks) != config.task_count:
        raise ValueError("Frozen task count differs from the configuration")
    user_prompts = json.loads((inputs / "user-prompts.json").read_text())
    system_prompt = (inputs / "system-prompt.txt").read_text()
    bootstrap = docs_bootstrap(
        json.loads((inputs / "docs-corpus.json").read_text())
    ) + resources_bootstrap(json.loads((inputs / "resources.json").read_text()))
    # Validate every captured byte and refuse a rerun only after the plan is checked.
    start_evaluation(manifest, directory.parent)
    records = []
    failure = None
    report = {}
    with tempfile.TemporaryDirectory(prefix="cute-diagnostic-broker-") as temporary:
        proxy = SubLLMProxy(
            socket_path=Path(temporary) / "broker.sock",
            max_request_bytes=config.broker_request_bytes,
        )
        try:
            await proxy.start()
            for task_index, task in enumerate(tasks):
                for rollout_index in range(plan["rollouts_per_example"]):
                    root_dir = directory / f"task-{task_index:05d}" / f"rollout-{rollout_index:05d}"
                    root_dir.mkdir(parents=True, exist_ok=False)
                    record = {"status": "not_started", "final_answer": None, "events": []}
                    runtime = None
                    try:
                        feedback = CompileFeedback(
                            task,
                            config.verification,
                            root_dir,
                            worker_command=worker_command,
                            capture_compiler_artifacts=True,
                        )
                        runtime = NativeRuntime(
                            client,
                            config.model_id,
                            proxy,
                            limits=NativeLimits(
                                max_depth=config.max_depth,
                                max_turns=config.max_turns,
                                max_children=config.max_children,
                                max_live_children=config.max_live_children,
                                max_concurrent_requests=config.max_concurrent_requests,
                                max_model_calls=config.max_model_calls,
                                request_timeout=config.request_timeout,
                                episode_timeout=config.episode_timeout,
                                max_prompt_chars=config.max_prompt_chars,
                            ),
                            budget_policy=ContextBudgetPolicy(
                                server_context_tokens=config.server_context_tokens,
                                output_fraction=config.output_fraction,
                                max_generated_tokens=config.max_generated_tokens,
                                logprobs=config.logprobs,
                            ),
                            prompt_counter=VllmPromptCounter(client, config.server_context_tokens),
                            backend_factory=backend_factory,
                            bootstrap_code=bootstrap,
                            system_prompt=system_prompt,
                            enable_thinking=config.enable_thinking,
                            stream_responses=True,
                            checkpoint_path=root_dir / "checkpoint.json",
                            child_repl_at_limit=True,
                            compile_candidate=feedback,
                            turn_notice=cute_turn_notice,
                            event_path=root_dir / "native-events.jsonl",
                        )
                        record = await runtime.run(
                            user_prompts[task["task_id"]], task_context(task)
                        )
                        submitted = record.get("final_answer")
                        if submitted is not None:
                            if isinstance(submitted, str):
                                (root_dir / "final-source.py").write_text(submitted)
                                record["final_source_sha256"] = hashlib.sha256(
                                    submitted.encode()
                                ).hexdigest()
                            record["host_final_smoke"] = {
                                "origin": "host_final_smoke",
                                "model_action": False,
                                "result": await feedback(
                                    "host-final-smoke:" + runtime.root.id, submitted
                                ),
                            }
                    except BaseException as error:
                        record["diagnostic_error"] = {
                            "type": type(error).__name__,
                            "detail": repr(error),
                        }
                        raise
                    finally:
                        try:
                            if runtime is not None:
                                await runtime.close()
                        finally:
                            if runtime is not None and runtime.root is not None:
                                partial = runtime.records[runtime.root.id]
                                partial.update(
                                    {
                                        key: value
                                        for key, value in record.items()
                                        if key not in partial
                                    }
                                )
                                record = partial
                            record.update(
                                example_id=task["task_id"],
                                task_source_sha256=task["source"]["sha256"],
                                task_index=task_index,
                                rollout_index=rollout_index,
                                correct=None,
                                cute_verified=None,
                                speedup=None,
                            )
                            (root_dir / "task.json").write_text(json.dumps(record, indent=2))
                            records.append(record)
        except BaseException as error:
            failure = {"type": type(error).__name__, "detail": repr(error)}
            raise
        finally:
            try:
                await proxy.stop()
            except BaseException as error:
                failure = {"type": type(error).__name__, "detail": repr(error)}
                raise
            finally:
                statuses = Counter(record.get("status", "unknown") for record in records)
                smoke = Counter(
                    record.get("host_final_smoke", {})
                    .get("result", {})
                    .get("status", "not_submitted")
                    for record in records
                )
                planned = len(tasks) * plan["rollouts_per_example"]
                report = {
                    "run_id": manifest["run_id"],
                    "experiment_id": manifest["experiment_id"],
                    "mode": MODE,
                    "status": "failed"
                    if failure
                    else "completed"
                    if statuses.get("completed", 0) == planned
                    else "incomplete",
                    "failure": failure,
                    "selected_tasks": len(tasks),
                    "rollouts_per_example": plan["rollouts_per_example"],
                    "planned_rollouts": planned,
                    "recorded_rollouts": len(records),
                    "root_statuses": dict(statuses),
                    "final_smoke_statuses": dict(smoke),
                    "usage": token_usage(records),
                    "config": config.payload(),
                    "diagnostic_plan": plan,
                    "GPU_receipt": receipt,
                    "manifest_sha256": manifest["manifest_sha256"],
                    "correct": None,
                    "cute_verified": None,
                    "speedup": None,
                    "scope": "Independent root diagnostics only. Compiled means one forward completed, not numerical correctness, timing, or CuTe provenance. Final smoke is a separate host action. No automatic retry/resume. Backup verification is external.",
                }
                (directory / "report.json").write_text(json.dumps(report, indent=2))
    return report


async def run_diagnostic(
    directory: Path,
    *,
    base_url: str,
    sandbox_spec: Path,
    receipt_path: Path,
    host_python: str = sys.executable,
    api_key: str | None = None,
) -> dict:
    directory, sandbox_spec = directory.resolve(), sandbox_spec.resolve()
    manifest = json.loads((directory / "provenance.json").read_text())
    config = EvalConfig(**manifest["config"])
    receipt = json.loads(receipt_path.read_text())
    check_diagnostic_receipt(receipt, config, manifest)
    check_evidence_files(receipt, receipt_path.parent)
    if receipt.get("sandbox_spec_sha256") != hashlib.sha256(sandbox_spec.read_bytes()).hexdigest():
        raise ValueError("Diagnostic receipt does not bind this sandbox specification")
    spec = SandboxSpec.read(sandbox_spec)
    spec.validate_host()
    wrapper = directory / "repl-python"
    write_repl_wrapper(wrapper, sandbox_spec, host_python)
    async with httpx.AsyncClient(
        base_url=base_url.rstrip("/") + "/",
        timeout=config.request_timeout,
        headers={"Authorization": "Bearer " + (api_key or "EMPTY")},
    ) as client:
        await check_server_identity(client, config, receipt)
        return await diagnose_prepared(
            directory,
            client=client,
            worker_command=[host_python, "-m", "cute_rlm.sandbox", "--spec", str(sandbox_spec)],
            backend_factory=isolated_backend_factory(config, str(wrapper)),
            receipt=receipt,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--run", type=Path, metavar="DIRECTORY")
    parser.add_argument("--project", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--rollouts-per-example", type=int)
    parser.add_argument("--base-url")
    parser.add_argument("--sandbox-spec", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--host-python", default=sys.executable)
    args = parser.parse_args()
    required = (
        ("project", "config", "run_id", "output_root", "rollouts_per_example")
        if args.prepare
        else ("base_url", "sandbox_spec", "receipt")
    )
    if any(getattr(args, name) is None for name in required):
        parser.error(
            "This action requires " + ", ".join("--" + name.replace("_", "-") for name in required)
        )
    if args.prepare:
        result = prepare_diagnostic(
            args.project.resolve(),
            EvalConfig.read(args.config),
            args.run_id,
            args.output_root.resolve(),
            rollouts_per_example=args.rollouts_per_example,
        )
    else:
        if args.rollouts_per_example is not None:
            parser.error("Rollout count is frozen during --prepare and cannot be changed by --run")
        result = asyncio.run(
            run_diagnostic(
                args.run,
                base_url=args.base_url,
                sandbox_spec=args.sandbox_spec,
                receipt_path=args.receipt,
                host_python=args.host_python,
                api_key=os.environ.get("OPENAI_API_KEY"),
            )
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
