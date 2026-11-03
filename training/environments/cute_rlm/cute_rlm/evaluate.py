"""Native CuTe evaluation: preparation is offline; execution requires GPU evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from cute_rlm.config import EvalConfig, check_gpu_receipt, required_gpu_checks, serving_arguments
from cute_rlm.prompts import CUTE_SYSTEM_PROMPT, first_user_message
from cute_rlm.resources import build_resources, resources_bootstrap


def prepare(project: Path, config: EvalConfig, run_id: str, output_root: Path) -> dict:
    from kernel_factory import prepare_evaluation

    from cute_rlm.dataset import load_tasks
    from cute_rlm.docs import load_docs

    tasks = load_tasks(project / config.task_manifest)
    for task in tasks:
        source = task.get("source", {})
        if (
            source.get("repo") != config.dataset_id
            or source.get("revision") != config.dataset_revision
        ):
            raise ValueError("Task source does not match the pinned cudaLLM dataset")
    if len(tasks) != config.task_count:
        raise ValueError("Freeze the selected task manifest before preparing the run")
    # Selection was done by the manifest command, not again during inference.
    if len({task["task_id"] for task in tasks}) != len(tasks):
        raise ValueError("Duplicate task IDs")
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
    docs_manifest = json.loads((project / config.docs_manifest).read_text())
    doc_paths = [
        str(
            ((project / config.docs_manifest).parent / item["path"])
            .resolve()
            .relative_to(project.resolve())
        )
        for item in docs_manifest["files"]
    ]
    corpus = json.dumps(docs, sort_keys=True, ensure_ascii=False).encode()
    captured_bytes = {
        ".eval-inputs/docs-corpus.json": corpus,
        ".eval-inputs/resources.json": json.dumps(build_resources(config), sort_keys=True).encode(),
        ".eval-inputs/resolved-config.json": json.dumps(config.payload(), sort_keys=True).encode(),
        ".eval-inputs/system-prompt.txt": CUTE_SYSTEM_PROMPT.encode(),
        ".eval-inputs/user-prompts.json": json.dumps(
            {task["task_id"]: first_user_message(task, config) for task in tasks}, sort_keys=True
        ).encode(),
    }
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
    manifest = prepare_evaluation(
        project,
        "laguna-xs-2-1-cute-knowledge-baseline",
        run_id,
        output_root,
        {"id": config.model_id, "revision": config.model_revision, "dtype": "bfloat16"},
        {"id": config.dataset_id, "revision": config.dataset_revision},
        config.payload(),
        config.task_manifest,
        code_paths,
        docs_paths=[config.docs_manifest, *doc_paths],
        input_bytes=captured_bytes,
        runtime={"status": "not_run", "GPU_validation": "pending"},
        prompt_contract={
            "native_tools": ["repl"],
            "task_data": "REPL context",
            "system_prompt_input": ".eval-inputs/system-prompt.txt",
            "user_prompts_input": ".eval-inputs/user-prompts.json",
            "shared_resources_input": ".eval-inputs/resources.json",
            "shared_resources_variable": "resources",
            "selected_logprobs": True,
            "compaction": False,
            "compile_feedback": {
                "helper": "compile_candidate(source)",
                "call_quota": None,
                "scope": "compile and one forward; no numerical comparison or timing",
            },
            "turn_warning_start": max(1, config.max_turns - 4)
            if config.max_turns is not None
            else None,
            "streamed_response_evidence": True,
            "checkpoint_scope": "manual infrastructure recovery; no automatic retries",
            "child_depth": "root 0, recursive REPL child 1; no grandchildren",
            "final_submission": "answer['content'] = best_source; answer['ready'] = True",
        },
    )
    directory = output_root / run_id
    # Materialized docs make the capture self-contained, rather than merely pinning filenames.
    readiness = {
        "status": "CPU_prepared_GPU_pending",
        "config_sha256": config.sha256(),
        "manifest_sha256": manifest["manifest_sha256"],
        "docs_corpus_sha256": hashlib.sha256(corpus).hexdigest(),
        "checks": {
            key: {"status": "pending", "evidence": None} for key in required_gpu_checks(config)
        },
        "serving_arguments": serving_arguments("/path/to/pinned/model", config),
        "guards": {
            "model_calls": config.max_model_calls,
            "turns_per_session": config.max_turns,
            "child_admissions": config.max_children,
            "note": "Explicit provisional guards, not model limits",
        },
    }
    (directory / "readiness.json").write_text(json.dumps(readiness, indent=2))
    return readiness


async def evaluate_prepared(
    directory: Path,
    *,
    client,
    worker_command: list[str],
    backend_factory,
    gpu_receipt: dict,
    cute_observer=None,
) -> dict:
    from kernel_factory import start_evaluation
    from rlm_train.native import NativeLimits, NativeRuntime
    from rlm_train.proxy import SubLLMProxy
    from rlm_train.token_budget import ContextBudgetPolicy, VllmPromptCounter

    from cute_rlm.dataset import load_tasks, task_context
    from cute_rlm.docs import docs_bootstrap
    from cute_rlm.feedback import CompileFeedback
    from cute_rlm.verifier import verify_submission

    manifest = json.loads((directory / "provenance.json").read_text())
    config = EvalConfig(**manifest["config"])
    check_gpu_receipt(gpu_receipt, config, manifest["manifest_sha256"])
    if cute_observer is None:
        raise ValueError(
            "A protected CuTe compile/launch observer is required before scored evaluation"
        )
    captured = {}
    for name in ("docs-corpus.json", "system-prompt.txt", "user-prompts.json", "resources.json"):
        relative = ".eval-inputs/" + name
        content = (directory / "inputs" / relative).read_bytes()
        expected = next(item["sha256"] for item in manifest["inputs"] if item["path"] == relative)
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError(f"Frozen evaluation input changed: {relative}")
        captured[name] = content
    corpus_bytes = captured["docs-corpus.json"]
    system_prompt = captured["system-prompt.txt"].decode()
    user_prompts = json.loads(captured["user-prompts.json"])
    bootstrap = docs_bootstrap(json.loads(corpus_bytes)) + resources_bootstrap(
        json.loads(captured["resources.json"])
    )
    tasks = load_tasks(directory / "inputs" / config.task_manifest)
    start_evaluation(manifest, directory.parent)
    import tempfile

    broker_directory = tempfile.TemporaryDirectory(prefix="cute-broker-")
    proxy = SubLLMProxy(
        socket_path=Path(broker_directory.name) / "broker.sock",
        max_request_bytes=config.broker_request_bytes,
    )
    await proxy.start()
    results = []
    try:
        for index, task in enumerate(tasks):
            feedback = CompileFeedback(
                task,
                config.verification,
                directory / f"compile-feedback-{index:05d}",
                worker_command=worker_command,
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
                stream_responses=True,
                checkpoint_path=directory / f"checkpoint-{index:05d}.json",
                child_repl_at_limit=True,
                compile_candidate=feedback,
                turn_notice=cute_turn_notice,
                event_path=directory / "native-events.jsonl",
            )
            try:
                record = await runtime.run(
                    user_prompts[task["task_id"]],
                    task_context(task),
                )
                record["example_id"] = task["task_id"]
                if record.get("final_answer") is not None:
                    try:
                        record["verification"] = await verify_submission(
                            task,
                            record["final_answer"],
                            config.verification,
                            directory / f"verify-{index:05d}",
                            worker_command=worker_command,
                            cute_observer=cute_observer,
                        )
                        record["correct"] = record["verification"].get("status") == "correct"
                    except Exception as error:
                        record["verification"] = {"status": "infrastructure", "detail": repr(error)}
            finally:
                if runtime.root is not None:
                    record = runtime.records[runtime.root.id]
                    record["example_id"] = task["task_id"]
                    (directory / f"task-{index:05d}.json").write_text(json.dumps(record, indent=2))
                    results.append(record)
    finally:
        await proxy.stop()
        broker_directory.cleanup()
    ledgers = [
        event
        for record in results
        for session in [record, *record.get("child_sessions", [])]
        for event in session.get("events", [])
    ]
    unique = {event["call_id"]: event for event in ledgers}
    token_usage = {}
    for branch in ("root", "child", "plain"):
        calls = [event for event in unique.values() if event.get("branch") == branch]
        token_usage[branch] = {}
        for metric in ("prompt_tokens", "completion_tokens", "total_tokens"):
            values = [(event.get("response", {}).get("usage") or {}).get(metric) for event in calls]
            recorded = [value for value in values if type(value) is int and value >= 0]
            token_usage[branch][metric] = {
                "sum": sum(recorded) if recorded else 0 if not calls else None,
                "recorded": len(recorded),
                "total": len(calls),
            }
    report = {
        "run_id": manifest["run_id"],
        "experiment_id": manifest["experiment_id"],
        "status": "evaluated",
        "completed_tasks": len(results),
        "selected_tasks": len(tasks),
        "correct": sum(r.get("correct", False) for r in results),
        "model_id": config.model_id,
        "model_revision": config.model_revision,
        "config": config.payload(),
        "GPU_receipt": gpu_receipt,
        "usage": token_usage,
        "scope": "Inference evaluation only; factory finalization requires a verified backup receipt",
    }
    (directory / "report.json").write_text(json.dumps(report, indent=2))
    return report


def isolated_backend_factory(config: EvalConfig, worker_python: str):
    import os

    from rlm_train.repl.subprocess import SubprocessReplBackend

    environment = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": "/nonexistent",
        "PYTHONPATH": str(Path(__file__).resolve().parents[3] / "src"),
        "RLM_TRAIN_READONLY_CONTEXT": "1",
        "RLM_TRAIN_EXEC_TIMEOUT_S": str(config.repl_exec_timeout),
        "RLM_TRAIN_BROKER_TIMEOUT_S": str(config.broker_timeout),
    }

    def factory(**kwargs):
        return SubprocessReplBackend(
            python=worker_python,
            worker_env=environment,
            startup_timeout=config.worker_startup_timeout,
            **kwargs,
        )

    return factory


def cute_turn_notice(turn: int, max_turns: int | None) -> str | None:
    """Warn on the last five numbered requests, including the final opportunity."""
    if max_turns is None or turn < max(1, max_turns - 4):
        return None
    remaining = max_turns - turn + 1
    if turn == max_turns:
        return (
            f"FINAL TURN {turn}/{max_turns}. Submit your best complete ModelNew now: "
            "answer['content'] = best_source; answer['ready'] = True. "
            "Do not spend this final turn on another compile check or exploration."
        )
    return (
        f"Turn {turn}/{max_turns}: {remaining} model turns remain, including this one. "
        f"Finish corrections and submit before the final turn ({max_turns}) if possible. "
        "Set answer['content'] = best_source and answer['ready'] = True to submit. "
        "Do not wait for more feedback once your remaining turns are needed to finalize."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(
        args.project.resolve(), EvalConfig.read(args.config), args.run_id, args.output_root
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
