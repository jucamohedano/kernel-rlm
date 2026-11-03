import hashlib
import importlib
import json
from pathlib import Path


def test_split_card_import_keeps_topology_and_exact_native_step(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "training/environments/kernel_rlm"))
    artifacts = importlib.import_module("kernel_trl.artifacts")
    source = tmp_path / "source"
    source.mkdir()
    (source / "report.json").write_text(
        json.dumps(
            {
                "run_id": "rtx-baseline",
                "optimizer_steps": 1,
                "title": "RTX TP1 baseline",
                "gpu_topology": "RTX GPU0 learner; GPU1 inference",
                "results": [
                    {"task_id": "task", "optimizer_step": 18, "metrics": {"train_loss": 0.1}}
                ],
            }
        )
    )
    (source / "inputs.json").write_text(
        json.dumps(
            {
                "context_tokens": 16384,
                "learner_world_size": 1,
                "inference_tensor_parallel_size": 1,
                "child_prompt_tokens": 4096,
                "child_reply_tokens": 4096,
                "child_tools": True,
                "gpu_memory_total_mib_by_index": [97887, 97887],
            }
        )
    )
    (source / "update.log").write_text("{'loss': 0.1, 'entropy': 0.2}\n")
    (source / "gpu-telemetry.csv").write_text(
        "index,timestamp,memory.used,utilization.gpu,power.draw\n"
        "0,2026/10/20 10:31:00.000,70000 MiB,50 %,200 W\n"
    )
    destination = tmp_path / "normalized"
    artifacts.import_run(source, destination)
    report = json.loads((destination / "report.json").read_text())
    assert report["title"] == "RTX TP1 baseline"
    assert report["gpu_topology"] == "RTX GPU0 learner; GPU1 inference"
    metrics = [
        json.loads(line) for line in (destination / "metrics.jsonl").read_text().splitlines()
    ]
    assert [entry["step"] for entry in metrics] == [18]
    assert metrics[0]["entropy"] == 0.2
    config = json.loads((destination / "provenance.json").read_text())["config"]
    assert config["learner_world_size"] == 1
    assert config["inference_tensor_parallel_size"] == 1
    assert config["child_tools"] is True
    gpu = json.loads((destination / "gpu_stats.jsonl").read_text())["gpus"][0]
    assert gpu["memory_total_mib"] == 97887


def test_saved_messages_and_reasoning_survive_without_ui_redesign(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "training/environments/kernel_rlm"))
    artifacts = importlib.import_module("kernel_trl.artifacts")
    directory = tmp_path / "group-0-trajectory-0"
    directory.mkdir()

    def save(name, value):
        (directory / name).write_text(json.dumps(value))

    save("task.json", {"name": "real-task"})
    save(
        "submission.json",
        {
            "messages": [
                {"role": "system", "content": "all system instructions"},
                {"role": "assistant", "content": "visible <script>"},
            ],
            "submission": "cuda source",
            "termination_reason": "submitted",
        },
    )
    save("score.json", {"reward": 0.8, "verification": {"correct": True}})
    response = {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": "visible <script>",
                    "reasoning": "saved reasoning",
                },
            }
        ],
        "usage": {"completion_tokens": 20},
    }
    (directory / "calls.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "branch": "root",
                        "response": response,
                        "request": {"max_tokens": 777},
                        "elapsed_seconds": 1.0,
                    }
                ),
                json.dumps({"branch": "child", "request": {"max_tokens": 99}}),
            ]
        )
        + "\n"
    )
    (directory / "child-progress.jsonl").write_text(
        json.dumps({"call_id": "child", "stage": "generating", "completion_tokens": 257}) + "\n"
    )
    trace = artifacts.episode_trace(directory, "run")
    assert trace["info"]["kernel_child_progress"][0]["completion_tokens"] == 257
    assert trace["info"]["repl_transcript"][0]["content"] == "all system instructions"
    assert trace["nodes"][1]["message"]["reasoning_content"] == "saved reasoning"
    assert trace["calls"][0]["node"] == 1
    assert trace["calls"][0]["request"]["max_tokens"] == 777
    assert trace["info"]["rlm_child_calls"][0]["request"]["max_tokens"] == 99
    assert "kernel_subcalls" not in trace["info"]
    assert trace["rewards"]["kernel_reward"]["score"] == 0.8
    (directory / "score.json").unlink()
    assert artifacts.episode_trace(directory, "run")["rewards"]["kernel_reward"]["score"] is None


def test_import_accepts_collector_report_without_lifecycle_events(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "training/environments/kernel_rlm"))
    artifacts = importlib.import_module("kernel_trl.artifacts")
    source = tmp_path / "source"
    source.mkdir()
    (source / "report.json").write_text(
        json.dumps({"run_id": "20261018T235332Z-4b40f63d", "optimizer_steps": 0})
    )
    (source / "inputs.json").write_text("{}")
    rollouts = source / "rollouts"
    rollouts.mkdir()
    rewards = [0.8, 0.1, 0.7, 0.6, 0.1, 0.1, 0.7, 0.8]
    (rollouts / "groups.json").write_text(
        json.dumps([{"task": "task-1", "rewards": rewards, "admitted": True}])
    )
    for index, reward in enumerate(rewards):
        episode = rollouts / f"group-0-trajectory-{index}"
        episode.mkdir()
        (episode / "task.json").write_text(json.dumps({"name": "sample"}))
        (episode / "submission.json").write_text(json.dumps({"messages": []}))
        (episode / "score.json").write_text(json.dumps({"reward": reward}))
    destination = tmp_path / "imported"

    result = artifacts.import_run(source, destination)

    provenance = json.loads((destination / "provenance.json").read_text())
    assert result["trajectories"] == 8
    assert provenance["created_at_utc"] is None
    assert provenance["config"]["group_size"] == 8
    assert provenance["config"]["max_optimizer_steps"] == 0
    assert (destination / "events.jsonl").read_text() == ""


def test_import_recurses_full_run_groups_and_hashes_rlm_sources(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "training/environments/kernel_rlm"))
    artifacts = importlib.import_module("kernel_trl.artifacts")
    source = tmp_path / "source"
    source.mkdir()
    dataset = b'{"_task_id":"task-next"}\n'
    (source / "inputs").mkdir()
    (source / "inputs/lvl1_seed42.jsonl").write_bytes(dataset)
    sources = {
        "kernel_trl/__init__.py": (
            "training/environments/kernel_rlm/kernel_trl/__init__.py",
            b"env",
        ),
        "rlm/__init__.py": ("rlm/__init__.py", b"core"),
        "training/src/rlm_train/__init__.py": ("training/src/rlm_train/__init__.py", b"worker"),
    }
    hashes = {}
    for relative, (repo_relative, content) in sources.items():
        path = source / "repo" / repo_relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        hashes[relative] = hashlib.sha256(content).hexdigest()
    (source / "inputs.json").write_text(
        json.dumps(
            {
                "source_sha256": hashes,
                "dataset_sha256": hashlib.sha256(dataset).hexdigest(),
            }
        )
    )
    groups = [
        {"task": "task-first", "rewards": [0.1, 0.8], "admitted": True},
        {
            "task": "task-next",
            "rewards": [0.1, 0.9],
            "admitted": True,
            "group_id": "g-next",
            "episodes": [{"episode_id": "e-next", "attempt_id": "a1"}],
        },
    ]
    report = {
        "run_id": "20261019T000001Z-curriculum",
        "status": "paused",
        "optimizer_steps": 2,
        "groups": groups,
        "results": [
            {
                "task_id": "task-first",
                "status": "trained",
                "optimizer_step": 1,
                "metrics": {"train_loss": -0.2},
            },
            {
                "task_id": "task-next",
                "status": "trained",
                "optimizer_step": 2,
                "metrics": {"train_loss": -0.1},
            },
        ],
    }
    (source / "report.json").write_text(json.dumps(report))
    (source / "events.jsonl").write_text(
        json.dumps({"phase": "driver_started", "utc": "2026-10-19T00:00:00+00:00"}) + "\n"
    )
    attempt = source / "groups/step-002-task-next/collection/rollouts/group-0-trajectory-0"
    attempt.mkdir(parents=True)
    identity = {"group_id": "g-next", "episode_id": "e-next", "attempt_id": "a1"}
    (attempt / "task.json").write_text(json.dumps({"name": "task-next", **identity}))
    (attempt / "submission.json").write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "task"},
                    {"role": "assistant", "content": "answer"},
                ],
                "submission": "kernel source",
                "termination_reason": "submitted",
                **identity,
            }
        )
    )
    (attempt / "score.json").write_text(
        json.dumps({"reward": 0.9, "verification": {"correct": True}, **identity})
    )
    call = {
        "branch": "root",
        "elapsed_seconds": 1.0,
        "response": {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {"role": "assistant", "content": "answer"},
                }
            ],
            "usage": {"completion_tokens": 4},
        },
    }
    (attempt / "calls.jsonl").write_text(json.dumps(call) + "\n")
    (attempt / "feedback-test.jsonl").write_text(json.dumps({"feedback": "compiled"}) + "\n")
    (attempt / "program.log").write_text("REPL finished")
    group_dir = source / "groups/step-002-task-next"
    (group_dir / "update.log").write_text(
        "{'loss': '-0.1', 'grad_norm': '0.3', 'learning_rate': '1e-05'}\n"
    )
    (group_dir / "training-telemetry.csv").write_text(
        "index, timestamp, memory.used [MiB], utilization.gpu [%], power.draw [W]\n"
        "0, 2026/10/19 00:01:00.000, 1000 MiB, 50 %, 100.0 W\n"
    )
    destination = tmp_path / "imported"

    result = artifacts.import_run(source, destination)

    traces = json.loads((destination / "traces.jsonl").read_text())["traces"]
    effective = json.loads((destination / "effective/traces.jsonl").read_text())["traces"]
    metrics = [
        json.loads(line) for line in (destination / "metrics.jsonl").read_text().splitlines()
    ]
    telemetry = [
        json.loads(line) for line in (destination / "gpu_stats.jsonl").read_text().splitlines()
    ]
    manifest = json.loads((destination / "provenance.json").read_text())
    assert result["trajectories"] == 1
    assert traces[0]["info"]["feedback_results"] == [{"feedback": "compiled"}]
    assert traces[0]["info"]["program_log"] == "REPL finished"
    assert effective == traces
    assert {item["step"] for item in metrics} == {1, 2}
    assert next(item for item in metrics if item["step"] == 2)["grad_norm"] == 0.3
    assert telemetry[0]["gpus"][0]["memory_used_mib"] == 1000
    assert {item["role"] for item in manifest["inputs"]} == {"dataset", "executed source"}


def test_import_multigroup_training_metrics_feedback_and_nested_telemetry(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "training/environments/kernel_rlm"))
    artifacts = importlib.import_module("kernel_trl.artifacts")
    source = tmp_path / "source"
    source.mkdir()
    groups = [
        {"task": "task-first", "rewards": [0.2, 0.8], "admitted": True},
        {
            "task": "task-next",
            "rewards": [0.1, 0.9],
            "admitted": True,
            "group_id": "g-next",
            "episodes": [{"episode_id": "e-next", "attempt_id": "a1"}],
        },
    ]
    (source / "report.json").write_text(
        json.dumps(
            {
                "run_id": "20261019T000002Z-curriculum",
                "status": "paused",
                "optimizer_steps": 2,
                "groups": groups,
                "results": [
                    {
                        "task_id": "task-first",
                        "status": "trained",
                        "optimizer_step": 1,
                        "metrics": {"train_loss": -0.2},
                    },
                    {
                        "task_id": "task-next",
                        "status": "trained",
                        "optimizer_step": 2,
                        "metrics": {"train_loss": -0.1},
                    },
                ],
            }
        )
    )
    (source / "events.jsonl").write_text(
        json.dumps({"phase": "driver_started", "utc": "2026-10-19T00:00:00+00:00"}) + "\n"
    )
    data = b'{"_task_id":"task-next"}\n'
    (source / "inputs/lvl1_seed42.jsonl").parent.mkdir(parents=True)
    (source / "inputs/lvl1_seed42.jsonl").write_bytes(data)
    sources = {
        "kernel_trl/__init__.py": (
            "training/environments/kernel_rlm/kernel_trl/__init__.py",
            b"env",
        ),
        "rlm/__init__.py": ("rlm/__init__.py", b"core"),
        "training/src/rlm_train/__init__.py": ("training/src/rlm_train/__init__.py", b"worker"),
    }
    source_hashes = {}
    for relative, (repository_path, content) in sources.items():
        path = source / "repo" / repository_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        source_hashes[relative] = hashlib.sha256(content).hexdigest()
    (source / "inputs.json").write_text(
        json.dumps(
            {
                "source_sha256": source_hashes,
                "dataset_sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    )
    group = source / "groups/step-002-task-next/collection/rollouts/group-0-trajectory-0"
    group.mkdir(parents=True)
    identity = {"group_id": "g-next", "episode_id": "e-next", "attempt_id": "a1"}
    (group / "task.json").write_text(json.dumps({"name": "task-next", **identity}))
    (group / "submission.json").write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "task"},
                    {"role": "assistant", "content": "answer"},
                ],
                "submission": "kernel source",
                "termination_reason": "submitted",
                **identity,
            }
        )
    )
    (group / "score.json").write_text(
        json.dumps({"reward": 0.9, "verification": {"correct": True}, **identity})
    )
    call = {
        "branch": "root",
        "elapsed_seconds": 1.0,
        "response": {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {"role": "assistant", "content": "answer"},
                }
            ],
            "usage": {"completion_tokens": 4},
        },
    }
    (group / "calls.jsonl").write_text(json.dumps(call) + "\n")
    (group / "feedback-test.jsonl").write_text(json.dumps({"feedback": "compiled"}) + "\n")
    (group / "program.log").write_text("REPL finished")
    step = source / "groups/step-002-task-next"
    (step / "update.log").write_text(
        "{'loss': '-0.1', 'grad_norm': '0.3', 'learning_rate': '1e-05'}\n"
    )
    (step / "training-telemetry.csv").write_text(
        "index, timestamp, memory.used [MiB], utilization.gpu [%], power.draw [W]\n"
        "0, 2026/10/19 00:01:00.000, 1000 MiB, 50 %, 100.0 W\n"
    )
    destination = tmp_path / "imported"

    result = artifacts.import_run(source, destination)

    traces = json.loads((destination / "traces.jsonl").read_text())["traces"]
    effective = json.loads((destination / "effective/traces.jsonl").read_text())["traces"]
    metrics = [
        json.loads(line) for line in (destination / "metrics.jsonl").read_text().splitlines()
    ]
    telemetry = [
        json.loads(line) for line in (destination / "gpu_stats.jsonl").read_text().splitlines()
    ]
    manifest = json.loads((destination / "provenance.json").read_text())

    assert result["trajectories"] == 1
    assert traces[0]["info"]["feedback_results"] == [{"feedback": "compiled"}]
    assert traces[0]["info"]["program_log"] == "REPL finished"
    assert effective == traces
    assert {item["step"] for item in metrics} == {1, 2}
    assert next(item for item in metrics if item["step"] == 2)["grad_norm"] == 0.3
    assert telemetry[0]["gpus"][0]["memory_used_mib"] == 1000
    assert {item["role"] for item in manifest["inputs"]} == {"dataset", "executed source"}
