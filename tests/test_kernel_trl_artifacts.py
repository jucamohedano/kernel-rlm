import importlib
import json
from pathlib import Path


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
        json.dumps({"branch": "root", "response": response, "elapsed_seconds": 1.0}) + "\n"
    )
    trace = artifacts.episode_trace(directory, "run")
    assert trace["info"]["repl_transcript"][0]["content"] == "all system instructions"
    assert trace["nodes"][1]["message"]["reasoning_content"] == "saved reasoning"
    assert trace["calls"][0]["node"] == 1
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
