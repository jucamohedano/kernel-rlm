import importlib.util
import json
from pathlib import Path

SERVER = Path(__file__).parents[1] / "training/environments/kernel_rlm/dashboard/server.py"
SPEC = importlib.util.spec_from_file_location("kernel_dashboard", SERVER)
dashboard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dashboard)


def test_nested_rollout_navigation_and_escaping(tmp_path):
    run = tmp_path / "20261008-test"
    run.mkdir()
    (run / "report.json").write_text(json.dumps({"run_id": run.name, "status": "failed"}))
    trace = {
        "task": {"data": {"name": "addition"}},
        "rewards": {"kernel_reward": {"score": 0}},
        "info": {
            "kernel_submission": "",
            "repl_transcript": [
                {"role": "system", "content": "system prompt"},
                {"role": "assistant", "content": "<script>alert(1)</script>"},
            ],
        },
        "nodes": [{"message": {"tool_calls": [{"name": "repl"}]}}],
        "calls": [{"node": 0, "finish_reason": "length", "usage": {"completion_tokens": 10}}],
    }
    (run / "traces.jsonl").write_text(
        "\n".join(json.dumps({"traces": [trace]}) for index in range(15))
    )
    page = dashboard.render_run(tmp_path, run.name, 14)
    assert "15 / 15" in page
    assert 'value="14" selected' in page
    assert "0.000" in page
    assert "1 length-limited calls" in page
    assert "No final submission" in page
    assert "Structured tool calls" in page
    assert "system prompt" in page
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page
    assert "<script>alert(1)</script>" not in page
    assert "<details open" not in page


def test_empty_run_and_out_of_range_selection(tmp_path):
    run = tmp_path / "empty"
    run.mkdir()
    (run / "report.json").write_text('{"run_id":"empty"}')
    page = dashboard.render_run(tmp_path, "empty", 999)
    assert "No trajectory messages downloaded" in page
    assert "No series recorded" in page
    assert "Run not found" in dashboard.render_run(tmp_path, "missing")


def test_failed_trace_without_final_transcript():
    page = dashboard.render_timeline(
        {
            "nodes": [
                {"message": {"role": "user", "content": "initial prompt"}},
                {"message": {"role": "assistant", "content": "partial response"}},
            ],
            "calls": [{"node": 1, "finish_reason": "stop"}],
        }
    )
    assert "initial prompt" in page
    assert "partial response" in page


def test_child_calls_do_not_shift_parent_token_labels():
    page = dashboard.render_timeline(
        {
            "info": {
                "repl_transcript": [{"role": "assistant", "content": "parent reply"}],
                "kernel_model_calls": [
                    {"finish_reason": "stop", "usage": {"completion_tokens": 15}}
                ],
                "kernel_subcalls": [{"prompt": "child prompt", "response": "<child reply>"}],
            },
            "calls": [{"finish_reason": "length", "usage": {"completion_tokens": 2048}}],
            "nodes": [
                {"parent": None, "message": {"role": "system", "content": "root"}},
                {"parent": None, "message": {"role": "system", "content": "child policy"}},
            ],
        }
    )
    assert "15 output tokens" in page
    assert "2048 output tokens" not in page
    assert "child prompt" in page
    assert "child policy" in page
    assert "&lt;child reply&gt;" in page
    assert "<details open" not in page


def test_index_has_descriptive_labels_outcomes_and_search(tmp_path):
    run = tmp_path / "frozen-probe" / "20261008T225956Z-5733f992"
    run.mkdir(parents=True)
    (run / "report.json").write_text(
        json.dumps(
            {
                "run_id": run.name,
                "status": "group_passed",
                "elapsed_seconds": 600,
                "model_id": "<script>unsafe</script>",
                "groups": [{"rewards": [0.7, 0.7, 0.7, 0], "correct": 3}],
            }
        )
    )
    page = dashboard.render_index(tmp_path)
    assert "Fixed syntax check · three correct submissions" in page
    assert "3/4 correct final submissions" in page
    assert "10.0 min" in page
    assert 'id="run-search"' in page
    assert "<script>unsafe</script>" not in page
    assert "&lt;script&gt;unsafe&lt;/script&gt;" in page
    assert run.name in page
    (run / "report.json").write_text(json.dumps({"run_id": run.name, "groups": {"legacy": 4}}))
    assert "No completed rollouts recorded" in dashboard.render_index(tmp_path)
