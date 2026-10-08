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
