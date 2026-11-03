"""Child accounting and per-call limits use recorded evidence only."""

import json

from dashboard import data


def test_new_child_ledger_wins_and_rejections_are_not_model_calls():
    child = {
        "call_id": "c",
        "request": {"max_tokens": 300},
        "response": {
            "id": "response-c",
            "usage": {"prompt_tokens": 10, "completion_tokens": 234, "total_tokens": 244},
        },
    }
    trace = {
        "calls": [],
        "info": {
            "rlm_child_calls": [child, child, {"error": "budget exhausted"}],
            "kernel_subcalls": [{"usage": {"total_tokens": 999}}],
        },
    }
    usage = data.token_usage(trace)
    assert usage["children"]["total_tokens"] == {"sum": 244, "recorded": 1, "total": 1}
    assert usage["children"]["output_cap_tokens"] == {"sum": 300, "recorded": 1, "total": 1}
    trace["info"]["rlm_child_calls"] = []
    assert data.token_usage(trace)["children"]["total_tokens"]["sum"] == 0


def test_root_and_child_limits_are_joined_from_recorded_requests():
    root = {
        "call_id": "root",
        "node": 0,
        "usage": {"prompt_tokens": 10, "completion_tokens": 234, "total_tokens": 244},
    }
    trace = {
        "id": "t",
        "calls": [root],
        "nodes": [{"message": {"role": "assistant"}}],
        "info": {
            "repl_transcript": [{"role": "assistant"}],
            "evaluation": {"events": [{"call_id": "root", "request": {"max_tokens": 300}}]},
            "rlm_child_calls": [
                {
                    "request": {"max_tokens": 50},
                    "response": {
                        "usage": {"prompt_tokens": 5, "completion_tokens": 20, "total_tokens": 25}
                    },
                }
            ],
        },
    }
    run = {"id": "r", "report": {}, "provenance": {"config": {"model_context_tokens": 1000}}}
    detail = data.trajectory_detail(run, trace, 0)
    root_event, child_event = detail["events"]
    assert root_event["token_limits"]["output_tokens"] == {
        "limit": 300,
        "source": "request.max_tokens",
    }
    assert root_event["token_limits"]["context_tokens"]["limit"] == 1000
    assert child_event["source"] == "rlm_child_calls"
    assert child_event["records"][0]["token_limits"]["output_tokens"]["limit"] == 50
    assert child_event["records"][0]["token_limits"]["context_tokens"]["limit"] == 1000
    assert data.call_limits({})["output_tokens"]["limit"] is None
    assert data.call_limits({})["context_tokens"]["limit"] is None


def test_native_session_tree_usage_is_read_without_an_export(tmp_path):
    path = tmp_path / "run"
    path.mkdir()
    (path / "report.json").write_text(json.dumps({"run_id": "run"}))
    record = {
        "session_id": "root",
        "example_id": "task",
        "model": "m",
        "messages": [],
        "final_answer": "ok",
        "events": [
            {
                "call_id": "r",
                "response": {
                    "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
                },
            }
        ],
        "child_sessions": [
            {
                "session_id": "child",
                "events": [
                    {
                        "call_id": "c",
                        "response": {
                            "usage": {
                                "prompt_tokens": 7,
                                "completion_tokens": 4,
                                "total_tokens": 11,
                            }
                        },
                    }
                ],
            }
        ],
    }
    (path / "task-000.json").write_text(json.dumps(record))
    runs = data.run_records(tmp_path)
    traces, warnings = data.saved_traces(runs[0]["_reader"])
    assert not warnings
    assert runs[0]["trajectory_count"] == 1
    assert data.run_records(tmp_path, count_trajectories=False)[0]["evidence_available"] is True
    usage = data.token_usage(traces[0])
    assert usage["root"]["total_tokens"]["sum"] == 5
    assert usage["children"]["total_tokens"]["sum"] == 11
    assert usage["all"]["total_tokens"]["sum"] == 16


def test_missing_output_limits_do_not_become_a_global_context_budget():
    usage = data.token_usage(
        {
            "calls": [{"usage": {"completion_tokens": 10}}, {"request": {"max_tokens": 100}}],
            "info": {"rlm_child_calls": []},
        }
    )
    assert usage["root"]["output_tokens"] == {"sum": 10, "recorded": 1, "total": 2}
    assert usage["root"]["output_cap_tokens"] == {"sum": 100, "recorded": 1, "total": 2}
    assert "context_window_tokens" not in usage["all"]
