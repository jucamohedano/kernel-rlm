from dashboard import data


def test_usage_coverage_zero_unknown_and_single_root_source():
    trace = {
        "info": {
            "kernel_model_calls": [
                {"usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}},
                {"usage": {"prompt_tokens": 4}},
            ]
        },
        "calls": [{"usage": {"prompt_tokens": 900}}],
        "nodes": [],
    }
    usage = data.token_usage(trace)
    assert usage["root"]["input_tokens"] == {"sum": 4, "recorded": 2, "total": 2}
    assert usage["root"]["output_tokens"] == {"sum": 0, "recorded": 1, "total": 2}
    assert usage["root"]["total_tokens"] == {"sum": 0, "recorded": 1, "total": 2}
    assert usage["children"]["input_tokens"] == {"sum": None, "recorded": 0, "total": None}


def test_historical_calls_and_child_response_or_direct_usage():
    trace = {
        "calls": [{"usage": {"prompt_tokens": 3, "completion_tokens": 2}}],
        "info": {
            "kernel_child_progress": [{"response": {"usage": {"prompt_tokens": 999}}}],
            "kernel_subcalls": [
                {
                    "response": {
                        "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}
                    }
                },
                {"usage": {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9}},
            ],
        },
    }
    usage = data.token_usage(trace)
    assert usage["root"]["input_tokens"]["sum"] == 3
    assert usage["children"]["input_tokens"] == {"sum": 12, "recorded": 2, "total": 2}
    assert usage["all"]["output_tokens"]["sum"] == 5


def test_children_without_ledger_are_unknown_but_empty_ledger_is_zero():
    missing = data.token_usage({"info": {}})
    empty = data.token_usage({"info": {"kernel_subcalls": []}})
    assert missing["children"]["input_tokens"]["total"] is None
    assert empty["children"]["input_tokens"] == {"sum": 0, "recorded": 0, "total": 0}
    assert missing["children"]["input_tokens"]["sum"] is None


def test_per_metric_coverage_marks_partial_and_duration_stays_run_scoped():
    trace = {
        "info": {
            "kernel_model_calls": [
                {"usage": {"prompt_tokens": 4, "completion_tokens": 2}},
                {"usage": {"prompt_tokens": 3}},
            ]
        }
    }
    metrics = data.trajectory_metrics(trace)
    assert metrics["token_usage"]["root"]["output_tokens"] == {"sum": 2, "recorded": 1, "total": 2}
    assert metrics["token_usage"]["root"]["input_tokens"]["sum"] == 7
    assert metrics["token_usage"]["root"]["output_tokens"]["sum"] == 2


def test_missing_final_verifier_timing_is_not_synthesized():
    detail = data.trajectory_detail(
        {"id": "run", "report": {}, "experiments": []}, {"id": "t", "info": {}}, 0
    )
    assert detail["verification"] == {}
    assert detail["metrics"]["token_usage"]["children"]["input_tokens"]["sum"] is None


def test_per_call_elapsed_sum_has_coverage_and_is_not_wall_time():
    usage = data.token_usage(
        {
            "info": {
                "kernel_model_calls": [
                    {"elapsed_seconds": 2.0, "usage": {"prompt_tokens": 1}},
                    {"elapsed_seconds": 3.0, "usage": {"prompt_tokens": 2}},
                ],
                "kernel_subcalls": [
                    {"elapsed_seconds": 4.0, "response": {"usage": {"prompt_tokens": 3}}}
                ],
            }
        }
    )
    assert usage["root"]["elapsed_seconds"] == {"sum": 5.0, "recorded": 2, "total": 2}
    assert usage["children"]["elapsed_seconds"] == {"sum": 4.0, "recorded": 1, "total": 1}
    assert usage["all"]["elapsed_seconds"]["sum"] == 9.0


def test_missing_root_call_ledger_is_unknown_not_zero():
    usage = data.token_usage({"info": {}})
    assert usage["root"]["input_tokens"] == {"sum": None, "recorded": 0, "total": None}
    assert usage["root"]["elapsed_seconds"]["total"] is None
