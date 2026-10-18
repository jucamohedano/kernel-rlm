import ast
import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parents[1] / "training/environments/kernel_rlm"
SPEC = importlib.util.spec_from_file_location(
    "kernel_diagnostics", ROOT / "kernel_rlm/diagnostics.py"
)
diagnostics = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostics)


def test_union_does_not_count_overlaps_twice():
    assert diagnostics.union_seconds([(0, 10), (2, 8), (5, 15), (20, 25)]) == 20
    assert diagnostics.union_seconds([]) == 0
    with pytest.raises(ValueError):
        diagnostics.union_seconds([(2, 1)])


def test_percentiles_and_missing_measurements():
    assert diagnostics.distribution([]) is None
    assert diagnostics.distribution([7])["p95_seconds"] == 7
    assert diagnostics.distribution([0, 10])["p50_seconds"] == 5


def test_literal_recovery_does_not_execute_candidate_code():
    source = "source = 'CUDA_SOURCE = ' + repr('kernel')\ncuda_feedback(source)"
    trace = {
        "info": {
            "repl_transcript": [
                {"tool_calls": [{"function": {"arguments": json.dumps({"code": source})}}]}
            ]
        }
    }
    assert diagnostics.feedback_sources(trace) == ["CUDA_SOURCE = 'kernel'"]
    with pytest.raises(ValueError):
        diagnostics.literal_text(
            ast.parse("__import__('os').system('false')", mode="eval").body, {}
        )


def test_failed_assignment_does_not_reuse_stale_source():
    code = "source='old'\nsource=unknown_function()\ncuda_feedback(source)"
    trace = {
        "info": {
            "repl_transcript": [
                {"tool_calls": [{"function": {"arguments": json.dumps({"code": code})}}]}
            ]
        }
    }
    assert diagnostics.feedback_sources(trace) == [None]


def test_embedded_sources_ignore_python_wrapper_but_not_kernel_changes():
    source = "CPP_SOURCE='cpp'\nCUDA_SOURCE='cuda'\n"
    assert diagnostics.embedded_source_hashes(source) == diagnostics.embedded_source_hashes(
        source + "\n"
    )
    assert diagnostics.embedded_source_hashes(source) != diagnostics.embedded_source_hashes(
        source.replace("'cuda'", "'other'")
    )
    assert diagnostics.embedded_source_hashes("malformed(") is None


def test_recovery_tracks_literal_string_updates():
    code = "source='first'\nsource=source+' second'\nsource+=' third'\ncuda_feedback(source)"
    trace = {
        "info": {
            "repl_transcript": [
                {"tool_calls": [{"function": {"arguments": json.dumps({"code": code})}}]}
            ]
        }
    }
    assert diagnostics.feedback_sources(trace) == ["first second third"]


def test_profile_deduplicates_and_keeps_admission_separate(tmp_path):
    traces = [
        {
            "id": f"episode-{index}",
            "task": {"data": {"name": "test"}},
            "info": {
                "kernel_reward": index,
                "kernel_submission": "pass",
                "kernel_verify": {"correct": index == 1},
            },
            "timing": {
                "start": 0,
                "agent": {"model": {"duration": 2}, "harness": {"duration": 3}},
                "scoring": {"start": 5, "end": 7},
            },
            "calls": [{"time": {"start": 0, "end": 2}}],
        }
        for index in range(2)
    ]
    for cohort, selected in (("all", traces + traces), ("effective", traces[1:])):
        path = tmp_path / f"rollouts/step_2/train/{cohort}/traces.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"traces": selected}) + "\n")
    report = diagnostics.profile_run(tmp_path)
    assert report["kind"] == "diagnostic_not_experiment"
    assert len(report["episodes"]) == 2
    assert report["wall_clock"]["model_request_union_seconds"] == 2
    assert report["wall_clock"]["scoring_union_seconds"] == 2
    assert report["wall_clock"]["collection_span_seconds"] == 7
    assert report["task_cohorts"][0]["admitted"] == 1
    assert "final_compile_elapsed_seconds" not in report["stages"]


def test_missing_exports_fail_without_generating_results(tmp_path):
    with pytest.raises(ValueError):
        diagnostics.profile_run(tmp_path)
    with pytest.raises(ValueError):
        diagnostics.select_cases(tmp_path)


def test_replay_has_no_training_or_model_dispatch_and_strict_limits():
    tree = ast.parse((ROOT / "modal_app/cache_diagnostic.py").read_text())
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert "prepare_run" not in ast.unparse(tree)
    for name, timeout in (("cpu_replay", 480), ("gpu_replay", 180)):
        keywords = {
            item.arg: ast.literal_eval(item.value)
            for item in functions[name].decorator_list[0].keywords
            if item.arg in {"timeout", "retries", "max_containers", "gpu", "scaledown_window"}
        }
        assert keywords["timeout"] == timeout
        assert keywords["retries"] == 0
        assert keywords["max_containers"] == 1
        assert 2 <= keywords["scaledown_window"] <= 3600
        if name == "gpu_replay":
            assert keywords["gpu"] == "A100-80GB"
        else:
            assert "gpu" not in keywords
    remote_targets = [
        ast.unparse(node.func.value)
        for node in ast.walk(functions["main"])
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "remote"
    ]
    assert remote_targets == ["cpu_replay", "gpu_replay"]


@pytest.mark.parametrize("v1", [False, True])
@pytest.mark.parametrize("source", ["\nCPP_SOURCE='cpp'\nCUDA_SOURCE='cuda'\n", " \n\t"])
def test_scoring_preserves_source_bytes_and_rejects_blank_content(v1, source):
    filename = "kernel_rlm_v1/taskset.py" if v1 else "kernel_rlm/env.py"
    name = "kernel_reward" if v1 else "score"
    method = next(
        node
        for node in ast.walk(ast.parse((ROOT / filename).read_text()))
        if isinstance(node, ast.AsyncFunctionDef) and node.name == name
    )
    method.decorator_list = []
    method.returns = None
    for argument in method.args.args:
        argument.annotation = None
    if method.args.kwarg:
        method.args.kwarg.annotation = None
    seen = []

    def verify(reference, submitted):
        seen.append(submitted)
        return {"compiled": True, "extension_used": True, "correct": True, "speedup": 1}

    async def to_thread(function, *args):
        return function(*args)

    async def read(*args, **kwargs):
        return json.dumps({"submission": source, "repl_calls": 1, "messages": []})

    namespace = {
        "asyncio": SimpleNamespace(to_thread=to_thread),
        "json": json,
        "verify_cuda": verify,
        "validate_submission_source": lambda value: (True, ""),
        "mask_child_branches": lambda trace: None,
        "os": SimpleNamespace(environ={}),
    }
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), "scoring-diagnostic", "exec"), namespace
    )
    trace = SimpleNamespace(info={})
    state = {"rlm_final_answer": source}
    if v1:
        task = SimpleNamespace(data=SimpleNamespace(context="reference", speed_reward_weight=0.1))
        coroutine = namespace[name](task, trace, SimpleNamespace(read=read))
    else:
        coroutine = namespace[name]({"context": "reference"}, state)
    reward = asyncio.run(coroutine)
    if source.strip():
        assert seen == [source]
        assert reward == pytest.approx(0.8)
        if v1:
            assert trace.info["kernel_submission"] == source
        else:
            assert state["kernel_submission_chars"] == len(source)
    else:
        assert seen == []
        assert reward == 0
