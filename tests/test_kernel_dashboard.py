import hashlib
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
    assert "GPU telemetry not downloaded" in page
    assert "<polyline" not in page
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
    page = dashboard.render_index(tmp_path, view="runs")
    assert "Fixed syntax check · three correct submissions" in page
    assert "3/4 correct final submissions" in page
    assert "10.0 min" in page
    assert 'id="run-search"' in page
    assert "<script>unsafe</script>" not in page
    assert "&lt;script&gt;unsafe&lt;/script&gt;" in page
    assert run.name in page
    (run / "report.json").write_text(json.dumps({"run_id": run.name, "groups": {"legacy": 4}}))
    assert "Outcome unavailable in downloaded evidence" in dashboard.render_index(
        tmp_path, view="runs"
    )


def test_project_guide_is_available_from_dashboard():
    page = dashboard.render_project_guide()
    assert "Train Laguna to write CUDA kernels" in page
    assert "Project brief" in page


def test_dates_sort_across_categories_and_legacy_ids_still_open(tmp_path):
    older = tmp_path / "z-contract" / "20261007T120000Z-old"
    newer = tmp_path / "a-training" / "20261016T120000Z-new"
    for directory in (older, newer):
        directory.mkdir(parents=True)
        (directory / "report.json").write_text("{}")
    runs = dashboard.discover_runs(tmp_path)
    assert [run["id"] for run in runs] == [newer.name, older.name]
    assert "16 Oct 2026" in dashboard.run_date(runs[0])
    assert "Run not found" not in dashboard.render_run(tmp_path, f"a-training/{newer.name}")


def test_prime_all_and_effective_are_deduplicated_with_separate_admission(tmp_path):
    directory = tmp_path / "live-grpo" / "20261016T120000Z-test"
    directory.mkdir(parents=True)
    (directory / "report.json").write_text(json.dumps({"run_id": directory.name}))
    traces = [
        {
            "id": f"episode-{index}",
            "info": {
                "kernel_termination": "submitted",
                "kernel_reward": 0.9 if index >= 4 else 0.1,
                "kernel_verify": {"correct": index >= 4},
                "kernel_model_calls": [{"finish_reason": "length" if index == 0 else "stop"}],
            },
        }
        for index in range(8)
    ]
    for cohort, values in (("all", traces), ("effective", traces[4:])):
        path = directory / "rollouts/step_2/train" / cohort / "traces.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text("\n".join(json.dumps({"traces": [trace]}) for trace in values))
    summary = dashboard.run_summary(dashboard.discover_runs(tmp_path)[0])
    assert summary["completed"] == summary["submitted"] == 8
    assert summary["correct"] == summary["admitted"] == 4
    assert summary["model_calls"] == 8
    assert summary["truncated_calls"] == 1
    page = dashboard.render_index(tmp_path, view="runs")
    assert "4/8 correct final submissions" in page
    assert "4 / 8 admitted for training" in page


def test_legacy_trace_correctness_is_not_reported_as_missing(tmp_path):
    directory = tmp_path / "model-rollout" / "20261001T120000Z-test"
    directory.mkdir(parents=True)
    (directory / "trace.json").write_text(json.dumps({"verification": {"correct": True}}))
    summary = dashboard.run_summary(dashboard.discover_runs(tmp_path)[0])
    assert summary["completed"] == summary["correct"] == 1


def test_real_metric_names_and_single_point_do_not_invent_a_curve(tmp_path):
    directory = tmp_path / "20261016T120000Z-test"
    directory.mkdir()
    (directory / "metrics.jsonl").write_text(
        json.dumps(
            {
                "producer": "trainer",
                "step": 2,
                "loss/mean": -0.0023,
                "entropy/all/mean": 0.13,
            }
        )
        + "\n"
    )
    run = dashboard.discover_runs(tmp_path)[0]
    assert dashboard.run_summary(run)["optimizer_step"] == 2
    page = dashboard.render_run(tmp_path, directory.name)
    assert "One saved point; no trend established" in page
    assert "Optimizer step" in page
    assert "-0.0023" in page
    assert "<polyline" not in page
    assert "Entropy (nats)" in page


def test_gpu_telemetry_has_units_labels_and_duration_provenance(tmp_path):
    directory = tmp_path / "20261016T120000Z-test"
    directory.mkdir()
    (directory / "report.json").write_text("{}")
    samples = [
        {
            "timestamp_utc": timestamp,
            "gpus": [
                {
                    "index": 0,
                    "memory_used_mib": memory,
                    "memory_total_mib": 81920,
                    "gpu_utilization_percent": 25,
                },
            ],
        }
        for timestamp, memory in (("2026-10-16T12:00:00Z", 4096), ("2026-10-16T12:01:00Z", 8192))
    ]
    (directory / "gpu_stats.jsonl").write_text("\n".join(json.dumps(sample) for sample in samples))
    run = dashboard.discover_runs(tmp_path)[0]
    summary = dashboard.run_summary(run)
    assert summary["duration"] == 60
    assert "excludes queue time" in summary["duration_source"]
    page = dashboard.render_gpu_charts(run)
    assert "GPU memory (GiB)" in page
    assert "GPU utilization (%)" in page
    assert "Minutes since first sample" in page
    assert "GPU 0" in page
    assert "gpu_stats.jsonl" in page


def test_lineage_uses_explicit_resume_and_retry_relations_only(tmp_path, monkeypatch):
    parent = "20261016T100000Z-parent"
    failed = "20261016T110000Z-failed"
    retry = "20261016T120000Z-retry"
    unrelated = "20261016T130000Z-unrelated"
    for identity in (parent, failed, retry, unrelated):
        directory = tmp_path / identity
        directory.mkdir()
        report = {"run_id": identity, "title": identity}
        if identity in (failed, retry):
            report.update(resumed_from=parent, resume_step=1)
        (directory / "report.json").write_text(json.dumps(report))
    monkeypatch.setattr(
        dashboard,
        "experiment_catalog",
        lambda: [
            {
                "relationships": [
                    {
                        "from": failed,
                        "to": retry,
                        "kind": "retry",
                        "label": "Retry, not checkpoint inheritance",
                    },
                ]
            }
        ],
    )
    page = dashboard.render_lineage(tmp_path)
    assert page.count('class="lineage-edge"') == 3
    assert page.count("Resume requested from step 1 checkpoint") == 2
    assert "Retry, not checkpoint inheritance" in page
    assert unrelated not in page


def test_prepared_experiment_has_no_invented_live_results(tmp_path):
    page = dashboard.render_index(tmp_path, "reply-budget")
    assert "prepared" in page
    assert "4,096" in page
    assert "No run launched for this change" in page
    assert "not established" in page
    assert "<details open" not in page


def test_guide_shares_navigation_without_copying_document():
    page = dashboard.render_guide_workspace()
    assert 'class="app-frame"' in page
    assert 'class="guide"' in page
    assert "Train Laguna to write CUDA kernels" in page
    assert page.count("<!doctype html>") == 1
    assert "<math" in page


def test_prepared_manifest_is_discoverable_and_snapshots_are_whitelisted(tmp_path):
    directory = tmp_path / "20261017T120000Z-prepared"
    directory.mkdir()
    content = "<script>source, not executable HTML</script>"
    manifest = {
        "run_id": directory.name,
        "experiment_id": "reply-budget",
        "created_at_utc": "2026-10-17T12:00:00Z",
        "pins": {"model_id": "test"},
        "inputs": [
            {
                "path": "config.toml",
                "role": "config",
                "sha256": hashlib.sha256(content.encode()).hexdigest(),
                "bytes": len(content),
            }
        ],
    }
    (directory / "provenance.json").write_text(json.dumps(manifest))
    (directory / "inputs").mkdir()
    (directory / "inputs/config.toml").write_text(content)
    runs = dashboard.discover_runs(tmp_path)
    assert runs[0]["report"]["status"] == "prepared"
    assert "asset=config.toml" in dashboard.render_provenance(tmp_path, directory.name)
    page = dashboard.render_input(tmp_path, directory.name, "config.toml")
    assert "matches manifest" in page
    assert "<script>source" not in page
    assert "&lt;script&gt;source" in page
    assert "Input not registered" in dashboard.render_input(
        tmp_path, directory.name, "../../etc/passwd"
    )
    (directory / "inputs/config.toml").write_text("modified")
    assert "MISMATCH" in dashboard.render_input(tmp_path, directory.name, "config.toml")
    assert "Inputs &amp; setup" in dashboard.render_run(tmp_path, directory.name)


def test_historical_runs_do_not_gain_fake_provenance(tmp_path):
    directory = tmp_path / "20261016T120000Z-old"
    directory.mkdir()
    (directory / "report.json").write_text("{}")
    page = dashboard.render_provenance(tmp_path, directory.name)
    assert "Not captured before launch" in page
    assert "No exact commit" in page


def test_main_navigation_has_only_three_sections(tmp_path):
    page = dashboard.render_index(tmp_path, "checkpoint-resume")
    navigation = page.split('<nav aria-label="Workspace">')[1].split("</nav>")[0]
    assert navigation.count("<a ") == 3
    assert "Overview" in navigation
    assert "Experiments" in navigation
    assert "Learning report" in navigation
    assert "Provenance" not in navigation
    assert "Lineage" not in navigation
    assert "<h2>Intent</h2>" in page
    assert "<h2>Conclusion</h2>" in page
    assert "section=lineage" in page


def test_preparation_is_not_counted_as_execution(tmp_path):
    directory = tmp_path / "20261018T120000Z-prepared"
    directory.mkdir()
    (directory / "provenance.json").write_text(
        json.dumps({"run_id": directory.name, "experiment_id": "reply-budget", "inputs": []})
    )
    page = dashboard.render_index(tmp_path, "reply-budget")
    assert "0 executions" in page
    assert "Prepared inputs · no execution started" in page
    assert "<tr data-run>" not in page
    assert f"run={directory.name}&amp;tab=setup" in page
    assert "<details open" not in page


def test_run_tabs_are_bookmarkable_and_scoped_to_experiment(tmp_path):
    directory = tmp_path / "20261018T120000Z-test"
    directory.mkdir()
    (directory / "report.json").write_text(json.dumps({"experiment_id": "checkpoint-resume"}))
    page = dashboard.render_run(tmp_path, directory.name, tab="setup")
    assert 'href="/?experiment=checkpoint-resume#runs"' in page
    assert 'id="tab-setup" aria-controls="setup" data-tab="setup" aria-selected="true"' in page
    assert 'id="trace" hidden' in page
    assert "No exact commit or dataset hash is inferred" in page
    fallback = dashboard.render_run(tmp_path, directory.name, tab="invalid")
    assert 'data-tab="trace" aria-selected="true"' in fallback
    assert "<details open" not in page


def test_lineage_scopes_run_edges_and_distinguishes_research_questions(tmp_path):
    for name, report in (
        (
            "resume",
            {"experiment_id": "checkpoint-resume", "resumed_from": "original", "status": "passed"},
        ),
        ("unrelated", {"experiment_id": "other", "resumed_from": "other-parent"}),
    ):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "report.json").write_text(json.dumps(report))
    page = dashboard.render_index(tmp_path, "checkpoint-resume", section="lineage")
    assert "Follow-up research question" in page
    assert "does not imply inherited model weights" in page
    assert "Checkpoint and execution relationships" in page
    assert 'href="/?run=original"' in page
    assert "other-parent" not in page
    assert "<details open" not in page
