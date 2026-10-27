import json
import threading
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path

from dashboard import data
from dashboard.server import make_handler


def request(server, path):
    connection = HTTPConnection(*server.server_address)
    connection.request("GET", path)
    response = connection.getresponse()
    body = response.read()
    connection.close()
    return response.status, response.getheader("Content-Type"), body


def test_registry_and_missing_receipt_are_explicit():
    experiments = data.experiments()
    assert all({"id", "title", "status", "runs"} <= set(item) for item in experiments)
    assert all(item["campaign"] and item["category"] and item["scope_note"] for item in experiments)
    campaign, warnings = data.campaign(None)
    assert campaign["status"] is None
    assert "not configured" in warnings[0]


def test_invalid_receipt_does_not_fallback(tmp_path):
    receipt = tmp_path / "receipt.json"
    receipt.write_text("not json")
    campaign, warnings = data.campaign(receipt)
    assert campaign["status"] is None
    assert warnings


def test_final_run_base_model_has_evidence_and_does_not_imply_checkpoint():
    run = {"id": "20261020T230200Z-final-l1-pass"}
    trace = {"info": {"kernel_subcalls": [{"model": "kernel-policy-30"}]}}
    detail = data.trajectory_detail(run, trace, 0)
    assert detail["model"] == "poolside/Laguna-XS.2"
    assert detail["model_scope"] == "base"
    assert detail["model_revision"] == "a397bde04501591d0ac3d74a5d9d0a4ec298ffb8"
    assert detail["model_source"].endswith("/trl-l1-final-pass/provenance/base_model")
    assert detail["model_evidence"]["path"] == "notes/results/20261021-final-pass-completion.json"
    assert detail["model_evidence"]["model_field"] == "retirement.pinned_base_repo"
    assert detail["raw"]["info"]["kernel_subcalls"][0]["model"] == "kernel-policy-30"
    assert "checkpoint" not in detail


def test_recorded_models_take_precedence_over_curated_base():
    run = {"id": "20261020T230200Z-final-l1-pass", "report": {"model_id": "reported-model"}}
    detail = data.trajectory_metadata(run, {"info": {"model": "trace-model"}}, 0)
    assert detail["model"] == "trace-model"
    assert detail["model_source"] == "trace.info.model"
    assert detail["model_scope"] == "recorded"
    report_detail = data.trajectory_metadata(run, {}, 0)
    assert report_detail["model"] == "reported-model"
    assert report_detail["model_source"] == "report.model_id"
    assert report_detail["model_scope"] == "recorded"


def test_missing_and_ambiguous_model_declarations_remain_unknown(monkeypatch):
    assert data.trajectory_metadata({"id": "unrelated-run"}, {}, 0)["model"] is None
    declaration = {
        "id": "a",
        "runs": ["ambiguous"],
        "provenance": {"base_model": {"id": "model-a"}},
        "conclusion": "Uses model-a, not a structured source of identity.",
    }
    monkeypatch.setattr(data, "experiments", lambda: [declaration, {**declaration, "id": "b"}])
    result = data.trajectory_metadata({"id": "ambiguous"}, {}, 0)
    assert result["model"] is None
    assert result["model_source"] is None
    monkeypatch.setattr(data, "experiments", lambda: [{**declaration, "provenance": {}}])
    assert data.trajectory_metadata({"id": "ambiguous"}, {}, 0)["model"] is None


def test_run_api_and_trajectory_use_the_same_base_model(tmp_path):
    run = tmp_path / "20261020T230200Z-final-l1-pass"
    run.mkdir()
    (run / "report.json").write_text(json.dumps({"run_id": run.name}))
    (run / "traces.jsonl").write_text(json.dumps({"traces": [{"id": "sample"}]}))
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(tmp_path))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, _, body = request(server, f"/api/runs/{run.name}")
        assert status == 200
        payload = json.loads(body)
        for key in ("model", "model_scope", "model_source", "model_revision", "model_evidence"):
            assert payload["run"][key] == payload["trajectories"][0][key]
        assert payload["run"]["model"] == "poolside/Laguna-XS.2"
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_real_trace_normalization():
    trace = {
        "id": "t1",
        "task": {"data": {"name": "task"}},
        "nodes": [{"message": {"role": "assistant", "content": "node answer"}}],
        "info": {
            "kernel_submission": "x",
            "kernel_termination": "submitted",
            "kernel_verify": {"correct": False, "speedup": 2.5},
            "repl_transcript": [{"role": "assistant", "content": "transcript answer"}],
            "kernel_subcalls": [{"request": "child"}],
        },
    }
    detail = data.trajectory_detail({"id": "r"}, trace, 0)
    assert detail["task_source"] is None
    assert detail["outcome"]["status"] == "submitted"
    assert detail["outcome"]["speedup"] == 2.5
    assert detail["events"][0]["content"] == "transcript answer"
    assert len([event for event in detail["events"] if event.get("source") == "nodes"]) == 0
    assert detail["events"][-1]["source"] == "kernel_subcalls"


def test_http_api_routes_and_writing_safety(tmp_path):
    run = tmp_path / "r1"
    run.mkdir()
    (run / "report.json").write_text(json.dumps({"run_id": "r1", "status": "done"}))
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(tmp_path))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, _, body = request(server, "/api/runs")
        assert status == 200 and json.loads(body)["runs"][0]["id"] == "r1"
        status, content_type, font = request(server, "/fonts/InterVariable.woff2")
        assert status == 200 and content_type == "font/woff2" and font[:4] == b"wOF2"
        status, content_type, script = request(server, "/theme.js")
        assert status == 200 and "javascript" in content_type
        assert script == (Path(__file__).parents[1] / "dashboard/theme.js").read_bytes()
        status, _, _ = request(server, "/fonts/../../README.md")
        assert status == 404
        status, _, body = request(server, "/api/home")
        assert status == 200 and json.loads(body)["campaign"]["status"] is None
        status, _, body = request(server, "/api/writing")
        assert status == 200
        documents = json.loads(body)["documents"]
        literature = next(doc for doc in documents if doc["id"] == "literature-review")
        assert literature["path"] == "docs/research/literature-review.md"
        status, _, body = request(server, "/api/writing/literature-review")
        assert status == 200
        document = json.loads(body)
        assert (
            document["markdown"]
            == (Path(__file__).parents[1] / "docs/research/literature-review.md").read_text()
        )
        assert len(document["version"]) == 64
        status, _, _ = request(server, "/api/writing/../../AGENTS")
        assert status == 404
        status, _, _ = request(server, "/api/runs/nope")
        assert status == 404
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_completion_receipt_normalizes_counts_and_avoids_false_hardware(tmp_path):
    receipt = tmp_path / "completion.json"
    receipt.write_text(
        json.dumps(
            {
                "training_status": "nondeferred_pass_finished",
                "trained_tasks": 35,
                "deferred_untrained": ["a", "b"],
                "gpu_processes": [],
                "checkpoint35_backup": {"archive_sha256": "abc"},
            }
        )
    )
    campaign, warnings = data.campaign(receipt)
    assert campaign["status"] == "nondeferred_pass_finished"
    assert campaign["trained"] == 35
    assert campaign["deferred"] == 2
    assert campaign["checkpoint"] is None
    assert campaign["hardware"] is None
    assert warnings == []


def test_broad_discovery_is_lightweight_and_reports_linked_missing_run(tmp_path):
    root = tmp_path / "live-grpo"
    root.mkdir()
    (root / "20260101T000000Z-sample").mkdir()
    (root / "20260101T000000Z-sample" / "report.json").write_text(
        json.dumps({"run_id": "20260101T000000Z-sample", "status": "completed"})
    )
    records = data.run_records(root)
    assert any(record["id"] == "20260101T000000Z-sample" for record in records)
    assert all("metrics" not in record.get("_reader", {}) for record in records)
    missing = next(record for record in records if record["id"] == "20261019T072942Z-recovery")
    assert missing["evidence_available"] is False
    assert "unavailable" in missing["warnings"][0]


def test_nested_all_and_effective_exports_are_deduplicated(tmp_path):
    from dashboard.data import run_records

    run_dir = tmp_path / "20260101T000000Z-sample"
    (run_dir / "effective").mkdir(parents=True)
    trace = {"id": "episode-1"}
    row = json.dumps({"traces": [trace]}) + "\n"
    (run_dir / "traces.jsonl").write_text(row)
    (run_dir / "effective" / "traces.jsonl").write_text(row)
    record = next(r for r in run_records(tmp_path) if r["id"] == run_dir.name)
    assert record["trajectory_count"] == 1


def test_reasoning_metadata_maps_sampled_root_calls_only():
    trace = {
        "id": "x",
        "info": {
            "repl_transcript": [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"function": {"name": "repl", "arguments": '{"code": "print(1)"}'}}
                    ],
                },
                {"role": "tool", "content": "1"},
            ],
            "kernel_model_calls": [
                {"node": 1, "usage": {"completion_tokens": 9}, "finish_reason": "tool_calls"},
                {"node": 3, "usage": {"completion_tokens": 100}, "finish_reason": "stop"},
            ],
        },
        "nodes": [
            {"message": {"role": "system"}},
            {"message": {"role": "assistant", "reasoning_content": "root rationale"}},
            {"message": {"role": "system"}},
            {"message": {"role": "assistant", "reasoning_content": "child rationale"}},
        ],
    }
    detail = data.trajectory_detail({"id": "x", "root": Path(".")}, trace, 0)
    root_event = detail["events"][0]
    assert root_event["reasoning"] == "root rationale"
    assert root_event["usage"]["completion_tokens"] == 9
    assert root_event["finish_reason"] == "tool_calls"
    assert "print(1)" in root_event["assistant_response"]
    assert "child rationale" not in json.dumps(detail["events"])


def test_selected_episode_is_the_only_full_payload(tmp_path):
    run = tmp_path / "20260101T000000Z-preview"
    run.mkdir()
    (run / "report.json").write_text(json.dumps({"run_id": run.name}))
    (run / "traces.jsonl").write_text(
        json.dumps(
            {
                "traces": [
                    {"id": "a", "info": {"repl_transcript": [{"role": "user", "content": "A"}]}},
                    {"id": "b", "info": {"repl_transcript": [{"role": "user", "content": "B"}]}},
                ]
            }
        )
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(tmp_path))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, _, body = request(server, f"/api/runs/{run.name}")
        assert status == 200
        assert all("raw" not in item for item in json.loads(body)["trajectories"])
        status, _, body = request(server, f"/api/runs/{run.name}?trajectory=b")
        records = json.loads(body)["trajectories"]
        assert [item["id"] for item in records if "raw" in item] == ["b"]
        status, _, body = request(server, "/api/runs/20261019T072942Z-recovery")
        assert status == 200
        assert json.loads(body)["run"]["evidence_available"] is False
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_run_detail_api_exposes_recorded_elapsed_scope_and_trajectory_metrics(tmp_path):
    run = tmp_path / "20261020T230200Z-final-l1-pass"
    run.mkdir()
    (run / "report.json").write_text(
        json.dumps(
            {
                "run_id": run.name,
                "status": "completed",
                "elapsed_seconds": 13819.226649284363,
                "elapsed_scope": "run",
            }
        )
    )
    trace = {
        "id": "t1",
        "info": {
            "kernel_model_calls": [
                {
                    "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
                    "elapsed_seconds": 2.5,
                }
            ],
            "kernel_subcalls": [
                {
                    "response": {
                        "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}
                    },
                    "elapsed_seconds": 1.25,
                }
            ],
        },
    }
    (run / "traces.jsonl").write_text(json.dumps({"traces": [trace]}))
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(tmp_path))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, _, body = request(server, f"/api/runs/{run.name}")
        result = json.loads(body)
        assert status == 200
        assert result["run"]["elapsed_seconds"] == 13819.226649284363
        assert result["run"]["elapsed_scope"] == "run"
        usage = result["trajectories"][0]["metrics"]["token_usage"]
        assert usage["root"]["elapsed_seconds"] == {"sum": 2.5, "recorded": 1, "total": 1}
        assert usage["children"]["elapsed_seconds"] == {"sum": 1.25, "recorded": 1, "total": 1}
    finally:
        server.shutdown()
        thread.join()
        server.server_close()
