"""Synthetic CPU receipt fixtures only; none is evidence of a real GPU run."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict
from types import SimpleNamespace

import pytest
from cute_rlm.icl.repl_experiment import ReplExperimentConfig
from cute_rlm.icl.repl_readiness import (
    MODE,
    REQUIRED_CHECKS,
    check_model_stopped,
    check_readiness,
    digest,
)
from cute_rlm.server_identity import model_card_sha256


def write_cpu_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    return {"path": path.name, "sha256": digest(path.read_bytes())}


def cpu_readiness_fixture(tmp_path, config=None):
    """Construct fictional observations without any GPU, process or HTTP access."""
    config = config or ReplExperimentConfig()
    run = tmp_path / "CPU_FIXTURE_NOT_GPU"
    run.mkdir()
    write_cpu_json(run / "config.json", asdict(config))
    plan = {"mode": MODE, "run_id": run.name, "artifacts": {}}
    write_cpu_json(run / "plan.json", plan)
    bindings = {
        "mode": MODE,
        "run_id": run.name,
        "plan_sha256": digest((run / "plan.json").read_bytes()),
        "config_sha256": digest((run / "config.json").read_bytes()),
        "sandbox_spec_sha256": digest(b"CPU fake sandbox spec"),
    }
    base_url = "http://cpu-fixture.invalid:8123/v1"
    card = {
        "id": config.model_id,
        "object": "model",
        "owned_by": "CPU_FIXTURE_NOT_GPU",
        "root": "/CPU_FIXTURE_NOT_GPU/weights/" + config.model_revision,
        "parent": None,
        "max_model_len": 262144,
    }
    identity = {
        "served_model": config.model_id,
        "model_revision": config.model_revision,
        "model_path": card["root"],
        "base_url": base_url,
        "model_card_sha256": model_card_sha256(card),
        "host_id": "CPU_FIXTURE_NOT_GPU:fake-boot-id",
        "port": 8123,
        "owned_processes": [
            {"pid": 101, "start_time": "100010", "role": "api"},
            {"pid": 102, "start_time": "100020", "role": "engine"},
            {"pid": 103, "start_time": "100030", "role": "worker"},
        ],
        "gpu_uuids": [
            "GPU-00000000-0000-0000-0000-000000000001",
            "GPU-00000000-0000-0000-0000-000000000002",
        ],
        "precision": "bf16",
        "quantization": None,
        "server_context_tokens": 262144,
        "tensor_parallel_size": 2,
        "startup_args": [
            "vllm",
            "serve",
            card["root"],
            "--served-model-name",
            config.model_id,
            "--dtype=bfloat16",
            "--max-model-len",
            "262144",
            "--tensor-parallel-size",
            "2",
            "--port",
            "8123",
        ],
    }
    index_ref = write_cpu_json(
        tmp_path / "CPU-model.safetensors.index.json",
        {"weight_map": {"CPU_FIXTURE_NOT_GPU.weight": "model-00001-of-00001.safetensors"}},
    )
    model_files = {
        "config.json": digest(b"CPU model config"),
        "model.safetensors.index.json": index_ref["sha256"],
        "model-00001-of-00001.safetensors": digest(b"CPU fictional weight shard"),
    }
    metadata_ref = write_cpu_json(
        tmp_path / "CPU-download-metadata.json",
        {
            "model_id": config.model_id,
            "model_revision": config.model_revision,
            "files": model_files,
            "CPU_FIXTURE_NOT_GPU": True,
        },
    )
    weights = {
        **bindings,
        "model_id": config.model_id,
        "model_revision": config.model_revision,
        "model_path": card["root"],
        "complete": True,
        "files": model_files,
        "download_metadata": metadata_ref,
        "weights_index": index_ref,
        "CPU_FIXTURE_NOT_GPU": True,
    }
    weights_ref = write_cpu_json(tmp_path / "CPU-weights-manifest.json", weights)
    identity["weights_manifest_sha256"] = weights_ref["sha256"]
    startup = {
        **bindings,
        "observed_server_identity": copy.deepcopy(identity),
        "CPU_FIXTURE_NOT_GPU": True,
    }
    startup_ref = write_cpu_json(tmp_path / "CPU-startup.json", startup)
    identity["startup_evidence_sha256"] = startup_ref["sha256"]
    receipt = {
        **bindings,
        "server_identity": identity,
        "startup_evidence": startup_ref,
        "weights_manifest": weights_ref,
        "checks": {},
        "CPU_FIXTURE_NOT_GPU": True,
    }
    for name in REQUIRED_CHECKS:
        evidence = {
            **bindings,
            "server_identity": identity,
            "check": name,
            "passed": True,
            "CPU_FIXTURE_NOT_GPU": True,
        }
        reference = write_cpu_json(tmp_path / f"CPU-{name}.json", evidence)
        receipt["checks"][name] = {"passed": True, "evidence": reference}
    receipt_path = tmp_path / "CPU-readiness.json"
    write_cpu_json(receipt_path, receipt)
    return SimpleNamespace(
        run=run,
        plan=plan,
        config=config,
        base_url=base_url,
        receipt=receipt,
        receipt_path=receipt_path,
        card=card,
    )


def cpu_stop_fixture(fixture):
    identity = fixture.receipt["server_identity"]
    started = {
        "run": str(fixture.run),
        "run_id": fixture.run.name,
        "model_id": fixture.config.model_id,
        "model_revision": fixture.config.model_revision,
        "config_sha256": fixture.receipt["config_sha256"],
        "base_url": fixture.base_url,
        "nonce": "CPU_FIXTURE_NOT_GPU_nonce",
        "server_identity": copy.deepcopy(identity),
    }
    write_cpu_json(
        fixture.run / "GENERATION_FINISHED.json",
        {"finished_at": "2026-11-02T01:00:00+00:00", "attempted": 24},
    )
    bindings = {
        key: fixture.receipt[key] for key in ("mode", "run_id", "plan_sha256", "config_sha256")
    }
    bindings.update(generation_nonce=started["nonce"], server_identity=copy.deepcopy(identity))
    observed = {
        **bindings,
        "observed_at": "2026-11-02T01:00:01+00:00",
        "pids_absent": [{**process, "absent": True} for process in identity["owned_processes"]],
        "port_closed": True,
        "gpu_memory_bytes": dict.fromkeys(identity["gpu_uuids"], 0),
        "CPU_FIXTURE_NOT_GPU": True,
    }
    evidence_path = fixture.receipt_path.with_name("CPU-stop-observation.json")
    reference = write_cpu_json(evidence_path, observed)
    receipt = {
        **bindings,
        "run": str(fixture.run),
        "model_id": fixture.config.model_id,
        "model_revision": fixture.config.model_revision,
        "model_stopped": True,
        "evidence": reference,
        "CPU_FIXTURE_NOT_GPU": True,
    }
    path = fixture.receipt_path.with_name("CPU-stopped.json")
    write_cpu_json(path, receipt)
    return SimpleNamespace(
        started=started, observed=observed, receipt=receipt, path=path, evidence_path=evidence_path
    )


@pytest.mark.parametrize(
    ("model_id", "revision"),
    [
        ("poolside/Laguna-XS-2.1", "c5f36269bbdbd3f27fddc9a9f9dbae0cf2cf57db"),
        (
            "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
            "a9904d24bcc1d289a1950fa9d2b978c47cf903b9",
        ),
    ],
)
def test_cpu_fixture_readiness_and_exact_stop_bindings(tmp_path, model_id, revision):
    fixture = cpu_readiness_fixture(
        tmp_path, ReplExperimentConfig(model_id=model_id, model_revision=revision)
    )
    receipt = check_readiness(
        fixture.run, fixture.plan, fixture.config, fixture.receipt_path, fixture.base_url
    )
    assert receipt["receipt_sha256"] == digest(fixture.receipt_path.read_bytes())
    assert receipt["server_identity"] == fixture.receipt["server_identity"]
    stopped = cpu_stop_fixture(fixture)
    checked = check_model_stopped(fixture.run, fixture.config, stopped.started, stopped.path)
    assert checked["receipt_sha256"] == digest(stopped.path.read_bytes())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("served_model", "wrong-model"),
        ("model_revision", "0" * 40),
        ("port", 9123),
        ("precision", "fp8"),
        ("server_context_tokens", 32768),
    ],
)
def test_cpu_fixture_rejects_wrong_observed_deployment(tmp_path, field, value):
    fixture = cpu_readiness_fixture(tmp_path)
    fixture.receipt["server_identity"][field] = value
    write_cpu_json(fixture.receipt_path, fixture.receipt)
    with pytest.raises(ValueError, match="server identity"):
        check_readiness(
            fixture.run, fixture.plan, fixture.config, fixture.receipt_path, fixture.base_url
        )


@pytest.mark.parametrize("field", ["run_id", "plan_sha256", "config_sha256"])
def test_cpu_fixture_rejects_wrong_frozen_binding(tmp_path, field):
    fixture = cpu_readiness_fixture(tmp_path)
    fixture.receipt[field] = "different-run-or-hash"
    write_cpu_json(fixture.receipt_path, fixture.receipt)
    with pytest.raises(ValueError, match="bind this run"):
        check_readiness(
            fixture.run, fixture.plan, fixture.config, fixture.receipt_path, fixture.base_url
        )


def test_cpu_fixture_rejects_evidence_from_other_spec_and_modified_or_escaping_files(tmp_path):
    fixture = cpu_readiness_fixture(tmp_path)
    check = fixture.receipt["checks"]["namespace_isolation"]
    path = tmp_path / check["evidence"]["path"]
    original = path.read_bytes()
    observed = json.loads(original)
    observed["sandbox_spec_sha256"] = digest(b"different CPU fixture sandbox")
    check["evidence"] = write_cpu_json(path, observed)
    write_cpu_json(fixture.receipt_path, fixture.receipt)
    with pytest.raises(ValueError, match="bind this run"):
        check_readiness(
            fixture.run, fixture.plan, fixture.config, fixture.receipt_path, fixture.base_url
        )
    path.write_bytes(original)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        check_readiness(
            fixture.run, fixture.plan, fixture.config, fixture.receipt_path, fixture.base_url
        )
    check["evidence"]["path"] = "../outside-evidence.json"
    write_cpu_json(fixture.receipt_path, fixture.receipt)
    with pytest.raises(ValueError, match="escapes"):
        check_readiness(
            fixture.run, fixture.plan, fixture.config, fixture.receipt_path, fixture.base_url
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("host_id", "different-host:boot"),
        ("port", 9123),
        ("gpu_uuids", ["GPU-other-1", "GPU-other-2"]),
        ("owned_processes", [{"pid": 101, "start_time": "REUSED_PID", "role": "api"}]),
    ],
)
def test_cpu_fixture_stop_rejects_other_server_identity(tmp_path, field, value):
    fixture = cpu_readiness_fixture(tmp_path)
    stopped = cpu_stop_fixture(fixture)
    stopped.receipt["server_identity"][field] = value
    write_cpu_json(stopped.path, stopped.receipt)
    with pytest.raises(ValueError, match="server identity"):
        check_model_stopped(fixture.run, fixture.config, stopped.started, stopped.path)


@pytest.mark.parametrize(
    "failure", ["pid_start_time", "port_open", "gpu_busy", "before_finish", "wrong_nonce"]
)
def test_cpu_fixture_stop_requires_observed_absence_idle_and_after_finish(tmp_path, failure):
    fixture = cpu_readiness_fixture(tmp_path)
    stopped = cpu_stop_fixture(fixture)
    if failure == "pid_start_time":
        stopped.observed["pids_absent"][0]["start_time"] = "REUSED_PID"
    elif failure == "port_open":
        stopped.observed["port_closed"] = False
    elif failure == "gpu_busy":
        stopped.observed["gpu_memory_bytes"][fixture.receipt["server_identity"]["gpu_uuids"][0]] = 1
    elif failure == "before_finish":
        stopped.observed["observed_at"] = "2026-11-02T00:59:59+00:00"
    else:
        stopped.observed["generation_nonce"] = "other-generation"
    stopped.receipt["evidence"] = write_cpu_json(stopped.evidence_path, stopped.observed)
    write_cpu_json(stopped.path, stopped.receipt)
    with pytest.raises(ValueError):
        check_model_stopped(fixture.run, fixture.config, stopped.started, stopped.path)


@pytest.mark.parametrize("failure", ["startup_model", "metadata_revision", "unindexed_shard"])
def test_cpu_fixture_requires_actual_model_argv_and_pinned_download_inventory(tmp_path, failure):
    fixture = cpu_readiness_fixture(tmp_path)
    identity = fixture.receipt["server_identity"]
    if failure == "startup_model":
        identity["startup_args"][2] = "/wrong/local/model"
        write_cpu_json(fixture.receipt_path, fixture.receipt)
    elif failure == "metadata_revision":
        path = tmp_path / "CPU-download-metadata.json"
        metadata = json.loads(path.read_bytes())
        metadata["model_revision"] = "0" * 40
        # Re-pin every enclosing document, so this tests revision binding, not a stale hash.
        weights_path = tmp_path / "CPU-weights-manifest.json"
        weights = json.loads(weights_path.read_bytes())
        weights["download_metadata"] = write_cpu_json(path, metadata)
        fixture.receipt["weights_manifest"] = write_cpu_json(weights_path, weights)
        identity["weights_manifest_sha256"] = fixture.receipt["weights_manifest"]["sha256"]
        write_cpu_json(fixture.receipt_path, fixture.receipt)
    else:
        weights_path = tmp_path / "CPU-weights-manifest.json"
        weights = json.loads(weights_path.read_bytes())
        weights["files"]["unindexed.safetensors"] = digest(b"extra CPU shard")
        metadata_path = tmp_path / "CPU-download-metadata.json"
        weights["download_metadata"] = write_cpu_json(
            metadata_path,
            {
                "model_id": fixture.config.model_id,
                "model_revision": fixture.config.model_revision,
                "files": weights["files"],
                "CPU_FIXTURE_NOT_GPU": True,
            },
        )
        fixture.receipt["weights_manifest"] = write_cpu_json(weights_path, weights)
        identity["weights_manifest_sha256"] = fixture.receipt["weights_manifest"]["sha256"]
        write_cpu_json(fixture.receipt_path, fixture.receipt)
    with pytest.raises(ValueError):
        check_readiness(
            fixture.run, fixture.plan, fixture.config, fixture.receipt_path, fixture.base_url
        )


def infrastructure_cpu_fixture(tmp_path):
    fixture = cpu_readiness_fixture(tmp_path)
    authorization = write_cpu_json(
        fixture.run / "USER_AUTHORIZATION.json",
        {
            "scope": "CPU fictional authorization",
            "readiness_profile": "infrastructure-only-v1",
            "behavioral_preflight": False,
        },
    )
    policy = write_cpu_json(
        fixture.run / "READINESS_POLICY.json",
        {
            "profile": "infrastructure-only-v1",
            "behavioral_preflight": False,
            "authorization_sha256": authorization["sha256"],
        },
    )
    fixture.plan["artifacts"].update(
        {
            "USER_AUTHORIZATION.json": authorization["sha256"],
            "READINESS_POLICY.json": policy["sha256"],
        }
    )
    write_cpu_json(fixture.run / "plan.json", fixture.plan)
    bindings = {
        key: fixture.receipt[key]
        for key in ("mode", "run_id", "config_sha256", "sandbox_spec_sha256")
    }
    bindings.update(
        plan_sha256=digest((fixture.run / "plan.json").read_bytes()),
        readiness_profile="infrastructure-only-v1",
        readiness_policy_sha256=policy["sha256"],
        authorization_sha256=authorization["sha256"],
    )
    identity = fixture.receipt["server_identity"]
    weights_path = tmp_path / "CPU-weights-manifest.json"
    weights = json.loads(weights_path.read_bytes())
    weights.update(bindings)
    fixture.receipt["weights_manifest"] = write_cpu_json(weights_path, weights)
    identity["weights_manifest_sha256"] = fixture.receipt["weights_manifest"]["sha256"]
    observed_identity = {
        key: value for key, value in identity.items() if key != "startup_evidence_sha256"
    }
    fixture.receipt["startup_evidence"] = write_cpu_json(
        tmp_path / "CPU-startup.json", {**bindings, "observed_server_identity": observed_identity}
    )
    identity["startup_evidence_sha256"] = fixture.receipt["startup_evidence"]["sha256"]
    from cute_rlm.icl.repl_readiness import INFRASTRUCTURE_CHECKS

    fixture.receipt.update(bindings, checks={})
    for name in INFRASTRUCTURE_CHECKS:
        reference = write_cpu_json(
            tmp_path / f"CPU-infra-{name}.json",
            {
                **bindings,
                "server_identity": identity,
                "check": name,
                "passed": True,
                "CPU_FIXTURE_NOT_GPU": True,
            },
        )
        fixture.receipt["checks"][name] = {"passed": True, "evidence": reference}
    write_cpu_json(fixture.receipt_path, fixture.receipt)
    return fixture


def test_cpu_infrastructure_profile_has_no_model_behavior_gate(tmp_path):
    fixture = infrastructure_cpu_fixture(tmp_path)
    receipt = check_readiness(
        fixture.run, fixture.plan, fixture.config, fixture.receipt_path, fixture.base_url
    )
    assert "native_tool_protocol" not in receipt["checks"]
    assert receipt["readiness_profile"] == "infrastructure-only-v1"


@pytest.mark.parametrize(
    "failure",
    [
        "authorization",
        "policy",
        "profile",
        "missing_profile",
        "extra_behavior_check",
        "missing_check",
    ],
)
def test_cpu_infrastructure_profile_requires_frozen_policy_authorization_and_checks(
    tmp_path, failure
):
    fixture = infrastructure_cpu_fixture(tmp_path)
    if failure == "authorization":
        (fixture.run / "USER_AUTHORIZATION.json").write_text("{}")
    elif failure == "policy":
        (fixture.run / "READINESS_POLICY.json").write_text("{}")
    elif failure == "profile":
        fixture.receipt["readiness_profile"] = "unknown"
    elif failure == "missing_profile":
        fixture.receipt.pop("readiness_profile")
    elif failure == "extra_behavior_check":
        fixture.receipt["checks"]["native_tool_protocol"] = {"passed": True}
    else:
        fixture.receipt["checks"].pop("chat_stream_transport")
    write_cpu_json(fixture.receipt_path, fixture.receipt)
    with pytest.raises(ValueError):
        check_readiness(
            fixture.run, fixture.plan, fixture.config, fixture.receipt_path, fixture.base_url
        )


def test_cpu_infrastructure_profile_cannot_be_selected_by_receipt_alone(tmp_path):
    fixture = cpu_readiness_fixture(tmp_path)
    fixture.receipt["readiness_profile"] = "infrastructure-only-v1"
    write_cpu_json(fixture.receipt_path, fixture.receipt)
    with pytest.raises(ValueError, match="frozen policy"):
        check_readiness(
            fixture.run, fixture.plan, fixture.config, fixture.receipt_path, fixture.base_url
        )


def test_cpu_infrastructure_profile_requires_explicit_authorization_content(tmp_path):
    fixture = infrastructure_cpu_fixture(tmp_path)
    authorization_path = fixture.run / "USER_AUTHORIZATION.json"
    authorization = write_cpu_json(authorization_path, {"scope": "old authorization"})
    policy_path = fixture.run / "READINESS_POLICY.json"
    policy = json.loads(policy_path.read_bytes())
    policy["authorization_sha256"] = authorization["sha256"]
    policy_ref = write_cpu_json(policy_path, policy)
    fixture.plan["artifacts"].update(
        {
            "USER_AUTHORIZATION.json": authorization["sha256"],
            "READINESS_POLICY.json": policy_ref["sha256"],
        }
    )
    write_cpu_json(fixture.run / "plan.json", fixture.plan)
    fixture.receipt["plan_sha256"] = digest((fixture.run / "plan.json").read_bytes())
    write_cpu_json(fixture.receipt_path, fixture.receipt)
    with pytest.raises(ValueError, match="Authorization does not approve"):
        check_readiness(
            fixture.run, fixture.plan, fixture.config, fixture.receipt_path, fixture.base_url
        )
