from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from cute_rlm.sandbox import (
    SandboxSpec,
    build_command,
    build_repl_command,
    map_phase_request,
    phase_command,
)


def binding(source: str, target: str) -> dict[str, str]:
    return {"source": source, "target": target}


def test_build_command_maps_candidate_inputs_readonly_and_never_mounts_host_root(tmp_path):
    inputs = tmp_path / "inputs"
    output = tmp_path / "candidate"
    inputs.mkdir()
    request = {
        "role": "candidate",
        "input_dir": str(inputs),
        "output_dir": str(output),
        "source": "x",
    }
    spec = {
        "bwrap": "/usr/bin/bwrap",
        "runtime_python": "/venv/bin/python",
        "readonly_bindings": [binding("/venv/lib/python", "/runtime/python")],
        "gpu_devices": [binding("/dev/nvidia0", "/dev/nvidia0")],
    }

    command = build_command(spec, request)

    assert command[:4] == ["/usr/bin/bwrap", "--unshare-all", "--die-with-parent", "--new-session"]
    assert ["--ro-bind", str(inputs), "/phase/inputs"] in [
        command[i : i + 3] for i, item in enumerate(command) if item == "--ro-bind"
    ]
    assert ["--bind", str(output), "/phase/output"] in [
        command[i : i + 3] for i, item in enumerate(command) if item == "--bind"
    ]
    assert ["--dev-bind", "/dev/nvidia0", "/dev/nvidia0"] in [
        command[i : i + 3] for i, item in enumerate(command) if item == "--dev-bind"
    ]
    assert not any(
        command[i : i + 3] == ["--ro-bind", "/", "/"]
        for i, item in enumerate(command)
        if item == "--ro-bind"
    )
    assert "--clearenv" in command
    assert "cute_rlm.verify_worker" in command


def test_reference_request_contains_only_reference_output_mount(tmp_path):
    request = {"role": "reference", "output_dir": str(tmp_path / "ref")}
    mapped, mounts = map_phase_request(request)
    assert mapped["output_dir"] == "/phase/output"
    assert mounts == [(str(tmp_path / "ref"), "/phase/output")]
    assert "input_dir" not in mapped


def test_candidate_paths_must_not_overlap(tmp_path):
    path = tmp_path / "same"
    with pytest.raises(ValueError, match="disjoint"):
        map_phase_request({"role": "candidate", "output_dir": str(path), "input_dir": str(path)})


def test_phase_command_and_spec_round_trip(tmp_path):
    path = tmp_path / "sandbox.json"
    path.write_text(
        json.dumps(
            {
                "bwrap": "/usr/bin/bwrap",
                "runtime_python": "/venv/bin/python",
                "readonly_bindings": [],
            }
        )
    )
    spec = SandboxSpec.read(path)
    assert spec.runtime_python == "/venv/bin/python"
    assert phase_command(path)[-2:] == ["--spec", str(path.resolve())]
    assert phase_command(spec)[-2:] == ["--spec", str(path.resolve())]


def test_repl_command_is_separate_no_gpu_and_broker_dynamic_env(tmp_path):
    import socket as socket_module

    socket_path = tmp_path / "broker.sock"
    broker = socket_module.socket(socket_module.AF_UNIX)
    broker.bind(str(socket_path))
    spec = {
        "bwrap": "/usr/bin/bwrap",
        "runtime_python": "/venv/bin/python",
        "ro_bindings": [],
    }
    command = build_repl_command(
        spec,
        runtime_code_bindings=[binding("/code", "/runtime/code")],
        python_args=["-u", "-m", "rlm_train.worker"],
        runtime_env={
            "RLM_TRAIN_PROXY_URL": f"unix://{socket_path}",
            "RLM_TRAIN_ROLLOUT_ID": "r1",
            "AWS_SECRET_ACCESS_KEY": "bad",
        },
    )
    assert "--dev-bind" not in command
    assert ["--bind", str(socket_path), "/run/rlm-root-broker.sock"] in [
        command[i : i + 3] for i, item in enumerate(command) if item == "--bind"
    ]
    assert "unix:///run/rlm-root-broker.sock" in command
    assert "r1" in command
    assert "bad" not in command
    assert command[-4:] == ["/venv/bin/python", "-u", "-m", "rlm_train.worker"]
    broker.close()
    with pytest.raises(ValueError, match="Unix-socket"):
        build_repl_command(spec, runtime_env={"RLM_TRAIN_PROXY_URL": "https://secret.example"})


def test_validate_host_requires_executable_runtime_and_existing_readonly_bindings(tmp_path):
    bwrap = tmp_path / "bwrap"
    python = tmp_path / "python"
    source = tmp_path / "pythonlib"
    bwrap.write_text("tool")
    python.write_text("tool")
    source.mkdir()
    bwrap.chmod(0o755)
    python.chmod(0o755)
    spec_path = tmp_path / "sandbox.json"
    spec_path.write_text(
        json.dumps(
            {
                "bwrap": str(bwrap),
                "runtime_python": str(python),
                "readonly_bindings": [binding(str(source), "/runtime/python")],
            }
        )
    )
    SandboxSpec.read(spec_path).validate_host()


def test_validate_host_rejects_missing_runtime_and_overlapping_mount_targets(tmp_path):
    executable = tmp_path / "tool"
    executable.write_text("tool")
    executable.chmod(0o755)
    source = tmp_path / "source"
    source.mkdir()
    spec = SandboxSpec(
        bwrap=str(executable),
        runtime_python=str(executable),
        readonly_bindings=(
            binding(str(source), "/runtime/python"),
            binding(str(source), "/runtime/python/site-packages"),
        ),
    )
    with pytest.raises(ValueError, match="overlap"):
        spec.validate_host()


def test_phase_builder_rejects_runtime_binding_containing_phase_path(tmp_path):
    output = tmp_path / "workspace" / "output"
    output.parent.mkdir()
    spec = {
        "bwrap": "/usr/bin/bwrap",
        "runtime_python": "/venv/bin/python",
        "readonly_bindings": [binding(str(output.parent), "/runtime/project")],
    }
    with pytest.raises(ValueError, match="phase paths"):
        build_command(spec, {"role": "reference", "output_dir": str(output)})


@pytest.mark.parametrize("target", ["/", "/home", "/etc"])
def test_runtime_source_link_cannot_hide_a_broad_host_mount(tmp_path, target):
    from cute_rlm.sandbox import _bindings

    link = tmp_path / "runtime-link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="broad host"):
        _bindings([binding(str(link), "/runtime/hidden")], "read-only")


@pytest.fixture
def phase_worker(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from cute_rlm import sandbox

    bwrap = tmp_path / "bwrap"
    bwrap.write_text("executable")
    bwrap.chmod(0o755)
    device = tmp_path / "device"
    device.touch()
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    spec = {
        "bwrap": str(bwrap),
        "runtime_python": "/venv/bin/python",
        "readonly_bindings": [],
        "gpu_devices": [binding(str(device), "/dev/nvidia0")],
    }
    request = {
        "role": "candidate",
        "input_dir": str(inputs),
        "output_dir": str(tmp_path / "output"),
        "compile_feedback": True,
        "profile_compile": True,
    }

    def launch(write_output):
        def run(command, **kwargs):
            assert command[-1] == "cute_rlm.verify_worker"
            mapped = json.loads(kwargs["input"])
            assert mapped["compile_feedback"] is True
            assert mapped["profile_compile"] is request["profile_compile"]
            output = next(
                Path(command[index + 1])
                for index, value in enumerate(command)
                if value == "--bind" and command[index + 2] == "/phase/output"
            )
            write_output(output)
            return SimpleNamespace(returncode=0, stdout=b'{"status":"compiled"}\n', stderr=b"")

        monkeypatch.setattr(sandbox.subprocess, "run", run)
        return sandbox._run(spec, request)

    return launch, request


@pytest.mark.parametrize("compiled", [False, True])
def test_phase_retains_compile_trace_directory_and_artifacts(phase_worker, compiled, capfd):
    from types import SimpleNamespace

    from cute_rlm.compile_trace import install_compile_trace

    ptx, cubin = b".visible .entry test_kernel() {}", b"test cubin"

    def write_output(output):
        cute = SimpleNamespace(
            compile=lambda *args, **kwargs: SimpleNamespace(__ptx__=ptx, __cubin__=cubin)
        )
        trace = install_compile_trace(cute, SimpleNamespace(), output)
        try:
            if compiled:
                cute.compile("kernel")
            trace.snapshot()
        finally:
            trace.close()

    launch, request = phase_worker
    assert launch(write_output) == 0
    assert json.loads(capfd.readouterr().out)["status"] == "compiled"
    output = Path(request["output_dir"])
    artifacts = output / "compile-artifacts"
    assert artifacts.is_dir()
    provenance = json.loads((output / "compile-provenance.json").read_text())
    if compiled:
        artifact = provenance["artifacts"][0]
        assert (output / artifact["ptx_path"]).read_bytes() == ptx
        assert (output / artifact["cubin_path"]).read_bytes() == cubin
    else:
        assert list(artifacts.iterdir()) == [] and provenance["artifacts"] == []


@pytest.mark.parametrize(
    "kind",
    [
        "file_link",
        "directory_link",
        "artifact_directory_link",
        "fifo",
        "other_directory",
        "unrequested",
    ],
)
def test_phase_rejects_unsafe_or_unexpected_artifact_entries(phase_worker, tmp_path, kind):
    def write_output(output):
        artifacts = output / "compile-artifacts"
        if kind == "artifact_directory_link":
            artifacts.symlink_to(tmp_path, target_is_directory=True)
            return
        artifacts.mkdir()
        if kind == "file_link":
            target = tmp_path / "host-file"
            target.write_text("host data")
            (artifacts / "artifact.ptx").symlink_to(target)
        elif kind == "directory_link":
            (artifacts / "linked").symlink_to(tmp_path, target_is_directory=True)
        elif kind == "fifo":
            os.mkfifo(artifacts / "pipe")
        elif kind == "other_directory":
            (output / "unexpected").mkdir()

    launch, request = phase_worker
    if kind == "unrequested":
        request["profile_compile"] = False
    with pytest.raises(RuntimeError, match="worker outputs"):
        launch(write_output)
    assert not Path(request["output_dir"]).exists()


@pytest.mark.parametrize("module", ["cute_rlm.verify_worker", "cute_rlm.icl.worker"])
def test_phase_worker_module_is_an_explicit_allowlist(tmp_path, module):
    spec = {"bwrap": "/usr/bin/bwrap", "runtime_python": "/venv/bin/python"}
    request = {
        "role": "reference",
        "output_dir": str(tmp_path / "output"),
        "worker_module": module,
    }
    assert build_command(spec, request)[-1] == module
    request["worker_module"] = "arbitrary.module"
    with pytest.raises(ValueError, match="worker module"):
        build_command(spec, request)


def test_worker_uses_explicit_trusted_gpu_visibility_for_pair_partition(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    spec = {
        "bwrap": "/usr/bin/bwrap",
        "runtime_python": "/venv/bin/python",
        "readonly_bindings": [],
        "gpu_devices": [binding("/dev/nvidia2", "/dev/nvidia2")],
        "runtime_settings": {"CUDA_VISIBLE_DEVICES": "GPU-12345678-1234-1234-1234-123456789abc"},
    }
    command = build_command(
        spec,
        {
            "role": "candidate",
            "input_dir": str(inputs),
            "output_dir": str(tmp_path / "outputs"),
            "worker_module": "cute_rlm.icl.worker",
        },
    )
    index = command.index("CUDA_VISIBLE_DEVICES")
    assert command[index - 1 : index + 2] == [
        "--setenv",
        "CUDA_VISIBLE_DEVICES",
        "GPU-12345678-1234-1234-1234-123456789abc",
    ]
    assert ["--dev-bind", "/dev/nvidia2", "/dev/nvidia2"] in [
        command[i : i + 3] for i, value in enumerate(command) if value == "--dev-bind"
    ]
