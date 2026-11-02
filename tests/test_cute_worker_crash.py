"""Native-worker recovery is limited to trusted lifecycle evidence; no GPU evidence here."""

import json
import shutil
import subprocess
import sys

import pytest
from cute_rlm import sandbox
from cute_rlm.verifier import launch_phase

ENV = {"PATH": "/usr/bin:/bin", "HOME": "/nonexistent"}


def receipt(code):
    return {
        "events": [{"child-pid": 123}, {"exit-code": code}],
        "exec_confirmed": True,
        "namespace_cleanup": True,
    }


@pytest.mark.parametrize("code", sorted(sandbox.NATIVE_WORKER_EXIT_CODES))
def test_native_exit_requires_fresh_independent_health_check(monkeypatch, code):
    calls = []
    command = ["/usr/bin/bwrap", "/venv/python", "-m", "cute_rlm.icl.worker"]
    request = {"source": "must never run in the trusted health probe"}

    def observe(actual, payload, env):
        calls.append((actual, payload))
        if payload is request:
            return subprocess.CompletedProcess(
                actual, code, b"partial output", b"native abort"
            ), receipt(code)
        assert payload is None
        assert actual[:-2] == command[:-2]
        assert actual[-2:] == ["-c", sandbox.cute_health_program()]
        assert request["source"] not in actual[-1]
        return subprocess.CompletedProcess(actual, 0, b'{"status":"healthy"}', b""), receipt(0)

    monkeypatch.setattr(sandbox, "observe_compile_worker", observe)
    process, result = sandbox.compile_worker_result(command, request, ENV)
    assert len(calls) == 2 and process.returncode == code
    assert result["status"] == "runtime_error" and result["error_type"] == "IsolatedWorkerCrash"
    assert result["worker_returncode"] == code
    assert "cause unknown" in result["detail"]
    assert "does not establish candidate causality" in result["diagnostic_limit"]
    assert result["stdout"] == "partial output" and result["stderr"] == "native abort"
    assert result["worker_lifecycle"]["namespace_cleanup"]
    assert not {"correct", "cute_verified", "compile_forward"} & result.keys()


@pytest.mark.parametrize(
    "code, confirmed", [(0, True), (1, True), (125, True), (137, True), (143, True), (134, False)]
)
def test_ordinary_or_unconfirmed_exit_never_recovers(monkeypatch, code, confirmed):
    calls = []

    def observe(command, request, env):
        calls.append(request)
        return subprocess.CompletedProcess(command, code, b"", b"failure"), {
            **receipt(code),
            "exec_confirmed": confirmed,
        }

    monkeypatch.setattr(sandbox, "observe_compile_worker", observe)
    process, result = sandbox.compile_worker_result(["bwrap", "python", "-m", "worker"], {}, ENV)
    assert process.returncode == code and result is None and len(calls) == 1


@pytest.mark.parametrize("kind", ["nonzero", "unconfirmed", "bad_json", "wrong_result", "cleanup"])
def test_post_crash_health_failure_stays_fatal(monkeypatch, kind):
    calls = []

    def observe(command, request, env):
        calls.append(request)
        if len(calls) == 1:
            return subprocess.CompletedProcess(command, 134, b"", b"abort"), receipt(134)
        if kind == "cleanup":
            raise RuntimeError("namespace cleanup could not be confirmed")
        code = 1 if kind == "nonzero" else 0
        output = (
            b"not JSON"
            if kind == "bad_json"
            else b'{"status":"unhealthy"}'
            if kind == "wrong_result"
            else b'{"status":"healthy"}'
        )
        return subprocess.CompletedProcess(command, code, output, b"health failed"), {
            **receipt(code),
            "exec_confirmed": kind != "unconfirmed",
        }

    monkeypatch.setattr(sandbox, "observe_compile_worker", observe)
    with pytest.raises((RuntimeError, ValueError)):
        sandbox.compile_worker_result(["bwrap", "python", "-m", "worker"], {}, ENV)
    assert len(calls) == 2


@pytest.mark.parametrize("kind", ["missing_exit", "mismatched_exit", "cleanup", "open_status"])
def test_control_envelope_cannot_turn_setup_or_cleanup_failure_into_recovery(monkeypatch, kind):
    envelope = {
        "returncode": 134,
        "stdout": '{"status":"forged"}',
        "stderr": "abort",
        "status": '{"child-pid":123}\n',
        "status_closed": True,
        "namespace_cleanup": True,
    }
    if kind != "missing_exit":
        envelope["status"] += (
            json.dumps({"exit-code": 139 if kind == "mismatched_exit" else 134}) + "\n"
        )
    if kind == "cleanup":
        envelope["namespace_cleanup"] = False
    if kind == "open_status":
        envelope["status_closed"] = False

    def run(command, **kwargs):
        assert command[:5] == ["/usr/bin/python3", "-I", "-S", "-c", sandbox.PHASE_CONTROL_PROGRAM]
        return subprocess.CompletedProcess(command, 0, json.dumps(envelope).encode(), b"")

    monkeypatch.setattr(sandbox.subprocess, "run", run)
    if kind in {"cleanup", "open_status"}:
        with pytest.raises(RuntimeError, match="cleanup|status"):
            sandbox.observe_compile_worker(["bwrap"], {}, ENV)
    else:
        process, lifecycle = sandbox.observe_compile_worker(["bwrap"], {}, ENV)
        assert process.returncode == 134 and not lifecycle["exec_confirmed"]


def test_privileged_control_opens_fds_after_sudo_and_keeps_existing_uid_drop(monkeypatch):
    command = [
        "/usr/bin/sudo",
        "-n",
        "/usr/bin/bwrap",
        "/usr/bin/setpriv",
        "--reuid=1000",
        "python",
        "-m",
        "worker",
    ]

    def run(actual, **kwargs):
        assert actual[:6] == ["/usr/bin/sudo", "-n", "/usr/bin/python3", "-I", "-S", "-c"]
        payload = json.loads(kwargs["input"])
        assert payload["command"] == command[2:]
        assert "--reuid=1000" in payload["command"]
        assert "pass_fds" not in kwargs
        envelope = {
            "returncode": 0,
            "stdout": "{}",
            "stderr": "",
            "status": '{"child-pid":123}\n{"exit-code":0}\n',
            "status_closed": True,
            "namespace_cleanup": True,
        }
        return subprocess.CompletedProcess(actual, 0, json.dumps(envelope).encode(), b"")

    monkeypatch.setattr(sandbox.subprocess, "run", run)
    process, lifecycle = sandbox.observe_compile_worker(command, {}, ENV)
    assert process.returncode == 0 and lifecycle["exec_confirmed"]


@pytest.fixture
def real_bwrap():
    executable = shutil.which("bwrap")
    if not executable:
        pytest.skip("bubblewrap unavailable")
    command = [
        executable,
        "--unshare-all",
        "--die-with-parent",
        "--new-session",
        "--ro-bind",
        "/usr",
        "/usr",
        "--symlink",
        "usr/lib",
        "/lib",
        "--symlink",
        "usr/lib64",
        "/lib64",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
    ]
    probe = subprocess.run([*command, "/usr/bin/true"], capture_output=True)
    if probe.returncode:
        pytest.skip("real bubblewrap namespace setup unavailable: " + probe.stderr.decode())
    return command


def test_real_bwrap_native_abort_and_candidate_json_cannot_forge_control(real_bwrap):
    code = "import os,resource; resource.setrlimit(resource.RLIMIT_CORE,(0,0)); os.write(1,b'{\"exit-code\":0}\\n'); os.write(2,b'native diagnostic'); os.abort()"
    process, lifecycle = sandbox.observe_compile_worker(
        [*real_bwrap, "/usr/bin/python3", "-I", "-S", "-c", code], None, ENV
    )
    assert process.returncode == 134 and lifecycle["exec_confirmed"]
    assert lifecycle["events"][-1]["exit-code"] == 134
    assert lifecycle["namespace_cleanup"]
    assert b"exit-code" in process.stdout and process.stderr == b"native diagnostic"


def test_real_bwrap_setup_failure_has_no_exec_receipt(real_bwrap):
    process, lifecycle = sandbox.observe_compile_worker(
        [*real_bwrap, "/usr/bin/missing-cute-worker"], None, ENV
    )
    assert process.returncode != 0 and not lifecycle["exec_confirmed"]
    assert lifecycle["namespace_cleanup"]


def test_real_bwrap_closes_control_fds_before_untrusted_command(real_bwrap):
    code = "import json,os; print(json.dumps([fd for fd in range(3,256) if os.path.exists('/proc/self/fd/'+str(fd))]))"
    process, lifecycle = sandbox.observe_compile_worker(
        [*real_bwrap, "/usr/bin/python3", "-I", "-S", "-c", code], None, ENV
    )
    assert process.returncode == 0 and lifecycle["exec_confirmed"]
    assert json.loads(process.stdout) == []


@pytest.mark.asyncio
async def test_icl_compile_timeout_cannot_continue_without_namespace_receipt():
    from types import SimpleNamespace

    request = {
        "role": "candidate",
        "compile_feedback": True,
        "worker_module": "cute_rlm.icl.worker",
    }
    command = [sys.executable, "-c", "import time; time.sleep(10)"]
    with pytest.raises(RuntimeError, match="namespace cleanup could not be confirmed"):
        await launch_phase(command, request, SimpleNamespace(timeout_s=0.1))
    with pytest.raises(TimeoutError):
        await launch_phase(command, {}, SimpleNamespace(timeout_s=0.1))


@pytest.mark.parametrize(
    "role, feedback, module, observed",
    [
        ("candidate", True, "cute_rlm.icl.worker", True),
        ("candidate", False, "cute_rlm.icl.worker", False),
        ("candidate", True, "cute_rlm.verify_worker", False),
        ("reference", True, "cute_rlm.icl.worker", False),
    ],
)
def test_recovery_boundary_does_not_change_reference_final_or_other_workers(
    tmp_path, monkeypatch, capfd, role, feedback, module, observed
):
    executable = tmp_path / "bwrap"
    executable.write_text("test executable")
    executable.chmod(0o755)
    device = tmp_path / "device"
    device.touch()
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    output = tmp_path / "output"
    spec = {
        "bwrap": str(executable),
        "runtime_python": "/venv/python",
        "gpu_devices": [{"source": str(device), "target": "/dev/nvidia0"}],
    }
    request = {
        "role": role,
        "compile_feedback": feedback,
        "worker_module": module,
        "output_dir": str(output),
    }
    if role == "candidate":
        request["input_dir"] = str(inputs)
    calls = []

    def normal(command, **kwargs):
        calls.append("normal")
        return subprocess.CompletedProcess(command, 0, b'{"status":"executed"}', b"")

    def recovered(command, mapped, env):
        calls.append("observed")
        assert mapped["role"] == "candidate" and mapped["compile_feedback"] is True
        return subprocess.CompletedProcess(command, 134, b"", b"abort"), {
            "status": "runtime_error",
            "error_type": "IsolatedWorkerCrash",
        }

    monkeypatch.setattr(sandbox.subprocess, "run", normal)
    monkeypatch.setattr(sandbox, "compile_worker_result", recovered)
    assert sandbox._run(spec, request) == 0
    result = json.loads(capfd.readouterr().out)
    assert calls == ["observed" if observed else "normal"]
    assert result["status"] == ("runtime_error" if observed else "executed")
    assert output.exists() is not observed


def test_trusted_cute_health_runner_uses_fresh_file_and_runs_compile_launch_entry(
    monkeypatch, capsys
):
    import ast
    import runpy
    from pathlib import Path
    from types import SimpleNamespace

    calls = []
    program = sandbox.cute_health_program()
    tree = ast.parse(program)
    trusted_source = ast.literal_eval(tree.body[0].value)
    ast.parse(trusted_source)

    class Model:
        def cuda(self):
            return self

        def eval(self):
            return self

        def __call__(self, a, b):
            calls.append(("compile_launch", a, b))

    def load(path):
        module = Path(path)
        assert module.read_text() == trusted_source
        calls.append(("module", module))
        print("trusted DSL compile log stays out of the JSON response")
        return {"ModelNew": Model}

    torch = SimpleNamespace(
        ones=lambda shape, **kwargs: (shape, kwargs),
        ones_like=lambda value: value,
        float32="float32",
        cuda=SimpleNamespace(synchronize=lambda: calls.append(("synchronize",))),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(runpy, "run_path", load)
    exec(compile(program, "<trusted-health>", "exec"), {})
    assert [entry[0] for entry in calls] == ["module", "compile_launch", "synchronize"]
    assert not calls[0][1].exists()
    assert json.loads(capsys.readouterr().out) == {"status": "healthy"}
