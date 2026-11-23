import hashlib
import json
import sys
from pathlib import Path

import pytest

from training.environments.kernel_rlm.research.ragged_gather_reduce import vm_smoke

UUID = "GPU-11111111-2222-3333-4444-555555555555"
PINS = vm_smoke.SOURCE_SHA256


def make_capture(arrays):
    return b"ragged-observation-v1\n" + b"".join(
        " ".join(map(str, row)).encode("ascii") + b"\n" for row in arrays
    )


class StubVM:
    """Simulate process boundaries; never start a compiler, tool, or GPU process."""

    def __init__(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("UNRELATED_SECRET", "never copied")
        self.args = [
            "--run-author-controls",
            "--confirm-exclusive-disposable-l40s",
            "--gpu-uuid",
            UUID,
        ]
        for name in vm_smoke.TOOL_NAMES:
            program = tmp_path / name
            payload = (name + " stub executable").encode()
            program.write_bytes(payload)
            program.chmod(0o700)
            self.args += [
                f"--{name}",
                name,
                f"--{name}-sha256",
                hashlib.sha256(payload).hexdigest(),
            ]
        self.calls = []
        self.wires = []
        self.fixtures = {f.name: f for _, f, _ in vm_smoke.controls_plan()}
        self.device_row = f"NVIDIA L40S, {UUID}, 570.124.06, 8.9\n".encode()
        self.occupancy = b""
        self.fail_build = False
        self.clobber_build = False
        self.negative = vm_smoke.ProcessResult("exited", 0, make_capture(((7,), (1,)) * 2))
        monkeypatch.setattr(vm_smoke, "run_process", self.run)

    def run(self, argv, *, cwd, env, timeout, stdin=b"", stdout_cap=None):
        self.calls.append(argv)
        assert cwd.stat().st_mode & 0o077 == 0
        assert "UNRELATED_SECRET" not in env
        assert env["HOME"] == env["TMPDIR"] == str(cwd)
        program = Path(argv[0]).name
        if program == "nvidia-smi":
            assert f"--id={UUID}" in argv
            assert "--format=csv,noheader,nounits" in argv
            if "--query-compute-apps=pid" in argv:
                return vm_smoke.ProcessResult("exited", 0, self.occupancy)
            assert "--query-gpu=name,uuid,driver_version,compute_cap" in argv
            return vm_smoke.ProcessResult("exited", 0, self.device_row)
        assert env["CUDA_VISIBLE_DEVICES"] == UUID
        if "--version" in argv:
            assert program in ("nvcc", "host-cxx")
            return vm_smoke.ProcessResult("exited", 0, b"trusted test compiler 1.2.3\n")
        if program == "nvcc":
            assert vm_smoke.ARCH_FLAG in argv
            assert all(
                hashlib.sha256((cwd / name).read_bytes()).hexdigest() == expected
                for name, expected in PINS.items()
            )
            assert Path(argv[argv.index("-ccbin") + 1]).name == "host-cxx"
            source_args = [name for name in argv if name.endswith(".cu")]
            assert source_args in (
                ["control.cu", "device_check.cu"],
                ["wrong_offset.cu", "device_check.cu"],
            )
            if self.fail_build:
                return vm_smoke.ProcessResult("exited", 29, stderr=b"linker reported an error")
            name = argv[argv.index("-o") + 1]
            output = cwd / name
            output.write_bytes(name.encode())
            output.chmod(0o700)
            if self.clobber_build and name == "device_check_wrong_offset":
                (cwd / "device_check_control").write_bytes(b"different artifact")
            return vm_smoke.ProcessResult("exited", 0)

        assert len(argv) == 1 and stdout_cap == vm_smoke.MAX_OBSERVATION_BYTES
        assert timeout == vm_smoke.CONTROL_TIMEOUT
        (fixture,) = (f for f in self.fixtures.values() if f.to_wire().encode() == stdin)
        self.wires.append((program, fixture.name))
        if program == "device_check_wrong_offset":
            assert fixture.name == "mask-offset"
            return self.negative
        assert program == "device_check_control"
        return vm_smoke.ProcessResult(
            "exited", 0, make_capture((fixture.expected_sums, fixture.expected_counts) * 2)
        )


@pytest.fixture
def stub_vm(tmp_path, monkeypatch):
    return StubVM(tmp_path, monkeypatch)


def test_no_process_in_offline_preflight(stub_vm, capsys):
    assert vm_smoke.main(["--preflight"]) == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["schema"] == vm_smoke.SCHEMA
    assert receipt["mode"] == "offline"
    assert receipt["status"] == vm_smoke.OFFLINE_PREFLIGHT
    assert len(receipt["plan"]) == 8
    assert not receipt["device"] and not receipt["cases"] and not receipt["builds"]
    assert all(receipt["sources"][name] == digest for name, digest in PINS.items())
    assert not stub_vm.calls


@pytest.mark.parametrize(
    "missing", ["--confirm-exclusive-disposable-l40s", "--nvcc-sha256", "--gpu-uuid"]
)
def test_live_requires_explicit_approval_and_pins(stub_vm, capsys, missing):
    args = list(stub_vm.args)
    i = args.index(missing)
    del args[i : i + (1 if missing.startswith("--confirm") else 2)]
    with pytest.raises(SystemExit) as error:
        vm_smoke.main(args)
    assert error.value.code == 2
    assert not stub_vm.calls
    assert not capsys.readouterr().out


def test_wrong_binary_bytes_stop_before_any_process(stub_vm, capsys):
    Path("host-cxx").write_bytes(b"not the approved executable")
    assert vm_smoke.main(stub_vm.args) == 1
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["status"] == vm_smoke.AUTHOR_CONTROLS_FAILED
    assert receipt["error"]["stage"] == "toolchain"
    assert not stub_vm.calls


def test_source_pin_stops_before_any_process(stub_vm, capsys, monkeypatch):
    monkeypatch.setattr(vm_smoke, "SOURCE_SHA256", PINS | {"control.cu": "0" * 64})
    assert vm_smoke.main(stub_vm.args) == 1
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["error"]["stage"] == "sources"
    assert not stub_vm.calls


@pytest.mark.parametrize("problem", ["model", "capability", "unsupported_query", "busy"])
def test_unclear_or_wrong_device_stops_before_compiler(stub_vm, capsys, problem):
    if problem == "model":
        stub_vm.device_row = stub_vm.device_row.replace(b"L40S", b"L4")
    elif problem == "capability":
        stub_vm.device_row = stub_vm.device_row.replace(b"8.9", b"8.6")
    elif problem == "unsupported_query":
        stub_vm.device_row = stub_vm.device_row.replace(b"8.9", b"[N/A]")
    else:
        stub_vm.occupancy = b"1290\n"
    assert vm_smoke.main(stub_vm.args) == 1
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["status"] == vm_smoke.AUTHOR_CONTROLS_FAILED
    assert receipt["error"]["stage"] == "device"
    assert all(Path(call[0]).name == "nvidia-smi" for call in stub_vm.calls)
    assert not stub_vm.wires


def test_successful_author_controls_emit_bound_summary(stub_vm, capsys):
    assert vm_smoke.main(stub_vm.args) == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["status"] == vm_smoke.AUTHOR_CONTROLS_PASSED
    assert receipt["device"]["uuid"] == UUID
    assert receipt["error"] is None
    assert set(receipt["builds"]) == {"control", "wrong_offset"}
    assert stub_vm.wires == [
        *(("device_check_control", f.name) for f in stub_vm.fixtures.values()),
        ("device_check_wrong_offset", "mask-offset"),
    ]
    for case in receipt["cases"]:
        assert case["artifact_sha256"] == receipt["builds"][case["control"]]["sha256"]
        assert (
            case["input_sha256"]
            == hashlib.sha256(stub_vm.fixtures[case["fixture"]].to_wire().encode()).hexdigest()
        )
        assert case["expected"] == case["observed"]
        assert case["process"]["returncode"] == 0
    assert receipt["cases"][-1]["observed"] == "numerical_mismatch"
    assert (
        receipt["cases"][-1]["process"]["stdout_sha256"]
        == hashlib.sha256(stub_vm.negative.stdout).hexdigest()
    )


def test_changed_build_artifact_never_executes(stub_vm, capsys):
    stub_vm.clobber_build = True
    assert vm_smoke.main(stub_vm.args) == 1
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["error"]["stage"] == "capture"
    assert not stub_vm.wires


def test_build_failure_never_executes_controls(stub_vm, capsys):
    stub_vm.fail_build = True
    assert vm_smoke.main(stub_vm.args) == 1
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["error"]["stage"] == "build"
    assert receipt["builds"]["control"]["process"]["returncode"] == 29
    assert not stub_vm.wires


@pytest.mark.parametrize(
    ("negative", "observed"),
    [
        (vm_smoke.ProcessResult("exited", 3), "process_error"),
        (vm_smoke.ProcessResult("timeout"), "process_error"),
        (vm_smoke.ProcessResult("exited", 0, b"truncated"), "invalid_capture"),
        (vm_smoke.ProcessResult("exited", 0, make_capture(((34,), (2,)) * 2)), "match"),
        (vm_smoke.ProcessResult("exited", 0, make_capture(((8,), (1,)) * 2)), "numerical_mismatch"),
    ],
)
def test_bad_negative_capture_never_counts_as_passing_controls(stub_vm, capsys, negative, observed):
    stub_vm.negative = negative
    assert vm_smoke.main(stub_vm.args) == 1
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["status"] == vm_smoke.AUTHOR_CONTROLS_FAILED
    assert receipt["error"]["stage"] == "capture"
    assert receipt["cases"][-1]["observed"] == observed


def test_process_byte_cap_is_enforced_with_a_cpu_child(tmp_path):
    result = vm_smoke.run_process(
        [sys.executable, "-c", "import sys; sys.stdout.buffer.write(b'x' * 1000)"],
        cwd=tmp_path,
        env={},
        timeout=5,
        stdout_cap=20,
    )
    assert not result.ok
    assert result.status == "output_limit"
    assert result.stdout == b"x" * 20
