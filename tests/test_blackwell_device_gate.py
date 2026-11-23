import json
from dataclasses import replace

import pytest

from training.environments.kernel_rlm.research.blackwell_device_gate import author_smoke, gate


def native_elf(
    *, sm=120, osabi=0x41, abi=8, variant=False, kind=2, machine=190, text=b".text.vec_add"
):
    # Synthetic metadata for parser coverage. This is not a loadable CUDA module.
    names = b"\x00.shstrtab\x00" + text + b"\x00"
    # Match the OSABI/version bytes and non-SM flag bits of the captured NVCC author cubin.
    image = gate.ELF.pack(
        b"\x7fELF\x02\x01\x01" + bytes((osabi, abi)),
        kind,
        machine,
        1,
        0,
        0,
        gate.ELF.size,
        0x06000002 | sm << 8 | 8 * variant,
        gate.ELF.size,
        0,
        0,
        gate.SECTION.size,
        3,
        1,
    )
    strings = len(image) + 3 * gate.SECTION.size
    return (
        image
        + bytes(gate.SECTION.size)
        + gate.SECTION.pack(1, 3, 0, 0, strings, len(names), 0, 0, 1, 0)
        + gate.SECTION.pack(11, 1, 6, 0, strings + len(names), 16, 0, 0, 16, 0)
        + names
        + bytes(16)
    )


class Fake:
    def __init__(self, *, image=None, fault=None, mutate=None):
        self.blob = image if image is not None else native_elf()
        self.calls = []
        self.mem = {}
        self.fault = fault
        self.mutate = mutate

    def visit(self, stage):
        self.calls.append(stage)
        if self.fault == stage:
            raise RuntimeError(f"fake {stage} error")
        if self.fault == "watchdog" and stage == "completion":
            raise TimeoutError("fake watchdog")

    def build(self, source, descriptor):
        self.visit("build")
        assert source == gate.AUTHOR_SOURCE
        artifact = gate.Built(
            self.blob,
            gate.digest(source),
            gate.digest(descriptor),
            gate.digest(self.blob),
            "test compiler",
        )
        return self.mutate(artifact) if self.mutate else artifact

    def probe(self):
        self.visit("device")
        capability = (8, 9) if self.fault == "wrong_capability" else (12, 0)
        return gate.Device("synthetic", capability, 0, "mock-no-uuid")

    def load(self, image):
        self.visit("load")
        assert image is self.blob

    def entry(self, name):
        self.visit("entry")
        assert name == "vec_add"

    def allocate(self, initial):
        self.visit("prepare")
        addr = 0x1000 * (1 + len(self.mem))
        assert initial == gate.INITIAL[len(self.mem)]
        self.mem[addr] = bytearray(initial)
        return addr

    def launch(self, manifest, args):
        self.visit("launch")
        assert manifest.grid == (1, 1, 1) and manifest.block == (32, 1, 1)
        assert manifest.shared_bytes == 0
        assert args == (0x1010, 0x2010, 0x3010, 4)
        self.mem[0x3000][16:32] = gate.F32.pack(5, 4, 0, 2)
        if self.fault == "bad_result":
            self.mem[0x3000][16:32] = gate.F32.pack(5, 4, 1, 2)
            print("pass")
        if self.fault == "poison":
            self.mem[0x3000][16:20] = gate.POISON
        if self.fault == "changed_guard":
            self.mem[0x3000][32] ^= 1
        if self.fault == "changed_input":
            self.mem[0x1000][16] ^= 1

    def synchronize(self):
        self.visit("completion")

    def read(self, address, size):
        self.visit("d2h")
        assert size == 48
        data = bytes(self.mem[address])
        return data[:-1] if self.fault == "short_read" else data

    def close(self):
        self.visit("cleanup")
        self.mem.clear()


def failure(receipt, stage):
    assert receipt.stages[stage].status == "fail"
    for later in gate.STAGES[gate.STAGES.index(stage) + 1 :]:
        if later != "cleanup":
            assert receipt.stages[later].status == "not_run"
            assert receipt.stages[later].reason
    assert not receipt.author_hardware_correct and receipt.result is None
    assert receipt.stages["timing"].status == "not_run"


@pytest.mark.parametrize(
    "update",
    [
        {"block": [0, 1, 1]},
        {"block": [1025, 1, 1]},
        {"block": [True, 1, 1]},
        {"block": [32.0, 1, 1]},
        {"shared_bytes": 1},
        {"path": "a.cubin"},
        {"args": 0},
        {"args": ["b", "a", "out", "n"]},
        {"target": "sm_89"},
        {"n": 2**64},
        {"grid": [2, 1, 1]},
        {"entry": "other"},
        {"abi": "other"},
    ],
)
def test_closed_descriptor(update):
    fake = Fake()
    descriptor = json.dumps(json.loads(gate.DESCRIPTOR) | update).encode()
    failure(gate.MockGate(fake.build, fake).check(descriptor), "schema")
    assert not fake.calls


@pytest.mark.parametrize(
    "descriptor",
    [
        gate.DESCRIPTOR[:-1] + b',"entry":"vec_add"}',
        b"[]",
        b"{}",
        b"\xff",
        b" " * 2049,
    ],
)
def test_duplicate_missing_and_invalid_descriptor_bytes(descriptor):
    fake = Fake()
    failure(gate.MockGate(fake.build, fake).check(descriptor), "schema")
    assert not fake.calls


def test_only_exact_author_source_is_admitted():
    fake = Fake()
    failure(gate.MockGate(fake.build, fake).check(source=gate.AUTHOR_SOURCE + b"\n"), "admission")
    assert not fake.calls


@pytest.mark.parametrize(
    "image",
    [
        native_elf(osabi=41),  # decimal 41 is *not* 0x41
        native_elf(osabi=51, abi=7),  # old layout is outside this pinned smoke
        native_elf(abi=7),
        native_elf(kind=1),  # relocatable
        native_elf(machine=62),  # x86-64 ELF
        native_elf(sm=89),
        native_elf(sm=121),
        native_elf(variant=True),
        native_elf(text=b".text.ptx"),
        b".version 8.0\n.target sm_120\n",
        b"\xb1CbF" + bytes(128),
        native_elf()[:-50],
        native_elf(text=b".text.wrong"),
    ],
)
def test_bad_elf_cannot_reach_driver(image):
    fake = Fake(image=image)
    failure(gate.MockGate(fake.build, fake).check(), "architecture")
    assert fake.calls == ["build"]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda built: replace(built, image=built.image[:-1] + b"\xff"),
        lambda built: replace(built, source_sha256="wrong-source"),
        lambda built: replace(built, descriptor_sha256="wrong-descriptor"),
    ],
)
def test_unbound_bytes_cannot_reach_driver(mutate):
    fake = Fake(mutate=mutate)
    failure(gate.MockGate(fake.build, fake).check(), "architecture")
    assert fake.calls == ["build"]


FLOW = [
    "build",
    "device",
    "load",
    "entry",
    "prepare",
    "prepare",
    "prepare",
    "launch",
    "completion",
    "d2h",
    "d2h",
    "d2h",
    "cleanup",
]


def test_exact_readback_is_mock_author_only():
    fake = Fake()
    runner = gate.MockGate(fake.build, fake)
    receipt = runner.check()
    assert receipt.result == (5.0, 4.0, 0.0, 2.0)
    assert receipt.result_hex == "0000a040000080400000000000000040"
    assert receipt.role == "author_control" and receipt.backend == "mock"
    assert not receipt.author_hardware_correct and not receipt.quarantined
    assert receipt.cubin_sha256 == gate.digest(fake.blob)
    assert receipt.source_sha256 == gate.digest(gate.AUTHOR_SOURCE)
    assert receipt.descriptor_sha256 == gate.digest(gate.DESCRIPTOR)
    assert all(receipt.stages[name].status == "pass" for name in gate.STAGES[:-1])
    assert receipt.stages["timing"].status == "not_run"
    assert fake.calls == FLOW
    assert runner.check().run_id != receipt.run_id
    assert fake.calls == FLOW * 2


@pytest.mark.parametrize(
    "stage", ["build", "device", "load", "entry", "prepare", "launch", "completion", "d2h"]
)
def test_driver_fault_stages_stop_calls(stage):
    fake = Fake(fault=stage)
    receipt = gate.MockGate(fake.build, fake).check()
    failure(receipt, stage)
    assert fake.calls == FLOW[: FLOW.index(stage) + 1] + ([] if stage == "build" else ["cleanup"])
    assert receipt.quarantined == (stage != "build")


def test_wrong_device_prevents_load():
    fake = Fake(fault="wrong_capability")
    failure(gate.MockGate(fake.build, fake).check(), "device")
    assert fake.calls == ["build", "device", "cleanup"]


@pytest.mark.parametrize(
    ("fault", "stage", "reason"),
    [
        ("bad_result", "numerics", "exact reference"),
        ("poison", "numerics", "poison"),
        ("changed_guard", "integrity", "guard changed"),
        ("changed_input", "integrity", "immutable input"),
        ("short_read", "d2h", "exact allocation"),
    ],
)
def test_readback_bytes_not_worker_claims(fault, stage, reason, capsys):
    fake = Fake(fault=fault)
    receipt = gate.MockGate(fake.build, fake).check()
    failure(receipt, stage)
    assert reason in receipt.stages[stage].reason
    expected_calls = FLOW[: FLOW.index("d2h") + 1] + ["cleanup"] if fault == "short_read" else FLOW
    assert fake.calls == expected_calls
    if fault == "bad_result":
        assert capsys.readouterr().out == "pass\n"


def test_watchdog_stops_without_cleanup_or_retry():
    fake = Fake(fault="watchdog")
    runner = gate.MockGate(fake.build, fake)
    receipt = runner.check()
    failure(receipt, "completion")
    assert receipt.quarantined and receipt.stages["cleanup"].status == "not_run"
    assert runner.check().quarantined
    assert fake.calls == FLOW[: FLOW.index("d2h")]


def test_cleanup_error_never_promotes_mock():
    fake = Fake(fault="cleanup")
    receipt = gate.MockGate(fake.build, fake).check()
    assert receipt.stages["numerics"].status == "pass"
    assert receipt.stages["cleanup"].status == "fail" and receipt.quarantined
    assert not receipt.author_hardware_correct and receipt.stages["timing"].status == "not_run"


def test_hardware_build_path_with_process_calls_stubbed(monkeypatch):
    image = native_elf()
    nvcc, ccbin = author_smoke.Path("/author/bin/nvcc"), author_smoke.Path("/author/bin/g++")
    monkeypatch.setattr(author_smoke.subprocess, "check_output", lambda *a, **k: b"V13.3.73")

    def fake_compile(argv, *, cwd, env, **kwargs):
        assert argv == [
            nvcc,
            "--cubin",
            "--generate-code",
            "arch=compute_120,code=sm_120",
            "--compiler-bindir",
            ccbin,
            "author.cu",
            "-o",
            "author.cubin",
        ]
        assert (cwd / "author.cu").read_bytes() == gate.AUTHOR_SOURCE
        assert set(env) == {"PATH", "HOME", "TMPDIR", "LC_ALL"}
        (cwd / "author.cubin").write_bytes(image)

    monkeypatch.setattr(author_smoke.subprocess, "run", fake_compile)
    built = author_smoke.build_author(nvcc, ccbin, gate.AUTHOR_SOURCE, gate.DESCRIPTOR)
    assert built.image == image and built.cubin_sha256 == gate.digest(image)
    assert built.source_sha256 == gate.digest(gate.AUTHOR_SOURCE)
    assert built.descriptor_sha256 == gate.digest(gate.DESCRIPTOR)
