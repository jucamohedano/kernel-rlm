"""CPU-only tests for the closed KB30 ABI and author lifecycle."""

import json
import struct
from pathlib import Path

import pytest

TASK = (
    Path(__file__).resolve().parents[1]
    / "training/environments/kernel_rlm/research"
    / "kernelbench-gemm-groupnorm-hardtanh"
)


@pytest.fixture
def abi(monkeypatch):
    monkeypatch.syspath_prepend(str(TASK))
    import device_abi

    return device_abi


def test_full_manifest_and_exact_bytes(abi):
    req = b'{ "geometry": ["tile16x16","group256"], "plan":"staged", "abi":"kb30_f32_device_v1" }\n'
    d = abi.parse_descriptor(req)
    assert d.request is req and d.identity == abi.parse_descriptor(abi.AUTHOR_DESCRIPTOR).identity
    assert abi.BY_NAME["Z"].size == 33_554_432
    assert sum(b.size for b in d.buffers) == 369_197_056
    assert d.allocation_size == 369_200_640
    linear, finish = d.launches
    assert (linear.grid, linear.block, finish.grid, finish.block) == (
        (512, 64, 1),
        (16, 16, 1),
        (16, 1024, 1),
        (256, 1, 1),
    )
    pointers = {b.name: (i + 1) << 32 for i, b in enumerate(d.buffers)}
    assert linear.arguments(pointers) == tuple(pointers[n] for n in ("X", "W", "bias", "Z"))
    assert finish.arguments(pointers) == tuple(pointers[n] for n in ("Z", "gamma", "beta", "Y"))
    assert linear.param_layout == finish.param_layout == ((0, 8), (8, 8), (16, 8), (24, 8))
    source = abi.author_fragment("positive")
    tu, wrappers = abi.author_translation_unit(d, source)
    assert source in tu and wrappers in tu
    assert (
        b"void kb_linear(const float* X, const float* W, const float* bias, float* Z)" in wrappers
    )
    assert (
        b"void kb_finish(const float* Z, const float* gamma, const float* beta, float* Y)"
        in wrappers
    )
    for control in ("interleaved_groups", "sample_variance", "missing_producer"):
        assert abi.digest(abi.author_fragment(control)) != abi.digest(source)
    assert len({abi.digest(abi.author_fragment(control)) for control in abi.CONTROLS}) == 4
    alt = req.replace(b"tile16x16", b"tile32x8").replace(b"group256", b"group128")
    assert abi.parse_descriptor(alt).launches[0].grid == (256, 128, 1)
    assert abi.parse_descriptor(alt).launches[1].block == (128, 1, 1)


@pytest.mark.parametrize(
    "payload",
    [
        b"{}",
        b"[]",
        b"```json\n{}\n```",
        b"\xff",
        b" " * 4097,
        b'{"abi":"kb30_f32_device_v1","plan":"staged","geometry":["group256","tile16x16"]}',
        b'{"abi":"kb30_f32_device_v1","plan":"staged","geometry":["tile16x16"]}',
        b'{"abi":"kb30_f32_device_v1","plan":"staged","geometry":[true,"group256"]}',
        b'{"abi":"kb30_f32_device_v1","plan":"staged","plan":"staged","geometry":["tile16x16","group256"]}',
        b'{"abi":"kb30_f32_device_v1","plan":"fused","geometry":["group256"],"splits":2}',
        b'{"abi":"kb30_f32_device_v2","plan":"fused","geometry":["group256"]}',
        b'{"abi":"kb30_f32_device_v1","plan":"raw","geometry":["group256"]}',
    ],
)
def test_descriptor_rejects_invalid(abi, payload):
    with pytest.raises(ValueError):
        abi.parse_descriptor(payload)


def test_no_raw_overrides_fused_fails_before_build_and_fragment_framing(abi):
    def unreachable(*_):
        pytest.fail("compiler and driver must not be reached")

    d = json.loads(abi.AUTHOR_DESCRIPTOR)
    for key in ("path", "pointer", "args", "scratch_bytes", "target"):
        receipt, data = abi.run_trusted(
            unreachable, unreachable, "positive", {}, descriptor=abi.canonical({**d, key: 0})
        )
        assert receipt["stages"]["schema"]["status"] == "fail" and not data
    fused_bytes = abi.canonical({**d, "plan": "fused", "geometry": ["group256"]})
    fused = abi.parse_descriptor(fused_bytes)
    assert [s.entry for s in fused.launches] == ["kb_forward"] and "Z" not in [
        b.name for b in fused.buffers
    ]
    receipt, data = abi.run_trusted(
        unreachable, unreachable, "positive", {}, descriptor=fused_bytes
    )
    assert receipt["outcome"] == "unsupported" and receipt["scratch_bytes"] == 0 and not data
    with pytest.raises(LookupError, match="unavailable"):
        abi.validate_fragment(None)
    for raw, why in [
        (b"  ", "empty"),
        (b"```cuda\nsome text\n```", "fenced"),
        (b"no\0", "NUL-free"),
        (b"x" * (131073), "bounded"),
    ]:
        with pytest.raises(ValueError, match=why):
            abi.validate_fragment(raw)
    raw = b" // caf\xc3\xa9\r\n "
    assert abi.validate_fragment(raw) is raw
    for dims in [(True, 4), (0, 2), (2**62, 8)]:
        with pytest.raises(ValueError):
            abi.checked_product(*dims)
    with pytest.raises(ValueError, match="overlapping"):
        abi.check_range(512, 1024, [(256, 1024)])


def synthetic_elf(abi, *, entries=("kb_linear", "kb_finish"), sm=120, extra=None, point_oob=False):
    """Metadata only; cannot execute. Order of ELF sections is not launch order."""
    labels = [b".shstrtab", *(f".text.{name}".encode() for name in reversed(entries))]
    if extra:
        labels.append(extra)
    strings = b"\0" + b"\0".join(labels) + b"\0"
    pos = abi.ELF.size
    first_text = pos + len(strings)
    start = first_text + len(labels) - 1
    header = abi.ELF.pack(
        b"\x7fELF\x02\x01\x01\x41\x08",
        2,
        190,
        1,
        0,
        0,
        start,
        sm << 8,
        abi.ELF.size,
        0,
        0,
        abi.SECTION.size,
        1 + len(labels),
        1,
    )
    sections = [bytes(abi.SECTION.size)]
    for i, name in enumerate(labels):
        section_type, attrs = (3, 0) if i == 0 else (1, 4)
        offset, size = (pos, len(strings)) if i == 0 else (first_text + i - 1, 1)
        if point_oob and i == 1:
            offset += 2**50
        sections.append(
            abi.SECTION.pack(strings.index(name), section_type, attrs, 0, offset, size, 0, 0, 0, 0)
        )
    return header + strings + bytes(len(labels) - 1) + b"".join(sections)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sm": 121},
        {"sm": 90},
        {"entries": ("kb_linear",)},
        {"extra": b".nv_fatbin"},
        {"extra": b".ptx"},
        {"extra": b".text.unexpected"},
        {"point_oob": True},
    ],
)
def test_native_metadata_rejects(abi, kwargs):
    with pytest.raises(ValueError):
        abi.inspect_native(synthetic_elf(abi, **kwargs), ("kb_linear", "kb_finish"))


@pytest.fixture
def fake_job(abi, monkeypatch):
    # Exercise exact stage/argument orchestration without materializing >350 MiB on CPU.
    sizes = tuple(abi.Buffer(b.name, (4,), b.kind) for b in abi.BUFFERS)
    monkeypatch.setattr(abi, "BUFFERS", sizes)
    monkeypatch.setattr(abi, "INPUTS", tuple(b for b in sizes if b.kind == "input"))
    monkeypatch.setattr(abi, "BY_NAME", {b.name: b for b in sizes})
    image = synthetic_elf(abi)

    class FakeDriver:
        def __init__(self):
            self.calls, self.memory, self.bases = [], {}, []
            self.fail = None
            self.module = iter(self.variants)

        def event(self, key, value=None):
            self.calls.append((key, value))
            if key == self.fail:
                raise RuntimeError(f"fake {key} error")

        def probe(self):
            self.event("probe")
            return {"capability": (12, 0)}

        def load(self, data):
            self.event("load")
            assert data is image
            return next(self.module)

        def entry(self, module, slot):
            self.event("entry", slot.entry)
            return (module, slot.entry), {"static_shared_bytes": 0}

        def allocate(self, size):
            base = (1 + len(self.bases)) << 33
            self.bases.append(base)
            self.event("allocate", size)
            return base

        def write(self, address, data):
            self.memory[address] = data
            self.event("write", address)

        def poison(self, address, size, word):
            self.memory[address] = struct.pack("<I", word) * (size // 4)
            self.event("poison", address)

        def prepared(self):
            self.event("prepared")

        def launch(self, function, slot, args):
            self.event(slot.entry, args)
            variant, entry = function
            assert entry == slot.entry
            p = [b + abi.GUARD for b in self.bases]
            if entry == "kb_linear":
                assert args == (p[0], p[1], p[2], p[5])
                if variant not in ("missing_producer", "stale_producer"):
                    self.memory[args[-1]] = self.memory[args[0]]
            else:
                assert args == (p[5], p[3], p[4], p[6])
                self.memory[args[-1]] = self.memory[args[0]]
                if variant in ("interleaved_groups", "sample_variance"):
                    self.memory[args[-1]] = struct.pack("<4f", 0.5, 0.5, 0.5, 0.5)

        def synchronize(self):
            self.event("completion")
            if self.fail in ("guard", "immutable"):
                addr = self.bases[0] + (abi.GUARD if self.fail == "immutable" else 0)
                self.memory[addr] = b"!" * len(self.memory[addr])

        def read(self, address, size):
            self.event("read")
            return self.memory[address]

        def close(self):
            self.event("close")

    def job(control="positive", *, fault=None, descriptor=abi.AUTHOR_DESCRIPTOR):
        first, other = (
            struct.pack("<4f", 0.0, 0.0, 0.0, 0.0),
            struct.pack("<4f", 1.0, 1.0, 1.0, 1.0),
        )
        FakeDriver.variants = (
            ["positive", "stale_producer"] if control == "stale_producer" else [control]
        )
        driver = FakeDriver()
        driver.fail = fault
        next_x = other if control in ("positive", "stale_producer") else None

        def builder(variant, tu):
            if fault == "build":
                raise TimeoutError("fake compiler watchdog")
            return abi.Built(
                image, abi.digest(tu), "0" * 64 if fault == "digest" else abi.digest(image), {}
            )

        receipt, outputs = abi.run_trusted(
            builder,
            lambda: driver,
            control,
            {b.name: first for b in abi.INPUTS},
            next_x,
            descriptor=descriptor,
        )
        return driver, receipt, outputs, [first, other] if next_x else [first]

    return job


def mock_compare(z, y, expected):
    from math import isfinite, isnan

    z_f, y_f = struct.unpack("<4f", z), struct.unpack("<4f", y)
    return {
        "passed": y == expected,
        "scratch_finite": all(map(isfinite, z_f)),
        "output_finite": all(map(isfinite, y_f)),
        "scratch_all_nan": all(map(isnan, z_f)),
        "output_all_nan": all(map(isnan, y_f)),
        "interior_mismatched": int(y != expected),
    }


@pytest.mark.parametrize(
    "control",
    ["positive", "interleaved_groups", "sample_variance", "missing_producer", "stale_producer"],
)
def test_mock_run(abi, fake_job, control):
    driver, receipt, output, refs = fake_job(control)
    assert abi.compare_forwards(receipt, output, refs, mock_compare)
    calls = [name for name, _ in driver.calls]
    assert calls.count("allocate") == 7 and calls[-1] == "close"
    for i, name in enumerate(calls):
        if name == "kb_linear":
            assert calls[i - 1 : i + 4] == [
                "prepared",
                "kb_linear",
                "kb_finish",
                "completion",
                "read",
            ]
    last = receipt["forwards"][-1]
    assert last["stages"]["numerics"]["status"] == ("pass" if control == "positive" else "fail")
    assert receipt["stages"]["timing"]["status"] == "not_run"
    if control == "stale_producer":
        assert calls.count("poison") == 3
        assert last["z_sha256"] == receipt["forwards"][0]["z_sha256"]


@pytest.mark.parametrize(
    "fault, stage",
    [
        ("build", "build"),
        ("digest", "architecture"),
        ("load", "load"),
        ("entry", "entry"),
        ("kb_linear", "launch[0]"),
        ("kb_finish", "launch[1]"),
        ("completion", "completion"),
        ("read", "d2h"),
        ("guard", "integrity"),
        ("immutable", "integrity"),
        ("close", "cleanup"),
    ],
)
def test_failed_stages_do_not_qualify(abi, fake_job, fault, stage):
    driver, receipt, output, refs = fake_job(fault=fault)
    assert not abi.compare_forwards(receipt, output, refs, mock_compare)
    if stage in abi.STAGES:
        stages = receipt["forwards"][0]["stages"]
        assert stages["numerics"]["status"] == "not_run"
    else:
        stages = receipt["stages"]
    assert stages[stage]["status"] == "fail"
    if fault in ("build", "digest"):
        assert not driver.calls and not receipt["quarantined"]
    else:
        assert driver.calls[-1][0] == "close" and receipt["quarantined"]
