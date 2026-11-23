"""CPU-only ABI, synthetic semantics and author capture tests (no NVCC or libcuda)."""

import ctypes as c
import json
import math
import struct
from dataclasses import replace
from importlib import import_module
from pathlib import Path

import pytest

PKG = "training.environments.kernel_rlm.research.flashinfer-paged-gqa-decode"
abi = import_module(PKG + ".device_abi")
cases = import_module(PKG + ".author_cases")
runner = import_module(PKG + ".author_capture")


def test_strict_descriptor_and_exact_submission_bytes():
    raw = Path(abi.__file__).with_name("direct.descriptor.json").read_bytes()
    alternate = b" \n" + raw + b"\n"
    body = b"  // reviewed bytes\r\n__device__ void fi_decode_body();\n"
    sub = abi.decode_submission(body, alternate)
    assert sub.source is body and sub.descriptor_bytes is alternate
    assert sub.descriptor.encode() == raw
    for value in [b"{}", raw + b"{}", raw[:1] + b'"plan":"split",' + raw[1:], b" " * 4097]:
        with pytest.raises(ValueError):
            abi.parse_descriptor(value)
    for fields in [
        {"abi": "vec_add_f32_v1"},
        {"plan": "split", "splits": 8},
        {"splits": 2},
        {"geometry": ["head128", "head128"]},
        {"geometry": ["head256"]},
        {"geometry": [True]},
        {"entry": "fi_decode"},
        {"path": "x"},
        {"args": []},
        {"target": "sm_120"},
        {"B": 2},
    ]:
        with pytest.raises(ValueError):
            abi.parse_descriptor(json.dumps(json.loads(raw) | fields).encode())
    for value in [b"", b" \n", b"\xff", b"x\x00y", b"x" * 131073, b"```cuda\nx\n```", "str"]:
        with pytest.raises(ValueError):
            abi.decode_submission(value, raw)


@pytest.mark.parametrize(
    "fields",
    [
        {"B": True},
        {"B": 0},
        {"B": 17},
        {"P": 0},
        {"P": 513},
        {"M": -1},
        {"M": 1025},
        {"scale": True},
        {"scale": 0},
        {"scale": -1},
        {"scale": 1e-50},
        {"scale": 1e100},
        {"scale": float("nan")},
        {"scale": float("inf")},
        {"Q": b""},
        {"I": struct.pack("<3i", 1, 1, 2)},
        {"I": struct.pack("<3i", 0, 3, 2)},
        {"I": struct.pack("<3i", 0, 1, 1)},
        {"J": struct.pack("<2i", 0, -1)},
        {"J": struct.pack("<2i", 0, 2)},
        {"T": struct.pack("<2i", 1, 0)},
        {"T": struct.pack("<2i", 1, 65)},
    ],
)
def test_bad_fixtures_rejected(fields):
    with pytest.raises(ValueError):
        abi.prepare(replace(cases.analytic(), **fields))


def test_bad_lengths_finite_values_and_max_pages():
    f = cases.analytic()
    with pytest.raises(ValueError, match="finite"):
        abi.prepare(replace(f, V=b"\x80\x7f" + f.V[2:]))
    with pytest.raises(ValueError, match="unsafe"):
        abi.prepare(replace(f, V=b"\x7f\x7f" + f.V[2:]))
    with pytest.raises(ValueError, match="too many pages"):
        abi.prepare(
            replace(f, M=130, I=cases.pack_i32((0, 129, 130)), J=cases.pack_i32((0,) * 130))
        )
    with pytest.raises(ValueError, match="tail"):
        abi.prepare(replace(f, I=cases.pack_i32((0, 0, 2))))


def test_analytic_closed_form_checks_every_head_and_dimension():
    p = abi.prepare(cases.analytic())
    out_bytes, lse_bytes = cases.oracle(p)
    out = cases.unpack_bf16(out_bytes)
    lse = struct.unpack("<32f", lse_bytes)
    assert p.token_counts() == (1, 2)
    r = math.exp(0.5)
    for b in range(2):
        for h in range(16):
            kv = h // 8
            for d in range(128):
                a = 2 + (d % 8) / 8
                expect = a - kv if b == 0 else ((a - kv) + r * (-a - kv)) / (1 + r)
                index = (b * 16 + h) * 128 + d
                assert struct.unpack_from("<H", out_bytes, index * 2)[0] == cases.bf16(expect)
                assert out[index] != 0
            assert lse[b * 16 + h] == pytest.approx(((0.5 - kv) + b * math.log1p(r)) / math.log(2))


def f32(x):
    return c.c_float(x).value


def online_semantics(p, variant="positive"):
    """CPU specification of author recurrence, not execution of the CUDA fragment."""
    f = p.fixture
    Q, K, V = (cases.unpack_bf16(getattr(f, k)) for k in "QKV")
    out_bytes, lse_bytes = bytearray(), bytearray()
    for b in range(f.B):
        start, end = p.indptr[b : b + 2]
        for h in range(16):
            if start == end:
                out_bytes.extend(abi.ZERO_BF16 * 128)
                lse_bytes.extend(abi.NEG_INF)
                continue
            kv = h % 2 if variant == "wrong-head" else h // 8
            qoff = (b * 16 + h) * 128
            peak, mass, acc = -math.inf, 0.0, [0.0] * 128
            for page_slot in range(start, end):
                stop = p.tail[b] if page_slot + 1 == end and variant != "wrong-tail" else 64
                for t in range(stop):
                    off = ((p.indices[page_slot] * 64 + t) * 2 + kv) * 128
                    dot = sum(Q[qoff + d] * K[off + d] for d in range(128))
                    logit = f32(f32(dot * p.scale) * f32(math.log2(math.e)))
                    next_peak = max(peak, logit)
                    old, add = f32(2.0 ** (peak - next_peak)), f32(2.0 ** (logit - next_peak))
                    mass = f32(mass * old + add)
                    acc = [f32(x * old + add * V[off + d]) for d, x in enumerate(acc)]
                    peak = next_peak
            out_bytes.extend(struct.pack("<128H", *(cases.bf16(f32(x / mass)) for x in acc)))
            lse = peak + math.log2(mass)
            if variant == "wrong-lse":
                lse *= math.log(2)
            lse_bytes.extend(struct.pack("<f", lse))
    return bytes(out_bytes), bytes(lse_bytes)


@pytest.mark.parametrize(
    ("name", "counts"),
    [
        ("irregular", (1, 63, 64, 65, 153)),
        ("mixed-empty", (0, 63, 64, 65, 153)),
        ("all-empty", (0, 0)),
    ],
)
def test_online_semantics_against_independent_oracle(name, counts):
    p = abi.prepare(cases.fixture(name))
    assert p.token_counts() == counts
    assert cases.compare(p, online_semantics(p)) == {"O": 0, "L": 0}
    if name == "all-empty":
        assert p.indices == () and p.payloads["J"] == b"\xff" * 4
        out_bytes, lse_bytes = cases.oracle(p)
        assert out_bytes == b"\x00" * (2 * 16 * 128 * 2) and lse_bytes == abi.NEG_INF * 32
        with pytest.raises(ValueError, match="sentinel"):
            cases.compare(p, (out_bytes, b"\x00" * len(lse_bytes)))


def elf(entry=abi.ENTRY, sm=120):
    # Synthetic metadata only, not a loadable cubin.
    names = b"\0.shstrtab\0.text." + entry.encode() + b"\0"
    sections = runner.SECTION
    offset = runner.ELF.size + 3 * sections.size
    ident = b"\x7fELF\x02\x01\x01\x41\x08".ljust(16, b"\x00")
    return b"".join(
        (
            runner.ELF.pack(
                ident,
                2,
                190,
                1,
                0,
                0,
                runner.ELF.size,
                sm << 8,
                runner.ELF.size,
                0,
                0,
                sections.size,
                3,
                1,
            ),
            sections.pack(0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            sections.pack(1, 3, 0, 0, offset, len(names), 0, 0, 1, 0),
            sections.pack(11, 1, 4, 0, offset + len(names), 1, 0, 0, 1, 0),
            names,
            b"\x00",
        )
    )


@pytest.mark.parametrize(
    "image", [b"ptx", elf(sm=89), elf("bad"), elf("fi_decode.ptx"), elf()[:-1]]
)
def test_native_checks_before_driver(image):
    receipt = runner.run(
        "analytic", "positive", lambda _: (image, {}), lambda: pytest.fail("device touched")
    )
    assert receipt["stopped_at"] == "architecture"
    assert receipt["stages"]["device"] == "not_run"


class Driver:
    def __init__(self, prepared, expected, *, fail_at=None):
        self.prepared, self.expected = prepared, expected
        self.allocations = {}
        self.fail_at = fail_at
        self.events = []

    def probe(self):
        self.events.append("probe")
        return {"fake": True}

    def load(self, image):
        assert image is IMAGE
        self.events.append("load")

    def entry(self, launch):
        assert launch == abi.Launch((16, self.prepared.fixture.B, 1))
        self.events.append("entry")
        return {}

    def allocate(self, raw):
        self.events.append("allocate")
        ptr = 0x100000 * (len(self.allocations) + 1)
        self.allocations[ptr] = raw
        return ptr

    def launch(self, spec, args):
        self.events.append("launch")
        assert spec == self.prepared.launch
        assert tuple(type(arg) for arg in args) == tuple(arg.ctype for arg in abi.ARGS)
        a = [arg.value for arg in args]
        f = self.prepared.fixture
        expected = [ptr + 256 for ptr in self.allocations]
        assert a[:6] == expected[:6] and a[6:10] == [self.prepared.scale, f.B, f.P, f.M]
        assert a[10:] == expected[6:]
        for role, address in zip(abi.POINTERS, a[:6] + a[10:], strict=True):
            assert self.allocations[address - 256][256:-256] == self.prepared.payloads[role]
        if self.fail_at == "launch":
            raise RuntimeError("fake immediate failure")
        if self.fail_at != "numerics":
            for address, out in zip(a[10:], self.expected, strict=True):
                self.allocations[address - 256] = abi.PREFIX + out + abi.SUFFIX
        if self.fail_at == "guard":
            self.allocations[a[10] - 256] = b"!" + self.allocations[a[10] - 256][1:]
        if self.fail_at == "input":
            data = self.allocations[a[0] - 256]
            self.allocations[a[0] - 256] = data[:256] + bytes([data[256] ^ 1]) + data[257:]

    def synchronize(self):
        self.events.append("synchronize")
        if self.fail_at == "completion":
            raise RuntimeError("fake async failure")
        if self.fail_at == "timeout":
            raise TimeoutError("watchdog")

    def read(self, ptr, count):
        self.events.append("read")
        assert count == len(self.allocations[ptr])
        return self.allocations[ptr][1:] if self.fail_at == "d2h" else self.allocations[ptr]

    def close(self):
        self.events.append("close")
        if self.fail_at == "cleanup":
            raise RuntimeError("fake cleanup failure")


IMAGE = elf()


@pytest.mark.parametrize("variant", ["positive", *runner.CONTROL])
def test_mock_author_capture_and_separate_controls(variant):
    p = abi.prepare(cases.analytic())
    driver = Driver(p, online_semantics(p, variant))
    body, translation = runner.compose(variant)

    def builder(src):
        assert src == translation and body in src
        return IMAGE, {"fake": True}

    receipt = runner.run("analytic", variant, builder, lambda: driver)
    assert receipt["expected_outcome_met"] and receipt["backend"] == "mock"
    assert receipt["stages"]["timing"] == "not_run"
    assert receipt["stages"]["numerics"] == ("pass" if variant == "positive" else "fail")
    assert driver.events == [
        "probe",
        "load",
        "entry",
        *(["allocate"] * 8),
        "launch",
        "synchronize",
        *(["read"] * 8),
        "close",
    ]
    assert all(receipt["stages"][stage] == "pass" for stage in runner.STAGES[:-2])
    assert receipt["cubin_sha256"] == abi.digest(IMAGE)
    assert receipt["source_sha256"] == abi.digest(body)


@pytest.mark.parametrize(
    ("fail", "stopped"),
    [
        ("launch", "launch"),
        ("completion", "completion"),
        ("d2h", "d2h"),
        ("guard", "integrity"),
        ("input", "integrity"),
        ("cleanup", "cleanup"),
        ("numerics", "numerics"),
        ("timeout", "completion"),
    ],
)
def test_bad_capture_never_passes(fail, stopped):
    p = abi.prepare(cases.analytic())
    driver = Driver(p, cases.oracle(p), fail_at=fail)
    receipt = runner.run("analytic", "positive", lambda _: (IMAGE, {}), lambda: driver)
    assert not receipt["expected_outcome_met"] and receipt["stopped_at"] == stopped
    assert receipt["quarantined"] == (fail != "numerics")
    assert receipt["stages"]["timing"] == "not_run"
    assert driver.events.count("close") == (fail != "timeout")
    assert receipt["stages"]["numerics"] != "pass"


def test_parameter_offsets_and_driver_c_launch_packing_without_cuda():
    assert abi.PARAM_LAYOUT == (
        (0, 8),
        (8, 8),
        (16, 8),
        (24, 8),
        (32, 8),
        (40, 8),
        (48, 4),
        (52, 4),
        (56, 4),
        (60, 4),
        (64, 8),
        (72, 8),
    )
    p = abi.prepare(cases.analytic())
    values = p.arguments({role: i * 256 for i, role in enumerate(abi.POINTERS, start=1)})
    d = object.__new__(runner.AuthorDriver)  # deliberately do not run __init__ or load libcuda
    d.function, d.stream = object(), object()

    def call(name, types, function, *args):
        assert name == "cuLaunchKernel"
        assert len(types) == 11 and function is d.function
        assert args[:7] == (16, 2, 1, 128, 1, 1, 0) and args[7] is d.stream
        params = args[8]
        assert args[9] is None
        assert all(
            arg.ctype.from_address(params[i]).value == values[i].value
            for i, arg in enumerate(abi.ARGS)
        )

    d.call = call
    d.launch(p.launch, values)
