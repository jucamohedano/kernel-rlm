"""CPU-testable gate for one reviewed author kernel. Not a device sandbox."""

import hashlib
import json
import struct
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from threading import Lock
from typing import Literal, Protocol

ABI = "vec_add_f32_v1"
ENTRY = "vec_add"
TARGET = "sm_120"
CAPABILITY = (12, 0)
N = 4
GRID = (1, 1, 1)
BLOCK = (32, 1, 1)
SHARED_BYTES = 0
A = (1.0, -2.0, 3.0, 0.0)
B = (4.0, 6.0, -3.0, 2.0)
EXPECTED = (5.0, 4.0, 0.0, 2.0)
F32 = struct.Struct("<4f")
# First and last 16 bytes of each private allocation are checked, including inputs.
PREFIX = bytes.fromhex("b77efa11de05493242392bf0eb391816")
SUFFIX = bytes.fromhex("545c0991d4e98ab98b327761f3b70fc3")
POISON = bytes.fromhex("a500c07f")  # quiet NaN, byte-distinct from any expected result
INITIAL = tuple(PREFIX + data + SUFFIX for data in (F32.pack(*A), F32.pack(*B), POISON * N))
AUTHOR_SOURCE = b"""extern "C" __global__ void vec_add(const float* a, const float* b, float* out, int n) {
    int i = threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}
"""
MAX_CUBIN_BYTES = 1 << 20


@dataclass(frozen=True)
class Manifest:
    abi: str = ABI
    entry: str = ENTRY
    target: str = TARGET
    n: int = N
    grid: tuple[int, int, int] = GRID
    block: tuple[int, int, int] = BLOCK
    shared_bytes: int = SHARED_BYTES
    # Signature is implicit in the ABI, NOT caller-overridable.


MANIFEST = Manifest()
DESCRIPTOR = json.dumps(vars(MANIFEST), separators=(",", ":"), sort_keys=True).encode()


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def unique_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    obj: dict[str, object] = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError(f"duplicate manifest key: {key}")
        obj[key] = value
    return obj


def parse_manifest(payload: bytes) -> Manifest:
    if type(payload) is not bytes or not 0 < len(payload) <= 2048:
        raise ValueError("expected bounded UTF-8 manifest bytes")
    obj = json.loads(payload.decode("utf-8"), object_pairs_hook=unique_keys)
    expected = vars(MANIFEST)
    if type(obj) is not dict or obj.keys() != expected.keys():
        raise ValueError("closed vec_add_f32_v1 manifest fields required")
    for key, fixed in expected.items():
        value = obj[key]
        if isinstance(fixed, tuple):
            valid = (
                type(value) is list
                and len(value) == 3
                and all(type(dim) is int for dim in value)
                and tuple(value) == fixed
            )
        else:
            valid = type(value) is type(fixed) and value == fixed
        if not valid:
            raise ValueError(f"unsupported {key} for {ABI}")
    return MANIFEST


@dataclass(frozen=True)
class Built:
    """Trusted builder observation, not constructible from submission data."""

    image: bytes
    source_sha256: str
    descriptor_sha256: str
    cubin_sha256: str
    toolchain: str


@dataclass(frozen=True)
class Device:
    name: str
    capability: tuple[int, int]
    driver_version: int
    uuid: str


class Driver(Protocol):
    def probe(self) -> Device: ...
    def load(self, image: bytes) -> None: ...
    def entry(self, name: str) -> None: ...
    def allocate(self, initial: bytes) -> int: ...
    def launch(self, manifest: Manifest, args: tuple[int, int, int, int]) -> None: ...
    def synchronize(self) -> None: ...
    def read(self, address: int, size: int) -> bytes: ...
    def close(self) -> None: ...


# Pinned NVCC 13.3.73 sm_120 cubin: EI_OSABI is hex 0x41, EI_ABIVERSION is 8.
# In this Blackwell layout e_flags[15:8] is the real SM, bit 3 the accelerator variant.
ELF = struct.Struct("<16sHHIQQQIHHHHHH")
SECTION = struct.Struct("<IIQQQQIIQQ")


def inspect_native(image: bytes) -> str:
    """Check the trusted compiler's direct ELF output, not a validator for hostile cubins."""
    if type(image) is not bytes or not ELF.size <= len(image) <= MAX_CUBIN_BYTES:
        raise ValueError("expected bounded immutable cubin bytes")
    ident, kind, machine, version, _, _, start, flags, ehsize, _, _, shsize, count, names = (
        ELF.unpack_from(image)
    )
    if (
        ident[:7] != b"\x7fELF\x02\x01\x01"
        or ident[7:9] != b"\x41\x08"
        or (kind, machine, version, ehsize) != (2, 190, 1, ELF.size)  # ET_EXEC, EM_CUDA
    ):
        raise ValueError("expected direct native CUDA ELF, not PTX, fatbin or host object")
    if (flags >> 8) & 0xFF != 120 or flags & 0x8:
        raise ValueError("native ELF target must be base sm_120")
    if (
        shsize != SECTION.size
        or not 0 < names < count <= 256
        or start < ELF.size
        or start + shsize * count > len(image)
    ):
        raise ValueError("invalid cubin section table")
    sections = [SECTION.unpack_from(image, start + shsize * i) for i in range(count)]
    _, table_kind, _, _, offset, size, *_ = sections[names]
    if table_kind != 3 or offset + size > len(image):  # SHT_STRTAB
        raise ValueError("invalid cubin section names")
    strings = image[offset : offset + size]
    executable = []
    for index, section_type, attrs, _, offset, size, *_ in sections:
        end = strings.find(b"\x00", index)
        if end < 0 or (section_type != 8 and offset + size > len(image)):  # SHT_NOBITS
            raise ValueError("invalid cubin section")
        name = strings[index:end]
        if b"ptx" in name.lower() or b"fatbin" in name.lower():
            raise ValueError("PTX/fatbin section not allowed")
        if section_type == 1 and attrs & 4 and size:  # SHT_PROGBITS, SHF_EXECINSTR
            executable.append(name)
    if executable != [f".text.{ENTRY}".encode()]:
        raise ValueError("expected exactly the fixed native entry's text section")
    return TARGET


STAGES = (
    "schema",
    "admission",
    "build",
    "architecture",
    "device",
    "load",
    "entry",
    "prepare",
    "launch",
    "completion",
    "d2h",
    "integrity",
    "numerics",
    "cleanup",
    "timing",
)
StageStatus = Literal["not_run", "pass", "fail"]
Backend = Literal["mock", "hardware"]


@dataclass(frozen=True)
class Outcome:
    status: StageStatus
    reason: str = ""


@dataclass
class Receipt:
    backend: Backend = "mock"
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    role: Literal["author_control"] = "author_control"
    abi: str = ABI
    runner_version: str = "vec-add-gate/2"
    oracle_version: str = "fixed-f32-le-exact/1"
    fixture_sha256: str = digest(b"".join(INITIAL) + F32.pack(*EXPECTED))
    source_sha256: str | None = None
    descriptor_sha256: str | None = None
    cubin_sha256: str | None = None
    toolchain: str | None = None
    native_target: str | None = None
    device: Device | None = None
    result_hex: str | None = None
    result: tuple[float, ...] | None = None
    quarantined: bool = False
    stages: dict[str, Outcome] = field(default_factory=dict)

    @property
    def author_hardware_correct(self) -> bool:
        return self.backend == "hardware" and all(
            self.stages[name].status == "pass" for name in STAGES if name != "timing"
        )


Builder = Callable[[bytes, bytes], Built]


class MockGate:
    """Trusted in-process injection for offline tests, never hardware evidence."""

    def __init__(self, builder: Builder, driver: Driver):
        self.builder = builder
        self.driver = driver
        self.quarantined = False
        self.lock = Lock()

    def check(self, descriptor: bytes = DESCRIPTOR, source: bytes = AUTHOR_SOURCE) -> Receipt:
        with self.lock:
            receipt = run_check(
                self.builder,
                lambda: self.driver,
                descriptor,
                source,
                quarantined=self.quarantined,
            )
            self.quarantined = receipt.quarantined
        return receipt


def run_check(
    builder: Builder,
    make_driver: Callable[[], Driver],
    descriptor: bytes,
    source: bytes,
    *,
    quarantined: bool = False,
) -> Receipt:
    # Injected execution is mock unless the fixed author CLI owns the provenance.
    receipt = Receipt(quarantined=quarantined)
    receipt.stages = {name: Outcome("not_run", "unreached") for name in STAGES}
    receipt.stages["timing"] = Outcome("not_run", "author correctness only; no timing lane")
    if quarantined:
        for name in STAGES[:-1]:
            receipt.stages[name] = Outcome("not_run", "worker quarantined; no retry")
        return receipt

    driver = None
    timed_out = False
    stage = "schema"

    def mark_pass(next_stage: str) -> None:
        nonlocal stage
        receipt.stages[stage] = Outcome("pass")
        stage = next_stage

    try:
        manifest = parse_manifest(descriptor)
        receipt.descriptor_sha256 = digest(descriptor)
        mark_pass("admission")
        if type(source) is not bytes or source != AUTHOR_SOURCE:
            raise ValueError("only the exact reviewed author source is admitted")
        receipt.source_sha256 = digest(source)
        mark_pass("build")
        built = builder(source, descriptor)
        receipt.toolchain = built.toolchain
        mark_pass("architecture")
        image = built.image  # inspect and load this exact bytes object, no path re-open
        if (
            built.source_sha256 != receipt.source_sha256
            or built.descriptor_sha256 != receipt.descriptor_sha256
            or type(image) is not bytes
            or not 0 < len(image) <= MAX_CUBIN_BYTES
            or built.cubin_sha256 != digest(image)
        ):
            raise ValueError("trusted source/descriptor/cubin binding mismatch")
        receipt.cubin_sha256 = built.cubin_sha256
        receipt.native_target = inspect_native(image)
        mark_pass("device")
        driver = make_driver()
        receipt.device = driver.probe()
        if receipt.device.capability != CAPABILITY:
            raise ValueError("selected device must have compute capability 12.0")
        mark_pass("load")
        driver.load(image)
        mark_pass("entry")
        driver.entry(manifest.entry)
        mark_pass("prepare")
        a, b, out = (driver.allocate(initial) for initial in INITIAL)
        mark_pass("launch")
        driver.launch(manifest, (a + len(PREFIX), b + len(PREFIX), out + len(PREFIX), manifest.n))
        mark_pass("completion")
        driver.synchronize()
        mark_pass("d2h")
        readback = []
        for ptr, initial in zip((a, b, out), INITIAL, strict=True):
            raw = driver.read(ptr, len(initial))
            if type(raw) is not bytes or len(raw) != len(initial):
                raise ValueError("readback must contain exact allocation lengths")
            readback.append(raw)
        raw_a, raw_b, raw_out = readback
        start = len(PREFIX)
        end = start + F32.size
        result = raw_out[start:end]
        receipt.result_hex = result.hex()
        mark_pass("integrity")
        for role, raw in zip(("a", "b", "out"), readback, strict=True):
            if raw[:start] != PREFIX or raw[end:] != SUFFIX:
                raise ValueError(f"{role} guard changed")
        if (raw_a, raw_b) != INITIAL[:2]:
            raise ValueError("immutable input changed")
        mark_pass("numerics")
        if any(result[i : i + 4] == POISON for i in range(0, F32.size, 4)):
            raise ValueError("untouched output poison")
        if result != F32.pack(*EXPECTED):
            raise ValueError("result differs from exact reference f32 bytes")
        receipt.result = F32.unpack(result)
        mark_pass("cleanup")
    except Exception as exc:
        timed_out = isinstance(exc, TimeoutError)
        receipt.stages[stage] = Outcome("fail", str(exc)[:240] or type(exc).__name__)
        # Future real watchdogs must stop the worker from outside. Never retry or reset.
        receipt.quarantined = timed_out or stage in {
            "device",
            "load",
            "entry",
            "prepare",
            "launch",
            "completion",
            "d2h",
        }
        for name in STAGES[STAGES.index(stage) + 1 : -1]:
            receipt.stages[name] = Outcome("not_run", f"stopped at {stage}")
    finally:
        if driver is not None:
            if timed_out:
                receipt.stages["cleanup"] = Outcome("not_run", "watchdog; no more driver calls")
            else:
                try:
                    driver.close()
                    receipt.stages["cleanup"] = Outcome("pass")
                except Exception as exc:
                    receipt.stages["cleanup"] = Outcome(
                        "fail", str(exc)[:240] or type(exc).__name__
                    )
                    receipt.quarantined = True
    return receipt
