"""Closed KB30 device ABI and author-only correctness orchestration (no CUDA imports)."""

import hashlib
import json
import struct
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol
from uuid import uuid4

ABI = "kb30_f32_device_v1"
M, K, N, GROUPS = 1024, 8192, 8192, 16
GUARD = 256
MEMORY_CAP = 384 << 20
MAX_DESCRIPTOR = 4096
MAX_SOURCE = 128 << 10
MAX_CUBIN = 16 << 20
AUTHOR_DESCRIPTOR = (
    b'{"abi":"kb30_f32_device_v1","plan":"staged","geometry":["tile16x16","group256"]}'
)
BODIES_SHA256 = "dc8fdc401143fd8306e5a9a3793c6223efa04c0b9d5e4ded09f1670d660da94d"
ORIGINAL_SOURCE_SHA256 = "3e53d3484eb996260a88ed6b35feb4499ca81329ae05659d949807653e0d3446"
CONTROLS = (
    "positive",
    "interleaved_groups",
    "sample_variance",
    "missing_producer",
    "stale_producer",
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(data: object) -> bytes:
    return json.dumps(data, separators=(",", ":"), sort_keys=True, allow_nan=False).encode("utf-8")


def checked_product(*dims: int, cap: int = MEMORY_CAP) -> int:
    size = 1
    for dim in dims:
        if type(dim) is not int or dim <= 0 or size > cap // dim:
            raise ValueError("invalid dimension or size exceeds cap")
        size *= dim
    return size


@dataclass(frozen=True)
class Buffer:
    name: str
    shape: tuple[int, ...]
    kind: str  # input, FP32 Linear scratch, or output

    @property
    def size(self) -> int:
        return checked_product(*self.shape, 4)  # little-endian FP32


BUFFERS = (
    Buffer("X", (M, K), "input"),
    Buffer("W", (N, K), "input"),
    Buffer("bias", (N,), "input"),
    Buffer("gamma", (N,), "input"),
    Buffer("beta", (N,), "input"),
    Buffer("Z", (M, N), "scratch"),
    Buffer("Y", (M, N), "output"),
)
INPUTS = tuple(b for b in BUFFERS if b.kind == "input")
BY_NAME = {b.name: b for b in BUFFERS}


@dataclass(frozen=True)
class Slot:
    name: str
    roles: tuple[tuple[str, str], ...]
    presets: tuple[tuple[str, tuple[int, int, int], tuple[int, int, int]], ...]


GROUP_PRESETS = tuple((f"group{t}", (GROUPS, M, 1), (t, 1, 1)) for t in (128, 256))
PLANS = {
    "staged": (
        Slot(
            "kb_linear",
            (("X", "r"), ("W", "r"), ("bias", "r"), ("Z", "w")),
            (
                ("tile16x16", (N // 16, M // 16, 1), (16, 16, 1)),
                ("tile32x8", (N // 32, M // 8, 1), (32, 8, 1)),
            ),
        ),
        Slot("kb_finish", (("Z", "r"), ("gamma", "r"), ("beta", "r"), ("Y", "w")), GROUP_PRESETS),
    ),
    "fused": (
        Slot(
            "kb_forward",
            (
                ("X", "r"),
                ("W", "r"),
                ("bias", "r"),
                ("gamma", "r"),
                ("beta", "r"),
                ("Y", "w"),
            ),
            GROUP_PRESETS,
        ),
    ),
}


@dataclass(frozen=True)
class Launch:
    entry: str
    roles: tuple[tuple[str, str], ...]
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    shared_bytes: int = 0

    @property
    def param_layout(self) -> tuple[tuple[int, int], ...]:
        return tuple((8 * i, 8) for i in range(len(self.roles)))

    def arguments(self, allocations: dict[str, int]) -> tuple[int, ...]:
        return tuple(allocations[name] for name, _ in self.roles)

    def wrapper(self) -> bytes:
        sig = ", ".join(
            f"{'const ' if role == 'r' else ''}float* {name}" for name, role in self.roles
        )
        args = ", ".join(name for name, _ in self.roles)
        return (
            f'extern "C" __global__ void {self.entry}({sig}) {{\n'
            f"    {self.entry}_body({args});\n}}\n"
        ).encode("ascii")


@dataclass(frozen=True)
class Descriptor:
    plan: str
    geometry: tuple[str, ...]
    launches: tuple[Launch, ...]
    request: bytes

    @property
    def identity(self) -> str:
        return digest(canonical({"abi": ABI, "plan": self.plan, "geometry": self.geometry}))

    @property
    def buffers(self) -> tuple[Buffer, ...]:
        return tuple(b for b in BUFFERS if b.kind != "scratch" or self.plan == "staged")

    @property
    def allocation_size(self) -> int:
        size = sum(b.size + 2 * GUARD for b in self.buffers)
        if size > MEMORY_CAP:
            raise ValueError("guarded allocation cap exceeded")
        return size


def unique_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate descriptor field: {key}")
        result[key] = value
    return result


def parse_descriptor(payload: bytes) -> Descriptor:
    if type(payload) is not bytes or not 0 < len(payload) <= MAX_DESCRIPTOR:
        raise ValueError("expected bounded descriptor bytes")
    parsed = json.loads(payload.decode("utf-8"), object_pairs_hook=unique_keys)
    if type(parsed) is not dict or set(parsed) != {"abi", "plan", "geometry"}:
        raise ValueError("expected exactly abi, plan and geometry")
    abi, plan, geometry = parsed["abi"], parsed["plan"], parsed["geometry"]
    if abi != ABI or type(plan) is not str or plan not in PLANS:
        raise ValueError("unknown ABI or plan")
    slots = PLANS[plan]
    if type(geometry) is not list or len(geometry) != len(slots):
        raise ValueError("wrong geometry arity")
    launches = []
    for preset, slot in zip(geometry, slots, strict=True):
        resolved = next(
            ((grid, block) for name, grid, block in slot.presets if preset == name), None
        )
        if type(preset) is not str or resolved is None:
            raise ValueError(f"unsupported geometry for {slot.name}")
        launches.append(Launch(slot.name, slot.roles, *resolved))
    return Descriptor(plan, tuple(geometry), tuple(launches), payload)


def validate_fragment(source: bytes | None) -> bytes:
    if source is None:
        raise LookupError("source unavailable")
    if type(source) is not bytes or len(source) > MAX_SOURCE or b"\x00" in source:
        raise ValueError("expected bounded NUL-free UTF-8 source")
    text = source.decode("utf-8").strip()
    if not text:
        raise ValueError("empty source")
    if text.startswith("```") or text.startswith("~~~"):
        raise ValueError("fenced text is not device-body source")
    return source  # No unreviewed body admission; this only validates byte framing.


def author_fragment(control: str) -> bytes:
    if control not in CONTROLS:
        raise ValueError("unknown author control")
    raw = Path(__file__).with_name("device_bodies.cu").read_bytes()
    if digest(raw) != BODIES_SHA256:
        raise ValueError("reviewed author body template hash mismatch")
    constants, linear, finish = raw.split(b"// --- LINEAR / FINISH BOUNDARY ---\n")
    if control in ("missing_producer", "stale_producer"):
        linear = b"__device__ __forceinline__ void kb_linear_body(const float*, const float*, const float*, float*) {}\n"
    variant = {"interleaved_groups": b"true, false", "sample_variance": b"false, true"}
    return validate_fragment(
        constants + linear + finish.replace(b"/*VARIANT*/", variant.get(control, b"false, false"))
    )


def author_translation_unit(desc: Descriptor, source: bytes) -> tuple[bytes, bytes]:
    if desc.plan != "staged":
        raise NotImplementedError("fused is reserved; no implementation")
    wrappers = b"\n".join(slot.wrapper() for slot in desc.launches)
    return b"#include <cuda_runtime.h>\n\n" + source + b"\n" + wrappers, wrappers


ELF = struct.Struct("<16sHHIQQQIHHHHHH")
SECTION = struct.Struct("<IIQQQQIIQQ")


def inspect_native(image: bytes, entries: tuple[str, ...]) -> None:
    """Check direct NVCC output against the proof's native ELF format (not hostile ELF validation)."""
    if type(image) is not bytes or not ELF.size <= len(image) <= MAX_CUBIN:
        raise ValueError("expected bounded immutable cubin bytes")
    (
        ident,
        kind,
        machine,
        version,
        _,
        phstart,
        start,
        flags,
        ehsize,
        phsize,
        phnum,
        shsize,
        count,
        names,
    ) = ELF.unpack_from(image)
    if ident[:9] != b"\x7fELF\x02\x01\x01\x41\x08" or (kind, machine, version, ehsize) != (
        2,
        190,
        1,
        ELF.size,
    ):
        raise ValueError("expected direct native CUDA ELF")
    if (flags >> 8) & 0xFF != 120 or flags & 0x8:
        raise ValueError("native ELF target must be base sm_120")
    if phnum and (phsize != 56 or phstart < ELF.size or phstart + phsize * phnum > len(image)):
        raise ValueError("invalid cubin program table")
    if (
        shsize != SECTION.size
        or not 0 < names < count <= 256
        or not ELF.size <= start <= len(image) - shsize * count
    ):
        raise ValueError("invalid cubin section table")
    sections = [SECTION.unpack_from(image, start + shsize * i) for i in range(count)]
    _, table_kind, _, _, offset, size, *_ = sections[names]
    if table_kind != 3 or offset + size > len(image):
        raise ValueError("invalid cubin section names")
    strings = image[offset : offset + size]
    executable = []
    for index, section_type, attrs, _, offset, size, *_ in sections:
        end = strings.find(b"\x00", index)
        if end < 0 or (section_type != 8 and offset + size > len(image)):
            raise ValueError("invalid cubin section bounds")
        name = strings[index:end]
        if b"ptx" in name.lower() or b"fatbin" in name.lower():
            raise ValueError("PTX/fatbin section not allowed")
        if attrs & 4:
            if section_type != 1 or not size or offset < ELF.size:
                raise ValueError("invalid executable text")
            executable.append(name)
    if sorted(executable) != sorted(f".text.{e}".encode() for e in entries):
        raise ValueError("expected only the manifest's native entry text sections")


@dataclass(frozen=True)
class Built:
    image: bytes
    tu_sha256: str
    image_sha256: str
    toolchain: dict


class Driver(Protocol):
    def probe(self) -> dict: ...
    def load(self, image: bytes) -> object: ...
    def entry(self, module: object, launch: Launch) -> tuple[object, dict]: ...
    def allocate(self, size: int) -> int: ...
    def write(self, address: int, data: bytes) -> None: ...
    def poison(self, address: int, size: int, word: int) -> None: ...
    def prepared(self) -> None: ...
    def launch(self, function: object, launch: Launch, args: tuple[int, ...]) -> None: ...
    def synchronize(self) -> None: ...
    def read(self, address: int, size: int) -> bytes: ...
    def close(self) -> None: ...


GUARD_LOW = b"\x9d" * GUARD
GUARD_HIGH = b"\xa7" * GUARD
STAGES = ("prepare", "launch[0]", "launch[1]", "completion", "d2h", "integrity", "numerics")


def check_range(base: int, size: int, previous: list[tuple[int, int]]) -> None:
    end = base + size
    if (
        not 0 < base < end <= 2**64
        or base % GUARD
        or any(base < hi and lo < end for lo, hi in previous)
    ):
        raise ValueError("unaligned, overflowing or overlapping guarded allocation")


def run_trusted(
    builder: Callable[[str, bytes], Built],
    driver_factory: Callable[[], Driver],
    control: str,
    inputs: dict[str, bytes],
    next_x: bytes | None = None,
    *,
    descriptor: bytes = AUTHOR_DESCRIPTOR,
) -> tuple[dict, list[tuple[bytes, bytes]]]:
    """Fixed author bodies only. Results are raw bytes; no reference goes to the driver."""
    driver = None
    outputs = []
    receipt = {
        "run_id": str(uuid4()),
        "runner": "kb30-device-author/1",
        "abi": ABI,
        "role": "trusted_author_control",
        "backend": "unqualified_adapter",
        "control": control,
        "quarantined": False,
        "outcome": "not_qualified",
        "forwards": [],
        "artifacts": {},
    }
    stages = receipt["stages"] = {
        key: {"status": "not_run", "reason": "prerequisite not reached"}
        for key in (
            "schema",
            "admission",
            "build",
            "architecture",
            "device",
            "load",
            "entry",
            "cleanup",
        )
    }
    stages["timing"] = {"status": "not_run", "reason": "author correctness only; no timing lane"}
    current = "schema"
    status = stages

    def passed() -> None:
        status[current] = {"status": "pass"}

    try:
        desc = parse_descriptor(descriptor)
        receipt.update(
            plan=desc.plan,
            descriptor_sha256=desc.identity,
            request_sha256=digest(desc.request),
            launches=[asdict(slot) for slot in desc.launches],
            scratch_bytes=BY_NAME["Z"].size if desc.plan == "staged" else 0,
            guarded_allocation_bytes=desc.allocation_size,
        )
        passed()
        current = "admission"
        if desc.plan == "fused":
            raise NotImplementedError("fused is reserved; no implementation or fallback")
        if desc.geometry != ("tile16x16", "group256") or control not in CONTROLS:
            raise NotImplementedError("no reviewed author body for this geometry/control")
        checks = {
            "positive": ("positive", "positive"),
            "stale_producer": ("positive", "stale_producer"),
        }.get(control, (control,))
        if len(checks) != 1 + (next_x is not None):
            raise ValueError("incorrect number of X snapshots for this control")
        if set(inputs) != {b.name for b in INPUTS} or any(
            type(inputs[b.name]) is not bytes or len(inputs[b.name]) != b.size for b in INPUTS
        ):
            raise ValueError("incomplete or wrong-sized immutable FP32 input snapshots")
        initial_hashes = {name: digest(data) for name, data in inputs.items()}
        if next_x is not None and (
            type(next_x) is not bytes
            or len(next_x) != BY_NAME["X"].size
            or digest(next_x) == initial_hashes["X"]
        ):
            raise ValueError("second X must be a different full-size immutable snapshot")
        receipt["manifest_sha256"] = digest(
            canonical(
                {"buffers": [asdict(b) for b in desc.buffers], "launches": receipt["launches"]}
            )
        )
        receipt["author_predecessor_sha256"] = ORIGINAL_SOURCE_SHA256
        receipt["fixture_fp32_sha256"] = [initial_hashes]
        if next_x is not None:
            receipt["fixture_fp32_sha256"].append({**initial_hashes, "X": digest(next_x)})
        variants = {}
        for variant in dict.fromkeys(checks):
            source = author_fragment(variant)
            tu, wrappers = author_translation_unit(desc, source)
            variants[variant] = tu
            receipt["artifacts"][variant] = dict(
                source_sha256=digest(source), wrappers_sha256=digest(wrappers), tu_sha256=digest(tu)
            )
        passed()

        current = "build"
        built = {variant: builder(variant, tu) for variant, tu in variants.items()}
        passed()
        current = "architecture"
        for variant, artifact in built.items():
            meta = receipt["artifacts"][variant]
            if (
                artifact.tu_sha256 != meta["tu_sha256"]
                or digest(artifact.image) != artifact.image_sha256
            ):
                raise ValueError("build digest mismatch")
            inspect_native(artifact.image, tuple(slot.entry for slot in desc.launches))
            meta.update(
                cubin_sha256=artifact.image_sha256, target="sm_120", toolchain=artifact.toolchain
            )
        passed()
        current = "device"
        driver = driver_factory()
        device = receipt["device"] = driver.probe()
        if tuple(device["capability"]) != (12, 0):
            raise ValueError("requires compute capability 12.0")
        passed()
        current = "load"
        modules = {variant: driver.load(artifact.image) for variant, artifact in built.items()}
        passed()
        current = "entry"
        entries = {}
        for variant, module in modules.items():
            resolved = [driver.entry(module, slot) for slot in desc.launches]
            entries[variant] = [fn for fn, _ in resolved]
            receipt["artifacts"][variant]["function_resources"] = [meta for _, meta in resolved]
        passed()
        addresses: dict[str, int] = {}
        ranges: list[tuple[int, int]] = []
        previous_z = None
        for i, variant in enumerate(checks):
            stage = {
                key: {"status": "not_run", "reason": "prerequisite not reached"} for key in STAGES
            }
            row = {
                "control": variant,
                "stages": stage,
                "reuses_allocations": i > 0,
                "retains_previous_z": variant == "stale_producer",
            }
            receipt["forwards"].append(row)
            status = stage
            current = "prepare"
            if i == 0:
                for b in desc.buffers:
                    length = b.size + 2 * GUARD
                    base = driver.allocate(length)
                    check_range(base, length, ranges)
                    ranges.append((base, base + length))
                    addresses[b.name] = base + GUARD
                    driver.write(base, GUARD_LOW)
                    driver.write(base + GUARD + b.size, GUARD_HIGH)
                for name, initial in inputs.items():
                    driver.write(addresses[name], initial)
            else:
                assert next_x is not None
                driver.write(addresses["X"], next_x)
            if variant != "stale_producer":
                driver.poison(addresses["Z"], BY_NAME["Z"].size, 0x7FC10000 + i)
            driver.poison(addresses["Y"], BY_NAME["Y"].size, 0x7FC20000 + i)
            driver.prepared()
            passed()
            for slot_idx, (slot, function) in enumerate(
                zip(desc.launches, entries[variant], strict=True)
            ):
                current = f"launch[{slot_idx}]"
                driver.launch(function, slot, slot.arguments(addresses))
                passed()
            current = "completion"
            driver.synchronize()
            passed()
            current = "d2h"
            seen = {}
            guards = []
            for b in desc.buffers:
                addr = addresses[b.name]
                data = driver.read(addr, b.size)
                low = driver.read(addr - GUARD, GUARD)
                high = driver.read(addr + b.size, GUARD)
                if (len(data), len(low), len(high)) != (b.size, GUARD, GUARD):
                    raise ValueError(f"short readback: {b.name}")
                guards.append((b.name, low, high))
                seen[b.name] = digest(data) if b.kind == "input" else data
            passed()
            current = "integrity"
            for name, low, high in guards:
                if low != GUARD_LOW or high != GUARD_HIGH:
                    raise ValueError(f"guard mismatch: {name}")
            for name, expected in receipt["fixture_fp32_sha256"][i].items():
                if seen[name] != expected:
                    raise ValueError(f"immutable bytes changed: {name}")
            z, y = seen["Z"], seen["Y"]
            row.update(z_sha256=digest(z), y_sha256=digest(y))
            if variant == "stale_producer" and row["z_sha256"] != previous_z:
                raise ValueError("stale control unexpectedly changed prior Z")
            previous_z = row["z_sha256"]
            passed()
            outputs.append((z, y))
    except NotImplementedError as exc:
        status[current] = {"status": "not_run", "reason": str(exc)}
        receipt["outcome"] = "unsupported"
    except Exception as exc:
        status[current] = {"status": "fail", "reason": f"{type(exc).__name__}: {exc}"}
        receipt["quarantined"] = driver is not None or current == "device"
    finally:
        if driver is not None:
            try:
                driver.close()
                stages["cleanup"] = {"status": "pass"}
            except Exception as exc:
                stages["cleanup"] = {"status": "fail", "reason": str(exc)}
                receipt["quarantined"] = True
    return receipt, outputs


def compare_forwards(
    receipt: dict,
    outputs: list[tuple[bytes, bytes]],
    references: list[object],
    comparator: Callable,
) -> bool:
    """Independent supervisor comparison after the driver has closed."""
    if (
        receipt["quarantined"]
        or receipt["stages"]["cleanup"]["status"] != "pass"
        or len(outputs) != len(references)
        or not outputs
    ):
        return False
    expected_behaviour = True
    for row, (scratch, output), expected in zip(
        receipt["forwards"], outputs, references, strict=True
    ):
        numerics = row["stages"]["numerics"]
        try:
            report = row["comparison"] = comparator(scratch, output, expected)
            control = row["control"]
            if control == "positive":
                observed = report["passed"] and report["scratch_finite"]
            elif control == "missing_producer":
                observed = (
                    not report["passed"] and report["scratch_all_nan"] and report["output_all_nan"]
                )
            else:
                observed = (
                    not report["passed"]
                    and report["scratch_finite"]
                    and report["output_finite"]
                    and report["interior_mismatched"] > 0
                )
            numerics.update(
                status="pass" if report["passed"] and report["scratch_finite"] else "fail",
                reason="" if report["passed"] else "independent full-output comparison rejected",
            )
            row["expected_author_behaviour"] = bool(observed)
        except Exception as exc:
            numerics.update(status="fail", reason=f"invalid oracle comparison: {exc}")
            observed = False
        expected_behaviour &= bool(observed)
    receipt["outcome"] = (
        "expected_author_behaviour" if expected_behaviour else "unexpected_author_behaviour"
    )
    return expected_behaviour
