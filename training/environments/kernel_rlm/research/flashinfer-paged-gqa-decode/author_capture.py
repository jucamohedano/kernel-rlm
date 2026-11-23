"""One-shot direct GQA author capture; only run in a parent-approved test window."""

import argparse
import ctypes as c
import json
import os
import re
import struct
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import UUID, uuid4

from . import author_cases as cases
from . import device_abi as abi

TARGET = "sm_120"
NVCC_VERSION = "13.3.73"
MAX_CUBIN = 16 << 20
FLAGS = ("--cubin", "--generate-code=arch=compute_120,code=sm_120", "--std=c++17", "-O3")
REVIEWED_BODY = "fafe46aa6bb846f5622e338c89b028bacf6814c811c8797339f8adc2681b43fb"
# Separate fixed translations of the author body; none change addresses outside the fixture.
CONTROL = {
    "wrong-head": (b"h / 8", b"h % 2", frozenset(("O", "L"))),
    "wrong-lse": (b"m + log2f(mass)", b"(m + log2f(mass)) / log2e", frozenset(("L",))),
    "wrong-tail": (b"j + 1 == end ? T[b] : 64", b"64", frozenset(("O", "L"))),
}
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
    "cleanup",
    "numerics",
    "timing",
)
POINTER, DEVICE_PTR, INT, UINT, SIZE = c.c_void_p, c.c_uint64, c.c_int, c.c_uint, c.c_size_t
ELF = struct.Struct("<16sHHIQQQIHHHHHH")
SECTION = struct.Struct("<IIQQQQIIQQ")


def compose(variant: str) -> tuple[bytes, bytes]:
    with Path(__file__).with_name("author_direct.cuh").open("rb") as fp:
        body = fp.read((128 << 10) + 1)
    if abi.digest(body) != REVIEWED_BODY:
        raise ValueError("reviewed author fragment digest mismatch")
    if variant != "positive":
        before, after, _ = CONTROL[variant]
        if body.count(before) != 1:
            raise ValueError("reviewed control substitution must be unique")
        body = body.replace(before, after)
    return body, abi.HEADER + body + abi.WRAPPER


def inspect_native(image: bytes) -> None:
    """Proof-branch CUDA ELF checks, pinned to this task's sole entry."""
    if type(image) is not bytes or not ELF.size <= len(image) <= MAX_CUBIN:
        raise ValueError("expected a bounded native cubin as immutable bytes")
    ident, kind, machine, version, _, _, start, flags, ehsize, _, _, shsize, count, names = (
        ELF.unpack_from(image)
    )
    if (
        ident[:9] != b"\x7fELF\x02\x01\x01\x41\x08"
        or (kind, machine, version, ehsize) != (2, 190, 1, ELF.size)
        or ((flags >> 8) & 0xFF) != 120
        or flags & 0x8
    ):
        raise ValueError("expected native base sm_120 ELF, not PTX, fatbin or host ELF")
    if not (
        shsize == SECTION.size
        and 0 < names < count <= 256
        and ELF.size <= start <= len(image) - shsize * count
    ):
        raise ValueError("invalid cubin section table")
    sections = tuple(SECTION.unpack_from(image, start + shsize * i) for i in range(count))
    _, table_kind, _, _, offset, size, *_ = sections[names]
    if table_kind != 3 or offset + size > len(image):
        raise ValueError("invalid cubin string table")
    strings = image[offset : offset + size]
    executable = []
    for index, section_type, attrs, _, offset, size, *_ in sections:
        end = strings.find(b"\x00", index)
        if end < 0 or (section_type != 8 and offset + size > len(image)):
            raise ValueError("invalid cubin section bounds")
        name = strings[index:end]
        if b"ptx" in name.lower() or b"fatbin" in name.lower():
            raise ValueError("embedded PTX/fatbin section not allowed")
        if attrs & 4 and size:
            if section_type != 1:
                raise ValueError("unexpected executable section type")
            executable.append(name)
    if executable != [f".text.{abi.ENTRY}".encode()]:
        raise ValueError("expected exactly the fixed native entry's text section")


def absolute_executable(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or not path.is_file() or not os.access(path, os.X_OK):
        raise argparse.ArgumentTypeError("expected an absolute trusted executable path")
    return path.resolve()


def build(nvcc: Path, ccbin: Path, translation: bytes) -> tuple[bytes, dict]:
    with TemporaryDirectory(prefix="fi-direct-author-", dir="/var/tmp") as directory:
        work = Path(directory)
        env = {
            "PATH": f"{nvcc.parent}:/usr/bin:/bin",
            "HOME": directory,
            "TMPDIR": directory,
            "LC_ALL": "C",
        }
        options = {"cwd": work, "env": env, "stdin": subprocess.DEVNULL}
        try:
            version = subprocess.check_output([nvcc, "--version"], timeout=10, **options)
            if not re.search(rb"\bV" + re.escape(NVCC_VERSION.encode()) + rb"\b", version):
                raise ValueError(f"requires NVCC V{NVCC_VERSION}")
            (work / "author.cu").write_bytes(translation)
            with (work / "build.log").open("w+b") as log:
                argv = [nvcc, *FLAGS, "--compiler-bindir", ccbin, "author.cu", "-o", "author.cubin"]
                result = subprocess.run(argv, stdout=log, stderr=log, timeout=40, **options)
                if result.returncode:
                    log.seek(0)
                    raise ValueError(
                        f"nvcc exit {result.returncode}: " + log.read(1200).decode(errors="replace")
                    )
            with (work / "author.cubin").open("rb") as artifact:
                image = artifact.read(MAX_CUBIN + 1)
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError("author build exceeded watchdog") from exc
    return image, {
        "nvcc": str(nvcc),
        "nvcc_version": NVCC_VERSION,
        "ccbin": str(ccbin),
        "flags": FLAGS,
    }


class AuthorDriver:
    def __init__(self, expected_uuid: UUID):
        self.expected_uuid = expected_uuid
        self.lib = c.CDLL("libcuda.so.1")
        self.ordinal, self.context, self.stream, self.module, self.function = (
            INT(),
            POINTER(),
            POINTER(),
            POINTER(),
            POINTER(),
        )

    def status(self, name: str, types: tuple, *args) -> int:
        api = getattr(self.lib, name)
        api.argtypes, api.restype = types, INT
        return api(*args)

    def call(self, name: str, types: tuple, *args) -> None:
        status = self.status(name, types, *args)
        if status:
            raise RuntimeError(f"{name}: CUDA status {status}")

    def query(self, name: str, attr: int, target, target_type) -> int:
        value = INT()
        self.call(name, (POINTER, INT, target_type), c.byref(value), attr, target)
        return value.value

    def device_attr(self, attr: int) -> int:
        return self.query("cuDeviceGetAttribute", attr, self.ordinal, INT)

    def function_attr(self, attr: int) -> int:
        return self.query("cuFuncGetAttribute", attr, self.function, POINTER)

    def probe(self) -> dict:
        self.call("cuInit", (UINT,), 0)
        self.call("cuDeviceGet", (POINTER, INT), c.byref(self.ordinal), 0)
        name, uuid_bytes, version = c.create_string_buffer(128), (c.c_ubyte * 16)(), INT()
        self.call("cuDeviceGetName", (POINTER, INT, INT), name, len(name), self.ordinal)
        self.call("cuDeviceGetUuid_v2", (POINTER, INT), uuid_bytes, self.ordinal)
        self.call("cuDriverGetVersion", (POINTER,), c.byref(version))
        observed = UUID(bytes=bytes(uuid_bytes))
        gpu = name.value.decode("ascii")
        capability = (self.device_attr(75), self.device_attr(76))
        if (
            observed != self.expected_uuid
            or capability != (12, 0)
            or "RTX PRO 6000 Blackwell" not in gpu
        ):
            raise ValueError("ordinal 0 must be the assigned RTX PRO 6000 Blackwell (12,0)")
        return {
            "name": gpu,
            "uuid": f"GPU-{observed}",
            "capability": capability,
            "driver_api": version.value,
        }

    def load(self, image: bytes) -> None:
        self.call("cuCtxCreate_v2", (POINTER, UINT, INT), c.byref(self.context), 0, self.ordinal)
        self.call("cuStreamCreate", (POINTER, UINT), c.byref(self.stream), 0)
        self.call(
            "cuModuleLoadData",
            (POINTER, POINTER),
            c.byref(self.module),
            c.create_string_buffer(image, len(image)),
        )

    def entry(self, launch: abi.Launch) -> dict:
        self.call(
            "cuModuleGetFunction",
            (POINTER, POINTER, c.c_char_p),
            c.byref(self.function),
            self.module,
            abi.ENTRY.encode(),
        )
        types = (POINTER, SIZE, POINTER, POINTER)
        for i, expected in enumerate(abi.PARAM_LAYOUT):
            offset, size = SIZE(), SIZE()
            self.call("cuFuncGetParamInfo", types, self.function, i, c.byref(offset), c.byref(size))
            if (offset.value, size.value) != expected:
                raise ValueError(f"wrong parameter byte layout at {i}")
        offset, size = SIZE(), SIZE()
        status = self.status(
            "cuFuncGetParamInfo",
            types,
            self.function,
            len(abi.PARAM_LAYOUT),
            c.byref(offset),
            c.byref(size),
        )
        if status != 1:  # CUDA_ERROR_INVALID_VALUE for first out-of-range index
            raise ValueError(f"could not establish exact parameter count (CUDA status {status})")
        resources = {
            label: self.function_attr(attr)
            for label, attr in (
                ("max_threads", 0),
                ("shared_bytes", 1),
                ("local_bytes", 3),
                ("registers", 4),
                ("binary_version", 6),
            )
        }
        limits = tuple(self.device_attr(attr) for attr in range(1, 9))
        if (
            any(x > cap for x, cap in zip((*launch.block, *launch.grid), limits[1:7], strict=True))
            or launch.block[0] > min(limits[0], resources["max_threads"])
            or not 0 <= resources["shared_bytes"] <= min(limits[7], 48 << 10)
            or resources["binary_version"] != 120
        ):
            raise ValueError("function resource, launch geometry or binary version mismatch")
        return resources

    def allocate(self, raw: bytes) -> int:
        ptr = DEVICE_PTR()
        self.call("cuMemAlloc_v2", (POINTER, SIZE), c.byref(ptr), len(raw))
        self.call(
            "cuMemcpyHtoD_v2",
            (DEVICE_PTR, POINTER, SIZE),
            ptr,
            c.create_string_buffer(raw, len(raw)),
            len(raw),
        )
        return ptr.value

    def launch(self, spec: abi.Launch, values: tuple) -> None:
        params = (POINTER * len(values))(*(c.addressof(arg) for arg in values))
        self.call(
            "cuLaunchKernel",
            (POINTER, UINT, UINT, UINT, UINT, UINT, UINT, UINT, POINTER, POINTER, POINTER),
            self.function,
            *spec.grid,
            *spec.block,
            spec.shared_bytes,
            self.stream,
            params,
            None,
        )

    def synchronize(self) -> None:
        self.call("cuStreamSynchronize", (POINTER,), self.stream)

    def read(self, address: int, size: int) -> bytes:
        dest = c.create_string_buffer(size)
        self.call("cuMemcpyDtoH_v2", (POINTER, DEVICE_PTR, SIZE), dest, address, size)
        return dest.raw

    def close(self) -> None:
        if self.context:
            self.call("cuCtxDestroy_v2", (POINTER,), self.context)
            self.context = POINTER()


def capture(prepared: abi.Prepared, driver, advance) -> tuple[bytes, bytes]:
    """Only fixture operands enter this path; the CPU oracle runs after teardown."""
    initial = {role: abi.PREFIX + raw + abi.SUFFIX for role, raw in prepared.payloads.items()}
    bases = {}
    for role, raw in initial.items():
        base = driver.allocate(raw)
        if (
            not 0 < base <= (1 << 64) - len(raw)
            or base % abi.ALIGN
            or any(
                max(base, other) < min(base + len(raw), other + len(initial[key]))
                for key, other in bases.items()
            )
        ):
            raise ValueError("driver allocations must be aligned and disjoint")
        bases[role] = base
    args = prepared.arguments({role: base + abi.ALIGN for role, base in bases.items()})
    advance("launch")
    driver.launch(prepared.launch, args)
    advance("completion")
    driver.synchronize()
    advance("d2h")
    raw = {role: driver.read(base, len(initial[role])) for role, base in bases.items()}
    if any(
        type(data) is not bytes or len(data) != len(initial[role]) for role, data in raw.items()
    ):
        raise ValueError("readback must match every allocation length")
    advance("integrity")
    for role, data in raw.items():
        if data[: abi.ALIGN] != abi.PREFIX or data[-abi.ALIGN :] != abi.SUFFIX:
            raise ValueError(f"{role}: changed guard")
        if role not in ("O", "L") and data != initial[role]:
            raise ValueError(f"{role}: immutable payload changed")
    return tuple(raw[role][abi.ALIGN : -abi.ALIGN] for role in ("O", "L"))


def run(case: str, variant: str, builder, make_driver) -> dict:
    """Injected builder/driver results are mock by default; only main labels hardware."""
    receipt = {
        "backend": "mock",
        "role": "author_control",
        "run_id": str(uuid4()),
        "abi": abi.ABI,
        "plan": "direct",
        "runner": "fi-direct-author/1",
        "oracle": cases.ORACLE,
        "case": case,
        "variant": variant,
        "quarantined": False,
        "expected_outcome_met": False,
        "stages": dict.fromkeys(STAGES, "not_run"),
        "timing_reason": "author correctness only; no timing lane",
    }
    states = receipt["stages"]
    stage, driver, timed_out = "schema", None, False

    def advance(next_stage: str) -> None:
        nonlocal stage
        states[stage] = "pass"
        stage = next_stage

    try:
        descriptor = abi.parse_descriptor(abi.DIRECT.encode())
        prepared = abi.prepare(cases.fixture(case))
        receipt.update(
            descriptor_sha256=abi.digest(descriptor.encode()),
            fixture_sha256=prepared.fixture_sha256,
            launch=asdict(prepared.launch),
            token_counts=prepared.token_counts(),
            fixture_dims={k: getattr(prepared.fixture, k) for k in "BPM"},
            scale_fp32=prepared.scale,
            explicit_bytes=sum(len(raw) + 2 * abi.ALIGN for raw in prepared.payloads.values()),
        )
        advance("admission")
        if variant != "positive" and (variant not in CONTROL or case != "analytic"):
            raise ValueError("negative author controls are analytic-only")
        body, translation = compose(variant)
        submission = abi.decode_submission(body, descriptor.encode())
        receipt.update(
            source_sha256=abi.digest(submission.source),
            request_descriptor_sha256=abi.digest(submission.descriptor_bytes),
            translation_sha256=abi.digest(translation),
        )
        advance("build")
        image, receipt["toolchain"] = builder(translation)
        advance("architecture")
        inspect_native(image)
        receipt.update(cubin_sha256=abi.digest(image), target=TARGET)
        advance("device")
        driver = make_driver()
        receipt["device"] = driver.probe()
        advance("load")
        driver.load(image)  # same immutable bytes that were inspected and hashed
        advance("entry")
        receipt["resources"] = driver.entry(prepared.launch)
        advance("prepare")
        outputs = capture(prepared, driver, advance)
        advance("cleanup")
        owner, driver = driver, None  # exactly one close, even when close errors
        owner.close()
        advance("numerics")
        receipt["raw_output_le_hex"] = {
            role: raw.hex() for role, raw in zip(("O", "L"), outputs, strict=True)
        }
        mismatches = cases.compare(prepared, outputs)
        receipt["mismatches"] = mismatches
        failed = {role for role, n in mismatches.items() if n}
        expected = set() if variant == "positive" else CONTROL[variant][2]
        states[stage] = "fail" if failed else "pass"
        receipt["expected_outcome_met"] = failed == expected
        receipt["status"] = (
            "positive_match"
            if not failed and not expected
            else "expected_numeric_rejection"
            if failed == expected
            else "unexpected_numerics"
        )
    except Exception as exc:
        timed_out = isinstance(exc, TimeoutError)
        states[stage] = "fail"
        receipt.update(status="stopped", stopped_at=stage, error=str(exc)[:1400])
        receipt["quarantined"] = timed_out or stage in STAGES[4:13]
    finally:
        if driver is not None and not timed_out:
            try:
                driver.close()
                states["cleanup"] = "pass"
            except Exception as exc:
                states["cleanup"] = "fail"
                receipt.update(quarantined=True, cleanup_error=str(exc)[:500])
    receipt["not_run_reasons"] = {
        name: receipt["timing_reason"]
        if name == "timing"
        else "watchdog; no more driver calls"
        if timed_out and name == "cleanup"
        else f"stopped at {receipt.get('stopped_at', 'unknown')}"
        for name, status in states.items()
        if status == "not_run"
    }
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--approved-author-smoke", required=True, action="store_true")
    parser.add_argument("--nvcc", required=True, type=absolute_executable)
    parser.add_argument("--ccbin", required=True, type=absolute_executable)
    parser.add_argument("--device-uuid", required=True, type=lambda x: UUID(x.removeprefix("GPU-")))
    parser.add_argument("--case", required=True, choices=cases.CASES)
    parser.add_argument("--variant", default="positive", choices=("positive", *CONTROL))
    args = parser.parse_args()
    if sys.platform != "linux" or sys.byteorder != "little" or c.sizeof(POINTER) != 8:
        parser.error("requires Linux with a 64-bit little-endian host ABI")
    os.environ.update(CUDA_CACHE_DISABLE="1", CUDA_DISABLE_PTX_JIT="1")
    receipt = run(
        args.case,
        args.variant,
        lambda src: build(args.nvcc, args.ccbin, src),
        lambda: AuthorDriver(args.device_uuid),
    )
    receipt["backend"] = "hardware"
    print(json.dumps(receipt, allow_nan=False, sort_keys=True), flush=True)
    return 0 if receipt["expected_outcome_met"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
