"""One-shot trusted author control. Only run in an explicitly approved test window."""

import argparse
import ctypes as c
import json
import os
import re
import subprocess
from dataclasses import asdict, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import UUID

from .gate import (
    AUTHOR_SOURCE,
    DESCRIPTOR,
    MAX_CUBIN_BYTES,
    Built,
    Device,
    Manifest,
    digest,
    run_check,
)

NVCC_VERSION = "13.3.73"
NATIVE_FLAGS = ("--cubin", "--generate-code", "arch=compute_120,code=sm_120")


def absolute_executable(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or not path.is_file() or not os.access(path, os.X_OK):
        raise argparse.ArgumentTypeError("expected an absolute trusted executable path")
    return path


def build_author(nvcc: Path, ccbin: Path, source: bytes, descriptor: bytes) -> Built:
    if source != AUTHOR_SOURCE or descriptor != DESCRIPTOR:
        raise ValueError("the hardware smoke builds only the fixed author source/descriptor")
    with TemporaryDirectory(prefix="blackwell-author-", dir="/var/tmp") as scratch:
        work = Path(scratch)
        env = {
            "PATH": f"{nvcc.parent}:/usr/bin:/bin",
            "HOME": scratch,
            "TMPDIR": scratch,
            "LC_ALL": "C",
        }
        with (work / "build.log").open("wb") as log:
            try:
                version = subprocess.check_output(
                    [nvcc, "--version"],
                    cwd=work,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stderr=log,
                    timeout=10,
                )
                if re.search(rb"\bV" + re.escape(NVCC_VERSION.encode()) + rb"\b", version) is None:
                    raise ValueError(f"requires the preflight NVCC V{NVCC_VERSION}")
                (work / "author.cu").write_bytes(source)
                subprocess.run(
                    [
                        nvcc,
                        *NATIVE_FLAGS,
                        "--compiler-bindir",
                        ccbin,
                        "author.cu",
                        "-o",
                        "author.cubin",
                    ],
                    cwd=work,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=log,
                    timeout=40,
                    check=True,
                )
            except subprocess.TimeoutExpired as exc:
                raise TimeoutError("author build exceeded watchdog") from exc
        with (work / "author.cubin").open("rb") as artifact:
            image = artifact.read(MAX_CUBIN_BYTES + 1)
        if not 0 < len(image) <= MAX_CUBIN_BYTES:
            raise ValueError("author cubin exceeded size limit or was empty")
    return Built(
        image,
        digest(source),
        digest(descriptor),
        digest(image),
        f"NVCC {NVCC_VERSION} ({nvcc}); ccbin={ccbin}; {' '.join(NATIVE_FLAGS)}",
    )


PTR = c.c_void_p
DEVICE_PTR = c.c_uint64
INT = c.c_int
UINT = c.c_uint
SIZE = c.c_size_t


class AuthorDriver:
    """Linux CUDA Driver API, with resources owned by a single fresh private context."""

    def __init__(self, expected_uuid: UUID):
        self.expected_uuid = expected_uuid
        self.lib = c.CDLL("libcuda.so.1")
        self.ordinal = INT()
        self.context = PTR()
        self.stream = PTR()
        self.module = PTR()
        self.function = PTR()

    def call(self, name: str, types: tuple, *args) -> None:
        func = getattr(self.lib, name)
        func.argtypes = types
        func.restype = INT
        status = func(*args)
        if status:
            raise RuntimeError(f"{name}: CUDA status {status}")

    def probe(self) -> Device:
        self.call("cuInit", (UINT,), 0)
        self.call("cuDeviceGet", (PTR, INT), c.byref(self.ordinal), 0)
        name, device_uuid, version = c.create_string_buffer(128), (c.c_ubyte * 16)(), INT()
        self.call("cuDeviceGetName", (PTR, INT, INT), name, len(name), self.ordinal)
        self.call("cuDeviceGetUuid_v2", (PTR, INT), device_uuid, self.ordinal)
        gpu_uuid = UUID(bytes=bytes(device_uuid))
        if gpu_uuid != self.expected_uuid:
            raise ValueError(f"GPU ordinal 0 UUID mismatch (found GPU-{gpu_uuid})")
        capability = []
        for attr in (75, 76):  # CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_{MAJOR,MINOR}
            number = INT()
            self.call("cuDeviceGetAttribute", (PTR, INT, INT), c.byref(number), attr, self.ordinal)
            capability.append(number.value)
        self.call("cuDriverGetVersion", (PTR,), c.byref(version))
        return Device(
            name.value.decode("ascii"),
            (capability[0], capability[1]),
            version.value,
            f"GPU-{gpu_uuid}",
        )

    def load(self, image: bytes) -> None:
        # Gate checked the device capability and ELF. No link or JIT fallback.
        self.call("cuCtxCreate_v2", (PTR, UINT, INT), c.byref(self.context), 0, self.ordinal)
        self.call("cuStreamCreate", (PTR, UINT), c.byref(self.stream), 0)
        self.call(
            "cuModuleLoadData",
            (PTR, PTR),
            c.byref(self.module),
            c.create_string_buffer(image, len(image)),
        )

    def entry(self, name: str) -> None:
        self.call(
            "cuModuleGetFunction",
            (PTR, PTR, c.c_char_p),
            c.byref(self.function),
            self.module,
            name.encode("ascii"),
        )
        if hasattr(self.lib, "cuFuncGetParamInfo"):
            # Parameter byte layouts only; the reviewed declaration fixes types and roles.
            for i, expected in enumerate(((0, 8), (8, 8), (16, 8), (24, 4))):
                offset, size = SIZE(), SIZE()
                self.call(
                    "cuFuncGetParamInfo",
                    (PTR, SIZE, PTR, PTR),
                    self.function,
                    i,
                    c.byref(offset),
                    c.byref(size),
                )
                if (offset.value, size.value) != expected:
                    raise ValueError(f"driver reports unexpected parameter layout at index {i}")

    def allocate(self, initial: bytes) -> int:
        ptr = DEVICE_PTR()
        self.call("cuMemAlloc_v2", (PTR, SIZE), c.byref(ptr), len(initial))
        self.call(
            "cuMemcpyHtoD_v2",
            (DEVICE_PTR, PTR, SIZE),
            ptr,
            c.create_string_buffer(initial, len(initial)),
            len(initial),
        )
        return ptr.value

    def launch(self, manifest: Manifest, args: tuple[int, int, int, int]) -> None:
        a, b, out, n = args
        values = (DEVICE_PTR(a), DEVICE_PTR(b), DEVICE_PTR(out), c.c_int32(n))
        params = (PTR * len(values))(*(c.addressof(value) for value in values))
        self.call(
            "cuLaunchKernel",
            (PTR, UINT, UINT, UINT, UINT, UINT, UINT, UINT, PTR, PTR, PTR),
            self.function,
            *manifest.grid,
            *manifest.block,
            manifest.shared_bytes,
            self.stream,
            params,
            None,
        )

    def synchronize(self) -> None:
        self.call("cuStreamSynchronize", (PTR,), self.stream)

    def read(self, address: int, size: int) -> bytes:
        dest = c.create_string_buffer(size)
        self.call("cuMemcpyDtoH_v2", (PTR, DEVICE_PTR, SIZE), dest, address, size)
        return dest.raw

    def close(self) -> None:
        if self.context:
            self.call("cuCtxDestroy_v2", (PTR,), self.context)
            self.context = PTR()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--approved-author-smoke", required=True, action="store_true")
    parser.add_argument("--nvcc", required=True, type=absolute_executable)
    parser.add_argument("--ccbin", required=True, type=absolute_executable)
    parser.add_argument(
        "--device-uuid", required=True, type=lambda value: UUID(value.removeprefix("GPU-"))
    )
    args = parser.parse_args()
    if c.sizeof(PTR) != 8:
        parser.error("this smoke is pinned to a 64-bit host and sm_120")
    # Even under a pre-existing shell configuration, use no persistent JIT cache or JIT.
    os.environ["CUDA_CACHE_DISABLE"] = "1"
    os.environ["CUDA_DISABLE_PTX_JIT"] = "1"
    result = run_check(
        lambda source, desc: build_author(args.nvcc, args.ccbin, source, desc),
        lambda: AuthorDriver(args.device_uuid),
        DESCRIPTOR,
        AUTHOR_SOURCE,
    )
    receipt = replace(result, backend="hardware")
    print(json.dumps(asdict(receipt), allow_nan=False, sort_keys=True), flush=True)
    return 0 if receipt.author_hardware_correct else 1


if __name__ == "__main__":
    raise SystemExit(main())
