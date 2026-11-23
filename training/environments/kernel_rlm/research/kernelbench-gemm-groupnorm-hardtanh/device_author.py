"""Parent-only opt-in native KB30 author controls. No file input for device code."""

import argparse
import ctypes as c
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import UUID

from device_abi import (
    AUTHOR_DESCRIPTOR,
    CONTROLS,
    MAX_CUBIN,
    Built,
    Launch,
    author_fragment,
    author_translation_unit,
    checked_product,
    compare_forwards,
    digest,
    parse_descriptor,
    run_trusted,
)

NVCC_VERSION = "13.3.73"
FLAGS = ("--cubin", "--generate-code=arch=compute_120,code=sm_120", "--std=c++17", "-O3")
P, INT, U, D, S = c.c_void_p, c.c_int, c.c_uint, c.c_uint64, c.c_size_t


def executable(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or not path.is_file() or not os.access(path, os.X_OK):
        raise argparse.ArgumentTypeError("expected absolute trusted executable")
    return path.resolve(strict=True)


def file_digest(path: Path) -> str:
    with path.open("rb") as data:
        return hashlib.file_digest(data, "sha256").hexdigest()


def toolkit_fingerprint(nvcc: Path, ccbin: Path) -> dict:
    root = nvcc.parent.parent
    include = (root / "include").resolve(strict=True)
    if not (include / "cuda_runtime.h").is_file():
        raise ValueError("missing pinned CUDA toolkit header tree")
    header_hash = hashlib.sha256()
    files = sorted(path for path in include.rglob("*") if path.is_file())
    for path in files:
        header_hash.update(
            path.relative_to(include).as_posix().encode() + b"\0" + bytes.fromhex(file_digest(path))
        )
    tools = (
        nvcc,
        root / "bin/ptxas",
        root / "nvvm/bin/cicc",
        root / "nvvm/libdevice/libdevice.10.bc",
        ccbin,
    )
    return {
        "nvcc_version": NVCC_VERSION,
        "flags": FLAGS,
        "toolkit_files_sha256": {str(path): file_digest(path) for path in tools},
        "toolkit_include_tree_sha256": header_hash.hexdigest(),
        "include_file_count": len(files),
    }


COMPILER_LIMITS = (
    "/usr/bin/prlimit",
    "--as=4294967296",
    "--fsize=33554432",
    "--cpu=35:40",
    "--core=0",
    "--",
)


def build_author(nvcc: Path, ccbin: Path, control: str, tu: bytes) -> Built:
    fixed_tu, _ = author_translation_unit(
        parse_descriptor(AUTHOR_DESCRIPTOR), author_fragment(control)
    )
    if tu != fixed_tu:
        raise ValueError("only the exact reviewed author TU can be compiled")
    metadata = toolkit_fingerprint(nvcc, ccbin)
    with TemporaryDirectory(prefix="kb30-device-author-", dir="/var/tmp") as directory:
        work = Path(directory)
        env = {
            "PATH": f"{nvcc.parent}:/usr/bin:/bin",
            "HOME": directory,
            "TMPDIR": directory,
            "LC_ALL": "C",
        }
        call = dict(
            cwd=work,
            env=env,
            stdin=subprocess.DEVNULL,
            check=True,
            close_fds=True,
        )
        version = subprocess.run(
            [*COMPILER_LIMITS, nvcc, "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=10,
            **call,
        ).stdout
        if re.search(rb"\bV" + re.escape(NVCC_VERSION.encode()) + rb"\b", version) is None:
            raise ValueError(f"requires NVCC V{NVCC_VERSION}")
        (work / "author.cu").write_bytes(tu)
        with (work / "build.log").open("wb") as log:
            try:
                subprocess.run(
                    [
                        *COMPILER_LIMITS,
                        nvcc,
                        *FLAGS,
                        "--compiler-bindir",
                        ccbin,
                        "author.cu",
                        "-o",
                        "author.cubin",
                    ],
                    stdout=log,
                    stderr=log,
                    timeout=50,
                    **call,
                )
            except subprocess.CalledProcessError as exc:
                with (work / "build.log").open("rb") as output:
                    raise RuntimeError(output.read(4096).decode("utf-8", errors="replace")) from exc
        with (work / "author.cubin").open("rb") as artifact:
            image = artifact.read(MAX_CUBIN + 1)
        if not 0 < len(image) <= MAX_CUBIN:
            raise ValueError("cubin size outside native policy")
    return Built(image, digest(tu), digest(image), metadata)


class AuthorDriver:
    """Exact-byte CUDA Driver API adapter. One private context and one explicit stream."""

    def __init__(self, expected_uuid: UUID):
        self.expected_uuid = expected_uuid
        self.lib = c.CDLL("libcuda.so.1")
        self.dev, self.context, self.stream = INT(), P(), P()

    def result(self, name: str, types: tuple, *args) -> int:
        api = getattr(self.lib, name)
        api.argtypes, api.restype = types, INT
        return api(*args)

    def call(self, name: str, types: tuple, *args) -> None:
        status = self.result(name, types, *args)
        if status != 0:
            raise RuntimeError(f"{name}: CUDA status {status}")

    def device_attr(self, number: int) -> int:
        out = INT()
        self.call("cuDeviceGetAttribute", (P, INT, INT), c.byref(out), number, self.dev)
        return out.value

    def probe(self) -> dict:
        self.call("cuInit", (U,), 0)
        self.call("cuDeviceGet", (P, INT), c.byref(self.dev), 0)
        name, raw_uuid, version = c.create_string_buffer(128), (c.c_ubyte * 16)(), INT()
        self.call("cuDeviceGetName", (P, INT, INT), name, len(name), self.dev)
        self.call("cuDeviceGetUuid_v2", (P, INT), raw_uuid, self.dev)
        uuid = UUID(bytes=bytes(raw_uuid))
        label = name.value.decode("ascii")
        if uuid != self.expected_uuid or not label.startswith("NVIDIA RTX PRO 6000 Blackwell"):
            raise ValueError("unexpected GPU ordinal 0 identity")
        self.call("cuDriverGetVersion", (P,), c.byref(version))
        if version.value < 13000:
            raise ValueError("requires a qualified CUDA 13+ Driver API")
        return {
            "name": label,
            "uuid": f"GPU-{uuid}",
            "driver_version": version.value,
            "capability": (self.device_attr(75), self.device_attr(76)),
        }

    def load(self, image: bytes) -> object:
        if not self.context:
            self.call("cuCtxCreate_v2", (P, U, INT), c.byref(self.context), 0, self.dev)
            self.call("cuStreamCreate", (P, U), c.byref(self.stream), 1)  # nonblocking stream
        module = P()
        self.call("cuModuleLoadData", (P, P), c.byref(module), c.c_char_p(image))
        return module

    def function_attr(self, fn: object, number: int) -> int:
        out = INT()
        self.call("cuFuncGetAttribute", (P, INT, P), c.byref(out), number, fn)
        return out.value

    def entry(self, module: object, launch: Launch) -> tuple[object, dict]:
        fn = P()
        self.call(
            "cuModuleGetFunction",
            (P, P, c.c_char_p),
            c.byref(fn),
            module,
            launch.entry.encode("ascii"),
        )
        offset, size = S(), S()
        for i, expected in enumerate(launch.param_layout):
            self.call("cuFuncGetParamInfo", (P, S, P, P), fn, i, c.byref(offset), c.byref(size))
            if (offset.value, size.value) != expected:
                raise ValueError(f"parameter layout mismatch in {launch.entry} at {i}")
        if (
            self.result(
                "cuFuncGetParamInfo",
                (P, S, P, P),
                fn,
                len(launch.param_layout),
                c.byref(offset),
                c.byref(size),
            )
            != 1
        ):
            raise ValueError(f"unexpected parameter count or query status in {launch.entry}")
        block_caps = tuple(self.device_attr(i) for i in (2, 3, 4))
        grid_caps = tuple(self.device_attr(i) for i in (5, 6, 7))
        threads = min(self.device_attr(1), self.function_attr(fn, 0))
        checked_product(*launch.block, cap=threads)
        if any(d > cap for d, cap in zip(launch.grid, grid_caps, strict=True)) or any(
            d > cap for d, cap in zip(launch.block, block_caps, strict=True)
        ):
            raise ValueError("geometry exceeds Driver API limits")
        shared = self.function_attr(fn, 1)
        binary_version = self.function_attr(fn, 6)
        if (
            not 0 <= shared <= min(48 << 10, self.device_attr(8))
            or binary_version != 120
            or launch.shared_bytes != 0
        ):
            raise ValueError("unexpected loaded function target/shared resources")
        return fn, {
            "entry": launch.entry,
            "registers_per_thread": self.function_attr(fn, 4),
            "local_bytes_per_thread": self.function_attr(fn, 3),
            "static_shared_bytes": shared,
            "max_threads_per_block": threads,
            "binary_version": binary_version,
            "parameter_layout": launch.param_layout,
        }

    def allocate(self, size: int) -> int:
        address = D()
        self.call("cuMemAlloc_v2", (P, S), c.byref(address), size)
        return address.value

    def write(self, address: int, data: bytes) -> None:
        self.call("cuMemcpyHtoD_v2", (D, P, S), address, c.c_char_p(data), len(data))

    def poison(self, address: int, size: int, word: int) -> None:
        self.call("cuMemsetD32Async", (D, U, S, P), address, word, size // 4, self.stream)

    def prepared(self) -> None:
        # Also finish any pageable staging on the implicit copy stream.
        self.call("cuCtxSynchronize", ())

    def launch(self, function: object, launch: Launch, args: tuple[int, ...]) -> None:
        words = [D(arg) for arg in args]
        argv = (P * len(words))(*(c.addressof(word) for word in words))
        self.call(
            "cuLaunchKernel",
            (P, U, U, U, U, U, U, U, P, P, P),
            function,
            *launch.grid,
            *launch.block,
            launch.shared_bytes,
            self.stream,
            argv,
            None,
        )

    def synchronize(self) -> None:
        self.call("cuStreamSynchronize", (P,), self.stream)

    def read(self, address: int, size: int) -> bytes:
        dest = c.create_string_buffer(size)
        self.call("cuMemcpyDtoH_v2", (P, D, S), dest, address, size)
        return dest.raw

    def close(self) -> None:
        if self.context:
            self.call("cuCtxDestroy_v2", (P,), self.context)
            self.context = P()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--approved-author-smoke", action="store_true", required=True)
    parser.add_argument("--nvcc", type=executable, required=True)
    parser.add_argument("--ccbin", type=executable, required=True)
    parser.add_argument("--device-uuid", type=lambda x: UUID(x.removeprefix("GPU-")), required=True)
    parser.add_argument("--control", choices=CONTROLS, default="positive")
    parser.add_argument("--fixture", choices=("upstream", "diagnostic"), required=True)
    args = parser.parse_args()
    if args.control != "positive" and args.fixture != "diagnostic":
        parser.error("negative controls require the mixed diagnostic fixture")
    if c.sizeof(P) != 8:
        parser.error("requires a 64-bit host")
    os.environ.update(CUDA_CACHE_DISABLE="1", CUDA_DISABLE_PTX_JIT="1")
    from device_oracle import compare, fixtures

    try:
        inputs, next_x, refs, metadata = fixtures(args.fixture, args.control)
    except Exception as exc:
        print(
            json.dumps(
                {
                    "role": "trusted_author_control",
                    "outcome": "unavailable",
                    "stage": "fixture_preflight",
                    "reason": str(exc),
                },
                allow_nan=False,
            ),
            flush=True,
        )
        return 2
    receipt, outputs = run_trusted(
        lambda variant, tu: build_author(args.nvcc, args.ccbin, variant, tu),
        lambda: AuthorDriver(args.device_uuid),
        args.control,
        inputs,
        next_x,
    )
    receipt.update(backend="native_author_only", oracle=metadata)
    expected = compare_forwards(receipt, outputs, refs, compare)
    print(json.dumps(receipt, sort_keys=True, allow_nan=False), flush=True)
    return 0 if expected else 1


if __name__ == "__main__":
    raise SystemExit(main())
