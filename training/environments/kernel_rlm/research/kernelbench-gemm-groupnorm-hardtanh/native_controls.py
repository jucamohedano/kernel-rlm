"""Build and load only the checked-in author controls; no artifact input API."""

import ctypes as ct
import hashlib
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

PTR = ct.c_void_p
UINT = ct.c_uint
INT = ct.c_int
SOURCE = Path(__file__).with_name("controls.cu")


class AuthorControls:
    def __init__(self, scratch_dir: Path, nvcc: str, receipt: dict[str, Any]) -> None:
        compiler = shutil.which(nvcc)
        if compiler is None:
            raise RuntimeError(f"nvcc not found: {nvcc}")
        if any(os.environ.get(key) for key in ("NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS")):
            raise RuntimeError("unset NVCC_PREPEND_FLAGS and NVCC_APPEND_FLAGS")
        env = dict(os.environ, TMPDIR=str(scratch_dir))
        version = subprocess.check_output([compiler, "--version"], text=True, env=env)
        cubin = scratch_dir / "kb30-author-controls.sm120.cubin"
        args = [
            compiler,
            "--cubin",
            "--generate-code=arch=compute_120,code=sm_120",
            "--std=c++17",
            "-O3",
            str(SOURCE),
            "-o",
            str(cubin),
        ]
        receipt.update(
            build="started",
            nvcc_version=version.strip(),
            argv=args,
            source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        )
        result = subprocess.run(args, text=True, capture_output=True, env=env, check=False)
        if result.returncode:
            receipt.update(build="failed", returncode=result.returncode, stderr=result.stderr)
            raise RuntimeError("trusted sm_120 cubin build failed")
        payload = cubin.read_bytes()
        # CUDA ELF (EM_CUDA=190); require an actual cubin, not an input PTX string.
        if payload[:6] != b"\x7fELF\x02\x01" or int.from_bytes(payload[18:20], "little") != 190:
            receipt["build"] = "failed"
            raise RuntimeError("nvcc did not emit an ELF64 CUDA cubin")
        receipt.update(
            build="passed",
            cubin_sha256=hashlib.sha256(payload).hexdigest(),
            cubin_bytes=len(payload),
            load="started",
        )
        self.lib = ct.CDLL("libcuda.so.1")
        self.call("cuInit", (UINT,), 0)
        ver = INT()
        self.call("cuDriverGetVersion", (ct.POINTER(INT),), ct.byref(ver))
        receipt["cuda_driver_api_version"] = ver.value
        self.module = PTR()
        try:
            self.call(
                "cuModuleLoad",
                (ct.POINTER(PTR), ct.c_char_p),
                ct.byref(self.module),
                os.fsencode(cubin),
            )
            self.linear = self.function(b"kb30_linear")
            self.epilogue = self.function(b"kb30_epilogue")
        except Exception:
            receipt["load"] = "failed"
            self.close()
            raise
        receipt["load"] = "passed"

    def close(self) -> None:
        if self.module:
            self.call("cuModuleUnload", (PTR,), self.module)
            self.module = PTR()

    def call(self, name: str, types: tuple, *args: Any) -> None:
        fn = getattr(self.lib, name)
        fn.argtypes, fn.restype = types, INT
        code = fn(*args)
        if code:
            raise RuntimeError(f"{name}: CUDA driver error {code}")

    def function(self, name: bytes) -> PTR:
        fn = PTR()
        self.call(
            "cuModuleGetFunction",
            (ct.POINTER(PTR), PTR, ct.c_char_p),
            ct.byref(fn),
            self.module,
            name,
        )
        return fn

    def launch(self, fn: PTR, grid: tuple, block: tuple, stream: Any, *args: Any) -> None:
        # kernelParams points to per-argument storage for the duration of the driver call.
        argv = (PTR * len(args))(*(ct.addressof(arg) for arg in args))
        self.call(
            "cuLaunchKernel",
            (PTR, UINT, UINT, UINT, UINT, UINT, UINT, UINT, PTR, ct.POINTER(PTR), ct.POINTER(PTR)),
            fn,
            *grid,
            *block,
            0,
            stream.cuda_stream,
            argv,
            None,
        )

    def produce(self, x: Any, w: Any, b: Any, scratch: Any, stream: Any) -> None:
        self.launch(
            self.linear,
            (512, 64, 1),
            (16, 16, 1),
            stream,
            *(PTR(t.data_ptr()) for t in (x, w, b, scratch)),
        )

    def finish(
        self, scratch: Any, scale: Any, offset: Any, y: Any, stream: Any, fault: int = 0
    ) -> None:
        args = [PTR(t.data_ptr()) for t in (scratch, scale, offset, y)]
        self.launch(self.epilogue, (16, 1024, 1), (256, 1, 1), stream, *args, INT(fault))
