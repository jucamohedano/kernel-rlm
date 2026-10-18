"""Bounded two-process CUDA/NCCL diagnostic, without Torch or model weights."""

import argparse
import ctypes
import hashlib
import json
import multiprocessing
import os
import platform
import statistics
import struct
import time
from pathlib import Path

SIZES = (4096, 1048576, 16777216, 67108864)
ITERATIONS = 20
WARMUPS = 5
REPEATS = 3
NCCL_FLOAT32 = 7
NCCL_SUM = 0
WHEEL_URL = (
    "https://files.pythonhosted.org/packages/67/f4/"
    "58e4e91b6919367c7aafb8e36fce9aad1a3047e536bf7e2fd560927d3a4c/"
    "nvidia_nccl_cu13-2.29.7-py3-none-manylinux_2_18_x86_64.whl"
)
WHEEL_SHA256 = "edd81538446786ec3b73972543e53bb43bcaf0bfc8ef76cb679fcc390ffe136d"


class UniqueId(ctypes.Structure):
    _fields_ = [("bytes", ctypes.c_char * 128)]


def checked(function, *arguments) -> None:
    status = function(*arguments)
    if status != 0:
        raise RuntimeError(f"{function.__name__} failed with status {status}")


def fetch_nccl(directory: Path) -> dict:
    import urllib.request
    import zipfile

    if platform.machine() != "x86_64":
        raise ValueError("This pinned diagnostic wheel is for x86_64")
    directory.mkdir(parents=True, exist_ok=False)
    wheel = directory / "nccl.whl"
    digest = hashlib.sha256()
    with urllib.request.urlopen(WHEEL_URL, timeout=30) as response, wheel.open("xb") as stream:
        while block := response.read(1024 * 1024):
            digest.update(block)
            stream.write(block)
    if digest.hexdigest() != WHEEL_SHA256:
        raise ValueError("NCCL wheel checksum does not match the upstream lockfile")
    with zipfile.ZipFile(wheel) as archive:
        for member, destination in (
            ("nvidia/nccl/lib/libnccl.so.2", "libnccl.so.2"),
            ("nvidia/nccl/include/nccl.h", "nccl.h"),
        ):
            (directory / destination).write_bytes(archive.read(member))
    report = {
        "wheel_url": WHEEL_URL,
        "wheel_sha256": WHEEL_SHA256,
        "wheel_bytes": wheel.stat().st_size,
        "version": "2.29.7",
        "library_sha256": hashlib.sha256((directory / "libnccl.so.2").read_bytes()).hexdigest(),
    }
    (directory / "provenance.json").write_text(json.dumps(report, indent=2))
    return report


def load_libraries(cuda_path: str, nccl_path: str):
    cuda = ctypes.CDLL(cuda_path, mode=ctypes.RTLD_GLOBAL)
    nccl = ctypes.CDLL(nccl_path, mode=ctypes.RTLD_GLOBAL)
    pointer = ctypes.c_void_p
    pointer_out = ctypes.POINTER(pointer)
    integer_out = ctypes.POINTER(ctypes.c_int)
    size = ctypes.c_size_t
    signatures = {
        "cudaGetDeviceCount": [integer_out],
        "cudaDeviceCanAccessPeer": [integer_out, ctypes.c_int, ctypes.c_int],
        "cudaSetDevice": [ctypes.c_int],
        "cudaDeviceReset": [],
        "cudaMalloc": [pointer_out, size],
        "cudaFree": [pointer],
        "cudaHostAlloc": [pointer_out, size, ctypes.c_uint],
        "cudaFreeHost": [pointer],
        "cudaMemcpy": [pointer, pointer, size, ctypes.c_int],
        "cudaMemcpyAsync": [pointer, pointer, size, ctypes.c_int, pointer],
        "cudaMemset": [pointer, ctypes.c_int, size],
        "cudaStreamCreate": [pointer_out],
        "cudaStreamSynchronize": [pointer],
        "cudaStreamDestroy": [pointer],
        "cudaEventCreate": [pointer_out],
        "cudaEventRecord": [pointer, pointer],
        "cudaEventSynchronize": [pointer],
        "cudaEventElapsedTime": [ctypes.POINTER(ctypes.c_float), pointer, pointer],
        "cudaEventDestroy": [pointer],
        "cudaRuntimeGetVersion": [integer_out],
        "cudaDriverGetVersion": [integer_out],
    }
    for name, arguments in signatures.items():
        function = getattr(cuda, name)
        function.argtypes = arguments
        function.restype = ctypes.c_int
    collective = [pointer, pointer, size, ctypes.c_int]
    signatures = {
        "ncclGetVersion": [integer_out],
        "ncclGetUniqueId": [ctypes.POINTER(UniqueId)],
        "ncclCommInitRank": [pointer_out, ctypes.c_int, UniqueId, ctypes.c_int],
        "ncclAllReduce": [*collective, ctypes.c_int, pointer, pointer],
        "ncclAllGather": [*collective, pointer, pointer],
        "ncclReduceScatter": [*collective, ctypes.c_int, pointer, pointer],
        "ncclCommDestroy": [pointer],
    }
    for name, arguments in signatures.items():
        function = getattr(nccl, name)
        function.argtypes = arguments
        function.restype = ctypes.c_int
    return cuda, nccl


def buffer_layout(operation: str, payload_bytes: int) -> tuple[int, int, int]:
    if operation not in {"all_reduce", "all_gather", "reduce_scatter"}:
        raise ValueError("Unknown collective")
    if payload_bytes <= 0 or payload_bytes % 8:
        raise ValueError("Payload must be a positive multiple of eight bytes")
    count = payload_bytes // 4
    if operation == "all_gather":
        return count, payload_bytes, payload_bytes * 2
    if operation == "reduce_scatter":
        return count // 2, payload_bytes, payload_bytes // 2
    return count, payload_bytes, payload_bytes


def expected_result(operation: str, count: int) -> bytes:
    if operation == "all_gather":
        return struct.pack("f", 1.0) * count + struct.pack("f", 2.0) * count
    return struct.pack("f", 3.0) * count


def measure_collective(
    cuda, nccl, communicator, stream, rank: int, operation: str, payload_bytes: int
) -> dict:
    count, send_bytes, receive_bytes = buffer_layout(operation, payload_bytes)
    send = ctypes.c_void_p()
    receive = ctypes.c_void_p()
    started = ctypes.c_void_p()
    stopped = ctypes.c_void_p()
    checked(cuda.cudaMalloc, ctypes.byref(send), send_bytes)
    checked(cuda.cudaMalloc, ctypes.byref(receive), receive_bytes)
    checked(cuda.cudaEventCreate, ctypes.byref(started))
    checked(cuda.cudaEventCreate, ctypes.byref(stopped))
    try:
        initial = ctypes.create_string_buffer(struct.pack("f", float(rank + 1)) * (send_bytes // 4))
        checked(cuda.cudaMemcpy, send, initial, send_bytes, 1)

        def enqueue() -> None:
            arguments = [send, receive, count, NCCL_FLOAT32]
            if operation != "all_gather":
                arguments.append(NCCL_SUM)
            arguments.extend([communicator, stream])
            function = {
                "all_reduce": nccl.ncclAllReduce,
                "all_gather": nccl.ncclAllGather,
                "reduce_scatter": nccl.ncclReduceScatter,
            }[operation]
            checked(function, *arguments)

        for _iteration in range(WARMUPS):
            enqueue()
        checked(cuda.cudaStreamSynchronize, stream)
        timings = []
        for _repeat in range(REPEATS):
            checked(cuda.cudaEventRecord, started, stream)
            host_started = time.perf_counter()
            for _iteration in range(ITERATIONS):
                enqueue()
            checked(cuda.cudaEventRecord, stopped, stream)
            checked(cuda.cudaEventSynchronize, stopped)
            host_seconds = time.perf_counter() - host_started
            elapsed_ms = ctypes.c_float()
            checked(cuda.cudaEventElapsedTime, ctypes.byref(elapsed_ms), started, stopped)
            timings.append(
                {
                    "event_us_per_op": elapsed_ms.value * 1000 / ITERATIONS,
                    "host_us_per_op": host_seconds * 1e6 / ITERATIONS,
                }
            )
        result = ctypes.create_string_buffer(receive_bytes)
        checked(cuda.cudaMemcpy, result, receive, receive_bytes, 2)
        if result.raw != expected_result(operation, count):
            raise RuntimeError(f"Incorrect {operation} result on rank {rank}")
        return {
            "operation": operation,
            "payload_bytes": payload_bytes,
            "send_bytes_per_rank": send_bytes,
            "receive_bytes_per_rank": receive_bytes,
            "correct": True,
            "timings": timings,
        }
    finally:
        checked(cuda.cudaEventDestroy, started)
        checked(cuda.cudaEventDestroy, stopped)
        checked(cuda.cudaFree, send)
        checked(cuda.cudaFree, receive)


def rank_worker(
    rank: int, unique_bytes: bytes, directory: str, cuda_path: str, nccl_path: str
) -> None:
    output = Path(directory) / f"rank-{rank}.json"
    communicator = ctypes.c_void_p()
    stream = ctypes.c_void_p()
    try:
        cuda, nccl = load_libraries(cuda_path, nccl_path)
        checked(cuda.cudaSetDevice, rank)
        checked(cuda.cudaStreamCreate, ctypes.byref(stream))
        unique_id = UniqueId.from_buffer_copy(unique_bytes)
        checked(nccl.ncclCommInitRank, ctypes.byref(communicator), 2, unique_id, rank)
        rows = [
            measure_collective(cuda, nccl, communicator, stream, rank, operation, payload)
            for operation in ("all_reduce", "all_gather", "reduce_scatter")
            for payload in SIZES
        ]
        checked(nccl.ncclCommDestroy, communicator)
        checked(cuda.cudaStreamDestroy, stream)
        checked(cuda.cudaDeviceReset)
        output.write_text(json.dumps({"rank": rank, "status": "passed", "rows": rows}, indent=2))
    except BaseException as error:
        output.write_text(
            json.dumps(
                {"rank": rank, "status": "failed", "error": f"{type(error).__name__}: {error}"},
                indent=2,
            )
        )
        raise


def staged_transfer(cuda, source_device: int, destination_device: int) -> list[dict]:
    payload_bytes = max(SIZES)
    source = ctypes.c_void_p()
    destination = ctypes.c_void_p()
    host = ctypes.c_void_p()
    checked(cuda.cudaSetDevice, source_device)
    checked(cuda.cudaMalloc, ctypes.byref(source), payload_bytes)
    checked(cuda.cudaMemset, source, 37, payload_bytes)
    checked(cuda.cudaHostAlloc, ctypes.byref(host), payload_bytes, 0)
    checked(cuda.cudaSetDevice, destination_device)
    checked(cuda.cudaMalloc, ctypes.byref(destination), payload_bytes)
    try:
        rows = []
        for size in SIZES[1:]:

            def transfer(transfer_bytes: int = size) -> None:
                checked(cuda.cudaSetDevice, source_device)
                checked(cuda.cudaMemcpy, host, source, transfer_bytes, 2)
                checked(cuda.cudaSetDevice, destination_device)
                checked(cuda.cudaMemcpy, destination, host, transfer_bytes, 1)

            for _iteration in range(WARMUPS):
                transfer()
            elapsed = []
            for _repeat in range(REPEATS):
                started = time.perf_counter()
                for _iteration in range(ITERATIONS):
                    transfer()
                elapsed.append((time.perf_counter() - started) / ITERATIONS)
            result = ctypes.create_string_buffer(size)
            checked(cuda.cudaMemcpy, result, destination, size, 2)
            if result.raw != bytes([37]) * size:
                raise RuntimeError("Incorrect pinned-host staged transfer")
            median = statistics.median(elapsed)
            rows.append(
                {
                    "source_gpu": source_device,
                    "destination_gpu": destination_device,
                    "payload_bytes": size,
                    "correct": True,
                    "median_end_to_end_us": median * 1e6,
                    "logical_gb_per_second": size / median / 1e9,
                }
            )
        return rows
    finally:
        checked(cuda.cudaSetDevice, source_device)
        checked(cuda.cudaFree, source)
        checked(cuda.cudaSetDevice, destination_device)
        checked(cuda.cudaFree, destination)
        checked(cuda.cudaFreeHost, host)


def run_collectives(
    nccl, directory: Path, cuda_path: str, nccl_path: str, force_no_p2p: bool, timeout_s: int = 90
) -> dict:
    directory.mkdir()
    unique_id = UniqueId()
    checked(nccl.ncclGetUniqueId, ctypes.byref(unique_id))
    context = multiprocessing.get_context("spawn")
    saved_environment = {
        key: os.environ.get(key)
        for key in (
            "NCCL_DEBUG",
            "NCCL_DEBUG_SUBSYS",
            "NCCL_DEBUG_FILE",
            "NCCL_P2P_DISABLE",
        )
    }
    os.environ.update(
        NCCL_DEBUG="INFO",
        NCCL_DEBUG_SUBSYS="INIT,GRAPH,P2P,SHM,NET",
        NCCL_DEBUG_FILE=str(directory.resolve() / "nccl-%h-%p.log"),
    )
    if force_no_p2p:
        os.environ["NCCL_P2P_DISABLE"] = "1"
    workers = [
        context.Process(
            target=rank_worker, args=(rank, bytes(unique_id), str(directory), cuda_path, nccl_path)
        )
        for rank in range(2)
    ]
    try:
        for worker in workers:
            worker.start()
        deadline = time.monotonic() + timeout_s
        for worker in workers:
            worker.join(max(0, deadline - time.monotonic()))
        if any(worker.is_alive() for worker in workers):
            raise TimeoutError("NCCL communication exceeded its 90-second deadline")
        if any(worker.exitcode != 0 for worker in workers):
            raise RuntimeError("NCCL rank failed; inspect rank JSON and transport logs")
        ranks = [json.loads((directory / f"rank-{rank}.json").read_text()) for rank in range(2)]
        return {"status": "passed", "forced_p2p_disabled": force_no_p2p, "ranks": ranks}
    finally:
        for worker in workers:
            if worker.pid is not None:
                if worker.is_alive():
                    worker.kill()
                worker.join()
        for key, value in saved_environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def run_probe(cuda_path: str, nccl_path: str, directory: Path) -> dict:
    started = time.monotonic()
    directory.mkdir(parents=True, exist_ok=False)
    report = {
        "kind": "gpu-communication-maintenance",
        "model_loaded": False,
        "training_updates": 0,
        "single_host": True,
        "processes": 2,
        "iterations": ITERATIONS,
        "warmups": WARMUPS,
        "repeats": REPEATS,
        "sizes_bytes": list(SIZES),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "nccl_library": nccl_path,
        "cuda_library": cuda_path,
        "status": "running",
    }
    try:
        cuda, nccl = load_libraries(cuda_path, nccl_path)
        count = ctypes.c_int()
        checked(cuda.cudaGetDeviceCount, ctypes.byref(count))
        if count.value != 2:
            raise ValueError(f"Expected exactly two visible allocated GPUs, found {count.value}")
        report["device_count"] = count.value
        for name, function in (
            ("cuda_runtime_version", cuda.cudaRuntimeGetVersion),
            ("cuda_driver_api_version", cuda.cudaDriverGetVersion),
            ("nccl_version", nccl.ncclGetVersion),
        ):
            value = ctypes.c_int()
            checked(function, ctypes.byref(value))
            report[name] = value.value
        report["peer_access"] = []
        for source, destination in ((0, 1), (1, 0)):
            capable = ctypes.c_int()
            checked(cuda.cudaDeviceCanAccessPeer, ctypes.byref(capable), source, destination)
            report["peer_access"].append(
                {
                    "source_gpu": source,
                    "destination_gpu": destination,
                    "can_access": bool(capable.value),
                }
            )
        report["staged_transfers"] = staged_transfer(cuda, 0, 1) + staged_transfer(cuda, 1, 0)
        for device in range(2):
            checked(cuda.cudaSetDevice, device)
            checked(cuda.cudaDeviceReset)
        report["automatic"] = run_collectives(
            nccl, directory / "automatic", cuda_path, nccl_path, False
        )
        report["p2p_disabled"] = run_collectives(
            nccl, directory / "p2p-disabled", cuda_path, nccl_path, True
        )
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        (directory / "report.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch-nccl-dir", type=Path)
    parser.add_argument("--cuda-library")
    parser.add_argument("--nccl-library")
    parser.add_argument("--output-dir", type=Path)
    arguments = parser.parse_args()
    if arguments.fetch_nccl_dir:
        print(json.dumps(fetch_nccl(arguments.fetch_nccl_dir), indent=2))
        return
    if not all((arguments.cuda_library, arguments.nccl_library, arguments.output_dir)):
        parser.error("Provide all three library/output arguments, or --fetch-nccl-dir")
    print(
        json.dumps(
            run_probe(arguments.cuda_library, arguments.nccl_library, arguments.output_dir),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
