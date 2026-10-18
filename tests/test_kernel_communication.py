import ctypes
import hashlib
import importlib.util
import io
import struct
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE = (
    Path(__file__).parents[1] / "training/environments/kernel_rlm/kernel_rlm/communication_probe.py"
)


def load_probe():
    specification = importlib.util.spec_from_file_location("communication_probe", SOURCE)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "operation, expected",
    [
        ("all_reduce", (1024, 4096, 4096)),
        ("all_gather", (1024, 4096, 8192)),
        ("reduce_scatter", (512, 4096, 2048)),
    ],
)
def test_collective_buffer_sizes(operation, expected):
    assert load_probe().buffer_layout(operation, 4096) == expected


@pytest.mark.parametrize("size", [0, -8, 7, 4097])
def test_invalid_payload_sizes_are_rejected(size):
    with pytest.raises(ValueError):
        load_probe().buffer_layout("all_reduce", size)


def test_expected_result_checks_rank_order_and_reduction():
    probe = load_probe()
    assert probe.expected_result("all_gather", 2) == struct.pack("ffff", 1, 1, 2, 2)
    assert probe.expected_result("all_reduce", 2) == struct.pack("ff", 3, 3)
    assert probe.expected_result("reduce_scatter", 2) == struct.pack("ff", 3, 3)


def test_native_library_errors_are_not_silently_accepted():
    def failing_function():
        return 7

    with pytest.raises(RuntimeError, match="failing_function failed with status 7"):
        load_probe().checked(failing_function)


def test_ctypes_abi_and_pin_without_loading_cuda(monkeypatch):
    probe = load_probe()
    libraries = []

    class FakeLibrary:
        def __getattr__(self, name):
            value = SimpleNamespace(__name__=name)
            setattr(self, name, value)
            return value

    def create_library(*args, **kwargs):
        library = FakeLibrary()
        libraries.append(library)
        return library

    monkeypatch.setattr(ctypes, "CDLL", create_library)
    cuda, nccl = probe.load_libraries("cuda", "nccl")
    assert ctypes.sizeof(probe.UniqueId) == 128
    assert nccl.ncclCommInitRank.argtypes[2] is probe.UniqueId
    assert cuda.cudaMalloc.argtypes[1] is ctypes.c_size_t
    assert len(nccl.ncclAllReduce.argtypes) == 7
    assert len(nccl.ncclAllGather.argtypes) == 6
    assert probe.NCCL_FLOAT32 == 7
    assert probe.REPEATS == 3
    assert max(probe.SIZES) == 64 * 1024 * 1024
    assert len(probe.WHEEL_SHA256) == 64


def test_download_rejects_wrong_checksum(tmp_path, monkeypatch):
    import urllib.request

    probe = load_probe()
    monkeypatch.setattr(probe.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(b"bad wheel"))
    with pytest.raises(ValueError, match="checksum"):
        probe.fetch_nccl(tmp_path / "library")
    assert not (tmp_path / "library/libnccl.so.2").exists()


def test_download_extracts_only_the_pinned_library_and_header(tmp_path, monkeypatch):
    import urllib.request

    probe = load_probe()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("nvidia/nccl/lib/libnccl.so.2", b"native fixture")
        archive.writestr("nvidia/nccl/include/nccl.h", b"header fixture")
        archive.writestr("unexpected-script.py", b"never run")
    payload = buffer.getvalue()
    monkeypatch.setattr(probe.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(probe, "WHEEL_SHA256", hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(payload))
    report = probe.fetch_nccl(tmp_path / "library")
    assert report["library_sha256"] == hashlib.sha256(b"native fixture").hexdigest()
    assert not (tmp_path / "library/unexpected-script.py").exists()
    with pytest.raises(FileExistsError):
        probe.fetch_nccl(tmp_path / "library")


def test_timeout_kills_both_owned_ranks_and_restores_environment(tmp_path, monkeypatch):
    probe = load_probe()
    workers = []

    class FakeWorker:
        pid = None
        alive = True
        killed = False

        def start(self):
            self.pid = 123

        def join(self, timeout=None):
            pass

        def is_alive(self):
            return self.alive

        def kill(self):
            self.killed = True
            self.alive = False

    def create_worker(**kwargs):
        worker = FakeWorker()
        workers.append(worker)
        return worker

    monkeypatch.setattr(
        probe.multiprocessing, "get_context", lambda mode: SimpleNamespace(Process=create_worker)
    )
    monkeypatch.setenv("NCCL_P2P_DISABLE", "original")
    with pytest.raises(TimeoutError):
        probe.run_collectives(
            SimpleNamespace(ncclGetUniqueId=lambda *args: 0),
            tmp_path / "timed-out",
            "cuda",
            "nccl",
            True,
            timeout_s=0,
        )
    assert len(workers) == 2
    assert all(worker.killed for worker in workers)
    assert probe.os.environ["NCCL_P2P_DISABLE"] == "original"
