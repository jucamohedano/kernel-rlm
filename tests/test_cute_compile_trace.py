import hashlib
import json
from pathlib import Path

import pytest
from cute_rlm.compile_trace import install_compile_trace


class Compiled:
    def __init__(self, ptx=b".visible .entry kernel_main() {}", cubin=b"cubin"):
        self.__ptx__ = ptx
        self.__cubin__ = cubin

    def __call__(self, value):
        if getattr(self, "fail", False):
            raise RuntimeError("launch failed")
        return value + 1


class Compile:
    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if kwargs.get("fail"):
            raise RuntimeError("compile failed")
        return Compiled()

    def __getitem__(self, options):
        def typed(*args, **kwargs):
            self.calls.append((args, {"typed_options": options, **kwargs}))
            return Compiled()

        return typed


class Cute:
    def __init__(self):
        self.compile = Compile()


class Nvtx:
    def __init__(self, events):
        self.events = events

    def range_push(self, name):
        self.events.append(("push", name))

    def range_pop(self):
        self.events.append(("pop", None))


class Cuda:
    def __init__(self, events):
        self.nvtx = Nvtx(events)

    def synchronize(self):
        self.nvtx.events.append(("sync", None))


class Torch:
    def __init__(self):
        self.events = []
        self.cuda = Cuda(self.events)


def test_direct_compile_captures_artifacts_and_invocation(tmp_path: Path):
    cute, torch = Cute(), Torch()
    original = cute.compile
    trace = install_compile_trace(cute, torch, tmp_path)

    compiled = cute.compile("kernel", options="--some-option")
    assert compiled(4) == 5
    data = trace.snapshot()

    assert cute.compile("second", options="--keep-ptx").compiled
    artifact = data["artifacts"][0]
    assert artifact["entry_names"] == ["kernel_main"]
    ptx_path = tmp_path / artifact["ptx_path"]
    cubin_path = tmp_path / artifact["cubin_path"]
    assert ptx_path.read_bytes() == b".visible .entry kernel_main() {}"
    assert cubin_path.read_bytes() == b"cubin"
    assert artifact["ptx_sha256"] == hashlib.sha256(ptx_path.read_bytes()).hexdigest()
    direct_options = trace._original.calls[0][1]["options"]
    assert "--some-option" in direct_options
    assert "--keep-ptx" in direct_options and "--keep-cubin" in direct_options
    assert data["invocations"][0]["status"] == "completed"
    assert torch.events[0][0] == "push"
    assert torch.events[1] == ("sync", None)
    assert torch.events[2] == ("pop", None)
    assert json.loads((tmp_path / "compile-provenance.json").read_text()) == data

    trace.close()
    assert cute.compile is original


def test_typed_options_keep_original_subscription_semantics(tmp_path: Path):
    cute, torch = Cute(), Torch()
    trace = install_compile_trace(cute, torch, tmp_path)
    result = cute.compile["--typed-option"]("kernel")
    assert result(8) == 9
    recorded = cute.compile.trace._original.calls[0][1]
    assert recorded["typed_options"] == "--typed-option"
    assert "--keep-ptx" in recorded["options"]
    trace.close()


def test_failed_compiled_call_is_recorded_and_reraised(tmp_path: Path):
    cute, torch = Cute(), Torch()
    trace = install_compile_trace(cute, torch, tmp_path)
    compiled = cute.compile("kernel")
    compiled.compiled.fail = True
    with pytest.raises(RuntimeError, match="launch failed"):
        compiled(1)
    assert trace.snapshot()["invocations"][0]["status"] == "failed"
    trace.close()


def test_compile_failure_does_not_leave_wrapper_installed(tmp_path: Path):
    cute, torch = Cute(), Torch()
    original = cute.compile
    trace = install_compile_trace(cute, torch, tmp_path)
    with pytest.raises(RuntimeError, match="compile failed"):
        cute.compile("kernel", fail=True)
    trace.close()
    assert cute.compile is original
