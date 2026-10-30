"""Local, observational CuTe compile capture for the extra profiling pass.

This is not a security boundary: Python code can bypass or replace these wrappers.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any

_KEEP_FLAGS = ("--keep-ptx", "--keep-cubin")
_ENTRY = re.compile(r"\.entry\s+([A-Za-z_$][A-Za-z0-9_.$]*)")


def _append_flags(options: Any) -> str:
    if options is None:
        return " ".join(_KEEP_FLAGS)
    if not isinstance(options, str):
        raise TypeError("CuTe string options must be a string or None")
    return " ".join((options, *_KEEP_FLAGS)).strip()


def _options_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    updated = dict(kwargs)
    updated["options"] = _append_flags(updated.get("options"))
    return updated


def _blob(value: Any) -> bytes | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, memoryview):
        return value.tobytes()
    if isinstance(value, str):
        return value.encode("utf-8")
    return None


class _CompileTrace:
    def __init__(self, cute: Any, torch: Any, output_dir: Path):
        self.cute = cute
        self.torch = torch
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_dir = self.output_dir / "compile-artifacts"
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts: list[dict[str, Any]] = []
        self.invocations: list[dict[str, str]] = []
        self._artifact_count = 0
        self._invocation_count = 0
        self._original = cute.compile
        self._closed = False
        self._proxy = _CompileProxy(self)
        cute.compile = self._proxy

    def compile(self, *args: Any, **kwargs: Any) -> Any:
        result = self._original(*args, **_options_kwargs(kwargs))
        self._record_artifact(result)
        return _CompiledCall(self, result)

    def compile_item(self, options: Any) -> Any:
        return _CompileItem(self, options)

    def compile_typed(self, options: Any, *args: Any, **kwargs: Any) -> Any:
        compile_callable = self._original[options]
        result = compile_callable(*args, **_options_kwargs(kwargs))
        self._record_artifact(result)
        return _CompiledCall(self, result)

    def _record_artifact(self, compiled: Any) -> str:
        artifact_id = f"artifact-{self._artifact_count:04d}"
        self._artifact_count += 1
        ptx = _blob(getattr(compiled, "__ptx__", None))
        cubin = _blob(getattr(compiled, "__cubin__", None))
        entry_names: list[str] = []
        artifact: dict[str, Any] = {
            "id": artifact_id,
            "ptx_path": None,
            "ptx_sha256": None,
            "cubin_path": None,
            "cubin_sha256": None,
        }
        for name, blob in (("ptx", ptx), ("cubin", cubin)):
            path_key, hash_key = f"{name}_path", f"{name}_sha256"
            if blob is not None:
                path = self.artifact_dir / f"{artifact_id}.{name}"
                path.write_bytes(blob)
                artifact[path_key] = str(path.relative_to(self.output_dir))
                artifact[hash_key] = hashlib.sha256(blob).hexdigest()
                if name == "ptx":
                    entry_names = list(
                        dict.fromkeys(_ENTRY.findall(blob.decode("utf-8", errors="replace")))
                    )
        artifact["entry_names"] = entry_names
        self.artifacts.append(artifact)
        return artifact_id

    def invoke(self, compiled: Any, artifact_id: str, *args: Any, **kwargs: Any) -> Any:
        self._invocation_count += 1
        invocation_id = f"invocation-{self._invocation_count:04d}"
        range_name = f"cute_compile:{artifact_id}:{invocation_id}"
        invocation = {"range_name": range_name, "artifact_id": artifact_id, "status": "running"}
        self.invocations.append(invocation)
        nvtx = self.torch.cuda.nvtx
        pushed = False
        try:
            nvtx.range_push(range_name)
            pushed = True
            result = compiled(*args, **kwargs)
            self.torch.cuda.synchronize()
            invocation["status"] = "completed"
            return result
        except BaseException:
            invocation["status"] = "failed"
            raise
        finally:
            if pushed:
                nvtx.range_pop()

    def snapshot(self) -> dict[str, Any]:
        data = {
            "observer": "local-cute-compile-trace",
            "scope": "local-process-observational-extra-profile-pass",
            "attestation": False,
            "artifacts": copy.deepcopy(self.artifacts),
            "invocations": copy.deepcopy(self.invocations),
            "observer_parent_must_hash_and_correlate_ptx_nsight_names_before_compiled_true": True,
            "caller_owns_model_forward_range": True,
        }
        path = self.output_dir / "compile-provenance.json"
        path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return data

    def close(self) -> None:
        if not self._closed:
            self.cute.compile = self._original
            self._closed = True


class _CompileProxy:
    def __init__(self, trace: _CompileTrace):
        self.trace = trace

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.trace.compile(*args, **kwargs)

    def __getitem__(self, options: Any) -> Any:
        return self.trace.compile_item(options)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.trace._original, name)


class _CompileItem:
    def __init__(self, trace: _CompileTrace, options: Any):
        self.trace = trace
        self.options = options

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        call_kwargs = dict(kwargs)
        return self.trace.compile_typed(self.options, *args, **call_kwargs)


class _CompiledCall:
    def __init__(self, trace: _CompileTrace, compiled: Any):
        self.trace = trace
        self.compiled = compiled
        self.artifact_id = f"artifact-{trace._artifact_count - 1:04d}"

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.trace.invoke(self.compiled, self.artifact_id, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.compiled, name)


def install_compile_trace(cute: Any, torch: Any, output_dir: Path) -> _CompileTrace:
    """Install a reversible observer around ``cute.compile`` (profile pass only)."""
    return _CompileTrace(cute, torch, Path(output_dir))
