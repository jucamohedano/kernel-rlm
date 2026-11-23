"""Controller-side isolation and limited child-call transport for the pinned native worker.

Use a disposable supported Linux VM. These are per-attempt caps, not aggregate
VM cgroup policy, and do not protect against kernel or interpreter vulnerabilities.
The trusted Harness-CG controller still owns model choice and provider accounting.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import signal
import tempfile
from contextlib import ExitStack, suppress
from hashlib import sha256
from importlib import import_module
from pathlib import Path
from typing import Any

WORKER_SHA256 = "6321bce3e54f18da21a76210646477cc7c49cdf5d1ab6b9e7bd56748b1b90f9c"
BWRAP = Path("/usr/bin/bwrap")
SYSTEM_LIBS = Path("/usr/lib/x86_64-linux-gnu")
LOADER = Path("/lib64/ld-linux-x86-64.so.2")
SYSTEM_FILES = (
    "libc.so.6",
    "libdl.so.2",
    "libm.so.6",
    "libpthread.so.0",
    "librt.so.1",
    "libutil.so.1",
    "libseccomp.so.2",
)
FRAME_BYTES = 128 * 1024  # also the bound on the local relay's request body
PROMPT_BYTES = 32 * 1024
REPLY_BYTES = 512 * 1024
NATIVE_BYTES = 2 * 1024 * 1024
BATCH_COUNT = 16
CHILD_TIMEOUT = 90.0


class SandboxError(RuntimeError):
    """Sandbox setup, timeout, or JSONL transport failure; no fallback."""


def json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, allow_nan=False).encode("ascii")


class ChildGate:
    """One Unix socket, one rollout, two operations, one upstream request at a time."""

    def __init__(self, url: str, rollout_id: str, cap: int):
        port = re.fullmatch(r"http://127\.0\.0\.1:([0-9]{1,5})", url)
        if port is None or not 1 <= int(port[1]) <= 65535:
            raise ValueError(
                "Harness-CG proxy must be an IPv4 loopback HTTP origin with explicit port"
            )
        if re.fullmatch(r"[A-Za-z0-9_-]{1,64}", rollout_id) is None:
            raise ValueError("invalid rollout id")
        self.url = url
        self.prefix = f"/rollout/{rollout_id}/"
        self.cap = cap
        self.allowed = False
        self.counts = dict(
            reserved_prompts=0,
            completed_prompts=0,
            completed_requests=0,
            rejected_requests=0,
            upstream_errors=0,
        )
        self.connection: asyncio.Task | None = None
        self.writers: set[asyncio.StreamWriter] = set()
        self.session = None
        self.server: asyncio.AbstractServer | None = None

    def tally(self, key: str, count: int = 1) -> None:
        self.counts[key] = min(self.counts[key] + count, 2**31 - 1)

    async def start(self, socket_path: Path) -> None:
        aiohttp = import_module("aiohttp")
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=CHILD_TIMEOUT),
            trust_env=False,
            auto_decompress=False,
            skip_auto_headers={"Accept-Encoding"},
            cookie_jar=aiohttp.DummyCookieJar(),
        )
        self.server = await asyncio.start_unix_server(
            self.accept, path=socket_path, backlog=4, limit=FRAME_BYTES + 1
        )

    def accept(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        if not self.allowed or self.connection is not None or len(self.writers) >= 1:
            self.tally("rejected_requests")
            writer.transport.abort()
            return
        self.writers.add(writer)
        self.connection = asyncio.create_task(self.handle(reader, writer))

    def reject(self, message: str) -> dict:
        self.tally("rejected_requests")
        return {"error": message}

    def parse(self, frame: Any) -> tuple[str, str, list[str]] | None:
        if not isinstance(frame, dict) or set(frame) != {"path", "body"}:
            return None
        path, body = frame["path"], frame["body"]
        if (
            not isinstance(body, dict)
            or not isinstance(body.get("depth"), int)
            or isinstance(body.get("depth"), bool)
            or body["depth"] != 1
        ):
            return None
        if path == self.prefix + "llm_query" and set(body) == {"prompt", "depth"}:
            key, prompts = "prompt", [body["prompt"]]
        elif path == self.prefix + "llm_query_batched" and set(body) == {"prompts", "depth"}:
            key, prompts = "prompts", body["prompts"]
        else:
            return None
        if not isinstance(prompts, list) or not 1 <= len(prompts) <= BATCH_COUNT:
            return None
        if not all(isinstance(p, str) and len(p.encode("utf-8")) <= PROMPT_BYTES for p in prompts):
            return None
        return path, key, prompts

    async def forward(self, frame: Any) -> dict:
        parsed = self.parse(frame)
        if parsed is None:
            return self.reject("invalid child-call request")
        path, key, prompts = parsed
        if self.counts["reserved_prompts"] + len(prompts) > self.cap:
            return self.reject("sub-call budget exhausted")
        self.tally("reserved_prompts", len(prompts))
        assert self.session is not None
        client_error = import_module("aiohttp").ClientError
        try:
            async with self.session.post(
                self.url + path,
                json={key: prompts[0] if key == "prompt" else prompts, "depth": 1},
                allow_redirects=False,
            ) as response:
                if response.status == 413:
                    self.tally("upstream_errors")
                    return {"error": "native proxy request-size limit", "too_large": True}
                if response.status != 200:
                    raise ValueError("unexpected upstream status")
                data = bytearray()
                async for chunk in response.content.iter_chunked(16 * 1024):
                    if len(data) + len(chunk) > REPLY_BYTES:
                        raise ValueError("upstream reply exceeded byte limit")
                    data.extend(chunk)
                reply = json.loads(data)
            if not isinstance(reply, dict):
                raise ValueError("invalid native reply")
            if isinstance(reply.get("error"), str) and reply["error"]:
                self.tally("upstream_errors")
                return {
                    "error": "native child call failed",
                    "too_large": reply.get("too_large") is True,
                }
            if key == "prompt" and isinstance(reply.get("response"), str):
                result, done = {"response": reply["response"]}, 1
            elif key == "prompts" and isinstance(reply.get("responses"), list):
                responses = reply["responses"]
                if len(responses) != len(prompts) or not all(isinstance(r, str) for r in responses):
                    raise ValueError("invalid native batch reply")
                # Pinned proxy represents individual failures as text in ordered batches.
                # Don't expose arbitrary exception details from the trusted proxy.
                result = {
                    "responses": [
                        "Error: native child call failed" if text.startswith("Error:") else text
                        for text in responses
                    ]
                }
                done = sum(not text.startswith("Error:") for text in responses)
            else:
                raise ValueError("invalid native reply shape")
            if len(json_bytes(result)) > REPLY_BYTES:
                raise ValueError("forwarded reply exceeded byte limit")
        except (client_error, OSError, ValueError, TimeoutError, RecursionError):
            self.tally("upstream_errors")
            return {"error": "native child-call transport failed"}
        self.tally("completed_requests")
        self.tally("completed_prompts", done)
        return result

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            try:
                async with asyncio.timeout(2):
                    size = int.from_bytes(await reader.readexactly(4), "big")
                    if not 1 <= size <= FRAME_BYTES:
                        raise ValueError("invalid frame size")
                    frame = json.loads(await reader.readexactly(size))
                result = await self.forward(frame)
            except (
                TimeoutError,
                ValueError,
                UnicodeError,
                RecursionError,
                asyncio.IncompleteReadError,
            ):
                result = self.reject("invalid child-call frame")
            data = json_bytes(result)
            writer.write(len(data).to_bytes(4, "big") + data)
            await asyncio.wait_for(writer.drain(), 2)
        except (OSError, TimeoutError):
            pass
        finally:
            writer.transport.abort()
            self.writers.discard(writer)
            self.connection = None

    async def close(self) -> None:
        self.allowed = False
        if self.server is not None:
            self.server.close()
        if self.connection is not None:
            connection = self.connection
            connection.cancel()
            await asyncio.gather(connection, return_exceptions=True)
        for writer in self.writers:
            writer.transport.abort()
        if self.server is not None:
            await self.server.wait_closed()
        if self.session is not None:
            await self.session.close()


class SandboxedReplBackend:
    """One single-use, depth-1 backend implementing the native Harness-CG interface."""

    def __init__(
        self, worker_path: Path, python_runtime: Path, *, max_sub_calls: int, wall_seconds: float
    ):
        if (
            not isinstance(max_sub_calls, int)
            or isinstance(max_sub_calls, bool)
            or not 0 <= max_sub_calls <= 1024
        ):
            raise ValueError("max_sub_calls must be between 0 and 1024 (0 disables the gate)")
        if not math.isfinite(wall_seconds) or wall_seconds <= 0:
            raise ValueError("finite positive wall seconds are required")
        with Path(worker_path).open("rb") as worker_file:
            self.worker = worker_file.read(21461)
        if sha256(self.worker).hexdigest() != WORKER_SHA256:
            raise ValueError("Harness-CG worker identity mismatch")
        if os.geteuid() == 0 or not BWRAP.is_file() or BWRAP.stat().st_mode & 0o6000:
            raise SandboxError("rootless non-setuid /usr/bin/bwrap is required")
        self.runtime = Path(python_runtime).resolve(strict=True)
        if (
            not (self.runtime / "bin/python3.13").is_file()
            or not (self.runtime / "lib/python3.13").is_dir()
        ):
            raise SandboxError("a clean standalone CPython 3.13 runtime directory is required")
        self.cap = max_sub_calls
        self.wall_seconds = wall_seconds
        self.process: asyncio.subprocess.Process | None = None
        self.gate: ChildGate | None = None
        self.temp: tempfile.TemporaryDirectory | None = None
        self.stderr_task: asyncio.Task | None = None
        self.watchdog: asyncio.Task | None = None
        self.stderr_tail = b""
        self.lock = asyncio.Lock()
        self.read_lock = asyncio.Lock()
        self.stop_lock = asyncio.Lock()
        self.req = 0
        self.deadline = 0.0

    @property
    def broker_evidence(self) -> dict[str, int]:
        return {} if self.gate is None else dict(self.gate.counts)

    def command(self, worker_fd: int, socket_path: Path, rollout_id: str) -> list[str]:
        argv = [
            str(BWRAP),
            "--unshare-all",
            "--unshare-user",
            "--disable-userns",
            "--cap-drop",
            "ALL",
            "--die-with-parent",
            "--new-session",
            "--clearenv",
            "--size",
            str(64 * 1024**2),
            "--tmpfs",
            "/tmp",
            "--dev",
            "/dev",
            "--proc",
            "/proc",
            "--remount-ro",
            "/dev",
            "--remount-ro",
            "/proc",
            "--symlink",
            "usr/lib",
            "/lib",
        ]
        for src, dst in [
            (self.runtime / "bin/python3.13", "/runtime/bin/python3.13"),
            (self.runtime / "lib/libpython3.13.so.1.0", "/runtime/lib/libpython3.13.so.1.0"),
            (self.runtime / "lib/python3.13", "/runtime/lib/python3.13"),
            (LOADER, str(LOADER)),
            *((SYSTEM_LIBS / name, str(SYSTEM_LIBS / name)) for name in SYSTEM_FILES),
            (Path(__file__).with_name("sandbox_worker.py"), "/bootstrap.py"),
            (socket_path, "/broker.sock"),
        ]:
            argv.extend(("--ro-bind", str(src), dst))
        argv.extend(
            (
                "--size",
                "4096",
                "--tmpfs",
                "/runtime/lib/python3.13/site-packages",
                "--remount-ro",
                "/runtime/lib/python3.13/site-packages",
                "--ro-bind-data",
                str(worker_fd),
                "/worker.py",
                "--remount-ro",
                "/",
                "--chdir",
                "/tmp",
                "--setenv",
                "TMPDIR",
                "/tmp",
                "--",
                "/runtime/bin/python3.13",
                "-I",
                "-S",
                "-B",
                "/bootstrap.py",
                rollout_id,
            )
        )
        return argv

    async def start(self, proxy_url: str, rollout_id: str, depth: int = 1) -> None:
        if (
            self.gate is not None
            or not isinstance(depth, int)
            or isinstance(depth, bool)
            or depth != 1
        ):
            raise SandboxError("a fresh depth-1 REPL backend is required")
        self.gate = ChildGate(proxy_url, rollout_id, self.cap)
        loop = asyncio.get_running_loop()
        self.deadline = loop.time() + self.wall_seconds
        self.temp = tempfile.TemporaryDirectory(dir="/tmp", prefix="rg-repl-")
        socket_path = Path(self.temp.name) / "broker.sock"
        try:
            async with asyncio.timeout_at(min(self.deadline, loop.time() + 10)):
                await self.gate.start(socket_path)
                with ExitStack() as stack:
                    worker_data = stack.enter_context(tempfile.TemporaryFile())
                    worker_data.write(self.worker)
                    worker_data.seek(0)
                    fd = worker_data.fileno()
                    self.process = await asyncio.create_subprocess_exec(
                        *self.command(fd, socket_path, rollout_id),
                        env={},
                        start_new_session=True,
                        stdin=asyncio.subprocess.PIPE,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        pass_fds=(fd,),
                        limit=NATIVE_BYTES,
                        cwd=self.temp.name,
                    )
                self.stderr_task = asyncio.create_task(self.drain_stderr(self.process))
                await self.read_reply("_init")
            self.watchdog = asyncio.create_task(self.expire())
        except BaseException as ex:
            await self.stop()
            if isinstance(ex, asyncio.CancelledError):
                raise
            detail = self.stderr_tail.decode(errors="replace")[-1000:]
            raise SandboxError(f"isolated worker failed to start: {ex}; stderr={detail!r}") from ex

    async def expire(self) -> None:
        await asyncio.sleep(max(0.0, self.deadline - asyncio.get_running_loop().time()))
        await self.stop()

    async def drain_stderr(self, process: asyncio.subprocess.Process) -> None:
        assert process.stderr is not None
        while chunk := await process.stderr.read(4096):
            self.stderr_tail = (self.stderr_tail + chunk)[-4096:]

    async def read_reply(self, request_id: str) -> dict:
        process = self.process
        if process is None or process.stdout is None:
            raise SandboxError("isolated worker is not running")
        async with self.read_lock:
            line = await process.stdout.readuntil(b"\n")
        result = json.loads(line)
        if (
            not isinstance(result, dict)
            or result.get("id") != request_id
            or result.get("ok") is not True
        ):
            raise SandboxError("isolated worker returned an invalid native reply")
        return result

    async def request(self, kind: str, **payload) -> dict:
        async with self.lock:
            process = self.process
            gate = self.gate
            if process is None or process.stdin is None or gate is None:
                raise SandboxError("isolated worker is not running")
            self.req += 1
            request_id = f"r{self.req}"
            line = json_bytes({"id": request_id, "type": kind, **payload}) + b"\n"
            if len(line) > NATIVE_BYTES:
                raise SandboxError("native request exceeded byte limit")
            gate.allowed = kind == "exec" and self.cap > 0
            try:
                async with asyncio.timeout_at(self.deadline):
                    process.stdin.write(line)
                    await process.stdin.drain()
                    return await self.read_reply(request_id)
            except (Exception, asyncio.CancelledError) as ex:
                await self.stop()
                if isinstance(ex, asyncio.CancelledError):
                    raise
                raise SandboxError("isolated worker protocol failure or deadline exceeded") from ex
            finally:
                gate.allowed = False

    async def load_context(self, payload: Any, index: int | None = None) -> int:
        result = await self.request("load_context", payload=payload, index=index)
        return int(result["index"])

    async def execute(self, code: str):
        result = await self.request("exec", code=code)
        types = import_module("rlm_train.repl.base")
        return types.ExecResult(
            **{
                name: result[name]
                for name in types.ExecResult.__dataclass_fields__
                if name in result
            }
        )

    async def stop(self) -> None:
        async with self.stop_lock:
            timer, self.watchdog = self.watchdog, None
            if timer is not None and timer is not asyncio.current_task():
                timer.cancel()
            process, self.process = self.process, None
            if process is not None:
                if process.stdin is not None:
                    process.stdin.close()
                if process.returncode is None:
                    with suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGKILL)
            if self.gate is not None:
                await self.gate.close()
            if process is not None:
                # Drain unread pipe data so asyncio's process wait can finish after a flood.
                if process.stdout is not None:
                    async with self.read_lock:
                        while await process.stdout.read(65536):
                            pass
                await process.wait()
            task, self.stderr_task = self.stderr_task, None
            if task is not None:
                await task
            if self.temp is not None:
                self.temp.cleanup()
                self.temp = None
