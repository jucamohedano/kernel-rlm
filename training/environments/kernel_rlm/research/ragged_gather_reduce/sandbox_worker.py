"""Guest-only bootstrap; never import or run this as a controller module."""

from __future__ import annotations

import ctypes
import errno
import json
import os
import resource
import socket
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

FRAME_BYTES = 128 * 1024
REPLY_BYTES = 512 * 1024
BROKER_TIMEOUT = 92
PYTHON = "/runtime/bin/python3.13"


class ArgCompare(ctypes.Structure):
    _fields_ = [
        ("arg", ctypes.c_uint),
        ("op", ctypes.c_int),
        ("first", ctypes.c_uint64),
        ("second", ctypes.c_uint64),
    ]


def restrict() -> None:
    """Allow private IPv4 and Unix sockets, but no other socket families or new tasks."""
    library = ctypes.CDLL("/usr/lib/x86_64-linux-gnu/libseccomp.so.2")
    library.seccomp_init.argtypes = (ctypes.c_uint32,)
    library.seccomp_init.restype = ctypes.c_void_p
    library.seccomp_syscall_resolve_name.argtypes = (ctypes.c_char_p,)
    library.seccomp_rule_add_array.argtypes = (
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_int,
        ctypes.c_uint,
        ctypes.POINTER(ArgCompare),
    )
    library.seccomp_load.argtypes = (ctypes.c_void_p,)
    library.seccomp_release.argtypes = (ctypes.c_void_p,)
    policy = library.seccomp_init(0x7FFF0000)  # SCMP_ACT_ALLOW
    if not policy:
        raise RuntimeError("could not initialize sandbox seccomp")
    try:
        rules = [
            (name, None)
            for name in (
                b"clone",
                b"clone3",
                b"fork",
                b"vfork",
                b"setns",
                b"unshare",
                b"io_uring_setup",
            )
        ] + [
            (b"socket", ArgCompare(0, 6, int(socket.AF_INET), 0)),  # SCMP_CMP_GT
            (b"socket", ArgCompare(0, 2, int(socket.AF_UNIX), 0)),  # SCMP_CMP_LT
        ]
        for name, comparison in rules:
            number = library.seccomp_syscall_resolve_name(name)
            if (
                number < 0
                or library.seccomp_rule_add_array(
                    policy,
                    0x00050000 | errno.EPERM,
                    number,
                    int(comparison is not None),
                    comparison,
                )
                != 0
            ):
                raise RuntimeError(f"could not add sandbox seccomp rule for {name!r}")
        if library.seccomp_load(policy) != 0:
            raise RuntimeError("could not activate sandbox seccomp")
    finally:
        library.seccomp_release(policy)


class Relay(BaseHTTPRequestHandler):
    """The native worker's local HTTP shape, not a general proxy."""

    def setup(self) -> None:
        self.request.settimeout(2)
        super().setup()

    def do_POST(self) -> None:
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 1 <= size <= FRAME_BYTES:
                self.send_error(413)
                return
            body = self.rfile.read(size)
            if len(body) != size:
                self.send_error(400)
                return
            frame = json.dumps(
                {"path": self.path, "body": json.loads(body)}, separators=(",", ":")
            ).encode("ascii")
            if len(frame) > FRAME_BYTES:
                self.send_error(413)
                return
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                connection.settimeout(BROKER_TIMEOUT)
                connection.connect("/broker.sock")
                connection.sendall(len(frame).to_bytes(4, "big") + frame)
                with connection.makefile("rb") as stream:
                    size = int.from_bytes(stream.read(4), "big")
                    if not 1 <= size <= REPLY_BYTES:
                        raise ValueError("invalid broker reply size")
                    result = stream.read(size)
                    if len(result) != size:
                        raise ValueError("incomplete broker reply")
        except (OSError, ValueError, RecursionError):
            self.send_error(502, "child-call gate unavailable or disabled")
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(result)))
        self.end_headers()
        self.wfile.write(result)

    def log_message(self, fmt: str, *args: object) -> None:
        pass


def limits() -> None:
    for key, soft, hard in (
        (resource.RLIMIT_AS, 512 * 1024**2, 512 * 1024**2),
        (resource.RLIMIT_CPU, 20, 20),
        (resource.RLIMIT_FSIZE, 32 * 1024**2, 32 * 1024**2),
        (resource.RLIMIT_NOFILE, 64, 64),
        (resource.RLIMIT_CORE, 0, 0),
    ):
        resource.setrlimit(key, (soft, hard))


def main(rollout_id: str) -> None:
    limits()
    server = HTTPServer(("127.0.0.1", 0), Relay)
    _, port = server.server_address
    ready_read, ready_write = os.pipe()
    if os.fork() == 0:
        os.close(ready_read)
        try:
            restrict()
            with open("/dev/null", "rb+") as null:
                for fd in range(3):
                    os.dup2(null.fileno(), fd)
            os.write(ready_write, b"1")
            os.close(ready_write)
            server.serve_forever()
        finally:
            os._exit(1)
    os.close(ready_write)
    server.server_close()
    ready = os.read(ready_read, 1)
    os.close(ready_read)
    if ready != b"1":
        raise RuntimeError("isolated loopback relay failed to start")
    restrict()
    os.execve(
        PYTHON,
        [PYTHON, "-I", "-S", "-B", "-u", "/worker.py"],
        {
            "TMPDIR": "/tmp",
            "RLM_TRAIN_PROXY_URL": f"http://127.0.0.1:{port}",
            "RLM_TRAIN_ROLLOUT_ID": rollout_id,
            "RLM_TRAIN_DEPTH": "1",
            "RLM_TRAIN_REPL_MAX_MEM_GB": "0.5",
            "RLM_TRAIN_SUBCALL_TIMEOUT_S": str(BROKER_TIMEOUT + 1),
        },
    )


if __name__ == "__main__":
    main(sys.argv[1])
