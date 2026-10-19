"""Keep the private Trackio SSH tunnel connected without touching training."""

import argparse
import fcntl
import json
import os
import signal
import socket
import subprocess
import time
from pathlib import Path


def ssh_command(
    host: str, user: str, ssh_port: int, local_port: int, remote_port: int
) -> list[str]:
    if any(not 1 <= port <= 65535 for port in (ssh_port, local_port, remote_port)):
        raise ValueError("Ports must be between 1 and 65535")
    return [
        "ssh",
        "-N",
        "-o",
        "BatchMode=yes",
        "-o",
        "ExitOnForwardFailure=yes",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "ServerAliveCountMax=6",
        "-o",
        "ConnectTimeout=15",
        "-L",
        f"127.0.0.1:{local_port}:127.0.0.1:{remote_port}",
        "-p",
        str(ssh_port),
        f"{user}@{host}",
    ]


def stop_requested(signal_number, frame):
    raise KeyboardInterrupt()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--user", default="riftuser")
    parser.add_argument("--ssh-port", type=int, default=60008)
    parser.add_argument("--local-port", type=int, default=7861)
    parser.add_argument("--remote-port", type=int, default=7861)
    parser.add_argument("--state-dir", type=Path, default=Path("logs/runtime"))
    arguments = parser.parse_args()
    command = ssh_command(
        arguments.host,
        arguments.user,
        arguments.ssh_port,
        arguments.local_port,
        arguments.remote_port,
    )
    arguments.state_dir.mkdir(parents=True, exist_ok=True)
    with (arguments.state_dir / "trackio-tunnel.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Trackio tunnel supervisor is already running.", flush=True)
            return
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", arguments.local_port)) == 0:
                print("Local port is already in use; refusing a duplicate tunnel.", flush=True)
                return
        state = {
            "pid": os.getpid(),
            "command": command,
            "local_url": f"http://127.0.0.1:{arguments.local_port}/",
            "training_control": False,
        }
        (arguments.state_dir / "trackio-tunnel.json").write_text(json.dumps(state, indent=2))
        signal.signal(signal.SIGTERM, stop_requested)
        process = None
        try:
            while True:
                process = subprocess.Popen(command, stdin=subprocess.DEVNULL)
                result = process.wait()
                print(f"SSH tunnel exited {result}; retrying in 15 seconds.", flush=True)
                time.sleep(15)
        except KeyboardInterrupt:
            pass
        finally:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


if __name__ == "__main__":
    main()
