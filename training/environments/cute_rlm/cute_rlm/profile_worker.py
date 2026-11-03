"""Extra in-namespace Nsight pass; no timing from this pass is scored.

Profiler artifacts share the candidate namespace and are observational rather than
malicious-code-resistant. The outer host copies and validates them after exit.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def main():
    request = json.load(sys.stdin)
    output = Path(request["output_dir"])
    profile = output / "nsight"
    profile.mkdir(parents=True, exist_ok=False)
    binary = request["nsys_inside"]
    child_request = {**request, "output_dir": str(output / "worker")}
    command = [
        binary,
        "profile",
        "--trace=cuda-sw,nvtx",
        "--sample=none",
        "--discard-environment=true",
        "--force-overwrite=true",
        "--output=" + str(profile / "trace"),
        sys.executable,
        "-m",
        "cute_rlm.verify_worker",
    ]
    result = subprocess.run(
        command,
        input=json.dumps(child_request).encode(),
        capture_output=True,
        timeout=request["config"]["timeout_s"],
    )
    (profile / "worker-stdout.txt").write_bytes(result.stdout)
    (profile / "worker-stderr.txt").write_bytes(result.stderr)
    if result.returncode:
        sys.stderr.buffer.write(result.stderr)
        return result.returncode
    export = subprocess.run(
        [
            binary,
            "export",
            "--type",
            "sqlite",
            "--output",
            str(profile / "trace.sqlite"),
            str(profile / "trace.nsys-rep"),
        ],
        capture_output=True,
        timeout=request["config"]["timeout_s"],
    )
    if export.returncode:
        sys.stderr.buffer.write(export.stderr)
        return export.returncode
    print(json.dumps({"status": "profiled", "scope": "in-namespace-observational"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
