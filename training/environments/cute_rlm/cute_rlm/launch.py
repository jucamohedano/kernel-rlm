"""Explicit GPU-host launch entry point for a previously frozen CuTe evaluation."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import shlex
import sys
from pathlib import Path

import httpx

from cute_rlm.config import EvalConfig, check_gpu_receipt
from cute_rlm.evaluate import evaluate_prepared, isolated_backend_factory


def check_evidence_files(receipt: dict, root: Path) -> None:
    """Keep readiness evidence local, hash-pinned and outside candidate writes."""
    root = root.resolve()
    for name, check in receipt.get("checks", {}).items():
        evidence = check.get("evidence")
        if not isinstance(evidence, dict):
            raise ValueError(f"Readiness check {name} needs a pinned evidence file")
        relative = Path(evidence["path"])
        path = (root / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(root):
            raise ValueError("GPU readiness evidence escapes its declared directory")
        if hashlib.sha256(path.read_bytes()).hexdigest() != evidence.get("sha256"):
            raise ValueError(f"Readiness evidence changed: {name}")


def write_repl_wrapper(path: Path, spec_path: Path, python: str) -> None:
    """A private host-created executable, never supplied by generated model code."""
    if path.exists():
        raise FileExistsError(path)
    command = [python, "-m", "cute_rlm.sandbox", "--spec", str(spec_path), "--repl"]
    package_root = str(Path(__file__).resolve().parents[1])
    payload = (
        "#!/bin/sh\nPYTHONPATH="
        + shlex.quote(package_root)
        + "\nexport PYTHONPATH\nexec "
        + shlex.join(command)
        + ' -- "$@"\n'
    )
    with path.open("x") as stream:
        stream.write(payload)
    path.chmod(0o700)


async def run_prepared(
    directory: Path,
    *,
    base_url: str,
    receipt_path: Path,
    sandbox_spec: Path,
    observer_root: Path,
    host_python: str = sys.executable,
    api_key: str | None = None,
) -> dict:
    from cute_rlm.observer import NsightObserver
    from cute_rlm.sandbox import SandboxSpec
    from cute_rlm.server_identity import check_server_identity

    directory, sandbox_spec = directory.resolve(), sandbox_spec.resolve()
    manifest = json.loads((directory / "provenance.json").read_text())
    config = EvalConfig(**manifest["config"])
    receipt = json.loads(receipt_path.read_text())
    check_gpu_receipt(receipt, config, manifest["manifest_sha256"])
    check_evidence_files(receipt, receipt_path.parent)
    spec_bytes = sandbox_spec.read_bytes()
    if receipt.get("sandbox_spec_sha256") != hashlib.sha256(spec_bytes).hexdigest():
        raise ValueError("Readiness evidence does not bind this sandbox specification")
    spec = SandboxSpec.read(sandbox_spec)
    spec.validate_host()
    if spec.observer_root is None or Path(spec.observer_root).resolve() != observer_root.resolve():
        raise ValueError("Observer root must match the protected collector path in sandbox spec")
    observer = NsightObserver(observer_root.resolve())
    wrapper = directory / "repl-python"
    write_repl_wrapper(wrapper, sandbox_spec, host_python)
    command = [host_python, "-m", "cute_rlm.sandbox", "--spec", str(sandbox_spec)]
    async with httpx.AsyncClient(
        base_url=base_url.rstrip("/") + "/",
        timeout=config.request_timeout,
        headers={"Authorization": "Bearer " + (api_key or "EMPTY")},
    ) as client:
        await check_server_identity(client, config, receipt)
        return await evaluate_prepared(
            directory,
            client=client,
            worker_command=command,
            backend_factory=isolated_backend_factory(config, str(wrapper)),
            gpu_receipt=receipt,
            cute_observer=observer,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--sandbox-spec", type=Path, required=True)
    parser.add_argument("--observer-root", type=Path, required=True)
    parser.add_argument("--host-python", default=sys.executable)
    args = parser.parse_args()
    result = asyncio.run(
        run_prepared(
            args.directory,
            base_url=args.base_url,
            receipt_path=args.receipt,
            sandbox_spec=args.sandbox_spec,
            observer_root=args.observer_root,
            host_python=args.host_python,
            api_key=os.environ.get("OPENAI_API_KEY"),
        )
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
