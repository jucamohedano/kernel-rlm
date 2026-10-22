"""Hash the source used by TRL rollout and update receipts."""

import hashlib
from pathlib import Path


def source_hashes(source_root: Path) -> dict[str, str]:
    hashes = {}
    for package in (
        "kernel_trl",
        "kernel_rollout",
        "kernel_training",
        "kernel_rlm",
        "kernel_rlm_v1",
    ):
        for path in sorted((source_root / package).rglob("*.py")):
            hashes[str(path.relative_to(source_root))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    repository_root = source_root.parents[2]
    core_root = repository_root / "rlm"
    if not core_root.is_dir():
        raise ValueError("The source snapshot must include the RLM core package")
    for path in sorted(core_root.rglob("*.py")):
        relative = Path("rlm") / path.relative_to(core_root)
        hashes[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
    training_package = repository_root / "training/src/rlm_train"
    if not training_package.is_dir():
        raise ValueError("The source snapshot must include training/src/rlm_train")
    for path in sorted(training_package.rglob("*.py")):
        relative = Path("training/src/rlm_train") / path.relative_to(training_package)
        hashes[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes
