"""Package pinned curricula for the Hub without changing legacy JSONL inputs."""

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

CONFIG_FILES = {
    "lvl1_seed42": "lvl1_seed42.jsonl",
    "lvl1_seed42_easy_first": "lvl1_seed42_easy_first.jsonl",
    "lvl2_seed42": "lvl2_seed42.jsonl",
    "lvl3_seed42": "lvl3_seed42.jsonl",
}
UPSTREAM = "BytedTsinghua-SIA/CUDA-Agent-Ops-6K"


def digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()


def read_rows(content: bytes) -> list[dict]:
    rows = [json.loads(line) for line in content.decode("utf-8").splitlines() if line.strip()]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError("Curriculum must contain task objects")
    identities = [row.get("_task_id") for row in rows]
    if any(not isinstance(identity, str) or not identity for identity in identities):
        raise ValueError("Every task requires a nonempty string _task_id")
    if len(set(identities)) != len(identities):
        raise ValueError("Task IDs must be unique within each configuration")
    if any(not isinstance(row.get("code"), str) or not row["code"].strip() for row in rows):
        raise ValueError("Every task requires reference source")
    return rows


def build_release(source_dir: Path, output_dir: Path) -> dict:
    from datasets import Dataset

    inputs = {
        config: (source_dir / filename).read_bytes() for config, filename in CONFIG_FILES.items()
    }
    rows_by_config = {config: read_rows(content) for config, content in inputs.items()}
    original_l1 = sorted(rows_by_config["lvl1_seed42"], key=lambda row: row["_task_id"])
    ordered_l1 = sorted(rows_by_config["lvl1_seed42_easy_first"], key=lambda row: row["_task_id"])
    if original_l1 != ordered_l1:
        raise ValueError("L1 orderings must contain exactly the same task records")
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": 1,
        "upstream_repo": UPSTREAM,
        "upstream_revision": None,
        "identity_scope": "Exact preserved materializations; upstream revision was not recorded",
        "configs": {},
    }
    configurations = []
    for config, filename in CONFIG_FILES.items():
        rows = rows_by_config[config]
        original = output_dir / "original" / filename
        original.parent.mkdir(exist_ok=True)
        original.write_bytes(inputs[config])
        parquet = output_dir / "data" / config / "train.parquet"
        parquet.parent.mkdir(parents=True)
        Dataset.from_list(rows).to_parquet(str(parquet))
        if Dataset.from_parquet(str(parquet)).to_list() != rows:
            raise ValueError(f"Parquet roundtrip changed {config}")
        manifest["configs"][config] = {
            "rows": len(rows),
            "jsonl_path": str(original.relative_to(output_dir)),
            "jsonl_sha256": digest(inputs[config]),
            "parquet_path": str(parquet.relative_to(output_dir)),
            "parquet_sha256": digest(parquet.read_bytes()),
            "ordered_task_ids_sha256": digest(canonical([row["_task_id"] for row in rows])),
            "rows_sha256": digest(canonical(rows)),
        }
        configurations.append(
            f"  - config_name: {config}\n"
            f"    data_files:\n      - split: train\n        path: data/{config}/train.parquet\n"
        )
    card = (
        "---\nlicense: cc-by-4.0\nlanguage:\n  - en\n"
        "pretty_name: Kernel RLM preserved CUDA curriculum\nconfigs:\n"
        + "".join(configurations)
        + "---\n\n# Kernel RLM CUDA curriculum\n\n"
        f"Derived from [{UPSTREAM}](https://huggingface.co/datasets/{UPSTREAM}), "
        "whose dataset card declares CC-BY-4.0. Attribute the original dataset and its authors.\n\n"
        "These configurations preserve the selected seed42 task references, IDs, row order and "
        "all original columns. Changes from upstream are task selection, curriculum ordering, "
        "and Parquet packaging. L1 is single-op, L2 two-op, and L3 three-to-five-op under the "
        "original builder's ops-column classification, not KernelBench levels.\n\n"
        "The easy-first and original L1 configurations are alternate orders of the same selected "
        "tasks, not independent train/test sets. All configurations are intended training inputs; "
        "not all selected tasks received updates. No held-out evaluation or performance gain is claimed.\n\n"
        "The upstream download revision was not recorded and is deliberately unknown. Exact "
        "historical JSONL bytes and hashes are retained under original/ and manifest.json. "
        "Do not regenerate the historical curriculum from a mutable upstream revision.\n\n"
        "Use datasets.load_dataset(repo_id, name=config, split='train', revision=commit_sha) "
        "for Arrow/Parquet access. Existing training CLIs can consume an exact original JSONL "
        "materialized with python -m kernel_rlm.curriculum download.\n\n"
        "This release contains reference programs, not sampled model responses, reasoning traces, "
        "checkpoint weights, credentials or private experiment logs. Review attribution and "
        "publication contents before changing a private repository to public.\n"
    )
    (output_dir / "README.md").write_text(card)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def materialize(
    repo_id: str, config: str, revision: str, expected_sha256: str, output: Path
) -> Path:
    if config not in CONFIG_FILES:
        raise ValueError("Unknown curriculum configuration")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Pin a full immutable Hub commit SHA,not main or a tag")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("An expected original JSONL SHA256 is required")
    if output.exists():
        content = output.read_bytes()
        if digest(content) != expected_sha256:
            raise ValueError("Refusing to overwrite a different existing curriculum")
        read_rows(content)
        return output
    from huggingface_hub import hf_hub_download

    manifest_path = hf_hub_download(
        repo_id, "manifest.json", repo_type="dataset", revision=revision
    )
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest["schema_version"] != 1:
        raise ValueError("Unsupported curriculum manifest schema")
    record = manifest["configs"][config]
    filename = "original/" + CONFIG_FILES[config]
    if record["jsonl_path"] != filename or record["jsonl_sha256"] != expected_sha256:
        raise ValueError("Remote curriculum identity differs from the expected original")
    source = Path(hf_hub_download(repo_id, filename, repo_type="dataset", revision=revision))
    content = source.read_bytes()
    if digest(content) != expected_sha256:
        raise ValueError("Downloaded curriculum hash differs")
    rows = read_rows(content)
    if (
        len(rows) != record["rows"]
        or digest(canonical(rows)) != record["rows_sha256"]
        or digest(canonical([row["_task_id"] for row in rows])) != record["ordered_task_ids_sha256"]
    ):
        raise ValueError("Downloaded curriculum rows or order differ")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(content)
    try:
        os.link(temporary, output)
    finally:
        temporary.unlink()
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export")
    export.add_argument("--source-dir", type=Path, required=True)
    export.add_argument("--output-dir", type=Path, required=True)
    download = commands.add_parser("download")
    download.add_argument("--repo-id", required=True)
    download.add_argument("--config", choices=CONFIG_FILES, required=True)
    download.add_argument("--revision", required=True)
    download.add_argument("--expected-sha256", required=True)
    download.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.command == "export":
        print(json.dumps(build_release(arguments.source_dir, arguments.output_dir), indent=2))
    else:
        print(
            materialize(
                arguments.repo_id,
                arguments.config,
                arguments.revision,
                arguments.expected_sha256,
                arguments.output,
            )
        )


if __name__ == "__main__":
    main()
