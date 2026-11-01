"""Plain system/user prompts, with frozen, locally bundled source packets only."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from cute_rlm.icl.tasks import get_task

CONDITIONS = ("task_only", "api", "api_example")
ASSET_DIRECTORY = Path(__file__).parent / "assets"
SYSTEM = (
    "Write a standalone Python module defining torch.nn.Module class ModelNew. "
    "Implement the requested computation in Python CuTe DSL. "
    "Return only Python code, plain or in one python code fence."
)


def source_ledger() -> dict:
    """The provenance and license record is experiment metadata, not an API tool."""
    return json.loads((ASSET_DIRECTORY / "sources.json").read_text())


def _source(asset_id: str, ledger: dict) -> dict:
    record = next(entry for entry in ledger["assets"] if entry["id"] == asset_id)
    data = (ASSET_DIRECTORY / record["path"]).read_bytes()
    if hashlib.sha256(data).hexdigest() != record["sha256"]:
        raise ValueError(f"ICL packet asset digest mismatch: {asset_id}")
    result = dict(record, content=data.decode("utf-8"))
    if record["kind"] == "api":
        result["upstream_sources"] = ledger["upstream_api_sources"]
        result["sufficiency"] = ledger["packet_sufficiency"]
    if record["kind"] == "example":
        result["origin"] = ledger["example_source"]
    result["license_review"] = ledger["license_review"]
    return result


def render_prompt(task: dict, condition: str) -> dict:
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown ICL condition: {condition}")
    # Reject unknown operations, but use the frozen public spec supplied by prepare.
    get_task(task["id"])
    interface = task["interface"]
    public_cases = [
        {key: value for key, value in case.items() if key != "seed"} for case in task["cases"]
    ]
    user = (
        f"Task: {task['id']}\n{task['description']}\n\n"
        "Target: Python CuTe DSL 4.8.1 on NVIDIA L40S (SM89), CUDA GPU inputs.\n"
        f"Interface: class ModelNew(torch.nn.Module), {interface['constructor_signature']}, "
        f"{interface['forward_signature']}. No constructor arguments or state.\n"
        f"Inputs: {interface['inputs']}\nOutput: {interface['output']}\n"
        "Do not mutate inputs. Input values are finite pseudorandom FP32 samples.\n"
        f"Public test domain (dimensions, strides in elements, and Python scalar values): "
        f"{json.dumps(public_cases, sort_keys=True)}\n"
        f"Numerical tolerance: atol={task['atol']}, rtol={task['rtol']}.\n\n"
        "All required arithmetic must execute in an output-producing CuTe device kernel. "
        "Torch is allowed only for allocation, metadata and tensor/stream plumbing, "
        "not for computing any part of the answer. No Triton, C++ extensions or torch.compile. "
        "No timing requirement. Simple scalar SIMT kernels are allowed. "
        "The evaluator synchronizes before and after forward; no autograd is required. "
        "Implement all stated cases, not just one shape. "
        "There are no tools, external files, runtime context variables or compiler feedback.\n"
    )
    sources = []
    if condition != "task_only":
        ledger = source_ledger()
        sources = [_source("api-common.txt", ledger), _source(f"api-{task['family']}.txt", ledger)]
        user += "\nAPI packet (provided in full):\n" + "\n".join(s["content"] for s in sources)
        if condition == "api_example":
            example = _source("generic-2d-add.py.txt", ledger)
            sources.append(example)
            user += (
                "\nRelated worked example: generic 2D addition of two equally shaped contiguous "
                "tensors. It is not the target operation. The example uses a historically "
                "validated local adapter; adapt specialization/caching to this task's cases. "
                "It is a close arithmetic pattern for affine/row-bias tasks, so this arm does "
                "not measure transfer to unrelated algorithms.\n\n" + example["content"]
            )
    return {"system": SYSTEM, "user": user, "sources": sources}
