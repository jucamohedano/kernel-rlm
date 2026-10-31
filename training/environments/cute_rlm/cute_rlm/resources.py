"""Small indexed REPL resources, captured before dispatch and shared with children."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

from cute_rlm.config import EvalConfig
from cute_rlm.contracts import VerificationConfig

EXAMPLE_SHA256 = "479ab93b9982fa1fb82cec37c6820a5ac64c52610ec8cec80d687d57a0a11fe4"


def example_sections(source: str) -> dict[str, dict[str, int]]:
    """Character offsets into the exact verified source, not rewritten snippets."""
    lines = source.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    wanted = {"ModelNew", "elementwise_add", "elementwise_add_kernel"}
    sections = {}
    nodes = ast.parse(source).body
    imports = []
    for node in nodes:
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            break
        imports.append(node)
    if imports:
        sections["imports"] = {
            "start": offsets[imports[0].lineno - 1],
            "end": offsets[imports[-1].end_lineno],
        }
    for node in nodes:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in wanted:
            start = min([node.lineno, *(d.lineno for d in node.decorator_list)])
            sections[node.name] = {"start": offsets[start - 1], "end": offsets[node.end_lineno]}
    return {
        name: sections[name]
        for name in ("imports", "ModelNew", "elementwise_add", "elementwise_add_kernel")
    }


def build_resources(config: EvalConfig) -> dict:
    source = (Path(__file__).parent / "examples/elementwise_add.py.txt").read_text()
    if hashlib.sha256(source.encode()).hexdigest() != EXAMPLE_SHA256:
        raise ValueError(
            "Generic CuTe example source changed without updated verification provenance"
        )
    return {
        "target": {
            "gpu_model": config.gpu_model,
            "compute_capability": config.gpu_compute_capability,
            "architecture": config.gpu_arch,
            "tensor_parallel_size": config.tensor_parallel_size,
            "cuda_toolkit": config.cuda_toolkit_version,
            "cute_dsl": "4.8.1",
        },
        "validation": {
            key: value
            for key, value in VerificationConfig(**config.verification).payload().items()
            if key != "seed"
        },
        "limits": {
            "server_context_tokens": config.server_context_tokens,
            "output_fraction_of_remaining_context": config.output_fraction,
            "turns_per_session": config.max_turns,
            "generated_tokens_per_tree": config.max_generated_tokens,
            "episode_timeout_s": config.episode_timeout,
            "request_timeout_s": config.request_timeout,
            "model_calls_per_tree": config.max_model_calls,
            "child_admissions": config.max_children,
            "max_depth": config.max_depth,
            "live_children": config.max_live_children,
            "concurrent_inference_requests": config.max_concurrent_requests,
            "note": "Configured ceilings, not live remaining balances; no separate feedback-call quota.",
        },
        "example_index": [
            {
                "id": "elementwise_add",
                "purpose": "Generic 2D elementwise add: Torch allocation, tensor adaptation, cached JIT executor, tiled device launch.",
                "source_sha256": EXAMPLE_SHA256,
                "license": "BSD-3-Clause; NVIDIA notice retained in source",
                "origin": "NVIDIA CUTLASS generic Ampere elementwise-add example with the exact local ModelNew adapter used in the historical verification; not a verbatim upstream file.",
                "sections": example_sections(source),
                "verification": {
                    "run_id": "20261030T200639Z-xs21-cute-l40s-detailed4",
                    "gpu_model": "NVIDIA L40S",
                    "compute_capability": "8.9",
                    "cute_dsl": "4.8.1",
                    "torch": "2.13.0+cu130",
                    "status": "correct_and_CuTe_launch_observed",
                    "correctness_trials": 3,
                    "warmups": 5,
                    "timing_samples": 20,
                    "result_sha256": "311123fec84566584a04aff24c5545e7caa62267007c5b28624ea2f404813fd5",
                    "scope": "Historical generic fixture, not a benchmark task solution or a readiness check on the current VM. FP32 contiguous input fixture only; not universal dtype/layout/stream coverage.",
                },
            }
        ],
        "examples": {"elementwise_add": source},
        "delegation_example": (Path(__file__).parent / "examples/delegation.py.txt").read_text(),
        "api_notes": [
            "Use Python CuTe DSL (introduced in mid-2025): import cutlass.cute as cute; from cutlass.cute.runtime import from_dlpack. This is not the older CUTLASS CuTe C++ API.",
            "Helpers are live injected callables: use compile_candidate, search_docs, read_doc, SHOW_VARS and rlm_* directly in repl, without importing or redefining them. resources contains data, not helper methods.",
            "Example sections are character slices of examples[id]. Read the imports, ModelNew, JIT launch or device kernel as needed; do not dump the whole example.",
        ],
        "helper_help": {
            "search_docs": "search_docs(query, limit=2) -> hits with id/excerpt/start/end. Query 1..300 characters; limit 1..5.",
            "read_doc": "read_doc(doc_id, start=0, chars=1200) -> dict; read result['excerpt']. chars 1..1600.",
            "compile_candidate": "compile_candidate(source) -> dict, automatically printed once. status='compiled' is PASS for one compile/forward/synchronize smoke. Retain returned diagnostics for repairs; this does not submit.",
            "delegation": "rlm_query(prompt, blocking=False) -> child handle; rlm_status(handle), rlm_result(handle), rlm_wait(handle, timeout_s=30) collect it. Root context is a task dictionary; child context is its prompt string. Pass reference/source/feedback explicitly. Each child has its own REPL, helpers and resources, not root variables. See resources['delegation_example'].",
        },
    }


def resources_bootstrap(resources: dict) -> str:
    """Load only model-visible captured data; no task reference or example execution."""
    serialized = json.dumps(resources, ensure_ascii=False, sort_keys=True)
    return f"import json as _cute_resource_json\nresources = _cute_resource_json.loads({serialized!r})\n"
