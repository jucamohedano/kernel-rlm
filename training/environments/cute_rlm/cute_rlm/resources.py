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
            "Read example_index.sections: imports, ModelNew host adapter, elementwise_add JIT launch, then elementwise_add_kernel. Offsets are character slices of examples[id]; do not dump the full CLI example.",
            "Use the exact imports shown in the indexed header: import cutlass.cute as cute; from cutlass.cute.runtime import from_dlpack. Do not guess import cute or import cutlass.cute.dsl.",
            "Keep Torch allocation and state access in ordinary Python. Follow the example's adaptation boundary; JIT tensor arguments may already be CuTe tensors, not Torch tensors to adapt again.",
            "DLPack rejects requires-grad parameters. Adapt detached views when needed without replacing registered state or changing training/state semantics. torch.no_grad does not clear requires_grad.",
            "Device functions use kernel(...).launch(grid=..., block=...). Match caller stream ordering; the historical example is not proof of every stream configuration.",
            "Define decorated functions directly in source, not exec strings. Direct @cute.jit calls may compile implicitly; explicit cute.compile is not always required.",
            "A successful compile_candidate smoke can still be PyTorch-only. Numerical forward must call CuTe; final grading independently checks correctness, state and launch evidence.",
        ],
        "helper_help": {
            "search_docs": "search_docs(query, limit=2) -> list[dict]. Query 1..300 characters, limit 1..5. Hits contain id/title/sha256/excerpt/start/end character offsets; use the matching offset rather than rereading the file header.",
            "read_doc": "read_doc(doc_id, start=0, chars=1200) -> dict with excerpt/more/id/title/sha256. chars 1..1600. Print result['excerpt']; a dictionary cannot be sliced as source text.",
            "compile_candidate": "compile_candidate(source: str) -> dict, automatically printed. Inspect status/detail/error_type/traceback/stdout/stderr/launcher_stderr when present. One isolated forward/synchronize only, no reference outputs, correctness or timing; never submits.",
            "delegation": "rlm_query(prompt, blocking=False) creates a child with its own REPL and compile_candidate/docs/resources; rlm_status(handle), rlm_result(handle), rlm_wait(handle, timeout_s=30) manage it. Child context is its prompt string, not the root task dictionary; no root variables are inherited. Pass reference/complete source/contract/exact errors explicitly. resources['delegation_example'] is root REPL code for a bounded complete-model repair. Depth 1 permits root 0 and child 1, rejects grandchildren. Tree budgets are shared; a wait timeout does not cancel work, but root finalization does. llm_query(prompt) is plain completion without REPL helpers.",
        },
    }


def resources_bootstrap(resources: dict) -> str:
    """Load only model-visible captured data; no task reference or example execution."""
    serialized = json.dumps(resources, ensure_ascii=False, sort_keys=True)
    return f"import json as _cute_resource_json\nresources = _cute_resource_json.loads({serialized!r})\n"
