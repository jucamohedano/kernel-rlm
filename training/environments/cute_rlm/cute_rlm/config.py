"""Frozen configuration for a new CuTe evaluation; no workload starts on import."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

MODEL_ID = "poolside/Laguna-XS-2.1"
MODEL_REVISION = "c5f36269bbdbd3f27fddc9a9f9dbae0cf2cf57db"
DATASET_ID = "ByteDance-Seed/cudaLLM-data"
DATASET_REVISION = "8387cbead988c758f204f4ec9feaf0318f12a3a5"
REQUIRED_GPU_CHECKS = (
    "server_context",
    "kv_cache_capacity",
    "two_concurrent_long_requests",
    "cute_compile_launch",
    "reference_candidate_correctness",
    "timing_protocol",
    "selected_token_logprobs",
    "worker_isolation",
)


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass(frozen=True)
class EvalConfig:
    task_manifest: str
    docs_manifest: str
    verification: dict[str, Any]
    seed: int
    task_count: int
    model_id: str = MODEL_ID
    model_revision: str = MODEL_REVISION
    dataset_id: str = DATASET_ID
    dataset_revision: str = DATASET_REVISION
    server_context_tokens: int = 262144
    output_fraction: float = 0.9
    max_generated_tokens: int = 1000000
    max_depth: int = 2
    max_live_children: int = 8
    max_concurrent_requests: int = 2
    max_model_calls: int | None = 256
    max_turns: int | None = 32
    max_children: int | None = 32
    max_prompt_chars: int | None = None
    broker_timeout: float = 600
    broker_request_bytes: int = 8 * 1024**2
    repl_exec_timeout: float = 600
    worker_startup_timeout: float = 30
    request_timeout: float = 600
    episode_timeout: float = 1800
    enable_thinking: bool = True
    logprobs: bool = True
    compaction: bool = False

    def __post_init__(self):
        if type(self.task_count) is not int or self.task_count <= 0:
            raise ValueError("Choose a positive task_count before preparation")
        if type(self.broker_request_bytes) is not int or self.broker_request_bytes <= 0:
            raise ValueError("broker_request_bytes must be a positive integer")
        if type(self.seed) is not int:
            raise ValueError("seed must be an integer")
        if self.model_id != MODEL_ID or self.model_revision != MODEL_REVISION:
            raise ValueError("This evaluation targets the pinned fresh Laguna release")
        if self.dataset_id != DATASET_ID or self.dataset_revision != DATASET_REVISION:
            raise ValueError("This evaluation targets the pinned cudaLLM source")
        if self.server_context_tokens != 262144:
            raise ValueError("Do not silently lower the agreed 262144-token window")
        if not self.enable_thinking or not self.logprobs or self.compaction:
            raise ValueError("Use preserved thinking and selected logprobs, without compaction")
        if not self.task_manifest or not self.docs_manifest or not self.verification:
            raise ValueError("Task, documentation and verifier settings must be explicit")
        from rlm_train.native import NativeLimits
        from rlm_train.token_budget import ContextBudgetPolicy

        from cute_rlm.contracts import VerificationConfig

        VerificationConfig(**self.verification)
        NativeLimits(
            max_depth=self.max_depth,
            max_turns=self.max_turns,
            max_children=self.max_children,
            max_live_children=self.max_live_children,
            max_concurrent_requests=self.max_concurrent_requests,
            max_model_calls=self.max_model_calls,
            request_timeout=self.request_timeout,
            episode_timeout=self.episode_timeout,
            max_prompt_chars=self.max_prompt_chars,
        )
        for name in ("broker_timeout", "repl_exec_timeout", "worker_startup_timeout"):
            value = getattr(self, name)
            if isinstance(value, bool) or not __import__("math").isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        ContextBudgetPolicy(
            server_context_tokens=self.server_context_tokens,
            output_fraction=self.output_fraction,
            max_generated_tokens=self.max_generated_tokens,
            logprobs=self.logprobs,
        )

    @classmethod
    def read(cls, path: str | Path) -> EvalConfig:
        return cls(**json.loads(Path(path).read_text()))

    def payload(self) -> dict:
        return asdict(self)

    def sha256(self) -> str:
        return digest(self.payload())


def check_gpu_receipt(receipt: dict, config: EvalConfig, manifest_sha256: str) -> None:
    """A receipt records observed checks, not CPU simulation or requested settings."""
    if receipt.get("config_sha256") != config.sha256():
        raise ValueError("GPU readiness receipt does not match this configuration")
    if receipt.get("manifest_sha256") != manifest_sha256:
        raise ValueError("GPU readiness receipt does not match frozen inputs")
    for key in REQUIRED_GPU_CHECKS:
        check = receipt.get("checks", {}).get(key, {})
        if check.get("status") != "passed" or not check.get("evidence"):
            raise ValueError(f"GPU validation still required: {key}")
    if receipt.get("observed_context_tokens") != config.server_context_tokens:
        raise ValueError("Observed server window does not match requested context")
    if not receipt.get("hardware") or not receipt.get("software"):
        raise ValueError("GPU hardware and software evidence is required")


def serving_arguments(model_path: str, config: EvalConfig) -> list[str]:
    """Requested startup arguments; memory support still needs a GPU check."""
    return [
        "vllm",
        "serve",
        model_path,
        "--served-model-name",
        config.model_id,
        "--dtype",
        "bfloat16",
        "--max-model-len",
        str(config.server_context_tokens),
        "--max-num-seqs",
        str(config.max_concurrent_requests),
        "--generation-config",
        "vllm",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "poolside_v1",
        "--reasoning-parser",
        "poolside_v1",
        "--default-chat-template-kwargs",
        '{"enable_thinking":true}',
    ]
