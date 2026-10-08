"""CUDA task data and trusted scoring for Verifiers V1."""

import asyncio
import json
from pathlib import Path

import verifiers.v1 as vf
from kernel_rlm.submission_contract import KERNEL_TASK_INSTRUCTION, validate_submission_source
from kernel_rlm.verifier import verify_cuda

INSTRUCTION = KERNEL_TASK_INSTRUCTION


class KernelData(vf.TaskData):
    context: str


class KernelTask(vf.Task[KernelData]):
    @vf.reward(weight=1.0)
    async def kernel_reward(self, trace: vf.Trace, runtime: vf.Runtime) -> float:
        record = json.loads(await runtime.read("submission.json", max_bytes=1_000_000))
        source = record["submission"].strip()
        trace.info["kernel_submission"] = source
        trace.info["repl_calls"] = record["repl_calls"]
        trace.info["repl_transcript"] = record["messages"]
        if not source:
            return 0.0
        is_valid, validation_error = validate_submission_source(source)
        if not is_valid:
            trace.info["kernel_static"] = {
                "valid": False,
                "error": validation_error,
            }
            trace.info["kernel_verify"] = {
                "compiled": False,
                "extension_used": False,
                "correct": False,
                "max_diff": None,
                "speedup": None,
                "ref_ms": None,
                "kernel_ms": None,
                "error": f"static validation: {validation_error}",
            }
            return 0.0
        trace.info["kernel_static"] = {"valid": True, "error": ""}
        result = await asyncio.to_thread(verify_cuda, self.data.context, source)
        trace.info["kernel_verify"] = result
        reward = 0.1
        if result.get("compiled") and result.get("extension_used"):
            reward += 0.2
        if result.get("correct"):
            reward += 0.4
            reward += 0.1 * min(max(float(result.get("speedup") or 0), 0), 3)
        return reward


class KernelTasksetConfig(vf.TasksetConfig):
    manifest_path: str


class KernelTaskset(vf.Taskset[KernelTask, KernelTasksetConfig]):
    def load(self) -> list[KernelTask]:
        rows = [
            json.loads(line) for line in Path(self.config.manifest_path).read_text().splitlines()
        ]
        return [
            KernelTask(
                KernelData(
                    idx=index, name=str(row["_task_id"]), prompt=INSTRUCTION, context=row["code"]
                ),
                self.config.task,
            )
            for index, row in enumerate(rows)
        ]
