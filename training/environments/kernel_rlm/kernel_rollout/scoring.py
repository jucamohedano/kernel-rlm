"""Shared CUDA submission verification and unchanged terminal reward."""

import asyncio

from kernel_rollout.contract import terminal_reward


async def score_submission(task: dict, record: dict) -> tuple[float, dict]:
    from kernel_rlm.submission_contract import validate_submission_source
    from kernel_rlm.verification_state import VERIFICATION_PROTOCOL
    from kernel_rlm.verifier import verify_cuda

    source = record["submission"]
    if not source.strip():
        return 0.0, {"error": "No final submission", "verification_protocol": VERIFICATION_PROTOCOL}
    valid, error = validate_submission_source(source)
    if not valid:
        return 0.0, {
            "error": f"static validation: {error}",
            "verification_protocol": VERIFICATION_PROTOCOL,
        }
    result = await asyncio.to_thread(verify_cuda, task["context"], source)
    return terminal_reward(result, task.get("speed_reward_weight", 0.1)), result
