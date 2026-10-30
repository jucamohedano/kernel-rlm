"""Compile-feedback launcher diagnostics without GPU dependencies."""

import sys

import pytest
from cute_rlm.contracts import VerificationConfig
from cute_rlm.verifier import launch_phase


@pytest.mark.asyncio
async def test_feedback_preserves_launcher_diagnostics_without_changing_final_phase():
    command = [
        sys.executable,
        "-c",
        "import sys; print('{\"status\": \"compiled\"}'); print('compiler note', file=sys.stderr)",
    ]
    config = VerificationConfig(
        atol=0.001,
        rtol=0.001,
        seed=7,
        mode="preserve",
        correctness_runs=3,
        warmup_runs=5,
        timed_runs=20,
        timeout_s=5,
    )
    feedback = await launch_phase(command, {"compile_feedback": True}, config)
    assert feedback["status"] == "compiled"
    assert feedback["launcher_stderr"] == "compiler note\n"
    assert feedback["returncode"] == 0
    final = await launch_phase(command, {}, config)
    assert "launcher_stderr" not in final
    assert "returncode" not in final
