"""Trackio identities for per-update native trainer runs."""


def trainer_tracking_identity(
    execution_run_id: str, group_id: str, optimizer_step: int, attempt_id: str
) -> dict:
    """Return a unique Trackio leaf-run name and its lineage config.

    The execution run identifies the coordinator campaign. Every optimizer update
    gets its own Trackio run so local trainer steps never collide or reset.
    """
    if not execution_run_id or not group_id or not attempt_id:
        raise ValueError("Execution, group, and attempt identifiers are required")
    if type(optimizer_step) is not int or optimizer_step < 1:
        raise ValueError("Optimizer step must be a positive integer")
    producer = "rlm_update"
    name = (
        f"{execution_run_id}--{producer}--{group_id}--step-{optimizer_step}--attempt-{attempt_id}"
    )
    return {
        "name": name,
        "config": {
            "execution_run_id": execution_run_id,
            "producer": producer,
            "group_id": group_id,
            "policy_step": optimizer_step,
            "attempt_id": attempt_id,
            "run_kind": "native_update",
        },
    }
