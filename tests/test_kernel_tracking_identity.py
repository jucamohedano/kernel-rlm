from kernel_trl.tracking import trainer_tracking_identity


def test_update_runs_have_stable_distinct_lineage_identity():
    first = trainer_tracking_identity("campaign", "group-a", 4, "attempt-1")
    retry = trainer_tracking_identity("campaign", "group-a", 4, "attempt-2")
    next_step = trainer_tracking_identity("campaign", "group-a", 5, "attempt-1")

    assert first["name"] == "campaign--rlm_update--group-a--step-4--attempt-attempt-1"
    assert len({first["name"], retry["name"], next_step["name"]}) == 3
    assert first["config"] == {
        "execution_run_id": "campaign",
        "producer": "rlm_update",
        "group_id": "group-a",
        "policy_step": 4,
        "attempt_id": "attempt-1",
        "run_kind": "native_update",
    }


def test_update_identity_requires_explicit_ids_and_valid_step():
    for args in (
        ("", "group", 1, "attempt"),
        ("run", "", 1, "attempt"),
        ("run", "group", 0, "attempt"),
    ):
        try:
            trainer_tracking_identity(*args)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid identity was accepted")
