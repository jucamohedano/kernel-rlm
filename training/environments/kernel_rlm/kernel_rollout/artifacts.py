"""Version and identity checks for captured rollout artifacts (no trainer imports)."""

GROUP_IDENTITY = ("execution_run_id", "group_id", "attempt_id")
EPISODE_IDENTITY = (*GROUP_IDENTITY, "episode_id")


def artifact_version(artifact: dict) -> int | None:
    if "schema_version" not in artifact:
        return None
    if type(artifact["schema_version"]) is not int or artifact["schema_version"] != 1:
        raise ValueError("Unsupported artifact schema version")
    return 1


def require_identity(record: dict, keys: tuple[str, ...]) -> None:
    if any(not isinstance(record.get(key), str) or not record[key].strip() for key in keys):
        raise ValueError("Versioned artifacts require complete identity")


def validate_generation_identity(manifest: dict, record: dict, calls: list[dict]) -> None:
    version = artifact_version(manifest)
    if version is None:
        for key in (*GROUP_IDENTITY, "policy_version"):
            if (
                manifest.get(key) is not None
                and record.get(key) is not None
                and record[key] != manifest[key]
            ):
                raise ValueError("Recorded submission identity differs from generation manifest")
        for call in calls:
            for key in (*EPISODE_IDENTITY, "policy_version"):
                if (
                    record.get(key) is not None
                    and call.get(key) is not None
                    and call[key] != record[key]
                ):
                    raise ValueError("Recorded call identity differs from submission")
        return
    require_identity(manifest, GROUP_IDENTITY)
    if type(manifest["policy_version"]) is not int or manifest["policy_version"] < 0:
        raise ValueError("Invalid sampled policy version")
    require_identity(record, EPISODE_IDENTITY)
    if any(record[key] != manifest[key] for key in GROUP_IDENTITY):
        raise ValueError("Submission identity differs from generation manifest")
    if (
        type(record.get("policy_version")) is not int
        or record["policy_version"] != manifest["policy_version"]
    ):
        raise ValueError("Submission policy differs from generation manifest")
    seen = set()
    root_turns = set()
    for call in calls:
        require_identity(call, (*EPISODE_IDENTITY, "call_id", "model_call_id"))
        if any(call[key] != record[key] for key in EPISODE_IDENTITY):
            raise ValueError("Call identity differs from submission")
        if (
            type(call.get("policy_version")) is not int
            or call["policy_version"] != manifest["policy_version"]
        ):
            raise ValueError("Call policy differs from generation manifest")
        if call["call_id"] != call["model_call_id"] or call["call_id"] in seen:
            raise ValueError("Call identity is inconsistent or repeated")
        seen.add(call["call_id"])
        if call.get("branch") == "root":
            require_identity(call, ("root_turn_id",))
            if call["root_turn_id"] in root_turns:
                raise ValueError("Root turn identity is repeated within episode")
            root_turns.add(call["root_turn_id"])
