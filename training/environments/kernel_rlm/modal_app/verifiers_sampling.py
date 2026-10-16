"""Keep lower per-request output limits in pinned Verifiers training clients."""


def patch_verifiers_sampling_caps(package_root: str | None = None) -> None:
    import ast
    from importlib.metadata import version
    from importlib.util import find_spec
    from pathlib import Path

    if package_root is None:
        if version("verifiers") != "0.3.1":
            raise RuntimeError("Review the sampling-cap patch before changing Verifiers 0.3.1")
        spec = find_spec("verifiers")
        root = Path(spec.origin).parent
    else:
        root = Path(package_root)
    patches = (
        (
            "v1/dialects/chat.py",
            "        overrides = sampling.wire_args()\n",
            "        requested_cap = body.get('max_tokens', body.get('max_completion_tokens'))\n"
            "        if requested_cap is not None and overrides.get('max_tokens') is not None:\n"
            "            overrides['max_tokens'] = min(overrides['max_tokens'], requested_cap)\n",
        ),
        (
            "v1/clients/train.py",
            "        sampling_params = sampling.wire_args()\n",
            "        requested_cap = body.get('max_tokens', body.get('max_completion_tokens'))\n"
            "        if requested_cap is not None:\n"
            "            sampling_params['max_tokens'] = min(\n"
            "                requested_cap, sampling_params.get('max_tokens', requested_cap)\n"
            "            )\n",
        ),
    )
    for relative_path, anchor, addition in patches:
        path = root / relative_path
        source = path.read_text()
        if addition in source:
            continue
        if source.count(anchor) != 1:
            raise RuntimeError(f"Verifiers sampling-cap anchor changed: {relative_path}")
        patched = source.replace(anchor, anchor + addition, 1)
        ast.parse(patched)
        path.write_text(patched)
