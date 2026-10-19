"""Explicit context limits for new TRL runs and preserved historical samples."""

LEGACY_CONTEXT_TOKENS = 12288
DEFAULT_CONTEXT_TOKENS = 16384


def validate_context_tokens(value: int) -> int:
    if type(value) is not int or not LEGACY_CONTEXT_TOKENS <= value <= DEFAULT_CONTEXT_TOKENS:
        raise ValueError("TRL context must be between 12288 and 16384 tokens")
    return value
