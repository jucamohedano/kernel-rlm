"""Helpers for using private reasoning without replaying it as visible content."""

from __future__ import annotations

import re

_THINK_BLOCK = re.compile(r"<think>.*?(?:</think>|$)", flags=re.IGNORECASE | re.DOTALL)


def visible_response(content: str) -> str:
    """Remove unparsed think-tag blocks from an OpenAI-compatible response."""
    return _THINK_BLOCK.sub("", content).replace("</think>", "").strip()
