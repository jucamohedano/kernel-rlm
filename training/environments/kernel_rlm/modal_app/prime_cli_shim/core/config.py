"""Small configuration surface needed by Prime RL's optional monitor import."""

from __future__ import annotations

import os


class Config:
    @property
    def api_key(self) -> str | None:
        return os.getenv("PRIME_API_KEY")
