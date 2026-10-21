from typing import Any


def __getattr__(name: str) -> Any:
    if name == "load_environment":
        from kernel_rlm.env import load_environment

        return load_environment
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["load_environment"]
