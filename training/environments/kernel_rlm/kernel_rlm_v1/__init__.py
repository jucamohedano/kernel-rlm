"""Load Verifiers classes on demand so the REPL program can serve other trainers."""

from typing import Any


def __getattr__(name: str) -> Any:
    if name == "KernelHarness":
        from kernel_rlm_v1.harness import KernelHarness

        return KernelHarness
    if name == "KernelTaskset":
        from kernel_rlm_v1.taskset import KernelTaskset

        return KernelTaskset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["KernelTaskset", "KernelHarness"]
