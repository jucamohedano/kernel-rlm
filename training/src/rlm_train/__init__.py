from typing import Any

from rlm_train.repl import ExecResult, ReplBackend, SubprocessReplBackend


def __getattr__(name: str) -> Any:
    if name == "RLMTrainEnv":
        from rlm_train.env import RLMTrainEnv

        return RLMTrainEnv
    if name == "RLMTrainRubric":
        from rlm_train.rubric import RLMTrainRubric

        return RLMTrainRubric
    if name in {"ClientHandle", "SubLLMProxy"}:
        from rlm_train import proxy

        return getattr(proxy, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__version__ = "0.1.0"

__all__ = [
    "RLMTrainEnv",
    "RLMTrainRubric",
    "ReplBackend",
    "ExecResult",
    "SubprocessReplBackend",
    "SubLLMProxy",
    "ClientHandle",
]
