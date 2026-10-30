"""OOLONG data/scoring and native evaluation entrypoints."""


def load_environment(*args, **kwargs):
    from oolong.env import load_environment as load

    return load(*args, **kwargs)


__all__ = ["load_environment"]
