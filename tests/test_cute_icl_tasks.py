"""CPU-only oracle tests using a tiny NumPy-backed Torch fixture, never CUDA."""

import ast
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest
from cute_rlm.icl.tasks import build_inputs, expected_output, get_task, load_tasks, reference_source


class Tensor(np.ndarray):
    def clone(self):
        return self.copy().view(Tensor)

    def contiguous(self):
        return np.ascontiguousarray(self).view(Tensor)

    def transpose(self, dim0, dim1):
        return np.swapaxes(self, dim0, dim1).view(Tensor)

    def as_strided(self, shape, stride):
        return np.lib.stride_tricks.as_strided(
            self, shape=shape, strides=tuple(s * self.itemsize for s in stride)
        ).view(Tensor)

    def stride(self):
        return tuple(s // self.itemsize for s in self.strides)


class Generator:
    def __init__(self, device):
        assert device == "cpu"

    def manual_seed(self, seed):
        self.rng = np.random.default_rng(seed)
        return self


class Module:
    def __call__(self, *args):
        return self.forward(*args)


class FakeTorch:
    float32 = np.float32
    Generator = Generator
    nn = SimpleNamespace(Module=Module)
    exp = staticmethod(np.exp)
    sqrt = staticmethod(np.sqrt)
    matmul = staticmethod(np.matmul)
    mv = staticmethod(np.matmul)

    @staticmethod
    def randn(*shape, generator, dtype, device):
        assert device == "cpu"
        return generator.rng.standard_normal(shape).astype(dtype).view(Tensor)

    @staticmethod
    def clamp(x, min, max):
        return np.clip(x, min, max)

    @staticmethod
    def sum(x, dim):
        return np.sum(x, axis=dim)

    @staticmethod
    def max(x, dim):
        return SimpleNamespace(values=np.max(x, axis=dim))

    @staticmethod
    def mean(x, dim, keepdim):
        return np.mean(x, axis=dim, keepdims=keepdim)


@pytest.fixture
def torch():
    return FakeTorch()


def test_manifest_is_fixed_public_and_independent():
    tasks = load_tasks()
    assert len(tasks) == 12
    assert len({t["id"] for t in tasks}) == 12
    assert Counter(t["family"] for t in tasks) == dict.fromkeys(
        ("pointwise", "indexing", "reduction", "matrix"), 3
    )
    for task in tasks:
        assert task["constructor_args"] == []
        assert task["dtype"] == "float32"
        assert len(task["cases"]) == 4
        assert task["atol"] > 0 and task["rtol"] > 0
        assert not any("source" in key or "oracle" in key for key in task)
        source = reference_source(task)
        tree = ast.parse(source)
        model = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        forward = next(
            n for n in model.body if isinstance(n, ast.FunctionDef) and n.name == "forward"
        )
        assert [a.arg for a in forward.args.args][1:] == task["interface"]["forward_args"]
    tasks[0]["cases"][0]["n"] = -1
    assert get_task("affine")["cases"][0]["n"] == 1
    with pytest.raises(KeyError):
        get_task("unknown")


@pytest.mark.parametrize("task", load_tasks(), ids=lambda t: t["id"])
def test_all_cases_reproducible_and_trusted_source_agrees(task, torch):
    # Only execute our trusted reference source; never student code.
    source = reference_source(task).replace("import torch\n", "")
    namespace = {"torch": torch}
    exec(compile(source, "trusted-icl-reference", "exec"), namespace)
    model = namespace["Model"]()
    for case in task["cases"]:
        first = build_inputs(task, case, torch)
        second = build_inputs(task, case, torch)
        for a, b in zip(first, second, strict=True):
            np.testing.assert_array_equal(a, b)
        expected = expected_output(task, first, torch)
        np.testing.assert_allclose(model(*second), expected, atol=task["atol"], rtol=task["rtol"])
        assert expected.dtype == np.float32
        assert np.all(np.isfinite(expected))
        for value in first:
            if isinstance(value, np.ndarray):
                assert not np.shares_memory(expected, value)
        if task["id"] == "strided_affine":
            assert first[0].stride() == (case["stride_m"], case["stride_n"])
            assert expected.flags.c_contiguous
        if task["id"] in {"row_sum", "row_max", "matvec"}:
            assert expected.shape == (case["m"],)
        if task["id"] == "transpose_copy":
            assert expected.shape == (case["n"], case["m"])
            assert expected.flags.c_contiguous


def test_known_small_oracles(torch):
    x = np.array([[1, -2, 3], [4, -5, 6]], dtype=np.float32).view(Tensor)
    bias = np.array([0.5, 1, -1], dtype=np.float32).view(Tensor)
    np.testing.assert_array_equal(expected_output(get_task("row_sum"), [x], torch), [2, 5])
    np.testing.assert_array_equal(expected_output(get_task("row_max"), [x], torch), [3, 6])
    np.testing.assert_array_equal(expected_output(get_task("row_bias"), [x, bias], torch), x + bias)
    np.testing.assert_array_equal(expected_output(get_task("matvec"), [x, bias], torch), [-4.5, -9])
    B = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float32).view(Tensor)
    C = np.array([[4, 1], [10, 1]], dtype=np.float32)
    np.testing.assert_array_equal(expected_output(get_task("matmul"), [x, B], torch), C)
    np.testing.assert_array_equal(
        expected_output(get_task("matmul_bias"), [x, B, bias[:2]], torch), C + bias[:2]
    )
    actual = expected_output(get_task("row_normalize"), [x, 0.25], torch)
    np.testing.assert_allclose(
        actual, x / np.sqrt(np.array([[14 / 3], [77 / 3]]) + 0.25), rtol=1e-6
    )


def test_native_torch_cpu_cases_if_installed():
    """Real Torch CPU tensors, when available; no CUDA availability check or call."""
    native = pytest.importorskip("torch", reason="Native Torch is not installed on CPU host")
    for task in load_tasks():
        namespace = {}
        exec(compile(reference_source(task), "trusted-icl-reference", "exec"), namespace)
        model = namespace["Model"]()
        for case in task["cases"]:
            inputs = build_inputs(task, case, native)
            repeated = build_inputs(task, case, native)
            for x, y in zip(inputs, repeated, strict=True):
                if isinstance(x, native.Tensor):
                    assert x.device.type == "cpu"
                    assert x.dtype == native.float32
                    native.testing.assert_close(x, y, atol=0, rtol=0)
            output = expected_output(task, inputs, native)
            native.testing.assert_close(
                model(*repeated), output, atol=task["atol"], rtol=task["rtol"]
            )
            if task["id"] in {"affine", "binary_gate", "clamp"}:
                shape = (case["n"],)
            elif task["id"] in {"row_sum", "row_max", "matvec"}:
                shape = (case["m"],)
            elif task["id"] == "transpose_copy":
                shape = (case["n"], case["m"])
            else:
                shape = (case["m"], case["n"])
            assert tuple(output.shape) == shape
            assert output.dtype == native.float32
            assert output.is_contiguous()
            assert bool(native.isfinite(output).all())
            for value in inputs:
                if isinstance(value, native.Tensor):
                    assert output.untyped_storage().data_ptr() != value.untyped_storage().data_ptr()
            if task["id"] == "strided_affine":
                assert inputs[0].stride() == (case["stride_m"], case["stride_n"])
