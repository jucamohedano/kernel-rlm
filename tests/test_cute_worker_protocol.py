"""Pure worker protocol checks; these are not CUDA execution tests."""

from cute_rlm.verify_worker import place_inputs, seed_all


def test_seed_helper_resets_python_numpy_and_torch_states():
    calls = []

    class Random:
        @staticmethod
        def seed(value):
            calls.append(("numpy", value))

    class Numpy:
        random = Random()

    class Cuda:
        @staticmethod
        def manual_seed_all(value):
            calls.append(("cuda", value))

    class Torch:
        cuda = Cuda()

        @staticmethod
        def manual_seed(value):
            calls.append(("torch", value))

    seed_all(Torch(), Numpy(), 7)
    assert calls == [("numpy", 7), ("torch", 7), ("cuda", 7)]


def test_input_placement_changes_only_tensor_device_metadata():
    value = [{"tensor_dtype": "torch.float32", "device": "cpu", "array": [1, 2]}, 3]
    placed = place_inputs(value, "cuda")
    assert placed[0]["device"] == "cuda"
    assert placed[0]["array"] == value[0]["array"] and placed[1] == 3
    assert value[0]["device"] == "cpu"


def test_cross_tree_storage_aliases_fail_closed_before_serialization():
    from types import SimpleNamespace

    import pytest
    from cute_rlm.verify_worker import reject_cross_tree_aliases

    class Tensor:
        device = "cuda:0"

        def __init__(self, pointer):
            self.pointer = pointer

        def numel(self):
            return 1

        def untyped_storage(self):
            return SimpleNamespace(data_ptr=lambda: self.pointer)

    torch = SimpleNamespace(Tensor=Tensor)
    reject_cross_tree_aliases(torch, [Tensor(1)], {"result": Tensor(2)}, {"weight": Tensor(3)})
    with pytest.raises(NotImplementedError, match="alias contract"):
        reject_cross_tree_aliases(torch, [Tensor(1)], {"result": Tensor(1)}, {})
