import numpy as np
import pytest
from cute_rlm.outputs import compare_trees, encode_tree, load_tree, save_tree


def test_safe_roundtrip_preserves_nested_types_dtype_and_shape(tmp_path):
    tree = {
        "nested": ([np.arange(6, dtype=np.int16).reshape(2, 3), None], (True, 7, 1.25, "x")),
        "empty": np.array([], dtype=np.float32),
    }
    path = tmp_path / "outputs.bin"
    save_tree(path, tree)
    restored = load_tree(path)

    assert type(restored) is dict
    assert type(restored["nested"]) is tuple
    assert type(restored["nested"][0]) is list
    assert type(restored["nested"][1]) is tuple
    assert restored["nested"][1] == (True, 7, 1.25, "x")
    np.testing.assert_array_equal(restored["nested"][0][0], tree["nested"][0][0])
    assert restored["nested"][0][0].dtype == np.dtype("int16")
    assert restored["nested"][0][0].shape == (2, 3)
    assert restored["empty"].dtype == np.dtype("float32")
    assert restored["empty"].shape == (0,)


def test_compare_rejects_shape_and_dtype_differences():
    assert not compare_trees(
        np.zeros((2,), dtype=np.float32), np.zeros((1, 2), dtype=np.float32), 0, 0
    )["correct"]
    result = compare_trees(np.zeros(2, dtype=np.float32), np.zeros(2, dtype=np.float64), 0, 0)
    assert not result["correct"] and "dtype mismatch" in result["reason"]


def test_bool_and_int_values_are_exact_and_types_do_not_coerce():
    assert not compare_trees(True, 1, 1, 1)["correct"]
    assert not compare_trees(np.array([True]), np.array([False]), 100, 100)["correct"]
    assert not compare_trees(
        np.array([1], dtype=np.int32), np.array([2], dtype=np.int32), 100, 100
    )["correct"]


def test_float_comparison_requires_finite_values_and_reports_difference():
    result = compare_trees(np.array([1.0]), np.array([1.01]), atol=0.02, rtol=0)
    assert result["correct"]
    assert result["max_abs_diff"] == pytest.approx(0.01)
    for value in (np.nan, np.inf, -np.inf):
        result = compare_trees(np.array([value]), np.array([value]), atol=1, rtol=1)
        assert not result["correct"]
        assert "non-finite" in result["reason"]


def test_encoding_rejects_object_arrays_and_shared_storage():
    arrays = {}
    with pytest.raises(TypeError, match="object-dtype"):
        encode_tree(np.array([object()], dtype=object), arrays)
    backing = np.arange(5)
    with pytest.raises(ValueError, match="sharing memory"):
        encode_tree([backing[:3], backing[2:]], {})


def test_npz_is_read_without_pickle_and_object_payload_is_rejected(tmp_path):
    path = tmp_path / "bad.npz"
    import json

    metadata = np.frombuffer(
        json.dumps({"type": "array", "name": "payload"}).encode(), dtype=np.uint8
    )
    np.savez(path, payload=np.array([object()], dtype=object), __metadata__=metadata)
    with pytest.raises(ValueError, match="Object arrays cannot be loaded"):
        load_tree(path)
