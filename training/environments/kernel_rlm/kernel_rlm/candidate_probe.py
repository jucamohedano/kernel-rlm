"""CPU-only real-Torch proof of constructor state and output comparison."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace


def check() -> dict:
    import torch

    from kernel_rlm.candidate import candidate_forward, clone_inputs, compare_outputs

    torch.set_num_threads(2)
    reference = torch.nn.Linear(3, 2).eval()
    source_before = {name: value.clone() for name, value in reference.state_dict().items()}

    class ModelNew(torch.nn.Linear):
        pass

    run = candidate_forward(SimpleNamespace(ModelNew=ModelNew), reference, [3, 2], "cpu")
    for name, value in run.state_dict().items():
        assert torch.equal(value, reference.state_dict()[name])
        assert value.data_ptr() != reference.state_dict()[name].data_ptr()
    inputs = [torch.randn(4, 3)]
    copied = clone_inputs(inputs)
    assert inputs[0].data_ptr() != copied[0].data_ptr()
    with torch.no_grad():
        assert compare_outputs(reference(*inputs), run(*copied), 1e-3, 1e-4)[0]
        run.weight.add_(1)
    assert all(
        torch.equal(value, reference.state_dict()[name]) for name, value in source_before.items()
    )
    expected = {"out": (torch.tensor([1.0, 2.0]), torch.tensor([7, 9]))}
    assert compare_outputs(expected, clone_inputs(expected), 1e-3, 1e-4) == (True, 0.0, "")
    assert not compare_outputs(
        expected, {"out": (torch.tensor([1.0, 3.0]), torch.tensor([7, 9]))}, 1e-3, 1e-4
    )[0]
    assert not compare_outputs(torch.tensor([1]), torch.tensor([1.0]), 1e-3, 1e-4)[0]
    assert not compare_outputs(
        torch.tensor([float("nan")]), torch.tensor([float("nan")]), 1e-3, 1e-4
    )[0]
    assert compare_outputs(torch.tensor([1 + 2j]), torch.tensor([1 + 3j]), 1e-3, 1e-4)[1] == 1
    assert compare_outputs(
        torch.tensor([], dtype=torch.float32), torch.tensor([], dtype=torch.float32), 1e-3, 1e-4
    )[0]
    try:
        candidate_forward(
            SimpleNamespace(ModelNew=lambda *args: torch.nn.Linear(4, 2)), reference, [3, 2], "cpu"
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("Mismatched reference state must be rejected")
    assert not torch.cuda.is_initialized()
    return {
        "status": "passed",
        "cpu_only": True,
        "constructor_state_copied_exactly": True,
        "reference_state_and_inputs_not_aliased": True,
        "nested_tensor_outputs_checked": True,
        "dtype_nonfinite_complex_and_empty_outputs_checked": True,
        "mismatched_state_rejected": True,
        "not_a_cuda_or_reward_quality_test": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    report = check()
    arguments.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
