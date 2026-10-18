"""Reference-state handoff and structured output checks for CUDA candidates."""

import copy


def clone_inputs(value):
    import torch

    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, list):
        return [clone_inputs(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_inputs(item) for item in value)
    if isinstance(value, dict):
        return {key: clone_inputs(item) for key, item in value.items()}
    return copy.deepcopy(value)


def candidate_forward(submission, reference_model, init_inputs, device):
    if not hasattr(submission, "ModelNew"):
        return submission.cuda_forward
    import torch

    candidate = submission.ModelNew(*clone_inputs(init_inputs))
    if not isinstance(candidate, torch.nn.Module):
        raise ValueError("ModelNew must be a torch.nn.Module")
    candidate = candidate.to(device).eval()
    state = {name: value.detach().clone() for name, value in reference_model.state_dict().items()}
    candidate.load_state_dict(state, strict=True)
    return candidate


def compare_outputs(reference, candidate, atol: float, rtol: float) -> tuple[bool, float, str]:
    import torch

    if torch.is_tensor(reference):
        if not torch.is_tensor(candidate):
            return False, 0.0, "output must be a tensor"
        if reference.shape != candidate.shape or reference.dtype != candidate.dtype:
            return False, 0.0, "output shape or dtype mismatch"
        if reference.device != candidate.device:
            return False, 0.0, "output device mismatch"
        expected, actual = reference.reshape(-1), candidate.reshape(-1)
        maximum = 0.0
        correct = True
        for start in range(0, expected.numel(), 1048576):
            expected_chunk = expected[start : start + 1048576]
            actual_chunk = actual[start : start + 1048576]
            dtype = torch.complex128 if expected_chunk.is_complex() else torch.float64
            difference = (expected_chunk.to(dtype) - actual_chunk.to(dtype)).abs()
            if not torch.isfinite(difference).all():
                return False, maximum, "nonfinite output difference"
            if difference.numel():
                maximum = max(maximum, difference.max().item())
            equal = (
                torch.allclose(expected_chunk, actual_chunk, atol=atol, rtol=rtol)
                if expected_chunk.is_floating_point() or expected_chunk.is_complex()
                else torch.equal(expected_chunk, actual_chunk)
            )
            correct = correct and equal
        return correct, maximum, "" if correct else "numerical mismatch"
    if isinstance(reference, (tuple, list)):
        if not isinstance(candidate, (tuple, list)) or isinstance(reference, list) != isinstance(
            candidate, list
        ):
            return False, 0.0, "output sequence type mismatch"
        if len(reference) != len(candidate):
            return False, 0.0, "output sequence length mismatch"
        pairs = zip(reference, candidate, strict=True)
    elif isinstance(reference, dict):
        if not isinstance(candidate, dict) or reference.keys() != candidate.keys():
            return False, 0.0, "output mapping keys mismatch"
        pairs = ((reference[key], candidate[key]) for key in reference)
    else:
        correct = type(reference) is type(candidate) and reference == candidate
        return correct, 0.0, "" if correct else "output metadata mismatch"
    maximum = 0.0
    for expected, actual in pairs:
        correct, difference, error = compare_outputs(expected, actual, atol, rtol)
        maximum = max(maximum, difference)
        if not correct:
            return False, maximum, error
    return True, maximum, ""
