import ast
from types import SimpleNamespace

import pytest


@pytest.fixture
def torch():
    return pytest.importorskip("torch")


@pytest.mark.parametrize(
    "module_name,shape",
    [
        ("BatchNorm1d", (2, 3, 7)),
        ("BatchNorm2d", (2, 3, 5, 7)),
        ("BatchNorm3d", (2, 3, 4, 5, 7)),
        ("SyncBatchNorm", (2, 3, 5, 7)),
    ],
)
def test_identity_passes_old_state_but_fails_challenge(torch, module_name, shape):
    from kernel_rlm.candidate import compare_outputs
    from kernel_rlm.verification_state import prepare_batchnorm_state

    model = getattr(torch.nn, module_name)(3).eval()
    values = torch.linspace(-3, 3, steps=shape[-1]).expand(shape).contiguous()
    assert compare_outputs(model(values), values, 1e-3, 1e-4)[0]
    cases = prepare_batchnorm_state(model, 0)
    assert len(cases) == 1
    assert not compare_outputs(model(values), values, 1e-3, 1e-4)[0]
    assert torch.all(model.running_var > 0)


@pytest.mark.parametrize("affine", [True, False])
@pytest.mark.parametrize("track_running_stats", [True, False])
def test_legitimate_candidate_receives_identical_state(torch, affine, track_running_stats):
    from kernel_rlm.candidate import candidate_forward, compare_outputs
    from kernel_rlm.verification_state import prepare_batchnorm_state

    class ModelNew(torch.nn.Module):
        def __init__(self, features):
            super().__init__()
            self.norm = torch.nn.BatchNorm2d(
                features, affine=affine, track_running_stats=track_running_stats
            )

        def forward(self, values):
            return self.norm(values)

    reference = ModelNew(3).eval()
    prepare_batchnorm_state(reference, 2)
    candidate = candidate_forward(SimpleNamespace(ModelNew=ModelNew), reference, [3], "cpu")
    values = torch.arange(210, dtype=torch.float32).reshape(2, 3, 5, 7) / 100
    assert compare_outputs(reference(values), candidate(values), 1e-3, 1e-4)[0]
    for key, value in reference.state_dict().items():
        assert torch.equal(candidate.state_dict()[key], value)


def test_challenges_are_repeatable_distinct_and_do_not_change_global_rng(torch):
    from kernel_rlm.verification_state import prepare_batchnorm_state

    model = torch.nn.Sequential(torch.nn.BatchNorm1d(4)).eval()
    rng = torch.random.get_rng_state().clone()
    first = prepare_batchnorm_state(model, 0)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert first == prepare_batchnorm_state(model, 0)
    assert first != prepare_batchnorm_state(model, 1)
    assert first == prepare_batchnorm_state(model, 0)


def test_unrelated_state_is_unchanged_and_batchnorm_dtype_preserved(torch):
    from kernel_rlm.verification_state import prepare_batchnorm_state

    model = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.BatchNorm1d(3)).double()
    weight = model[0].weight.detach().clone()
    bias = model[0].bias.detach().clone()
    prepare_batchnorm_state(model, 0)
    assert torch.equal(model[0].weight, weight)
    assert torch.equal(model[0].bias, bias)
    assert model[1].running_var.dtype == torch.float64
    assert model[1].weight.requires_grad
    assert model[1].num_batches_tracked.item() == 0


def test_runner_applies_state_before_reference_and_candidate_and_gates_timing():
    from kernel_rlm.verifier import _RUNNER

    tree = ast.parse(_RUNNER)
    loop = next(
        node for node in ast.walk(tree) if isinstance(node, ast.For) and node.target.id == "trial"
    )
    calls = [
        node.func.id
        for node in sorted(ast.walk(loop), key=lambda entry: getattr(entry, "lineno", 0))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    assert calls.index("prepare_batchnorm_state") < calls.index("candidate_forward")
    guarded = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If) and ast.unparse(node.test) == "correct and benchmark_enabled"
    ]
    assert len(guarded) == 1
    assert "prepare_batchnorm_state(model, trials - 1)" in ast.unparse(guarded[0])


@pytest.mark.parametrize("trials", [0, -1, True, 1.5])
def test_verifier_rejects_empty_or_invalid_trial_count(trials):
    from kernel_rlm.verifier import verify_cuda

    with pytest.raises(ValueError, match="trials"):
        verify_cuda("reference", "candidate", n_trials=trials)
