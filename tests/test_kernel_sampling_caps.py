import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parents[1] / "training/environments/kernel_rlm/modal_app"


@pytest.fixture
def patched_package(tmp_path):
    spec = importlib.util.spec_from_file_location("sampling_cap_patch", ROOT / "verifiers_sampling.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    dialect = tmp_path / "v1/dialects/chat.py"
    client = tmp_path / "v1/clients/train.py"
    dialect.parent.mkdir(parents=True)
    client.parent.mkdir(parents=True)
    dialect.write_text(
        "class ChatDialect:\n"
        "    def apply_overrides(self, body, model, sampling):\n"
        "        overrides = sampling.wire_args()\n"
        "        max_token_keys = {'max_tokens', 'max_completion_tokens'}\n"
        "        steered = {key: value for key, value in body.items() "
        "if max_token_keys.isdisjoint(overrides) or key not in max_token_keys}\n"
        "        return {**steered, **overrides, 'model': model}\n"
    )
    client.write_text(
        "class TrainClient:\n"
        "    def parameters(self, body, sampling):\n"
        "        sampling_params = sampling.wire_args()\n"
        "        return sampling_params\n"
    )
    module.patch_verifiers_sampling_caps(str(tmp_path))
    return module, tmp_path, dialect, client


@pytest.mark.parametrize("requested", [128, 512, 2048, 4096])
@pytest.mark.parametrize("field", ["max_tokens", "max_completion_tokens"])
def test_request_cap_survives_both_interception_stages(patched_package, requested, field):
    _module, _root, dialect, client = patched_package
    namespace = {}
    exec(compile(dialect.read_text(), str(dialect), "exec"), namespace)
    exec(compile(client.read_text(), str(client), "exec"), namespace)
    sampling = SimpleNamespace(wire_args=lambda: {"max_tokens": 2048, "temperature": 1.0})
    body = namespace["ChatDialect"]().apply_overrides({field: requested}, "policy", sampling)
    parameters = namespace["TrainClient"]().parameters(body, sampling)
    assert body["max_tokens"] == min(requested, 2048)
    assert parameters["max_tokens"] == min(requested, 2048)
    assert parameters["temperature"] == 1.0


def test_sampling_patch_is_idempotent_and_fails_on_source_drift(patched_package):
    module, root, dialect, client = patched_package
    original = dialect.read_text(), client.read_text()
    module.patch_verifiers_sampling_caps(str(root))
    assert original == (dialect.read_text(), client.read_text())
    client.write_text("class Changed: pass\n")
    with pytest.raises(RuntimeError, match="anchor changed"):
        module.patch_verifiers_sampling_caps(str(root))
