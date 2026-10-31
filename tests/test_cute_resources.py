"""CPU checks for indexed, captured CuTe examples and execution resources."""

import ast
import hashlib
import json

from cute_rlm.resources import EXAMPLE_SHA256, build_resources, resources_bootstrap

from tests.test_cute_eval_preparation import config


def test_resources_are_config_derived_and_exclude_hidden_verifier_data():
    cfg = config()
    resources = build_resources(cfg)
    assert resources["target"]["gpu_model"] == cfg.gpu_model
    assert resources["target"]["compute_capability"] == cfg.gpu_compute_capability
    assert resources["limits"]["turns_per_session"] == cfg.max_turns
    assert resources["limits"]["generated_tokens_per_tree"] == cfg.max_generated_tokens
    assert resources["validation"]["atol"] == cfg.verification["atol"]
    assert "seed" not in resources["validation"]
    assert set(resources) == {
        "target",
        "validation",
        "limits",
        "example_index",
        "examples",
        "delegation_example",
        "api_notes",
        "helper_help",
    }


def test_exact_historical_example_is_data_with_indexed_complete_definitions():
    resources = build_resources(config())
    entry = resources["example_index"][0]
    source = resources["examples"][entry["id"]]
    assert hashlib.sha256(source.encode()).hexdigest() == entry["source_sha256"] == EXAMPLE_SHA256
    assert entry["verification"]["compute_capability"] == "8.9"
    for name, span in entry["sections"].items():
        section = source[span["start"] : span["end"]]
        definitions = ast.parse(section).body
        if name == "imports":
            assert all(isinstance(node, (ast.Import, ast.ImportFrom)) for node in definitions)
            assert any(
                isinstance(node, ast.Import)
                and any(
                    alias.name == "cutlass.cute" and alias.asname == "cute" for alias in node.names
                )
                for node in definitions
            )
        else:
            assert definitions[0].name == name
    namespace = {}
    exec(resources_bootstrap(resources), namespace)
    assert namespace["resources"] == resources
    assert "ModelNew" not in namespace  # Example is never executed by the reasoning REPL.
    assert json.loads(json.dumps(resources)) == resources
