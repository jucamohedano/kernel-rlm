import importlib.util
from pathlib import Path

ROOT = Path(__file__).parents[1] / "training/environments/kernel_rlm/kernel_trl"


def load_probe():
    specification = importlib.util.spec_from_file_location("trl_cpu_probe", ROOT / "cpu_probe.py")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_append_only_trajectory_can_keep_exact_sampling_prefixes():
    probe = load_probe()
    assert probe.prefix_compatible(
        [
            {"prompt_ids": [1, 2], "response_ids": [3, 4]},
            {"prompt_ids": [1, 2, 3, 4, 5], "response_ids": [6]},
        ]
    )


def test_removing_past_thoughts_breaks_the_single_sequence_contract():
    probe = load_probe()
    assert not probe.prefix_compatible(
        [
            {"prompt_ids": [1, 2], "response_ids": [30, 31, 4]},
            {"prompt_ids": [1, 2, 4, 5], "response_ids": [6]},
        ]
    )


def test_cpu_probe_does_not_import_training_libraries_at_module_load():
    probe = load_probe()
    assert probe.TRL_REVISION == "33f9e462728b98f7f91d38b99328e81adde2faa0"
