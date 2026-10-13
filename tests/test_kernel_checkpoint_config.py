import tomllib
from pathlib import Path


def test_short_run_checkpoint_schedule_and_retention():
    config = tomllib.loads(
        (
            Path(__file__).parents[1] / "training/environments/kernel_rlm/configs/grpo_level1.toml"
        ).read_text()
    )
    assert config["ckpt"] == {"interval": 2, "keep_last": 2}
    assert config["output_dir"] == "/traces/outputs"
    assert config["max_steps"] == 3
    periodic = list(
        range(config["ckpt"]["interval"], config["max_steps"], config["ckpt"]["interval"])
    )
    assert periodic + [config["max_steps"]] == [2, 3]
    assert "ckpt" not in config["trainer"]
    assert "ckpt" not in config["orchestrator"]
    assert (
        config["orchestrator"]["train"]["source"][0]["env"]["taskset"]["speed_reward_weight"] == 0.1
    )


def test_training_matches_probe_generation_and_bounds_concurrency():
    config = tomllib.loads(
        (
            Path(__file__).parents[1] / "training/environments/kernel_rlm/configs/grpo_level1.toml"
        ).read_text()
    )
    orchestrator = config["orchestrator"]
    harness = orchestrator["train"]["source"][0]["env"]["agent"]["harness"]
    assert orchestrator["train"]["sampling"]["max_completion_tokens"] == 2048
    assert harness["max_tokens"] == 2048
    assert harness["max_turns"] == 5
    assert harness["enable_subcall"] is True
    assert harness["subcall_probe"] is False
    assert orchestrator["renderer"]["enable_thinking"] is False
    assert orchestrator["concurrency"] == {"initial_inflight": 8, "max_inflight": 8}
    assert orchestrator["concurrency"]["max_inflight"] >= orchestrator["group_size"]
    assert config["seq_len"] == config["inference"]["vllm"]["max_model_len"] == 8192
