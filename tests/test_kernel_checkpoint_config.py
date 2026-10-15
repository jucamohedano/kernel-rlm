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
    assert config["max_steps"] == 6
    periodic = list(
        range(config["ckpt"]["interval"], config["max_steps"], config["ckpt"]["interval"])
    )
    assert periodic + [config["max_steps"]] == [2, 4, 6]
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
    assert orchestrator["batch_size"] == 4
    assert orchestrator["group_size"] == 4
    assert orchestrator["train"]["sampling"]["max_completion_tokens"] == 2048
    assert harness["max_tokens"] == 2048
    assert harness["max_turns"] == 5
    assert harness["enable_subcall"] is True
    assert harness["max_subcalls"] == 2
    assert harness["subcall_probe"] is False
    assert orchestrator["renderer"]["enable_thinking"] is True
    assert harness["enable_thinking"] is True
    assert orchestrator["concurrency"] == {"initial_inflight": 4, "max_inflight": 4}
    assert orchestrator["concurrency"]["max_inflight"] >= orchestrator["group_size"]
    assert config["seq_len"] == config["inference"]["vllm"]["max_model_len"] == 8192
    assert config["inference"]["vllm"]["gpu_memory_utilization"] == 0.86
    assert config["inference"]["vllm"]["reasoning_parser"] == "poolside_v1"
    assert (
        config["inference"]["vllm"]["lora_target_modules"]
        == config["trainer"]["model"]["lora"]["target_modules"]
    )
