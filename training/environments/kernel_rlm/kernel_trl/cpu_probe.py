"""Exercise actual pinned TRL on a tiny random CPU model, without downloads."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
from pathlib import Path

TRL_REVISION = "33f9e462728b98f7f91d38b99328e81adde2faa0"


def prefix_compatible(turns: list[dict]) -> bool:
    for previous, following in zip(turns, turns[1:], strict=False):
        prefix = previous["prompt_ids"] + previous["response_ids"]
        if following["prompt_ids"][: len(prefix)] != prefix:
            return False
    return True


def run_probe(source: Path, work_dir: Path) -> dict:
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != TRL_REVISION or importlib.metadata.version("trl") != "1.9.2":
        raise ValueError("The CPU probe requires the pinned TRL v1.9.2 source and package")
    trainer_source = source / "trl/trainer/grpo_trainer.py"
    installed = Path(
        importlib.metadata.distribution("trl").locate_file("trl/trainer/grpo_trainer.py")
    )
    if installed.read_bytes() != trainer_source.read_bytes():
        raise ValueError("Installed TRL trainer differs from the pinned source")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TRL_EXPERIMENTAL_SILENCE"] = "1"

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
    from trl import GRPOConfig, GRPOTrainer

    torch.set_num_threads(2)
    torch.manual_seed(42)
    vocabulary = {
        "[PAD]": 0,
        "[BOS]": 1,
        "[EOS]": 2,
        "[UNK]": 3,
        "task": 4,
        "answer": 5,
        "correct": 6,
        "feedback": 7,
        "wrong": 8,
    }
    backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
        unk_token="[UNK]",
    )
    config = LlamaConfig(
        vocab_size=len(vocabulary),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=64,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        attn_implementation="eager",
    )
    model = get_peft_model(
        LlamaForCausalLM(config),
        LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
            bias="none",
        ),
    )
    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad], lr=0.01
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    observed = []

    def rollout(prompts, trainer):
        count = len(prompts)
        observed.append(count)
        return {
            "prompt_ids": [[4] for index in range(count)],
            "completion_ids": [[6 if index % 2 else 8, 7, 2] for index in range(count)],
            "logprobs": [[-1.0, -1.0, -1.0] for index in range(count)],
            "env_mask": [[1, 0, 1] for index in range(count)],
            "fixture_reward": [float(index % 4) / 3 for index in range(count)],
        }

    def reward(fixture_reward, **kwargs):
        return fixture_reward

    arguments = GRPOConfig(
        output_dir=str(work_dir),
        use_cpu=True,
        bf16=False,
        fp16=False,
        per_device_train_batch_size=4,
        num_generations=4,
        max_steps=1,
        max_completion_length=8,
        report_to="none",
        save_strategy="no",
        logging_steps=1,
        disable_tqdm=True,
        use_vllm=False,
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward,
        args=arguments,
        train_dataset=Dataset.from_dict({"prompt": ["task"] * 4}),
        processing_class=tokenizer,
        rollout_func=rollout,
        optimizers=(optimizer, scheduler),
    )
    dispatched = trainer._generate(["task"] * 4)
    assert dispatched[2] == [[1, 0, 1]] * 4
    assert trainer.optimizer is optimizer
    result = trainer.train()
    changed = [
        name
        for name, parameter in trainer.model.named_parameters()
        if name in before and not torch.equal(before[name], parameter.detach())
    ]
    assert trainer.state.global_step == 1 and changed
    assert all(parameter.device.type == "cpu" for parameter in trainer.model.parameters())
    nonprefix = [
        {"prompt_ids": [1, 4], "response_ids": [10, 11, 12, 5]},
        {"prompt_ids": [1, 4, 5, 7], "response_ids": [6, 2]},
    ]
    assert not prefix_compatible(nonprefix)
    return {
        "kind": "backend-compatibility-check",
        "trl_revision": revision,
        "source_sha256": hashlib.sha256(trainer_source.read_bytes()).hexdigest(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in (
                "trl",
                "torch",
                "transformers",
                "accelerate",
                "datasets",
                "peft",
            )
        },
        "device": "cpu",
        "pretrained_downloads": 0,
        "gpu_jobs_launched": 0,
        "random_model_optimizer_steps": trainer.state.global_step,
        "changed_adapter_tensors": len(changed),
        "train_loss": result.training_loss,
        "custom_rollout_dispatch_passed": True,
        "environment_loss_mask_passed": True,
        "injected_optimizer_used": True,
        "optimizer_fixture": "SGD hook test, not project Muon",
        "lora": {"rank": 8, "alpha": 16, "targets": ["q_proj", "k_proj", "v_proj", "o_proj"]},
        "observed_rollout_prompt_counts": observed,
        "thought_removal_counterexample": {
            "turns": nonprefix,
            "single_append_only_sequence_preserves_sampling_context": False,
            "note": "Synthetic token example, not a Laguna rollout or a measured model failure",
        },
        "remaining_gates": [
            "Exact per-turn RLM sampling contexts and trajectory-level GRPO grouping",
            "Laguna loader and pinned Moonlight Muon integration",
            "Trainer offload during inference; vLLM sleep alone does not remove trainer weights",
            "Avoid full-model LoRA merge/sync; prove two-L40S memory fit",
            "Compact checkpoint resume and factory launch capture",
        ],
        "scope": "Actual TRL CPU step and API hooks only; no CUDA, vLLM, Laguna or learning-quality validation",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trl-source", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.output.exists() or arguments.work_dir.exists():
        raise ValueError("Use fresh CPU probe output and work paths")
    report = run_probe(arguments.trl_source, arguments.work_dir)
    arguments.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
