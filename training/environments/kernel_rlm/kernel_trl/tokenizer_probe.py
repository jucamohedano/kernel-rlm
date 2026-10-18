"""CPU-only native TRL prompt preparation check using the pinned real tokenizer."""

import argparse
import json
import os
from pathlib import Path


def check(checkpoint: Path) -> dict:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Run this check with CUDA_VISIBLE_DEVICES='' before importing Torch")

    import torch
    from datasets import Dataset
    from kernel_skyrl.laguna_probe import tiny_model
    from transformers import AutoModelForCausalLM
    from trl import GRPOConfig, GRPOTrainer

    from kernel_trl.phase_probe import tokenizer_and_prompt

    tokenizer, messages = tokenizer_and_prompt(checkpoint)
    configuration = tiny_model().config
    configuration.vocab_size = len(tokenizer)
    model = AutoModelForCausalLM.from_config(configuration)

    def reward(completions, **kwargs):
        return [0.0] * len(completions)

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward,
        train_dataset=Dataset.from_dict({"prompt": [messages] * 4}),
        args=GRPOConfig(
            use_cpu=True,
            bf16=False,
            fp16=False,
            per_device_train_batch_size=4,
            num_generations=4,
            max_steps=1,
            report_to="none",
            save_strategy="no",
            chat_template_kwargs={"enable_thinking": True},
        ),
    )
    actual, images, fields = trainer._tokenize_prompts([messages])
    expected = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, enable_thinking=True, return_dict=True
    )["input_ids"]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
    )
    old = tokenizer(formatted)["input_ids"]
    assert actual == [expected]
    assert old[:2] == [tokenizer.bos_token_id] * 2
    assert expected[0] == tokenizer.bos_token_id and expected[1] != tokenizer.bos_token_id
    assert formatted.endswith("<think>")
    assert torch.cuda.memory_allocated() == 0
    return {
        "status": "passed",
        "kind": "cpu-only-native-trl-tokenizer-check",
        "no_training_or_generation_performed": True,
        "no_gpu_allocated": True,
        "old_double_bos_reproduced": True,
        "new_prompt_exactly_matches_template_tokenization": True,
        "thinking_marker_present": True,
        "old_prompt_ids": old,
        "new_prompt_ids": expected,
        "rendered_prompt": formatted,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise ValueError("Use a fresh output path")
    result = check(arguments.checkpoint)
    arguments.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
