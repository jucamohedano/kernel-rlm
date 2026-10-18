"""CPU-only native TRL loss/gradient check, not a project-optimizer replacement."""

import argparse
import json
import os
from pathlib import Path


def check(directory: Path) -> dict:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Use CUDA_VISIBLE_DEVICES='' for the CPU gradient proof")
    import torch
    from datasets import Dataset
    from kernel_skyrl.contract import RootSample
    from kernel_skyrl.laguna_policy import attach_adapters
    from kernel_skyrl.laguna_probe import tiny_model
    from trl import GRPOConfig

    from kernel_trl.phase_probe import tokenizer_and_prompt
    from kernel_trl.replay_trainer import rlm_grpo_trainer
    from kernel_trl.turn_credit import tensors_for_turn, training_rows

    torch.set_num_threads(2)
    torch.manual_seed(42)
    tokenizer, prompt = tokenizer_and_prompt(None)
    model = attach_adapters(tiny_model().to(torch.bfloat16)).float()
    values = []
    for trajectory, reward in enumerate([0.0, 0.2, 0.8, 1.0]):
        samples = [RootSample([2, 10], [20 + trajectory, 2], [-4.8, -4.8], "stop")]
        if trajectory % 2:
            samples.append(
                RootSample([2, 40 + trajectory, 7], [30 + trajectory, 2], [-4.8, -4.8], "stop")
            )
        values.append({"reward": reward, "samples": samples})
    group = training_rows(values)
    parent = rlm_grpo_trainer(None, lambda optimizer: None, group)

    class CPUGradientProof(parent):
        def create_optimizer(self, model=None):
            if self.optimizer is None:
                self.optimizer = torch.optim.SGD(
                    [parameter for parameter in self.model.parameters() if parameter.requires_grad],
                    lr=1e-3,
                )
            return self.optimizer

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    gas = len(group["rows"]) // world_size
    trainer = CPUGradientProof(
        model=model,
        processing_class=tokenizer,
        reward_funcs=lambda completions, **kwargs: [0.0] * len(completions),
        train_dataset=Dataset.from_dict({"prompt": [prompt] * gas}),
        args=GRPOConfig(
            output_dir=str(directory),
            use_cpu=True,
            bf16=False,
            fp16=False,
            gradient_checkpointing=False,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=gas,
            steps_per_generation=gas,
            num_generations=4,
            max_steps=1,
            max_completion_length=8,
            loss_type="dapo",
            router_aux_loss_coef=0.0,
            beta=0.0,
            temperature=1.0,
            max_grad_norm=0.0,
            lr_scheduler_type="constant",
            learning_rate=1e-3,
            report_to="none",
            save_strategy="no",
            disable_tqdm=True,
        ),
    )
    trainer.current_gradient_accumulation_steps = gas
    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    for row in group["rows"]:
        (
            trainer.compute_loss(model, tensors_for_turn(row, group["trainable_tokens"], "cpu"))
            / world_size
        ).backward()
    expected = {
        name: before[name] - 1e-3 * parameter.grad
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    model.zero_grad(set_to_none=True)
    trainer.train()
    errors = [
        float((parameter.detach() - expected[name]).abs().max())
        for name, parameter in trainer.model.named_parameters()
        if name in expected
    ]
    assert max(errors) < 1e-7, errors
    assert trainer.state.global_step == 1
    assert not torch.cuda.is_initialized()
    return {
        "status": "passed",
        "kind": "CPU native TRL per-turn gradient and accumulation proof",
        "probe_optimizer": "SGD for analytic comparison only; production remains Muon",
        "real_rows": group["real_rows"],
        "padded_rows": len(group["rows"]) - group["real_rows"],
        "exact_nonprefix_contexts": True,
        "max_parameter_error": max(errors),
        "no_gpu_allocated": True,
        "world_size": world_size,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        arguments.output = arguments.output.with_name(
            arguments.output.stem + "-rank-" + os.environ["RANK"] + arguments.output.suffix
        )
    if arguments.output.exists():
        raise ValueError("Use a fresh output path")
    result = check(arguments.output.parent / "cpu-proof-trainer")
    arguments.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
