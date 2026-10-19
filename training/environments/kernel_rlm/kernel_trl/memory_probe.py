"""Synthetic long-sequence memory checks, never sampled data or optimizer updates."""

from kernel_trl.context import validate_context_tokens


def probe_group(sequence_tokens: int, policy_version: int) -> dict:
    validate_context_tokens(sequence_tokens)
    completion_tokens = 4096
    prompt = [2 + index % 256 for index in range(sequence_tokens - completion_tokens)]
    completion = [2 + index % 256 for index in range(completion_tokens)]
    row = {
        "prompt_ids": prompt,
        "completion_ids": completion,
        "old_per_token_logps": [-1.0] * completion_tokens,
        "advantage": 1.0,
        "trajectory": 0,
        "turn": 0,
        "terminal_reward": 0.0,
        "policy_version": policy_version,
        "padding": False,
    }
    return {
        "rows": [{**row, "trajectory": index} for index in range(4)],
        "rewards": [0.0] * 4,
        "trainable_tokens": completion_tokens * 4,
        "context_tokens": sequence_tokens,
        "policy_version": policy_version,
        "synthetic_memory_probe_only": True,
    }


def measure_backward(trainer, group: dict) -> dict:
    import torch

    trainer.model.train()
    row = group["rows"][0]
    for _iteration in range(2):
        loss = trainer.training_step(trainer.model, row)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite loss during memory-only backward check")
    torch.cuda.synchronize()
    return {
        "sequence_tokens": group["context_tokens"],
        "prompt_tokens": len(row["prompt_ids"]),
        "reply_tokens": len(row["completion_ids"]),
        "accumulated_backward_passes": 2,
        "synthetic_inputs_not_rollouts": True,
        "optimizer_steps": 0,
        "free_device_bytes": torch.cuda.mem_get_info()[0],
        "total_device_bytes": torch.cuda.mem_get_info()[1],
    }
