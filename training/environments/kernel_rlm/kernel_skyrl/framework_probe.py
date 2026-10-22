"""One bounded native skyrl update on two GPUs; no inference service or task training."""

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

from kernel_training import diagnostics, laguna_policy
from kernel_training.diagnostics import observe_updates, run_framework_probe
from kernel_training.laguna_policy import audit_storage

from kernel_skyrl import framework_hooks
from kernel_skyrl.framework_hooks import skyrl_muon_strategy


def skyrl_update(model, directory: Path, records: list) -> dict:
    import numpy as np
    import torch
    import torch.distributed as dist
    from skyrl.backends.skyrl_train.utils.ppo_utils import (
        compute_grpo_outcome_advantage,
        ppo_policy_loss,
    )
    from skyrl.backends.skyrl_train.workers.model_wrapper import HFModelWrapper
    from skyrl.train.config import AlgorithmConfig, FSDPConfig, ModelConfig, OptimizerConfig

    configuration = ModelConfig()
    configuration.lora.rank = 8
    configuration.lora.alpha = 16
    strategy = skyrl_muon_strategy()(
        FSDPConfig(
            fsdp_size=2, wrap_policy={"transformer_layer_cls_to_wrap": ["LagunaDecoderLayer"]}
        ),
        optimizer_config=OptimizerConfig(lr=1e-5, weight_decay=0.01, offload_after_step=False),
        model_config=configuration,
        num_training_steps=1,
    )
    strategy.setup_distributed()
    wrapper = HFModelWrapper(model, use_flash_attention_2=False, remove_microbatch_padding=False)
    wrapper.logprobs_chunk_size = 16
    wrapper, optimizer, scheduler = strategy.prepare((wrapper, None, None))
    observe_updates(optimizer, wrapper, directory, records)
    wrapper.train()
    mask = torch.tensor([[1, 0, 1]] * 4, device="cuda", dtype=torch.float32)
    rewards = torch.zeros_like(mask)
    rewards[:, -1] = torch.tensor([0, 1 / 3, 2 / 3, 1], device="cuda")
    advantages, returns = compute_grpo_outcome_advantage(rewards, mask, np.array([0] * 4))
    losses = []
    algorithm = AlgorithmConfig(use_kl_loss=False)
    for microbatch in range(2):
        sample = dist.get_rank() * 2 + microbatch
        sequences = torch.tensor([[2, 10, 11 + sample % 2, 7, 2]], device="cuda")
        logprobs = wrapper(sequences, num_actions=3, attention_mask=torch.ones_like(sequences))
        loss, metrics = ppo_policy_loss(
            logprobs,
            logprobs.detach(),
            advantages[sample : sample + 1],
            algorithm,
            mask[sample : sample + 1],
        )
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite native SkyRL policy loss")
        strategy.backward(loss / 2, wrapper, optimizer)
        losses.append(float(loss.detach()))
    norm = strategy.optimizer_step(optimizer, wrapper, scheduler)
    if norm is None or not bool(torch.isfinite(norm)):
        raise ValueError("Native SkyRL gradient clipping did not return a finite norm")
    return {
        "native_path": "FSDPStrategy.prepare/backward/optimizer_step + HFModelWrapper + GRPO advantages/PPO loss",
        "losses": losses,
        "gradient_norm": float(diagnostics.local_tensor(norm)),
        "optimizer_steps": len(records),
        "ray_orchestration_tested": False,
        "model_inventory_after": audit_storage(wrapper),
    }


def run(arguments, directory: Path) -> dict:
    return run_framework_probe(
        arguments,
        directory,
        backend="skyrl",
        update=lambda model, mesh, output, records: skyrl_update(model, output, records),
        versions=("torch", "transformers", "peft", "dion"),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", nargs="?", choices=("skyrl",), default="skyrl")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    directory = arguments.output_dir
    directory.mkdir(parents=True, exist_ok=True)
    rank = os.environ["RANK"]
    snapshot = directory / "source" / f"rank-{rank}"
    snapshot.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for module in (framework_hooks, laguna_policy, diagnostics):
        path = Path(module.__file__)
        shutil.copy2(path, snapshot / path.name)
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    shutil.copy2(__file__, snapshot / Path(__file__).name)
    started = time.monotonic()
    try:
        result = run(arguments, directory)
    except Exception as error:
        (directory / f"failure-rank-{rank}.json").write_text(
            json.dumps(
                {
                    "type": type(error).__name__,
                    "error": str(error),
                    "elapsed_seconds": time.monotonic() - started,
                },
                indent=2,
            )
        )
        raise
    result.update(elapsed_seconds=time.monotonic() - started, source_hashes=hashes, rank=int(rank))
    (directory / f"report-rank-{rank}.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
