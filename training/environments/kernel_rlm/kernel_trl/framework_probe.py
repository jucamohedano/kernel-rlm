"""One bounded native trl update on two GPUs; no inference service or task training."""

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

from kernel_trl import framework_hooks
from kernel_trl.framework_hooks import (
    materialize_meta_adapters,
    materialize_nonpersistent_buffers,
    trl_muon_trainer,
)


def trl_update(model, mesh, directory: Path, records: list) -> dict:
    import torch
    import torch.distributed as dist
    from datasets import Dataset
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast
    from trl import GRPOConfig

    vocabulary = {f"token{index}": index for index in range(128)}
    tokenizer_backend = Tokenizer(WordLevel(vocabulary, unk_token="token3"))
    tokenizer_backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_backend,
        pad_token="token9",
        bos_token="token2",
        eos_token="token2",
        unk_token="token3",
    )

    def rollout(prompts, trainer):
        samples = [dist.get_rank() * 2 + index for index in range(len(prompts))]
        sequences = torch.tensor(
            [[2, 10, 11 + sample % 2, 7, 2] for sample in samples], device="cuda"
        )
        with torch.no_grad():
            scores = (
                trainer.model(sequences, use_cache=False).logits[:, 1:-1].float().log_softmax(-1)
            )
            logprobs = scores.gather(-1, sequences[:, 2:].unsqueeze(-1)).squeeze(-1).tolist()
        return {
            "prompt_ids": [[2, 10] for sample in samples],
            "completion_ids": [[11 + sample % 2, 7, 2] for sample in samples],
            "logprobs": logprobs,
            "env_mask": [[1, 0, 1] for sample in samples],
            "fixture_reward": [sample / 3 for sample in samples],
        }

    def reward(fixture_reward, **kwargs):
        return fixture_reward

    arguments = GRPOConfig(
        output_dir=str(directory / "trainer-output"),
        bf16=True,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        num_generations=4,
        max_steps=1,
        max_completion_length=3,
        learning_rate=1e-5,
        weight_decay=0.01,
        lr_scheduler_type="constant",
        beta=0.0,
        report_to="none",
        save_strategy="no",
        logging_steps=1,
        disable_tqdm=True,
        gradient_checkpointing=False,
        use_vllm=False,
        fsdp="full_shard auto_wrap",
        fsdp_config={
            "version": 2,
            "transformer_layer_cls_to_wrap": ["LagunaDecoderLayer"],
            "cpu_ram_efficient_loading": True,
            "activation_checkpointing": True,
            "reshard_after_forward": True,
        },
    )
    trainer_class = trl_muon_trainer(
        mesh, lambda optimizer: observe_updates(optimizer, model, directory, records)
    )
    trainer = trainer_class(
        model=model,
        reward_funcs=reward,
        args=arguments,
        train_dataset=Dataset.from_dict({"prompt": ["token10"] * 4}),
        processing_class=tokenizer,
        rollout_func=rollout,
    )
    result = trainer.train()
    if trainer.state.global_step != 1 or len(records) != 1:
        raise ValueError("Expected exactly one native TRL optimizer update")
    return {
        "native_path": "GRPOTrainer.train + Accelerate FSDP2 + custom rollout/env_mask + Muon optimizer hook",
        "global_step": trainer.state.global_step,
        "metrics": result.metrics,
        "log_history": trainer.state.log_history,
        "model_inventory_after": audit_storage(trainer.model),
        "synthetic_rollouts_not_kernel_tasks": True,
        "rollout_logprobs_scored_by_current_policy": True,
    }


def prepare_model(model):
    materialized = materialize_meta_adapters(model)
    buffers = materialize_nonpersistent_buffers(model)
    return materialized, buffers


def run(arguments, directory: Path) -> dict:
    return run_framework_probe(
        arguments,
        directory,
        backend="trl",
        update=trl_update,
        prepare=prepare_model,
        versions=("torch", "transformers", "peft", "accelerate", "trl", "dion"),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", nargs="?", choices=("trl",), default="trl")
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
