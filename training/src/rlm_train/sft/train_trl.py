"""Run one local, synthetic CPU step with the pinned TRL SFTTrainer."""

import argparse
import hashlib
import inspect
import json
import math
import os
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import torch
from datasets import Dataset
from tokenizers import Tokenizer, models
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast, TrainerCallback
from trl import SFTConfig, SFTTrainer

from rlm_train.sft.bundle import Bundle, collate_group, load_bundle, masked_nll_sum

TRL_COMMIT = "68ad0529f6b79b4091b4d0f14cbc721f5c0c1b63"
SFT_SOURCE_SHA256 = "7ce7d7e045da9cc830d04131e75a51e12e293d04beb4979c104320497f324ab3"
VERSIONS = {
    "trl": "1.19.0.dev0",
    "transformers": "5.23.0",
    "accelerate": "1.16.0",
    "torch": "2.15.0+cpu",
}
TRACKIO_VERSION = "0.40.0"
TRACKIO_PROJECT = "xs21-synthetic-trl"
LR = 0.01
VOCAB_SIZE = 16
METRICS = ("masked_nll", "target_tokens", "trajectory_groups")


def _check_runtime():
    for package, pin in VERSIONS.items():
        if version(package) != pin:
            raise ValueError(f"unsupported {package} version: require {pin}")
    if (
        hashlib.sha256(Path(inspect.getfile(SFTTrainer)).read_bytes()).hexdigest()
        != SFT_SOURCE_SHA256
    ):
        raise ValueError(f"SFTTrainer source differs from tested TRL {TRL_COMMIT}")
    if any(
        int(os.environ.get(key, default)) != default
        for key, default in (
            ("WORLD_SIZE", 1),
            ("LOCAL_WORLD_SIZE", 1),
            ("RANK", 0),
            ("LOCAL_RANK", -1),
        )
    ):
        raise ValueError(
            "only a single local CPU process is supported; do not use a distributed launcher"
        )


@dataclass(frozen=True)
class _TrajectoryCollator:
    bundle: Bundle

    def __call__(self, samples):
        if len(samples) != 1:
            raise ValueError("one whole trajectory per loader item is required")
        b = self.bundle
        return collate_group(
            b.groups[samples[0]["group_index"]],
            pad_id=b.pad_id,
            member_cap=b.member_cap,
            padded_slots_cap=b.padded_slots_cap,
        )


def _check_args(args):
    if (
        not args.use_cpu
        or args.world_size != 1
        or args.gradient_accumulation_steps != 1
        or args.per_device_train_batch_size != 1
        or args.max_steps != 1
        or args.packing
        or args.eval_packing
        or args.padding_free
        or args.max_length is not None
        or args.dataset_kwargs != {"skip_prepare_dataset": True}
        or args.remove_unused_columns
        or args.completion_only_loss
        or args.assistant_only_loss
        or args.use_liger_kernel
        or args.torch_compile
        or args.deepspeed
        or args.fsdp
        or args.bf16
        or args.fp16
        or args.push_to_hub
        or args.report_to
        or args.loss_type != "nll"
        or args.label_smoothing_factor
    ):
        raise ValueError(
            "only the verified single-group, unpacked, unit-accumulation synthetic CPU path is supported"
        )


class _GroupedTrainer(SFTTrainer):
    # Transformers must neither pass its denominator to the model nor rescale this loss.
    loss_is_scaled_for_ga = True

    def __init__(self, *, bundle, args, model, processing_class):
        _check_args(args)
        super().__init__(
            model=model,
            args=args,
            processing_class=processing_class,
            train_dataset=Dataset.from_dict({"group_index": list(range(len(bundle.groups)))}),
            data_collator=_TrajectoryCollator(bundle),
        )
        if (
            self.accelerator.num_processes != 1
            or self.accelerator.device.type != "cpu"
            or self.accelerator.gradient_accumulation_steps != 1
            or self.accelerator.distributed_type.value != "NO"
        ):
            raise ValueError("unexpected Accelerate execution mode")
        self.model_accepts_loss_kwargs = False
        self.step_metrics = None

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if num_items_in_batch is not None or set(inputs) != {
            "input_ids",
            "attention_mask",
            "labels",
        }:
            raise ValueError("unexpected Trainer denominator or grouped batch fields")
        out = model(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], use_cache=False
        )
        summed_nll, n = masked_nll_sum(out.logits, inputs["labels"])
        count = int(n)
        if count <= 0:
            raise ValueError("empty target mask in trajectory")
        loss = summed_nll / count
        scalar = float(loss.detach())
        if not math.isfinite(scalar):
            raise ValueError("nonfinite masked NLL")
        self.step_metrics = {"masked_nll": scalar, "target_tokens": count, "trajectory_groups": 1}
        return (loss, out) if return_outputs else loss


def _tiny_model(bundle):
    if not 0 <= bundle.pad_id < VOCAB_SIZE or any(
        token >= VOCAB_SIZE
        for group in bundle.groups
        for member in group.members
        for token in member.input_ids
    ):
        raise ValueError("synthetic token IDs do not fit the tiny model vocabulary")
    torch.manual_seed(17)
    return GPT2LMHeadModel(
        GPT2Config(
            vocab_size=VOCAB_SIZE,
            n_positions=bundle.member_cap,
            n_embd=16,
            n_layer=1,
            n_head=2,
            pad_token_id=bundle.pad_id,
            bos_token_id=None,
            eos_token_id=None,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
            attn_implementation="eager",
        )
    )


def _make_trainer(bundle, output_dir):
    _check_runtime()
    model = _tiny_model(bundle)
    # Required by SFTTrainer's constructor; never called on training data.
    vocab = {f"t{i}": i for i in range(VOCAB_SIZE)}
    processing_class = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(models.WordLevel(vocab)), pad_token=f"t{bundle.pad_id}"
    )
    args = SFTConfig(
        output_dir=str(output_dir),
        use_cpu=True,
        max_steps=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        dataloader_num_workers=0,
        seed=17,
        learning_rate=LR,
        warmup_steps=0,
        optim="sgd",
        lr_scheduler_type="constant",
        max_grad_norm=0.0,
        weight_decay=0.0,
        bf16=False,
        fp16=False,
        activation_checkpointing=False,
        report_to=[],
        logging_strategy="no",
        save_strategy="no",
        eval_strategy="no",
        disable_tqdm=True,
        push_to_hub=False,
        packing=False,
        eval_packing=False,
        padding_free=False,
        max_length=None,
        assistant_only_loss=False,
        completion_only_loss=False,
        loss_type="nll",
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False,
    )
    return _GroupedTrainer(bundle=bundle, args=args, model=model, processing_class=processing_class)


class _NumericTrackioCallback(TrainerCallback):
    def __init__(self, trainer, run):
        self.trainer, self.run = trainer, run

    def on_step_end(self, args, state, control, **kwargs):
        metrics = self.trainer.step_metrics
        if (
            metrics is None
            or set(metrics) != set(METRICS)
            or any(
                type(value) not in (int, float) or not math.isfinite(value)
                for value in metrics.values()
            )
        ):
            raise ValueError("Trackio only accepts the numeric aggregate allowlist")
        self.run.log(metrics, step=state.global_step)


@contextmanager
def _local_trackio(output_dir):
    destination = (output_dir / "trackio").resolve()
    # Trackio may resolve remote destinations via its environment, even without explicit arguments.
    forbidden = (
        "TRACKIO_SPACE_ID",
        "TRACKIO_SERVER_URL",
        "TRACKIO_BUCKET_ID",
        "TRACKIO_DATASET_ID",
        "TRACKIO_REGISTRY_BUCKET_ID",
        "SPACE_ID",
        "SPACE_HOST",
    )
    if any(os.environ.get(key) for key in forbidden):
        raise ValueError("remote Trackio configuration is not allowed")
    try:
        installed = version("trackio")
    except ModuleNotFoundError as e:
        raise ValueError(
            f"Trackio is optional and not installed; require trackio=={TRACKIO_VERSION}"
        ) from e
    if installed != TRACKIO_VERSION:
        raise ValueError(f"unsupported Trackio version: require {TRACKIO_VERSION}")
    destination.mkdir(exist_ok=False)
    previous = os.environ.get("TRACKIO_DIR")
    os.environ["TRACKIO_DIR"] = str(destination)
    try:
        import trackio
        from trackio import context_vars, utils

        if utils.TRACKIO_DIR.resolve() != destination or context_vars.current_run.get() is not None:
            raise ValueError("use a fresh process for isolated local Trackio logging")
        run = trackio.init(
            project=TRACKIO_PROJECT,
            name="single-step",
            embed=False,
            auto_log_cpu=False,
            auto_log_gpu=False,
            config={},
            resume="never",
        )
        try:
            yield run
        finally:
            trackio.finish()
    finally:
        if previous is None:
            os.environ.pop("TRACKIO_DIR", None)
        else:
            os.environ["TRACKIO_DIR"] = previous


def run_smoke(directory, manifest_sha256, output_dir, *, allow_synthetic=False, track=False):
    bundle = load_bundle(directory, manifest_sha256, allow_synthetic=allow_synthetic)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    trainer = _make_trainer(bundle, output_dir / "trainer")
    with _local_trackio(output_dir) if track else nullcontext() as run:
        if run is not None:
            trainer.add_callback(_NumericTrackioCallback(trainer, run))
        result = trainer.train()
    if result.global_step != 1 or trainer.step_metrics is None:
        raise RuntimeError("expected exactly one synthetic optimizer step")
    return {"optimizer_steps": result.global_step, **trainer.step_metrics}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-synthetic", action="store_true", required=True)
    parser.add_argument(
        "--trackio", action="store_true", help="log numeric aggregates to local SQLite only"
    )
    args = parser.parse_args(argv)
    try:
        result = run_smoke(
            args.bundle,
            args.manifest_sha256,
            args.output_dir,
            allow_synthetic=args.allow_synthetic,
            track=args.trackio,
        )
    except ValueError as e:
        parser.error(str(e))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
