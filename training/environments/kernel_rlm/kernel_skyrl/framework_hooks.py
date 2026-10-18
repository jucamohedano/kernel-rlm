"""Native trainer hooks for the bounded Laguna BF16/Muon compatibility gates."""

import hashlib
from pathlib import Path

from kernel_skyrl.laguna_policy import build_muon


def enable_lazy_padding_import(source: Path) -> dict:
    path = source / "skyrl/backends/skyrl_train/workers/model_wrapper.py"
    original = path.read_text()
    eager = "from flash_attn.bert_padding import pad_input, unpad_input\n"
    if original.count(eager) != 1:
        raise ValueError("Expected the exact pinned SkyRL padding import, once")
    replacement = (
        "def pad_input(*args, **kwargs):\n"
        "    from flash_attn.bert_padding import pad_input as implementation\n"
        "    return implementation(*args, **kwargs)\n\n"
        "def unpad_input(*args, **kwargs):\n"
        "    from flash_attn.bert_padding import unpad_input as implementation\n"
        "    return implementation(*args, **kwargs)\n\n"
    )
    updated = original.replace(eager, replacement)
    path.write_text(updated)
    return {
        "path": str(path.relative_to(source)),
        "original_sha256": hashlib.sha256(original.encode()).hexdigest(),
        "patched_sha256": hashlib.sha256(updated.encode()).hexdigest(),
        "reason": "Padded SDPA does not use FlashAttention padding helpers; lazy import only",
        "flash_attention_required_if_padding_helpers_called": True,
    }


def skyrl_muon_strategy():
    from skyrl.backends.skyrl_train.distributed.fsdp_strategy import FSDPStrategy
    from transformers import get_scheduler

    class LagunaMuonStrategy(FSDPStrategy):
        def prepare(self, *models_or_model_optim_pairs):
            if len(models_or_model_optim_pairs) != 1:
                raise ValueError("The gate prepares exactly one policy")
            pair = models_or_model_optim_pairs[0]
            if not isinstance(pair, tuple) or len(pair) != 3 or pair[1:] != (None, None):
                raise ValueError("Supply a policy and no pre-existing optimizer/scheduler")
            configuration = self.optimizer_config
            if configuration is None:
                raise ValueError("Explicit optimizer settings are required")
            self.optimizer_config = None
            try:
                model, optimizer, scheduler = super().prepare(pair)
            finally:
                self.optimizer_config = configuration
            if optimizer is not None or scheduler is not None:
                raise ValueError("Upstream unexpectedly constructed an optimizer")
            optimizer = build_muon(
                model, self.device_mesh, configuration.lr, configuration.weight_decay
            )
            scheduler = get_scheduler(
                configuration.scheduler,
                optimizer,
                num_warmup_steps=configuration.num_warmup_steps,
                num_training_steps=self.total_training_steps,
            )
            return model, optimizer, scheduler

    return LagunaMuonStrategy


def trl_muon_trainer(mesh, observe_optimizer):
    from trl import GRPOTrainer

    class LagunaMuonGRPOTrainer(GRPOTrainer):
        def create_optimizer(self, model=None):
            if self.optimizer is None:
                self.optimizer = build_muon(
                    self.model if model is None else model,
                    mesh,
                    self.args.learning_rate,
                    self.args.weight_decay,
                )
                observe_optimizer(self.optimizer)
            return self.optimizer

    return LagunaMuonGRPOTrainer
