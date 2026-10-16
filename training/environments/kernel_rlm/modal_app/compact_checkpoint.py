"""Save LoRA training state without duplicating frozen pretrained parameters."""


def patch_compact_lora_checkpoints(prime_root: str = "/root/prime_rl") -> None:
    import ast
    from pathlib import Path

    root = Path(prime_root)
    path = root / "trainer/ckpt.py"
    source = path.read_text()
    marker = "    def checkpoint_options(self):\n"
    filter_marker = "        frozen_fqns = set()\n"
    if marker in source and filter_marker not in source:
        raise RuntimeError("Rebuild the image from pinned Prime source before upgrading this patch")
    if marker not in source:
        replacements = (
            (
                "from torch.distributed.checkpoint.state_dict import get_state_dict, set_model_state_dict, set_state_dict",
                "from torch.distributed.checkpoint.state_dict import StateDictOptions, _get_fqns, get_state_dict, set_model_state_dict, set_state_dict",
            ),
            (
                "    def state_dict(self) -> dict[str, Any]:\n",
                "    def checkpoint_options(self):\n"
                "        compact = any('lora_' in name and parameter.requires_grad "
                "for name, parameter in self.model.named_parameters())\n"
                "        return StateDictOptions(strict=not compact)\n\n"
                "    def state_dict(self) -> dict[str, Any]:\n",
            ),
            (
                "get_state_dict(self.model, checkpoint_optimizers)",
                "get_state_dict(self.model, checkpoint_optimizers, options=self.checkpoint_options())\n"
                "        frozen_fqns = set()\n"
                "        if not self.checkpoint_options().strict:\n"
                "            for parameter_name, parameter in self.model.named_parameters():\n"
                "                if not parameter.requires_grad:\n"
                "                    frozen_fqns.update(\n"
                "                        name.replace('.base_layer.', '.')\n"
                "                        for name in _get_fqns(self.model, parameter_name)\n"
                "                    )\n"
                "            model_state_dict = {\n"
                "                name: tensor for name, tensor in model_state_dict.items()\n"
                "                if name not in frozen_fqns\n"
                "            }",
            ),
            (
                'set_model_state_dict(self.model, model_state_dict=state_dict["model"])',
                'set_model_state_dict(self.model, model_state_dict=state_dict["model"], options=self.checkpoint_options())',
            ),
            (
                '                optim_state_dict=state_dict["optimizers"],\n',
                '                optim_state_dict=state_dict["optimizers"],\n'
                "                options=self.checkpoint_options(),\n",
            ),
        )
        for anchor, replacement in replacements:
            if source.count(anchor) != 1:
                raise RuntimeError(f"Prime checkpoint anchor changed: {anchor}")
            source = source.replace(anchor, replacement, 1)
        ast.parse(source)
        path.write_text(source)
    path = root / "trainer/rl/train.py"
    source = path.read_text()
    anchor = "    loading_from_ckpt_later = checkpoint_step is not None\n"
    replacement = (
        "    loading_from_ckpt_later = checkpoint_step is not None and config.model.lora is None\n"
    )
    if replacement not in source:
        if source.count(anchor) != 1:
            raise RuntimeError("Prime resume/base-loading anchor changed")
        source = source.replace(anchor, replacement, 1)
        ast.parse(source)
        path.write_text(source)


def check_compact_checkpoint_roundtrip() -> dict:
    import copy
    import tempfile
    from pathlib import Path

    import torch
    from prime_rl.trainer.ckpt import AppState, Progress
    from prime_rl.trainer.models.layers.lora.base import (
        set_lora_num_tokens,
        set_multilora_scaling,
    )
    from prime_rl.trainer.models.layers.lora.multi_linear import MultiLoRALinear
    from torch.distributed.checkpoint import FileSystemReader, load, save
    from torch.distributed.checkpoint.state_dict import get_state_dict

    class TinyLoRA(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = MultiLoRALinear(
                torch.nn.Linear(4, 4, bias=False), rank=2, n_adapters=1, use_grouped_mm=False
            )
            self.register_buffer("counter", torch.ones(1))

        def forward(self, inputs):
            return self.proj(inputs)

    def components(base):
        model = TinyLoRA()
        model.proj.base_layer.load_state_dict(base)
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad], lr=0.01
        )
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=1)
        progress = Progress(step=1, total_tokens=16, total_samples=4)
        return model, optimizer, scheduler, progress

    def update(model, optimizer, scheduler, inputs):
        optimizer.zero_grad()
        model(inputs).square().mean().backward()
        optimizer.step()
        scheduler.step()

    torch.manual_seed(42)
    set_lora_num_tokens(torch.tensor([4]), reset_reference=True)
    set_multilora_scaling(torch.ones(1), reset_reference=True)
    base = copy.deepcopy(TinyLoRA().proj.base_layer.state_dict())
    model, optimizer, scheduler, progress = components(base)
    inputs = torch.randn(4, 4)
    update(model, optimizer, scheduler, inputs)
    trained = copy.deepcopy(model.state_dict())
    results = []
    with tempfile.TemporaryDirectory() as temporary:
        for legacy in (False, True):
            directory = Path(temporary) / ("legacy" if legacy else "compact")
            if legacy:
                model_state, optimizer_state = get_state_dict(model, [optimizer])
                payload = {
                    "app": {
                        "model": model_state,
                        "optimizers": optimizer_state,
                        "scheduler": scheduler.state_dict(),
                        "progress": {"step": 1, "total_tokens": 16, "total_samples": 4},
                    }
                }
            else:
                payload = {"app": AppState(model, [optimizer], scheduler, progress)}
            save(payload, checkpoint_id=str(directory))
            keys = list(FileSystemReader(str(directory)).read_metadata().state_dict_metadata)
            assert any("lora_B" in key for key in keys)
            assert ("app.model.proj.weight" in keys) is legacy
            restored, restored_optimizer, restored_scheduler, restored_progress = components(base)
            load(
                {
                    "app": AppState(
                        restored, [restored_optimizer], restored_scheduler, restored_progress
                    )
                },
                checkpoint_id=str(directory),
            )
            for key, tensor in trained.items():
                torch.testing.assert_close(restored.state_dict()[key], tensor, rtol=0, atol=0)
            assert restored_progress.step == 1
            reference = components(base)
            reference[0].load_state_dict(trained)
            reference[1].load_state_dict(copy.deepcopy(optimizer.state_dict()))
            reference[2].load_state_dict(copy.deepcopy(scheduler.state_dict()))
            update(reference[0], reference[1], reference[2], inputs)
            update(restored, restored_optimizer, restored_scheduler, inputs)
            for key, tensor in reference[0].state_dict().items():
                torch.testing.assert_close(restored.state_dict()[key], tensor, rtol=0, atol=0)
            results.append(
                {
                    "legacy_full_checkpoint": legacy,
                    "next_update_matches": True,
                    "bytes": sum(
                        file.stat().st_size for file in directory.rglob("*") if file.is_file()
                    ),
                }
            )
    return {
        "status": "passed",
        "roundtrips": results,
        "scope": "Real CPU DCP save/load with Prime MultiLoRALinear key-remapping hooks and next-update equivalence with AdamW; Laguna/Muon GPU resume still needs validation",
    }
