"""Native TRL hooks for BF16 Laguna and Moonlight Muon."""

from kernel_training.laguna_policy import build_muon


def materialize_meta_adapters(model) -> int:
    import torch

    count = 0
    for module in model.modules():
        for name, parameter in list(module.named_parameters(recurse=False)):
            if parameter.requires_grad and parameter.is_meta:
                module.register_parameter(
                    name,
                    torch.nn.Parameter(
                        torch.empty(parameter.shape, dtype=parameter.dtype, device="cpu")
                    ),
                )
                count += 1
    return count


def materialize_nonpersistent_buffers(model) -> list[str]:
    import torch
    import torch.distributed as dist

    restored = []
    for module_name, module in model.named_modules():
        for name in sorted(module._non_persistent_buffers_set):
            buffer = module._buffers.get(name)
            if buffer is None:
                continue
            if dist.get_rank() == 0:
                if buffer.is_meta:
                    raise ValueError("Rank zero must supply materialized nonpersistent buffers")
                value = buffer.detach().cuda()
            else:
                value = torch.empty(buffer.shape, dtype=buffer.dtype, device="cuda")
            dist.broadcast(value, src=0)
            if not torch.isfinite(value).all():
                raise ValueError("Nonfinite nonpersistent buffer")
            module.register_buffer(name, value.cpu(), persistent=False)
            restored.append(f"{module_name}.{name}")
    return restored


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
