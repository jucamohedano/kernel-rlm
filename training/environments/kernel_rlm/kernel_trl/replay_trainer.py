"""Native TRL PPO loss and trainer loop over a verified group of exact RLM turns."""

from kernel_skyrl.framework_hooks import trl_muon_trainer

from kernel_trl.turn_credit import tensors_for_turn


def first_row(rows):
    if len(rows) != 1:
        raise ValueError("This memory baseline uses one turn per microbatch")
    return rows[0]


def rlm_grpo_trainer(mesh, observe_optimizer, group: dict):
    import torch
    from torch.utils.data import DataLoader

    parent = trl_muon_trainer(mesh, observe_optimizer)

    class RLMGRPOTrainer(parent):
        def get_train_dataloader(self):
            loader = DataLoader(group["rows"], batch_size=1, shuffle=False, collate_fn=first_row)
            return self.accelerator.prepare(loader)

        def training_step(self, model, inputs, num_items_in_batch=None):
            if self.state.global_step != 0:
                raise ValueError("This bounded trainer refuses stale policy data")
            model.train()
            prepared = tensors_for_turn(inputs, group["trainable_tokens"], self.accelerator.device)
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, prepared)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite per-turn TRL loss")
            self.accelerator.backward(loss * self.accelerator.gradient_accumulation_steps)
            return loss.detach()

    return RLMGRPOTrainer
