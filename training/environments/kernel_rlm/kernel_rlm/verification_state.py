"""Reproducible nondefault BatchNorm state for numerical verification."""

import hashlib
import json

VERIFICATION_PROTOCOL = "batchnorm-state-v2"


def prepare_batchnorm_state(model, trial: int) -> list[dict]:
    import torch

    if type(trial) is not int or trial < 0:
        raise ValueError("Verification trial must be a nonnegative integer")
    module_types = (
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.nn.SyncBatchNorm,
    )
    cases = []
    with torch.no_grad():
        for name, module in model.named_modules():
            if type(module) not in module_types:
                continue
            identity = f"{VERIFICATION_PROTOCOL}:{trial}:{name}".encode()
            seed = int.from_bytes(hashlib.sha256(identity).digest()[:8], "big") % (2**63)
            generator = torch.Generator(device="cpu").manual_seed(seed)
            state = {}
            for key, lower, upper in (
                ("weight", 0.5, 1.5),
                ("bias", -0.75, 0.75),
                ("running_mean", -1.0, 1.0),
                ("running_var", 0.5, 2.0),
            ):
                tensor = getattr(module, key)
                if tensor is None:
                    continue
                values = torch.empty(tensor.shape, dtype=torch.float32, device="cpu")
                values.uniform_(lower, upper, generator=generator)
                tensor.copy_(values.to(device=tensor.device, dtype=tensor.dtype))
                state[key] = tensor.detach().cpu().float().tolist()
            if state:
                digest = hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()
                cases.append(
                    {"module": name, "seed": seed, "state_keys": list(state), "sha256": digest}
                )
    return cases
