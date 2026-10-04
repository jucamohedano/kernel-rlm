import torch


class Model(torch.nn.Module):
    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return left + right


def get_inputs() -> list[torch.Tensor]:
    return [
        torch.randn(4096, dtype=torch.float32),
        torch.randn(4096, dtype=torch.float32),
    ]


def get_init_inputs() -> list[object]:
    return []
