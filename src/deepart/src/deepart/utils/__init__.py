import torch

__all__ = [
    "get_device",
]


def get_device():
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    return device
