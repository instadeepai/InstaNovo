from typing import Optional

import torch
from omegaconf import DictConfig


def check_device(config: Optional[DictConfig] = None, device: Optional[str] = None) -> str:
    """Checks and returns the appropriate device for computation.

    Either `config` or `device` can be specified:
        - `device = check_device(config=config)`
        - `device = check_device(device=device)`

    Raises:
        RuntimeError: If MPS is detected and the device is not set to CPU.
        RuntimeError: If no GPU (CUDA/MPS) is detected but the device is not set to CPU.

    Returns:
        str: The selected device ('cpu', 'cuda', 'cuda:0', etc.).
    """
    if device is None:
        if config is None:
            raise ValueError("Either 'config' or 'device' must be specified.")
        device = config.get("device", "auto")

    device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device

    if device != "cpu" and not torch.cuda.is_available():
        if torch.backends.mps.is_available():
            raise RuntimeError(
                "MPS is available, but not supported. Please set device='cpu' in the config."
            )

        else:
            raise RuntimeError(
                "No GPU (CUDA or MPS) detected, but device is not set to 'cpu'."
                " Please set device='cpu' in the config."
            )

    return device
