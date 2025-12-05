from typing import Any, Dict, Optional

import torch
from omegaconf import DictConfig

from instanovo.__init__ import console
from instanovo.utils.colorlogging import ColorLog

logger = ColorLog(console, __name__).logger


def get_device_capabilities() -> Dict[str, bool]:
    """Check device capabilities.

    Returns:
        Dict containing availability flags for different device types.
    """
    return {
        "cuda": torch.cuda.is_available(),
        "mps": torch.backends.mps.is_available(),
    }


def detect_device() -> str:
    """Detect the best available device for computation.

    Returns:
        str: The selected device ('cpu', 'cuda', 'mps').
    """
    capabilities = get_device_capabilities()

    if capabilities["cuda"]:
        return "cuda"
    elif capabilities["mps"]:
        return "mps"
    else:
        return "cpu"


def get_device_config_updates(device: str) -> Dict[str, Any]:
    """Get configuration updates needed for the specified device.

    Args:
        device: The device type ('cpu', 'cuda', 'mps').

    Returns:
        Dict containing configuration updates for the device.
    """
    config_updates: Dict[str, Any] = {}

    if device == "cuda":
        config_updates.update(
            {
                "mps": False,
                "force_cpu": False,
            }
        )
    elif device == "mps":
        config_updates.update(
            {
                "mps": True,
                "force_fp32": True,
                "force_cpu": False,
            }
        )
        config_updates["model"] = {"peak_embedding_dtype": "float32"}
    elif device == "cpu":
        config_updates["force_cpu"] = True
    else:
        raise ValueError(f"Unknown device: {device}, no configuration updates applied.")

    return config_updates


def apply_device_config(config: DictConfig, device: Optional[str] = None) -> str:
    """Apply device-specific configuration to the provided config.

    Args:
        config: The configuration object to update.
        device: Optional device to use. If None, will auto-detect.

    Returns:
        str: The device that was applied.
    """
    if device is None:
        device = detect_device()

    config_updates = get_device_config_updates(device)

    for key, value in config_updates.items():
        if key == "model" and isinstance(value, dict):
            if "model" in config:
                for model_key, model_value in value.items():
                    config["model"][model_key] = model_value
        else:
            config[key] = value

    return device


def validate_and_configure_device(config: DictConfig) -> None:
    """Validate device configuration and apply necessary updates.

    Args:
        config: The configuration object to validate and update.
    """
    capabilities = get_device_capabilities()

    if capabilities["mps"] and not config.get("mps", False):
        logger.warning(
            "The Metal Performance Shaders (MPS) backend for Apple silicon devices is available, but not requested. "
            "See https://developer.apple.com/documentation/metalperformanceshaders for more information. "
            "Please set 'mps' to True in the configuration if you would like to use MPS."
        )

    if config.get("mps", False):
        if not capabilities["mps"]:
            logger.warning("MPS is not available, setting mps to False.")
            config["mps"] = False
        elif config.get("force_cpu", False):
            logger.warning("Force CPU is set to True, setting mps to False.")
            config["mps"] = False
        else:
            logger.info("MPS is set to True, forcing fp32. Note that performance on MPS may differ to performance on CUDA.")
            config["force_fp32"] = True  # Force fp32 if using mps

    elif not config.get("force_cpu", False) and not capabilities["cuda"]:
        logger.warning("CUDA is not available, setting force_cpu to True.")
        config["force_cpu"] = True


def check_device(config: Optional[DictConfig] = None) -> str:
    """Legacy function for backward compatibility.

    Args:
        config: Optional configuration object to update.

    Returns:
        str: The selected device.
    """
    if config is not None:
        return apply_device_config(config)
    else:
        return detect_device()
