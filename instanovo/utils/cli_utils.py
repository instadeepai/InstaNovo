from typing import List, Optional

from omegaconf import DictConfig

from instanovo.__init__ import console
from instanovo.utils.colorlogging import ColorLog

logger = ColorLog(console, __name__).logger


def compose_config(
    config_path: Optional[str] = None,
    config_name: Optional[str] = None,
    overrides: Optional[List[str]] = None,
) -> DictConfig:
    """Compose Hydra configuration with given overrides.

    Args:
        config_path: Relative path to config directory
        config_name: Name of the base config file
        overrides: List of Hydra override strings

    Returns:
        DictConfig: Composed configuration
    """
    from hydra import compose, initialize

    logger.info(f"Reading config from '{config_path}' with name '{config_name}'.")
    with initialize(config_path=config_path, version_base=None):
        cfg = compose(config_name=config_name, overrides=overrides, return_hydra_config=False)
        return cfg
