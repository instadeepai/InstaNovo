import copy
from typing import Any
from unittest.mock import patch

from omegaconf import DictConfig

from instanovo.diffusion.train import DiffusionTrainer
from instanovo.utils.device_handler import check_device


@patch("instanovo.diffusion.train.DiffusionTrainer.train", autospec=True)
def test_main(mock_train: Any, instanovoplus_config: DictConfig) -> None:
    """Test the main function call of train."""
    temp_config = copy.deepcopy(instanovoplus_config)
    check_device(config=temp_config)
    DiffusionTrainer(temp_config).train()
    mock_train.assert_called_once()
