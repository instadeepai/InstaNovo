from typing import Any
from unittest.mock import patch

import pytest
from omegaconf import DictConfig

from instanovo.diffusion.train import (
    main,
)


@patch("instanovo.diffusion.train._set_author_neptune_api_token")
@patch("instanovo.diffusion.train.train")
def test_main(mock_train: Any, mock_set_api_token: Any) -> None:
    """Test the main function call of train."""
    mock_config = DictConfig({})
    mock_config["n_gpu"] = 1

    main(mock_config)
    mock_set_api_token.assert_called_once()
    mock_train.assert_called_once_with(mock_config)

    mock_config["n_gpu"] = 2
    with pytest.raises(ValueError, match="n_gpu > 1 currently not supported."):
        main(mock_config)
