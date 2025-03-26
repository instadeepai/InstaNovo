import copy
import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
from lightning.pytorch.strategies import DDPStrategy
from omegaconf import DictConfig

from instanovo.transformer.train import (
    NeptuneSummaryWriter,
    WarmupScheduler,
    _get_strategy,
    _set_author_neptune_api_token,
    main,
    train,
)


@patch("lightning.pytorch.Trainer.fit")
def test_train(
    mock_fit: Any,
    instanovo_config: DictConfig,
    dir_paths: tuple[str, str],
) -> None:
    """Test the transformer training function with different run configurations."""
    root_dir, _ = dir_paths
    temp_config = copy.deepcopy(instanovo_config)

    # Check training when shards = False
    temp_config["use_shards"] = False
    train(temp_config)
    assert mock_fit.call_count == 1

    # Check training when valid_path = None
    temp_config["use_shards"] = True
    temp_config["valid_path"] = None
    train(temp_config)
    assert mock_fit.call_count == 2

    # Check training when instanovo_model is given
    temp_config["train_from_scratch"] = False
    temp_config["resume_checkpoint"] = os.path.join(root_dir, "model.ckpt")
    train(temp_config)
    assert mock_fit.call_count == 3

    # Check training when blacklist is given
    temp_config["train_from_scratch"] = True
    temp_config["blacklist"] = "./tests/instanovo_test_resources/example_data/blacklist.csv"
    train(temp_config)
    assert mock_fit.call_count == 4


@patch("instanovo.transformer.train._set_author_neptune_api_token")
@patch("instanovo.transformer.train.train")
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


def test_get_strategy() -> None:
    """Test the get strategy function for the Trainer."""
    with patch("torch.cuda.device_count", return_value=1):
        assert _get_strategy() == "auto"

    with patch("torch.cuda.device_count", return_value=2):
        strategy = _get_strategy()
        assert isinstance(strategy, DDPStrategy)


def test_set_neptune_token() -> None:
    """Test the setting of the neptune API variable."""
    with patch.dict(os.environ, {}, clear=True):
        _set_author_neptune_api_token()
        assert "VCS_AUTHOR_EMAIL" not in os.environ
        assert "NEPTUNE_API_TOKEN" not in os.environ

    with patch.dict(os.environ, {"VCS_AUTHOR_EMAIL": "a-test.email@example.com"}, clear=True):
        _set_author_neptune_api_token()
        assert os.environ["VCS_AUTHOR_EMAIL"] == "a-test.email@example.com"
        assert "NEPTUNE_API_TOKEN" not in os.environ

    with patch.dict(
        os.environ,
        {
            "VCS_AUTHOR_EMAIL": "a-test.email@example.com",
            "A_TEST_EMAIL__NEPTUNE_API_TOKEN": "example_api_token",
        },
        clear=True,
    ):
        _set_author_neptune_api_token()
        assert os.environ["NEPTUNE_API_TOKEN"] == "example_api_token"


def test_neptune_summary_writer(
    tmp_path: Path,
) -> None:
    """Test the neptune summary writer function."""
    mock_neptune_run = MagicMock()

    nsw = NeptuneSummaryWriter(log_dir=str(tmp_path), run=mock_neptune_run)
    assert nsw.run == mock_neptune_run

    nsw.add_scalar(tag="fake_tag", scalar_value=1.0, global_step=1)
    mock_neptune_run["fake_tag"].append.assert_called_once_with(1.0, step=1)


def test_warmup_scheduler() -> None:
    """Test linear warmup scheduler."""
    param = torch.nn.Parameter(torch.randn(1, requires_grad=True))
    mock_optimiser = torch.optim.Optimizer(params=[param], defaults={"lr": 0.1})

    ws = WarmupScheduler(optimizer=mock_optimiser, warmup=5)
    assert ws.warmup == 5
    assert ws.get_lr() == [0.0]
    assert ws.get_lr_factor(epoch=1) == 0.2
