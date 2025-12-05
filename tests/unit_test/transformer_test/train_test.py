import copy
import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import torch
from omegaconf import DictConfig

from instanovo.common.scheduler import WarmupScheduler
from instanovo.common.utils import NeptuneSummaryWriter, _set_author_neptune_api_token
from instanovo.transformer.train import TransformerTrainer
from instanovo.utils.device_handler import check_device


@patch("instanovo.transformer.train.TransformerTrainer.train", autospec=True)
def test_train(
    mock_train: Any,
    instanovo_config: DictConfig,
    dir_paths: tuple[str, str],
) -> None:
    """Test the transformer training function with different run configurations."""
    root_dir, _ = dir_paths
    temp_config = copy.deepcopy(instanovo_config)

    # Check training when valid_path = None
    temp_config.dataset["valid_path"] = None
    check_device(config=temp_config)
    TransformerTrainer(temp_config).train()
    assert mock_train.call_count == 1

    # Check training when instanovo_model is given
    temp_config["resume_checkpoint_path"] = os.path.join(root_dir, "model.ckpt")
    TransformerTrainer(temp_config).train()
    assert mock_train.call_count == 2

    # Check training when blacklist is given
    # TODO: Add back in
    # temp_config["resume_checkpoint_path"] = None
    # temp_config.dataset["blacklist"] = "tests/instanovo_test_resources/example_data/blacklist.csv"
    # TransformerTrainer(temp_config).train()
    # assert mock_train.call_count == 3


# Move to common tests
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


# Move to common tests
def test_neptune_summary_writer(
    tmp_path: Path,
) -> None:
    """Test the neptune summary writer function."""
    mock_neptune_run = MagicMock()

    nsw = NeptuneSummaryWriter(log_dir=str(tmp_path), run=mock_neptune_run)
    assert nsw.run == mock_neptune_run

    nsw.add_scalar(tag="fake_tag", scalar_value=1.0, global_step=1)
    mock_neptune_run["fake_tag"].append.assert_called_once_with(1.0, step=1)

    nsw.add_text(tag="fake_text_targ", text_string="my fake text", global_step=1)
    assert mock_neptune_run["fake_text_targ"].append.call_count == 1


# Move to common tests
def test_warmup_scheduler() -> None:
    """Test linear warmup scheduler."""
    param = torch.nn.Parameter(torch.randn(1, requires_grad=True))
    mock_optimiser = torch.optim.Optimizer(params=[param], defaults={"lr": 0.1})

    ws = WarmupScheduler(optimizer=mock_optimiser, warmup=5)
    assert ws.warmup == 5
    assert ws.get_lr() == [0.0]
    assert ws.get_lr_factor(epoch=1) == 0.2
