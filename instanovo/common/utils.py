import math
import os
import time
import traceback
from typing import Any, Dict, List, Optional

import neptune
from torch.utils.tensorboard import SummaryWriter

from instanovo.__init__ import console
from instanovo.utils.colorlogging import ColorLog
from instanovo.utils.data_handler import SpectrumDataFrame

logger = ColorLog(console, __name__).logger


# Used to record additional training state parameters
# This is only used for accelerate.save_state and
# accelerate.load_state. (resuming runs)
class TrainingState:
    """Training state for tracking training progress.

    This class is used by Accelerate to save and load training state during
    checkpointing and resuming training runs. It tracks the current epoch
    and global step of training.
    """

    def __init__(self) -> None:
        """Initialize training state with zeroed counters."""
        self._global_step: int = 0
        self._epoch: int = 0

    @property
    def global_step(self) -> int:
        """Get the current global step."""
        return self._global_step

    @property
    def epoch(self) -> int:
        """Get the current epoch."""
        return self._epoch

    def state_dict(self) -> dict[str, Any]:
        """Get the state dictionary for saving.

        Returns:
            dict[str, Any]: Dictionary containing the current training state.
        """
        return {
            "global_step": self.global_step,
            "epoch": self.epoch,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load state from a dictionary.

        Args:
            state_dict: Dictionary containing the training state to load.
        """
        self._global_step = state_dict["global_step"]
        self._epoch = state_dict["epoch"]

    def step(self) -> None:
        """Step the global step."""
        self._global_step += 1

    def step_epoch(self) -> None:
        """Step the epoch."""
        self._epoch += 1

    def unstep_epoch(self) -> None:
        """Unstep the epoch."""
        self._epoch -= 1


class Timer:
    """Timer for training and validation."""

    def __init__(self, total_steps: int | None = None):
        self.start_time = time.time()
        self.total_steps = total_steps
        self.current_step = 0

    def start(self) -> None:
        """Restart the timer."""
        self.start_time = time.time()
        self.current_step = 0

    def step(self) -> None:
        """Step the timer."""
        self.current_step += 1
        self.last_time = time.time()

    def get_delta(self) -> float:
        """Get the time delta since the timer was started."""
        return self.last_time - self.start_time

    def get_eta(self, current_step: int | None = None) -> float:
        """Get the estimated time to completion."""
        if self.total_steps is None:
            raise ValueError("Total steps is not set.")
        current_step = current_step or self.current_step
        if current_step == 0:
            return 0
        return self.get_delta() / current_step * max(self.total_steps - current_step, 0)

    def get_total_time(self) -> float:
        """Get the total time expected to complete all steps."""
        if self.total_steps is None:
            raise ValueError("Total steps is not set.")
        return self.get_delta() / self.current_step * self.total_steps

    def get_rate(self, current_step: int | None = None) -> float:
        """Get the rate of steps per second."""
        current_step = current_step or self.current_step
        return current_step / self.get_delta()

    def get_step_time(self, current_step: int | None = None) -> float:
        """Get the time per step."""
        current_step = current_step or self.current_step
        return self.get_delta() / current_step

    def get_time_str(self) -> str:
        """Get the time delta since the timer was started."""
        return Timer._format_time(self.get_delta())

    def get_eta_str(self, current_step: int | None = None) -> str:
        """Get the estimated time to completion."""
        current_step = current_step or self.current_step
        return Timer._format_time(self.get_eta(current_step))

    def get_total_time_str(self) -> str:
        """Get the total time expected to complete all steps."""
        return Timer._format_time(self.get_total_time())

    def get_rate_str(self, current_step: int | None = None) -> str:
        """Get the rate of steps per second."""
        current_step = current_step or self.current_step
        return f"{self.get_rate(current_step):.2f} steps/s"

    def get_step_time_rate_str(self, current_step: int | None = None) -> str:
        """Get the time per step."""
        current_step = current_step or self.current_step
        return f"{self.get_step_time(current_step):.2f} s/step"

    def get_step_time_str(self, current_step: int | None = None) -> str:
        """Get the time per step."""
        current_step = current_step or self.current_step
        return Timer._format_time(self.get_step_time(current_step))

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format time in seconds to HH:MM:SS."""
        seconds = int(seconds)
        return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


class NeptuneSummaryWriter(SummaryWriter):
    """Combine SummaryWriter with NeptuneWriter."""

    def __init__(self, log_dir: str, run: neptune.Run) -> None:
        super().__init__(log_dir=log_dir)
        self.run = run

    def add_scalar(self, tag: str, scalar_value: float, global_step: int | float | None = None) -> None:
        """Record scalar to tensorboard and Neptune."""
        # Check for NaN values - these indicate a serious training problem
        if math.isnan(scalar_value):
            error_msg = (
                f"NaN value detected when logging metric '{tag}' at step {global_step}. "
                f"This indicates a serious training problem (e.g., exploding gradients, division by zero). "
                f"Stopping training to prevent further issues.\n\n"
                f"Traceback showing where this NaN value originated:\n"
            )
            # Get the current stack trace
            stack_trace = traceback.format_stack()
            # Remove the last frame (this method) and show the relevant callers
            # Keep the last few frames that show where add_scalar was called from
            relevant_frames = stack_trace[:-1][-6:]  # Show last 6 frames before this method
            error_msg += "".join(relevant_frames)
            raise ValueError(error_msg)

        super().add_scalar(
            tag=tag,
            scalar_value=scalar_value,
            global_step=global_step,
        )
        self.run[tag].append(scalar_value, step=global_step)

    def add_text(
        self,
        tag: str,
        text_string: str,
        global_step: Optional[int] = None,
        walltime: Optional[float] = None,
    ) -> None:
        """Record text to tensorboard and Neptune."""
        super().add_text(tag=tag, text_string=text_string, global_step=global_step, walltime=walltime)

        self.run[tag] = text_string

    def add_hparams(
        self,
        hparam_dict: dict,
        metric_dict: dict,
        hparam_domain_discrete: Optional[Dict[str, List[Any]]] = None,
        run_name: Optional[str] = None,
        global_step: Optional[int] = None,
    ) -> None:
        """Add a set of hyperparameters to be compared in Neptune as for Tensorboard."""
        super().add_hparams(
            hparam_dict,
            metric_dict,
            hparam_domain_discrete=hparam_domain_discrete,
            run_name=run_name,
            global_step=global_step,
        )
        flatten_hparam = _flatten_dict_using_keypath(hparam_dict, base_keypath="params")
        for hparam, value in flatten_hparam.items():
            self.run[hparam] = value


def _set_author_neptune_api_token() -> None:
    """Set the variable NEPTUNE_API_TOKEN based on the email of commit author.

    It is useful on AIchor to have proper owner of each run.
    """
    try:
        author_email = os.environ["VCS_AUTHOR_EMAIL"]
    # we are not on AIchor
    except KeyError:
        logger.debug("We are not running on AIchor (https://aichor.ai/), not looking for Neptune API token.")
        return

    author_email, _ = author_email.split("@")
    author_email = author_email.replace("-", "_").replace(".", "_").upper()

    logger.info(f"Checking for Neptune API token under {author_email}__NEPTUNE_API_TOKEN.")
    try:
        author_api_token = os.environ[f"{author_email}__NEPTUNE_API_TOKEN"]
        os.environ["NEPTUNE_API_TOKEN"] = author_api_token
        logger.info(f"Set token for {author_email}.")
    except KeyError:
        logger.info(f"Neptune credentials for user {author_email} not found.")


def _get_filepath_mapping(file_groups: Dict[str, str]) -> Dict[str, str]:
    """Get filepath mapping for validation groups."""
    group_mapping = {}
    for group, path in file_groups.items():
        for fp in SpectrumDataFrame._convert_file_paths(path):
            group_mapping[fp] = group
    return group_mapping


def _flatten_dict_using_keypath(obj: Dict[str, Any], base_keypath: str = "", sep: str = "/") -> Dict[str, Any]:
    """Recursively flatten a nested mapping into a single-level dict with joined keys (usually called keypaths).

    Example:
        _flatten_dict_using_keypath({"a": {"b": 1}, "c": 2}) -> {"a/b": 1, "c": 2}
    """
    flatten: Dict[str, Any] = {}

    for key, value in obj.items():
        key_str = str(key)
        new_key = f"{base_keypath}{sep}{key_str}" if base_keypath else key_str

        if isinstance(value, dict):
            deeper = _flatten_dict_using_keypath(value, base_keypath=new_key, sep=sep)
            flatten.update(deeper)
        else:
            flatten[new_key] = value

    return flatten
