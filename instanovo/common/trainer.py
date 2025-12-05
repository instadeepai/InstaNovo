from __future__ import annotations

import datetime
import logging
import math
import os
import shutil
import sys
import traceback
from abc import ABCMeta, abstractmethod
from collections import Counter
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List

import neptune
import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.utils import DataLoaderConfiguration, InitProcessGroupKwargs, broadcast_object_list
from datasets import Dataset, Value
from datasets.utils.logging import disable_progress_bar
from dotenv import load_dotenv
from neptune.integrations.python_logger import NeptuneHandler
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter

from instanovo.__init__ import console, set_rank
from instanovo.common.dataset import DataProcessor
from instanovo.common.scheduler import CosineWarmupScheduler, FinetuneScheduler, WarmupScheduler
from instanovo.common.utils import (
    NeptuneSummaryWriter,
    Timer,
    TrainingState,
    _get_filepath_mapping,
    _set_author_neptune_api_token,
)
from instanovo.constants import ANNOTATED_COLUMN, ANNOTATION_ERROR, SHUFFLE_BUFFER_SIZE, MSColumns
from instanovo.inference import Decoder
from instanovo.utils.colorlogging import ColorLog
from instanovo.utils.data_handler import SpectrumDataFrame
from instanovo.utils.device_handler import validate_and_configure_device
from instanovo.utils.metrics import Metrics
from instanovo.utils.residues import ResidueSet
from instanovo.utils.s3 import S3FileHandler

load_dotenv()

# Automatic rank logger
logger = ColorLog(console, __name__).logger


class AccelerateDeNovoTrainer(metaclass=ABCMeta):
    """Trainer class that uses the Accelerate library."""

    @property
    def run_id(self) -> str:
        """Get the run ID.

        Returns:
            str: The run ID
        """
        return str(self._run_id)

    @property
    def s3(self) -> S3FileHandler:
        """Get the S3 file handler.

        Returns:
            S3FileHandler: The S3 file handler
        """
        return self._s3

    @property
    def global_step(self) -> int:
        """Get the current global training step.

        This represents the total number of training steps across all epochs.

        Returns:
            int: The current global step number
        """
        return int(self._training_state.global_step)

    @property
    def epoch(self) -> int:
        """Get the current training epoch.

        This represents the current epoch number in the training process.

        Returns:
            int: The current epoch number
        """
        return int(self._training_state.epoch)

    @property
    def training_state(self) -> TrainingState:
        """Get the training state."""
        return self._training_state

    def __init__(
        self,
        config: DictConfig,
    ) -> None:
        self.config = config
        self.enable_verbose_logging = self.config.get("enable_verbose_logging", True)
        if not self.config.get("enable_verbose_accelerate", True):
            logging.getLogger("accelerate").setLevel(logging.WARNING)

        # Hide progress bar from HF datasets
        disable_progress_bar()

        # Training state
        # Keeps track of the global step and epoch
        # Used for accelerate training state checkpointing
        self._training_state = TrainingState()

        self._run_id = self.config.get("run_name", "instanovo") + datetime.datetime.now().strftime("_%y_%m_%d_%H_%M")

        self.accelerator = self.setup_accelerator()

        self.log_if_verbose("Verbose logging enabled")

        if self.accelerator.is_main_process:
            logger.info(f"Config:\n{OmegaConf.to_yaml(self.config)}")

        self.residue_set = ResidueSet(
            residue_masses=self.config.residues.get("residues"),
            residue_remapping=self.config.dataset.get("residue_remapping", None),
        )
        logger.info(f"Vocab: {self.residue_set.index_to_residue}")

        # Initialise S3 file handler
        self._s3: S3FileHandler = S3FileHandler(verbose=self.config.get("enable_verbose_s3", True))

        self.train_dataset, self.valid_dataset, train_size, valid_size = self.load_datasets()

        logger.info(f"Data loaded from {train_size:,} training samples and {valid_size:,} validation samples (unfiltered values)")

        self.train_dataloader, self.valid_dataloader = self.build_dataloaders(self.train_dataset, self.valid_dataset)
        logger.info("Data loaders built")

        # Print sample batch
        self.print_sample_batch()

        logger.info("Setting up model...")
        self.model = self.setup_model()

        if self.accelerator.is_main_process:
            logger.info(f"Model has {sum(p.numel() for p in self.model.parameters()):,d} parameters")

        self.optimizer = self.setup_optimizer()
        self.lr_scheduler = self.setup_scheduler()

        self.decoder = self.setup_decoder()
        self.metrics = self.setup_metrics()

        # Optionally load a model state for fine-tuning
        # Note: will be overwritten by the accelerator state if resuming
        if self.config.get("resume_checkpoint_path", None) is not None:
            self.load_model_state()  # TODO check for loading on mps

        # Prepare for accelerated training
        (
            self.model,
            self.optimizer,
            self.lr_scheduler,
            self.train_dataloader,
            self.valid_dataloader,
        ) = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.lr_scheduler,
            self.train_dataloader,
            self.valid_dataloader,
        )
        # Make sure the training state is checkpointed
        self.accelerator.register_for_checkpointing(self._training_state)

        # Optionally load states if resuming a training run
        if self.config.get("resume_accelerator_state", None):
            # Resuming from an existing run
            self.load_accelerator_state()

        # Setup logging
        self.setup_neptune()
        self.setup_tensorboard()
        self._add_commit_message_to_monitoring_platform()
        self._add_config_summary_to_monitoring_platform()

        # Training control variables
        self.running_loss = None

        self.total_steps = self.config.get("training_steps", 2_500_000)

        # Setup finetuning scheduler
        if self.config.get("finetune", None):
            self.finetune_scheduler: FinetuneScheduler | None = FinetuneScheduler(
                self.model.state_dict(),
                self.config.get("finetune"),
            )
        else:
            self.finetune_scheduler = None

        self.steps_per_validation = self.config.get("validation_interval", 100_000)
        self.steps_per_checkpoint = self.config.get("checkpoint_interval", 100_000)

        # Print training control variables
        if self.accelerator.is_main_process:
            steps_per_epoch = train_size // self.config["train_batch_size"]
            logger.info("Training setup complete.")
            logger.info(f" - Steps per validation: {self.steps_per_validation:,d} ")
            logger.info(f" - Steps per checkpoint: {self.steps_per_checkpoint:,d} ")
            logger.info(f" - Total training steps: {self.total_steps:,d}")
            logger.info("Estimating steps per epoch based on unfiltered training set size:")
            logger.info(f" - Estimated steps per epoch: {steps_per_epoch:,d}")
            logger.info(f" - Estimated total epochs: {self.total_steps / steps_per_epoch:.1f}")

            if self.total_steps < steps_per_epoch:
                logger.warning("Total steps is less than estimated steps per epoch, this may result in less than one epoch during training")

        if self.global_step > 0:
            logger.info(f"Training will resume from epoch {self.epoch}, global_step {self.global_step}")

        self.last_validation_metric = None
        self.best_checkpoint_metric = None

        # Final sync after setup
        self.accelerator.wait_for_everyone()

    @abstractmethod
    def setup_model(self) -> nn.Module:
        """Setup the model."""
        ...

    @abstractmethod
    def setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup the optimizer."""
        ...

    @abstractmethod
    def setup_decoder(self) -> Decoder:
        """Setup the decoder."""
        ...

    @abstractmethod
    def setup_data_processors(self) -> tuple[DataProcessor, DataProcessor]:
        """Setup the data processor."""
        ...

    @abstractmethod
    def save_model(self, is_best_checkpoint: bool = False) -> None:
        """Save the model."""
        ...

    @abstractmethod
    def forward(self, batch: Any) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Forward pass for the model to calculate loss."""
        ...

    @abstractmethod
    def get_predictions(self, batch: Any) -> tuple[list[str] | list[list[str]], list[str] | list[list[str]]]:
        """Get the predictions for a batch."""
        ...

    @staticmethod
    def convert_interval_to_steps(interval: float | int, steps_per_epoch: int) -> int:
        """Convert an interval to steps.

        Args:
            interval (float | int): The interval to convert.
            steps_per_epoch (int): The number of steps per epoch.

        Returns:
            int: The number of steps.
        """
        if isinstance(interval, float):
            return int(interval * steps_per_epoch)
        else:
            raise ValueError(f"Invalid interval: {interval}")

    def log_if_verbose(self, message: str, level: str = "info") -> None:
        """Log a message if verbose logging is enabled."""
        if self.enable_verbose_logging:
            if level == "info":
                logger.info(message)
            elif level == "warning":
                logger.warning(message)
            elif level == "error":
                logger.error(message)
            elif level == "debug":
                logger.debug(message)
            else:
                raise ValueError(f"Invalid level: {level}")

    def setup_metrics(self) -> Metrics:
        """Setup the metrics."""
        return Metrics(self.residue_set, self.config.get("max_isotope_error", 1))

    def setup_accelerator(self) -> Accelerator:
        """Setup the accelerator."""
        timeout = timedelta(seconds=self.config.get("timeout", 3600))
        validate_and_configure_device(self.config)
        accelerator = Accelerator(
            cpu=self.config.get("force_cpu", False),
            mixed_precision="fp16" if torch.cuda.is_available() and not self.config.get("force_cpu", False) else "no",
            gradient_accumulation_steps=self.config.get("grad_accumulation", 1),
            dataloader_config=DataLoaderConfiguration(split_batches=True),
            kwargs_handlers=[InitProcessGroupKwargs(timeout=timeout)],
        )

        device = accelerator.device  # Important, this forces ranks to choose a device.

        if accelerator.num_processes > 1:
            set_rank(accelerator.local_process_index)

        if accelerator.is_main_process:
            logger.info(f"Python version: {sys.version}")
            logger.info(f"Torch version: {torch.__version__}")
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"Training with {accelerator.num_processes} devices")
            logger.info(f"Per-device batch size: {self.config['train_batch_size']}")
            logger.info(f"Gradient accumulation steps: {self.config['grad_accumulation']}")
            effective_batch_size = self.config["train_batch_size"] * accelerator.num_processes * self.config["grad_accumulation"]
            logger.info(f"Effective batch size: {effective_batch_size}")

        logger.info(f"Using device: {device}")

        return accelerator

    def build_dataloaders(self, train_dataset: Dataset, valid_dataset: Dataset) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """Setup the dataloaders."""
        train_processor, valid_processor = self.setup_data_processors()

        valid_processor.add_metadata_columns(["prediction_id"])
        if self.using_validation_groups:
            valid_processor.add_metadata_columns(["validation_group"])

        if self.config.get("use_shuffle_buffer", True):
            buffer_size = self.config.get("shuffle_buffer_size", SHUFFLE_BUFFER_SIZE)
            train_dataset = train_dataset.shuffle(buffer_size=buffer_size, seed=42)

        train_dataset = train_dataset.map(
            train_processor.process_row,
        )
        valid_dataset = valid_processor.process_dataset(valid_dataset)

        pin_memory = self.config.get("pin_memory", False)
        if self.accelerator.device == torch.device("cpu") or self.config.get("mps", False):
            pin_memory = False
        # Scale batch size by number of processes when using split_batches
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config["train_batch_size"] * self.accelerator.num_processes,
            collate_fn=train_processor.collate_fn,
            num_workers=self.config.get("num_workers", 8),
            pin_memory=pin_memory,
            prefetch_factor=self.config.get("prefetch_factor", None),
            drop_last=True,
        )
        valid_dataloader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=self.config["predict_batch_size"] * self.accelerator.num_processes,
            collate_fn=valid_processor.collate_fn,
            num_workers=self.config.get("num_workers", 8),
            pin_memory=pin_memory,
            prefetch_factor=self.config.get("prefetch_factor", None),
            drop_last=False,
        )
        return train_dataloader, valid_dataloader

    def setup_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler:
        """Setup the learning rate scheduler.

        Returns:
            torch.optim.lr_scheduler.LRScheduler: The learning rate scheduler
        """
        # Note: if split_batches is False, the scheduler will be called num_processes times
        # in each optimizer step. Therefore, we need to scale the scheduler steps by num_processes.
        # Default is split_batches is True
        if self.config.get("lr_scheduler", "warmup") == "warmup":
            warmup_steps = self.config.get("warmup_iters", 1000)  # * num_processes
            return WarmupScheduler(self.optimizer, warmup_steps)
        elif self.config.get("lr_scheduler", None) == "cosine":
            # Scale max_iters based on accumulation
            # train_dataloader is already scaled by num_processes
            max_iters = self.config.get("training_steps", 2_500_000)  # * num_processes
            warmup_steps = self.config.get("warmup_iters", 1000)  # * num_processes
            return CosineWarmupScheduler(self.optimizer, warmup_steps, max_iters)
        else:
            raise ValueError(f"Unknown lr_scheduler type '{self.config.get('lr_scheduler', None)}'")

    def setup_neptune(self) -> None:
        """Setup the neptune."""
        if not self.accelerator.is_main_process:
            self.neptune_run = None
            return

        if not self.config.get("use_neptune", True):
            self.neptune_run = None
            return

        _set_author_neptune_api_token()
        try:
            self.neptune_run = neptune.init_run(
                with_id=None,
                name=self.run_id,
                dependencies=str(Path(__file__).parent.parent.parent / "uv.lock"),
                tags=OmegaConf.to_object(self.config.get("tags", [])),
            )
            self.neptune_run.assign({"config": OmegaConf.to_yaml(self.config)})
            logger.addHandler(NeptuneHandler(run=self.neptune_run))
        except Exception as e:
            logger.warning(f"Failed to initialise neptune: {e}")
            self.neptune_run = None

    def setup_tensorboard(self) -> None:
        """Setup the tensorboard."""
        if not self.accelerator.is_main_process:
            self.sw = None
            return

        if S3FileHandler.register_tb():
            logs_path = os.environ["AICHOR_LOGS_PATH"]
        else:
            logs_path = self.config.get("tb_summarywriter", "runs") + self.run_id

        if self.neptune_run is not None:
            self.sw = NeptuneSummaryWriter(logs_path, self.neptune_run)
        else:
            self.sw = SummaryWriter(logs_path)
        logger.info(f"TensorBoard logs will be saved to {logs_path}")

    @property
    def _is_main_process_on_aichor(self) -> bool:
        """Return True if monitoring logging is configured and this is the main process."""
        return ("AICHOR_LOGS_PATH" in os.environ) and self.accelerator.is_main_process

    def _add_commit_message_to_monitoring_platform(self, commit_id_length: int = 7) -> None:
        """Add the git commit message to the monitoring platform."""
        if not self._is_main_process_on_aichor:
            logger.debug("Skipping config summary upload to Neptune: 'AICHOR_LOGS_PATH' not set or current process is not the main AICHOR process")
            return

        try:
            # Remove 'exp:' prefix if present in the commit message and also remove the space after it.
            git_commit_msg = os.environ["VCS_COMMIT_MESSAGE"].removeprefix("exp:").removeprefix(" ")
            commit_short_hash = os.environ["VCS_SHA"][:commit_id_length]
            self.sw.add_text(  # type: ignore[union-attr]
                "git/commit_message", f"{git_commit_msg} ({commit_short_hash})"
            )
        except (AttributeError, KeyError) as exc:
            logger.warning("Failed to write config summary to the monitoring plateform", exc_info=exc)

    def _add_config_summary_to_monitoring_platform(self) -> None:
        """Add the config summary to the monitoring platform."""
        if not self._is_main_process_on_aichor:
            logger.debug("Skipping config summary upload to Neptune: 'AICHOR_LOGS_PATH' not set or current process is not the main AICHOR process")
            return
        # https://github.com/pytorch/pytorch/blob/daca611465c93ac6b8147e6b7070ce2b4254cfc5/torch/utils/tensorboard/summary.py#L244 # noqa
        self.sw.add_hparams(  # type: ignore[union-attr]
            {k: v for k, v in self.config.items() if isinstance(v, (int, float, str))}, {}
        )

    def load_datasets(self) -> tuple[Dataset, Dataset, int, int]:
        """Load the training and validation datasets.

        Returns:
            tuple[SpectrumDataFrame, SpectrumDataFrame]:
                The training and validation datasets
        """
        validation_group_mapping = None
        dataset_config = self.config.get("dataset", {})
        try:
            logger.info("Loading training dataset...")
            train_sdf = SpectrumDataFrame.load(
                source=dataset_config.get("train_path"),
                source_type=dataset_config.get("source_type", "default"),
                lazy=dataset_config.get("lazy_loading", True),
                is_annotated=True,
                shuffle=True,
                partition=dataset_config.get("train_partition", None),
                column_mapping=dataset_config.get("column_remapping", None),
                max_shard_size=dataset_config.get("max_shard_size", 100_000),
                preshuffle_across_shards=dataset_config.get("preshuffle_shards", False),
                verbose=dataset_config.get("verbose_loading", True),
            )

            valid_path = dataset_config.get("valid_path", None)
            if valid_path is not None:
                if OmegaConf.is_dict(valid_path):
                    logger.info("Found grouped validation datasets.")
                    validation_group_mapping = _get_filepath_mapping(valid_path)
                    _valid_path = list(valid_path.values())
                else:
                    _valid_path = valid_path
            else:
                _valid_path = dataset_config.get("train_path")

            logger.info("Loading validation dataset...")
            valid_sdf = SpectrumDataFrame.load(
                _valid_path,
                lazy=dataset_config.get("lazy_loading", True),
                is_annotated=True,
                shuffle=False,
                partition=dataset_config.get("valid_partition", None),
                column_mapping=dataset_config.get("column_remapping", None),
                max_shard_size=dataset_config.get("max_shard_size", 100_000),
                add_source_file_column=True,  # used to track validation groups
                verbose=dataset_config.get("verbose_loading", True),
            )
        except ValueError as e:
            # More descriptive error message in predict mode.
            if str(e) == ANNOTATION_ERROR:
                raise ValueError("The sequence column is missing annotations, are you trying to run de novo prediction? Add the --denovo flag") from e
            else:
                raise

        # Split data if needed
        if dataset_config.get("valid_path", None) is None:
            logger.info("Validation path not specified, generating from training set.")
            sequences = list(train_sdf.get_unique_sequences())
            sequences = sorted({DataProcessor.remove_modifications(x) for x in sequences})
            train_unique, valid_unique = train_test_split(
                sequences,
                test_size=dataset_config.get("valid_subset_of_train"),
                random_state=42,
            )
            train_unique = set(train_unique)
            valid_unique = set(valid_unique)

            train_sdf.filter_rows(lambda row: DataProcessor.remove_modifications(row[ANNOTATED_COLUMN]) in train_unique)
            valid_sdf.filter_rows(lambda row: DataProcessor.remove_modifications(row[ANNOTATED_COLUMN]) in valid_unique)

            # Save splits
            if self.accelerator.is_main_process:
                split_path = os.path.join(self.config.get("model_save_folder_path", "./checkpoints"), "splits.csv")
                os.makedirs(os.path.dirname(split_path), exist_ok=True)
                splits_df = pd.DataFrame(
                    {
                        ANNOTATED_COLUMN: list(train_unique) + list(valid_unique),
                        "split": ["train"] * len(train_unique) + ["valid"] * len(valid_unique),
                    }
                )
                self.s3.upload_to_s3_wrapper(splits_df.to_csv, split_path, index=False)
                logger.info(f"Data splits saved to {split_path}")

        train_ds = train_sdf.to_dataset(force_unified_schema=True)
        valid_ds = valid_sdf.to_dataset(in_memory=True)

        # # Sample subsets if needed
        valid_subset = self.config.get("valid_subset", 1.0)
        if valid_subset < 1.0:
            valid_ds = valid_ds.train_test_split(test_size=valid_subset, seed=42)["test"]

        # Check residues
        if self.config.get("perform_data_checks", True):
            logger.info(f"Checking for unknown residues in {len(train_sdf) + len(valid_sdf):,d} rows.")
            supported_residues = set(self.residue_set.vocab)
            supported_residues.update(set(self.residue_set.residue_remapping.keys()))
            data_residues = set()
            data_residues.update(train_sdf.get_vocabulary(self.residue_set.tokenize))
            data_residues.update(valid_sdf.get_vocabulary(self.residue_set.tokenize))
            if len(data_residues - supported_residues) > 0:
                logger.warning(f"Found {len(data_residues - supported_residues):,d} unsupported residues! These rows will be dropped.")
                self.log_if_verbose(f"New residues found: \n{data_residues - supported_residues}")
                self.log_if_verbose(f"Residues supported: \n{supported_residues}")

                train_ds = train_ds.filter(
                    lambda row: all(residue in supported_residues for residue in set(self.residue_set.tokenize(row[ANNOTATED_COLUMN])))
                )
                valid_ds = valid_ds.filter(
                    lambda row: all(residue in supported_residues for residue in set(self.residue_set.tokenize(row[ANNOTATED_COLUMN])))
                )

            logger.info("Checking charge values...")
            # Check charge values
            precursor_charge_col = MSColumns.PRECURSOR_CHARGE.value

            if not train_sdf.check_values(1, self.config.get("max_charge", 10), precursor_charge_col):
                logger.warning("Found charge values out of range in training set. These rows will be dropped.")

                train_ds = train_ds.filter(
                    lambda row: (row[precursor_charge_col] <= self.config.get("max_charge", 10)) and (row[precursor_charge_col] > 0)
                )

            if not valid_sdf.check_values(1, self.config.get("max_charge", 10), precursor_charge_col):
                logger.warning("Found charge values out of range in validation set. These rows will be dropped.")
                valid_ds = valid_ds.filter(
                    lambda row: (row[precursor_charge_col] <= self.config.get("max_charge", 10)) and (row[precursor_charge_col] > 0)
                )

        # Create validation groups
        # Initialize validation groups if needed
        if validation_group_mapping is not None:
            logger.info("Computing validation groups.")
            validation_groups = [validation_group_mapping.get(row.get("source_file"), "no_group") for row in valid_ds]
            valid_ds = valid_ds.add_column("validation_group", validation_groups)

            logger.info("Sequences per validation group:")
            group_counts = Counter(validation_groups)
            for group, count in group_counts.items():
                logger.info(f" - {group}: {count:,d}")

            self.using_validation_groups = True
        else:
            self.using_validation_groups = False

        # Force add a unique prediction_id column
        # This will be used to order predictions and remove duplicates
        valid_ds = valid_ds.add_column("prediction_id", np.arange(len(valid_ds)), feature=Value("int32"))

        # Keep track of the train_sdf directory so it isn't garbage collected
        self._train_sdf = train_sdf

        return train_ds, valid_ds, len(train_sdf), len(valid_sdf)

    def print_sample_batch(self) -> None:
        """Print a sample batch of the training data."""
        if self.accelerator.is_main_process:
            # sample_batch = next(iter(self.train_dataloader))
            sample_batch = next(iter(self.train_dataloader))
            logger.info("Sample batch:")
            for key, value in sample_batch.items():
                if isinstance(value, torch.Tensor):
                    value_shape = value.shape
                    value_type = value.dtype
                else:
                    value_shape = len(value)
                    value_type = type(value)

                logger.info(f" - {key}: {value_type}, {value_shape}")

    def save_accelerator_state(self, is_best_checkpoint: bool = False) -> None:
        """Save the accelerator state."""
        checkpoint_dir = self.config.get("model_save_folder_path", "./checkpoints")

        if self.config.get("keep_accelerator_every_interval", False):
            checkpoint_path = os.path.join(
                checkpoint_dir,
                "accelerator_state",
                f"epoch_{self.epoch}_step_{self.global_step + 1}",
            )
        else:
            checkpoint_path = os.path.join(checkpoint_dir, "accelerator_state", "latest")
            if self.accelerator.is_main_process and Path(checkpoint_path).exists() and Path(checkpoint_path).is_dir():
                shutil.rmtree(checkpoint_path)

        if self.accelerator.is_main_process:
            os.makedirs(checkpoint_path, exist_ok=True)

        self.accelerator.save_state(checkpoint_path)

        logger.info(f"Saved accelerator state to {checkpoint_path}")

        if self.accelerator.is_main_process and S3FileHandler._aichor_enabled():
            for file in os.listdir(checkpoint_path):
                self.s3.upload(
                    os.path.join(checkpoint_path, file),
                    S3FileHandler.convert_to_s3_output(os.path.join(checkpoint_path, file)),
                )

        # Save best checkpoint and upload to S3
        if is_best_checkpoint and self.accelerator.is_main_process:
            best_checkpoint_path = os.path.join(checkpoint_dir, "accelerator_state", "best")
            if Path(best_checkpoint_path).exists() and Path(best_checkpoint_path).is_dir():
                shutil.rmtree(best_checkpoint_path)

            os.makedirs(best_checkpoint_path, exist_ok=True)

            for file in os.listdir(checkpoint_path):
                full_file = os.path.join(checkpoint_path, file)
                best_file = os.path.join(best_checkpoint_path, file)
                shutil.copy(full_file, best_file)
                if S3FileHandler._aichor_enabled():
                    self.s3.upload(
                        full_file,
                        S3FileHandler.convert_to_s3_output(best_file),
                    )

    def check_if_best_checkpoint(self) -> bool:
        """Check if the last validation metric is the best metric."""
        if self.config.get("checkpoint_metric", None) is None:
            return False

        if self.best_checkpoint_metric is None:
            self.best_checkpoint_metric = self.last_validation_metric
            return True

        if self.config.get("checkpoint_metric_mode", "min") == "min":
            is_best = self.last_validation_metric <= self.best_checkpoint_metric
        elif self.config.get("checkpoint_metric_mode", "min") == "max":
            is_best = self.last_validation_metric >= self.best_checkpoint_metric
        else:
            raise ValueError(f"Unknown checkpoint metric mode: {self.config.get('checkpoint_metric_mode', 'min')}")

        if is_best:
            self.best_checkpoint_metric = self.last_validation_metric

        return is_best

    def load_accelerator_state(self) -> None:
        """Load the accelerator state."""
        checkpoint_path = self.config.get("resume_accelerator_state", None)
        if checkpoint_path is None:
            return

        if not os.path.isdir(checkpoint_path) and not checkpoint_path.startswith("s3://"):
            raise ValueError(f"Accelerator state should be a directory of state files, got {checkpoint_path}")

        if S3FileHandler._aichor_enabled() and checkpoint_path.startswith("s3://"):
            # raise NotImplementedError("Loading accelerator state from S3 is not implemented.")

            if self.accelerator.is_main_process:
                local_path = os.path.join(self.s3.temp_dir.name, "accelerator_state")
                os.makedirs(local_path, exist_ok=True)
                logger.info(f"Downloading checkpoint files from {checkpoint_path} to {local_path}")

                # Download all files from the checkpoint folder
                checkpoint_files = self.s3.listdir(checkpoint_path)
                logger.info(f"Found {len(checkpoint_files)} files")
                for file in checkpoint_files:
                    if file.endswith("/"):  # Skip subdirectories
                        continue
                    local_file = os.path.join(local_path, os.path.basename(file))
                    self.s3.download(f"s3://{file}", local_file)
            else:
                local_path = None

            checkpoint_path = broadcast_object_list([local_path])[0]
            logger.info(f"Received checkpoint path: {checkpoint_path}")

            assert checkpoint_path is not None, "Failed to broadcast accelerator state across ranks"

        # Add safe globals
        torch.serialization.add_safe_globals(
            [
                np._core.multiarray.scalar,
                np.dtypes.Float64DType,
            ]
        )

        self.accelerator.load_state(checkpoint_path)
        logger.info(f"Loaded accelerator state from {checkpoint_path}")

    def load_model_state(self) -> None:
        """Load the model state."""
        checkpoint_path = self.config.get("resume_checkpoint_path", None)
        if checkpoint_path is None:
            return

        if os.path.isdir(checkpoint_path) and not checkpoint_path.startswith("s3://"):
            raise ValueError(f"Checkpoint path should be a file, got {checkpoint_path}")

        if self.accelerator.is_main_process:
            logger.info(f"Resuming model state from {checkpoint_path}")
            local_path = self.s3.get_local_path(checkpoint_path)
        else:
            local_path = None

        local_path = broadcast_object_list([local_path])[0]

        assert local_path is not None, "Failed to broadcast model state across ranks"

        # TODO: Switch to model.load(), implement model schema
        model_data = torch.load(local_path, weights_only=False, map_location="cpu")
        # TODO: Remove, only use state_dict
        if "model" in model_data:
            model_state = model_data["model"]
        else:
            model_state = model_data["state_dict"]

        # Check residues
        if "residues" in model_data:
            model_residues = dict(model_data["residues"].get("residues", {}))
        else:
            # Legacy format
            model_residues = dict(model_data["config"]["residues"])

        current_residues = self.config.residues.get("residues")
        if model_residues != current_residues:
            logger.warning(
                f"Checkpoint residues do not match current residues.\nCheckpoint residues: {model_residues}\nCurrent residues: {current_residues}"
            )
            logger.warning("Updating model state to match current residues.")
            model_state = self.update_vocab(model_state)

        model_state = self.update_model_state(model_state, model_data["config"])
        self.model.load_state_dict(model_state, strict=False)
        logger.info(f"Loaded model state from {local_path}")

    def update_model_state(self, model_state: dict[str, torch.Tensor], model_config: DictConfig) -> dict[str, torch.Tensor]:
        """Update the model state."""
        return model_state

    def update_vocab(self, model_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Update the vocabulary of the model."""
        # This should call `self._update_vocab` based on model implementation.
        raise NotImplementedError("Updating vocabulary is not implemented for the base trainer.")

    def _update_vocab(
        self,
        model_state: dict[str, torch.Tensor],
        target_layers: list[str],
        resolution: str = "delete",
    ) -> dict[str, torch.Tensor]:
        """Update the target heads of the model."""
        target_vocab_size = len(self.residue_set)
        current_model_state = self.model.state_dict()
        hidden_size = self.config.model.get("dim_model", 768)

        for layer in target_layers:
            if layer not in current_model_state:
                logger.warning(f"Layer {layer} not found in current model state.")
                continue
            tmp = torch.normal(
                mean=0,
                std=1.0 / np.sqrt(hidden_size),
                size=current_model_state[layer].shape,
                dtype=current_model_state[layer].dtype,
            )
            if "bias" in layer:
                # initialise bias to zeros
                tmp = torch.zeros_like(tmp)

            if resolution == "delete":
                del model_state[layer]
            elif resolution == "random":
                model_state[layer] = tmp
            elif resolution == "partial":
                tmp[:target_vocab_size] = model_state[layer][: min(tmp.shape[0], target_vocab_size)]
                model_state[layer] = tmp
            else:
                raise ValueError(f"Unknown residue_conflict_resolution type '{resolution}'")
        return model_state

    def train(self) -> None:
        """Train the model."""
        num_sanity_steps = self.config.get("num_sanity_val_steps", 0)
        if num_sanity_steps > 0:
            logger.info(f"Running sanity validation for {num_sanity_steps} steps...")
            self.validate_epoch(num_sanity_steps=num_sanity_steps, calculate_metrics=False)
            logger.info("Sanity validation complete.")

        if self.config.get("validate_before_training", False):
            logger.info("Running pre-validation...")
            self.validate_epoch()
            logger.info("Pre-validation complete.")

        self.train_timer = Timer(self.total_steps)
        is_first_epoch = True
        logger.info("Starting training...")
        while self.global_step < self.total_steps:
            self.train_epoch()
            self.training_state.step_epoch()
            if self.accelerator.is_main_process and is_first_epoch:
                is_first_epoch = False
                logger.info("First epoch complete:")
                logger.info(f"- Actual steps per epoch: {self.global_step}")
                logger.info(f"- Actual total epochs: {self.total_steps / self.global_step:.1f}")

        logger.info("Training complete.")

    def prepare_batch(self, batch: Iterable[Any]) -> Any:
        """Prepare a batch for training.

        Manually move tensors to accelerator.device since we do not
        prepare our dataloaders with the accelerator.

        Args:
            batch (Iterable[Any]): The batch to prepare.

        Returns:
            Any: The prepared batch
        """
        if isinstance(batch, dict):
            return {k: v.to(self.accelerator.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        elif isinstance(batch, (list, tuple)):
            return [v.to(self.accelerator.device) if isinstance(v, torch.Tensor) else v for v in batch]
        else:
            raise ValueError(f"Unsupported batch type: {type(batch)}")

    def train_epoch(self) -> None:
        """Train the model for one epoch."""
        total_loss = 0

        self.model.train()
        self.optimizer.zero_grad()
        self.running_loss = None

        epoch_timer = Timer()

        print_batch_size = True
        for batch_count, batch in enumerate(self.train_dataloader):
            if print_batch_size:
                # Confirm batch size during debugging
                self.log_if_verbose(f"Batch {batch_count} shape: {batch['spectra'].shape[0]}")
                print_batch_size = False

            with self.accelerator.accumulate(self.model):
                # Forward pass
                loss, loss_components = self.forward(batch)

                # Check for NaN/Inf in loss immediately after forward pass
                loss_value = loss.item()
                if math.isnan(loss_value) or math.isinf(loss_value):
                    error_msg = (
                        f"Invalid loss value detected: {loss_value} (NaN: {math.isnan(loss_value)}, Inf: {math.isinf(loss_value)}). "
                        f"This occurred at step {self.global_step + 1}, epoch {self.epoch}, batch {batch_count}. "
                        f"This indicates a serious training problem (e.g., exploding gradients, division by zero, numerical instability). "
                        f"Stopping training to prevent further issues.\n\n"
                        f"Loss components: {[f'{k}={v.item()}' for k, v in loss_components.items()]}\n\n"
                        f"Traceback showing where the loss was computed:\n"
                    )
                    stack_trace = traceback.format_stack()
                    # Show frames from the forward pass
                    relevant_frames = stack_trace[:-3][-8:]  # Show more frames to see the forward pass
                    error_msg += "".join(relevant_frames)
                    raise ValueError(error_msg)

                # Backward pass
                self.accelerator.backward(loss)

                # Update weights
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.get("gradient_clip_val", 10.0))
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

            # Update timer
            self.train_timer.step()

            # Update running loss
            # Exponentially weighted moving average to smooth noisy batch losses
            if self.running_loss is None:
                self.running_loss = loss.item()
            else:
                self.running_loss = 0.99 * self.running_loss + 0.01 * loss.item()

            total_loss += loss.item()

            # Log progress
            if (self.global_step + 1) % int(self.config.get("console_logging_steps", 2000)) == 0:
                lr = self.lr_scheduler.get_last_lr()[0]

                logger.info(
                    f"[TRAIN] "
                    f"[Epoch {self.epoch:02d}] "
                    f"[Step {self.global_step + 1:06d}/{self.total_steps:06d}] "
                    f"[{self.train_timer.get_time_str()}/{self.train_timer.get_eta_str()}, "
                    f"{self.train_timer.get_step_time_rate_str()}]: "
                    f"train_loss_raw={loss.item():.4f}, "
                    f"running_loss={self.running_loss:.4f}, LR={lr:.6f}"
                )

            # Log to tensorboard
            if (
                self.accelerator.is_main_process
                and self.sw is not None
                and (self.global_step + 1) % int(self.config.get("tensorboard_logging_steps", 500)) == 0
            ):
                lr = self.lr_scheduler.get_last_lr()[0]
                self.sw.add_scalar("train/loss_raw", loss.item(), self.global_step + 1)
                if self.running_loss is not None:
                    self.sw.add_scalar("train/loss_smooth", self.running_loss, self.global_step + 1)
                for k, v in loss_components.items():
                    if k == "loss":
                        continue
                    self.sw.add_scalar(f"train/{k}", v.item(), self.global_step + 1)
                self.sw.add_scalar("optim/lr", lr, self.global_step + 1)
                self.sw.add_scalar("optim/epoch", self.epoch, self.global_step + 1)

            if (self.global_step + 1) % self.steps_per_validation == 0:
                self.model.eval()
                self.validate_epoch()
                logger.info("Validation complete, resuming training...")
                self.model.train()

            if (self.global_step + 1) % self.steps_per_checkpoint == 0:
                is_best_checkpoint = self.check_if_best_checkpoint()
                self.save_model(is_best_checkpoint)
                if self.config.get("save_accelerator_state", False):
                    self.save_accelerator_state(is_best_checkpoint)

            self.training_state.step()

            # Update finetuning scheduler
            if self.finetune_scheduler is not None:
                self.finetune_scheduler.step(self.global_step)

            if self.global_step >= self.total_steps:
                break

        # Epoch complete
        self.accelerator.wait_for_everyone()

        epoch_timer.step()

        # Gather losses from all devices
        gathered_losses = self.accelerator.gather_for_metrics(torch.tensor(total_loss, device=self.accelerator.device))
        gathered_num_batches = self.accelerator.gather_for_metrics(torch.tensor(batch_count, device=self.accelerator.device))

        if self.accelerator.is_main_process and self.sw is not None:
            # Sum the losses and batch counts from all devices
            total_loss_all_devices = gathered_losses.sum().item()
            total_batches_all_devices = gathered_num_batches.sum().item()
            avg_loss = total_loss_all_devices / total_batches_all_devices

            self.sw.add_scalar("eval/train_loss", avg_loss, self.epoch)

            logger.info(f"[TRAIN] [Epoch {self.epoch:02d}] Epoch complete, total time {epoch_timer.get_time_str()}")

    def validate_epoch(self, num_sanity_steps: int | None = None, calculate_metrics: bool = True) -> None:
        """Validate for one epoch."""
        if self.valid_dataloader is None:
            return

        if self.accelerator.is_main_process:
            logger.info(f"[VALIDATION] [Epoch {self.epoch:02d}] Starting validation.")

        valid_epoch_step = 0
        valid_predictions: List[List[str] | str] = []
        valid_targets: List[List[str] | str] = []
        valid_groups: List[str] = []
        valid_prediction_ids: List[int] = []

        valid_metrics: Dict[str, List[float]] = {x: [] for x in ["valid_loss", "aa_er", "aa_prec", "aa_recall", "pep_recall"]}

        num_batches = len(self.valid_dataloader)

        valid_timer = Timer(num_batches)

        for batch_idx, batch in enumerate(self.valid_dataloader):
            if num_sanity_steps is not None and batch_idx >= num_sanity_steps:
                break

            with torch.no_grad(), self.accelerator.autocast():
                # Loss calculation
                loss, _ = self.forward(batch)
                # Get actual predictions
                y, targets = self.get_predictions(batch)

            valid_predictions.extend(y)
            valid_targets.extend(targets)
            valid_prediction_ids.extend([x.item() for x in batch["prediction_id"]])

            # Store validation groups if available
            if self.using_validation_groups:
                valid_groups.extend(batch["validation_group"])

            # Update metrics
            if self.metrics is not None:
                aa_prec, aa_recall, pep_recall, _ = self.metrics.compute_precision_recall(targets, y)
                aa_er = self.metrics.compute_aa_er(targets, y)

                valid_metrics["valid_loss"].append(loss.item())
                valid_metrics["aa_er"].append(aa_er)
                valid_metrics["aa_prec"].append(aa_prec)
                valid_metrics["aa_recall"].append(aa_recall)
                valid_metrics["pep_recall"].append(pep_recall)

            valid_epoch_step += 1

            valid_timer.step()

            # Log progress
            if (valid_epoch_step + 1) % int(self.config.get("console_logging_steps", 2000)) == 0:
                epoch_step = valid_epoch_step % num_batches

                logger.info(
                    f"[VALIDATION] "
                    f"[Epoch {self.epoch:02d}] "
                    f"[Step {self.global_step + 1:06d}] "
                    f"[Batch {epoch_step:05d}/{num_batches:05d}] "
                    f"[{valid_timer.get_time_str()}/{valid_timer.get_total_time_str()}, "
                    f"{valid_timer.get_step_time_rate_str()}]"
                )

        # Synchronize all processes at the end of validation
        # This ensures all ranks wait for the slowest rank to finish
        self.accelerator.wait_for_everyone()

        if not calculate_metrics:
            return

        # Gather predictions from all devices
        if valid_predictions:
            self.log_if_verbose("Gathering predictions from all devices")
            # Use use_gather_object=True for Python lists to ensure proper gathering
            valid_predictions = self.accelerator.gather_for_metrics(valid_predictions, use_gather_object=True)
            valid_targets = self.accelerator.gather_for_metrics(valid_targets, use_gather_object=True)
            valid_prediction_ids = self.accelerator.gather_for_metrics(valid_prediction_ids, use_gather_object=True)

            # Flatten nested lists if gather_for_metrics returned nested structure (one per device)
            # gather_for_metrics with use_gather_object=True returns a list of lists when num_processes > 1
            # Structure: [[pred1, pred2, ...], [pred3, pred4, ...], ...] where each inner list is from one device
            # We detect this by checking if we have num_processes or fewer top-level lists, and all are lists
            if self.accelerator.num_processes > 1 and valid_predictions:
                # If the length matches num_processes (or less, if some processes had no data)
                # and all elements are lists, it's likely the nested structure from gathering
                if len(valid_predictions) <= self.accelerator.num_processes and all(isinstance(item, list) for item in valid_predictions):
                    # Flatten the nested structure
                    valid_predictions = [item for sublist in valid_predictions for item in sublist]
                    valid_targets = [item for sublist in valid_targets for item in sublist]
                    valid_prediction_ids = [item for sublist in valid_prediction_ids for item in sublist]  # type: ignore[attr-defined]

            # Validate that all gathered lists have matching lengths after flattening
            if len(valid_predictions) != len(valid_targets) or len(valid_predictions) != len(valid_prediction_ids):
                raise ValueError(
                    f"Length mismatch after gathering predictions from all devices. "
                    f"valid_predictions: {len(valid_predictions)}, "
                    f"valid_targets: {len(valid_targets)}, "
                    f"valid_prediction_ids: {len(valid_prediction_ids)}. "
                    f"num_processes: {self.accelerator.num_processes}"
                )

            # Convert to numpy array for np.unique
            valid_prediction_ids_array = np.array(valid_prediction_ids)

            # Use valid_prediction_ids to remove duplicates
            # Find the indices of the first occurrence of each unique prediction_id
            _, idx = np.unique(valid_prediction_ids_array, return_index=True)

            # Store original length before deduplication for validation
            original_length = len(valid_predictions)

            # Validate indices are within bounds - this should never happen if lengths match
            max_idx = len(valid_predictions) - 1
            if len(idx) > 0 and idx.max() > max_idx:
                raise IndexError(
                    f"IndexError: max index {idx.max()} exceeds valid_predictions length {len(valid_predictions)}. "
                    f"valid_prediction_ids length: {len(valid_prediction_ids)}, "
                    f"unique prediction_ids: {len(idx)}, "
                    f"num_processes: {self.accelerator.num_processes}. "
                )

            valid_predictions = [valid_predictions[i] for i in idx]
            valid_targets = [valid_targets[i] for i in idx]

            self.log_if_verbose(f"Gathered {len(valid_predictions)} predictions")

            # Gather validation groups if available
            if self.using_validation_groups:
                valid_groups = self.accelerator.gather_for_metrics(valid_groups, use_gather_object=True)
                # Flatten nested structure if needed (same logic as above)
                if self.accelerator.num_processes > 1 and valid_groups:
                    if len(valid_groups) <= self.accelerator.num_processes and all(isinstance(item, list) for item in valid_groups):
                        valid_groups = [item for sublist in valid_groups for item in sublist]

                # Validate length matches the original length before deduplication
                if len(valid_groups) != original_length:
                    raise ValueError(
                        f"Length mismatch for valid_groups. "
                        f"valid_groups length: {len(valid_groups)}, "
                        f"expected length (before dedup): {original_length}, "
                        f"deduplicated predictions length: {len(idx)}. "
                    )
                valid_groups = [valid_groups[i] for i in idx]

            # Gather valid_metrics from all devices
            for metric, values in valid_metrics.items():
                valid_metrics[metric] = self.accelerator.gather_for_metrics(values)

        # Keep validation metrics for checkpointing
        checkpoint_metric = self.config.get("checkpoint_metric", None)
        if checkpoint_metric is not None:
            self.last_validation_metric = np.mean(valid_metrics[checkpoint_metric])

        # Log validation metrics
        if self.accelerator.is_main_process and self.metrics is not None:
            # Validation metrics are logged by epoch
            validation_step = self.global_step + 1

            if self.sw is not None:
                for k, v in valid_metrics.items():
                    self.sw.add_scalar(f"eval/{k}", np.mean(v), validation_step)

            logger.info(
                f"[VALIDATION] [Epoch {self.epoch:02d}] "
                f"[Step {self.global_step + 1:06d}] "
                f"train_loss={self.running_loss if self.running_loss else 0:.5f}, "
                f"valid_loss={np.mean(valid_metrics['valid_loss']):.5f}"
            )
            logger.info(f"[VALIDATION] [Epoch {self.epoch:02d}] [Step {self.global_step + 1:06d}] Metrics:")
            for metric in ["aa_er", "aa_prec", "aa_recall", "pep_recall"]:
                val = np.mean(valid_metrics[metric])
                logger.info(f"[VALIDATION] [Epoch {self.epoch:02d}] [Step {self.global_step + 1:06d}] - {metric:11s}{val:.3f}")

            # Validation group logging
            if self.using_validation_groups and valid_groups and self.sw is not None:
                preds = pl.Series(valid_predictions)
                targs = pl.Series(valid_targets)
                groups = pl.Series(valid_groups)

                assert len(preds) == len(groups)
                assert len(targs) == len(groups)

                for group in groups.unique():
                    idx = groups == group
                    logger.info(f"Computing group {group} with {idx.sum()} samples")
                    if idx.sum() > 0:  # Only compute metrics if we have samples for this group
                        aa_prec, aa_recall, pep_recall, _ = self.metrics.compute_precision_recall(targs.filter(idx), preds.filter(idx))
                        aa_er = self.metrics.compute_aa_er(targs.filter(idx), preds.filter(idx))
                        self.sw.add_scalar(f"eval/{group}_aa_er", aa_er, validation_step)
                        self.sw.add_scalar(f"eval/{group}_aa_prec", aa_prec, validation_step)
                        self.sw.add_scalar(f"eval/{group}_aa_recall", aa_recall, validation_step)
                        self.sw.add_scalar(f"eval/{group}_pep_recall", pep_recall, validation_step)

        self.accelerator.wait_for_everyone()
