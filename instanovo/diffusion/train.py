import datetime
import math
import os
import shutil
import sys
import tempfile
import time
import warnings
from pathlib import Path

import hydra
import neptune
import numpy as np
import pandas as pd
import polars as pl
import torch
from neptune.integrations.python_logger import NeptuneHandler
from neptune.internal.utils.git import GitInfo
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import instanovo.utils.s3 as s3
from instanovo.__init__ import console
from instanovo.constants import ANNOTATED_COLUMN, ANNOTATION_ERROR
from instanovo.diffusion.multinomial_diffusion import (
    DiffusionLoss,
    InstaNovoPlus,
    MassSpectrumTransFusion,
    cosine_beta_schedule,
)
from instanovo.inference.diffusion import DiffusionDecoder
from instanovo.transformer.dataset import (
    SpectrumDataset,
    collate_batch,
    remove_modifications,
)
from instanovo.transformer.train import (
    NeptuneSummaryWriter,
    WarmupScheduler,
    _format_time,
    _set_author_neptune_api_token,
)
from instanovo.utils import SpectrumDataFrame
from instanovo.utils.colorlogging import ColorLog
from instanovo.utils.device_handler import check_device
from instanovo.utils.metrics import Metrics
from instanovo.utils.residues import ResidueSet

logger = ColorLog(console, __name__).logger

CONFIG_PATH = Path(__file__).parent.parent / "configs"

warnings.filterwarnings("ignore", message=".*does not have many workers*")


def train(config: DictConfig) -> None:
    """Training function."""
    torch.manual_seed(config.get("seed", 101))
    torch.set_float32_matmul_precision("high")

    time_now = datetime.datetime.now().strftime("_%y_%m_%d_%H_%M")
    if s3.register_tb():  # when on aichor, s3_enabled True so register tb True
        config["tb_summarywriter"] = os.environ["AICHOR_LOGS_PATH"]
    else:
        config["tb_summarywriter"] = config["tb_summarywriter"] + time_now

    if config.get("report_to", "") == "neptune":
        if "NEPTUNE_API_TOKEN" not in os.environ:
            raise ValueError(
                "In the configuration file, 'report_to' is set to 'neptune', but no "
                "Neptune API token is found. Please set the NEPTUNE_API_TOKEN environment variable"
            )
        os.environ["NEPTUNE_PROJECT"] = "InstaDeep/denovo-sequencing"

        if "AICHOR_LOGS_PATH" in os.environ:
            # On AIchor the .git folder is not available so we cannot rely on neptune’s git
            # integration to log the git info as an artifact. Instead, we monkeypatch the function
            # used by neptune to retrieve the git info, and call it before creating the neptune run.
            neptune.metadata_containers.run.to_git_info = lambda git_ref: GitInfo(
                commit_id=os.environ["VCS_SHA"],
                message=os.environ["VCS_COMMIT_MESSAGE"],
                author_name=os.environ["VCS_AUTHOR_NAME"],
                author_email=os.environ["VCS_AUTHOR_EMAIL"],
                # not available as env variable
                commit_date=datetime.datetime.now(),
                dirty=False,
                branch=os.environ["VCS_REF_NAME"],
                remotes=None,
            )

        run = neptune.init_run(
            with_id=None,
            name=config.get("run_name", "no_run_name_specified") + time_now,
            dependencies=str(Path(__file__).parent.parent.parent / "uv.lock"),
            tags=OmegaConf.to_object(config.get("tags", ListConfig([]))),
        )
        run.assign({"config": OmegaConf.to_yaml(config)})
        sw = NeptuneSummaryWriter(config["tb_summarywriter"], run)
        logger.addHandler(NeptuneHandler(run=run))
    else:
        sw = SummaryWriter(config["tb_summarywriter"])

    logger.info("Starting diffusion training")

    # Transformer vocabulary
    residue_set = ResidueSet(
        residue_masses=config["residues"],
        residue_remapping=config["residue_remapping"],
    )
    logger.info(f"Vocab: {residue_set.index_to_residue}")

    logger.info("Loading data")

    try:
        train_sdf = SpectrumDataFrame.load(
            config.get("train_path"),
            lazy=config.get("lazy_loading", True),
            is_annotated=True,
            shuffle=True,
            partition=config.get("train_partition", None),
            column_mapping=config.get("column_remapping", None),
            max_shard_size=config.get("max_shard_size", 100_000),
            preshuffle_across_shards=config.get("preshuffle_shards", False),
            verbose=config.get("verbose_loading", True),
        )
        valid_sdf = SpectrumDataFrame.load(
            config.get("valid_path", None) or config.get("train_path"),
            lazy=config.get("lazy_loading", True),
            is_annotated=True,
            shuffle=False,
            partition=config.get("valid_partition", None),
            column_mapping=config.get("column_remapping", None),
            max_shard_size=config.get("max_shard_size", 100_000),
        )
    except ValueError as e:
        # More descriptive error message in predict mode.
        if str(e) == ANNOTATION_ERROR:
            raise ValueError(
                "The sequence column is missing annotations, "
                "are you trying to run de novo prediction? Add the --denovo flag"
            ) from e
        else:
            raise

    if config.get("valid_path", None) is None:
        logger.info("Validation path not specified, generating from training set.")
        sequences = list(train_sdf.get_unique_sequences())
        sequences = sorted({remove_modifications(x) for x in sequences})

        train_unique, valid_unique = train_test_split(
            sequences,
            test_size=config.get("valid_subset_of_train"),
            random_state=42,
        )
        train_unique = set(train_unique)
        valid_unique = set(valid_unique)

        train_sdf.filter_rows(lambda row: remove_modifications(row["sequence"]) in train_unique)
        valid_sdf.filter_rows(lambda row: remove_modifications(row["sequence"]) in valid_unique)
        # Save splits
        # TODO: Optionally load the data splits
        # TODO: Allow loading of data splits in `predict.py`
        # TODO: Upload to Aichor
        split_path = os.path.join(
            config.get("model_save_folder_path", "./checkpoints"), "splits.csv"
        )
        os.makedirs(os.path.dirname(split_path), exist_ok=True)
        pd.DataFrame(
            {
                "modified_sequence": list(train_unique) + list(valid_unique),
                "split": ["train"] * len(train_unique) + ["valid"] * len(valid_unique),
            }
        ).to_csv(str(split_path), index=False)
        logger.info(f"Data splits saved to {split_path}")

    # Check residues
    if config.get("perform_data_checks", True):
        logger.info(f"Checking for unknown residues in {len(train_sdf) + len(valid_sdf):,d} rows.")
        supported_residues = set(residue_set.vocab)
        supported_residues.update(set(residue_set.residue_remapping.keys()))
        data_residues = set()
        data_residues.update(train_sdf.get_vocabulary(residue_set.tokenize))
        data_residues.update(valid_sdf.get_vocabulary(residue_set.tokenize))
        if len(data_residues - supported_residues) > 0:
            logger.warning(
                "Unsupported residues found in evaluation set! These rows will be dropped."
            )
            logger.info(f"New residues found: \n{data_residues - supported_residues}")
            logger.info(f"Residues supported: \n{supported_residues}")
            original_size = (len(train_sdf), len(valid_sdf))
            train_sdf.filter_rows(
                lambda row: all(
                    residue in supported_residues
                    for residue in set(residue_set.tokenize(row[ANNOTATED_COLUMN]))
                )
            )
            valid_sdf.filter_rows(
                lambda row: all(
                    residue in supported_residues
                    for residue in set(residue_set.tokenize(row[ANNOTATED_COLUMN]))
                )
            )
            new_size = (len(train_sdf), len(valid_sdf))
            logger.warning(
                f"{original_size[0] - new_size[0]:,d} "
                f"({(original_size[0] - new_size[0]) / original_size[0] * 100:.2f}%) "
                "training rows dropped."
            )
            logger.warning(
                f"{original_size[1] - new_size[1]:,d} "
                f"({(original_size[1] - new_size[1]) / original_size[1] * 100:.2f}%) "
                "validation rows dropped."
            )

        # Check charge values:
        original_size = (len(train_sdf), len(valid_sdf))
        train_sdf.filter_rows(
            lambda row: (row["precursor_charge"] <= config.get("max_charge", 10))
            and (row["precursor_charge"] > 0)
        )
        if len(train_sdf) < original_size[0]:
            logger.warning(
                f"Found {original_size[0] - len(train_sdf)} rows in training set with charge > "
                f"{config.get('max_charge', 10)} or <= 0. These rows will be skipped."
            )

        valid_sdf.filter_rows(
            lambda row: (row["precursor_charge"] <= config.get("max_charge", 10))
            and (row["precursor_charge"] > 0)
        )
        if len(valid_sdf) < original_size[1]:
            logger.warning(
                f"Found {original_size[1] - len(valid_sdf)} rows in training set with charge > "
                f"{config.get('max_charge', 10)}. These rows will be skipped."
            )

    train_sdf.sample_subset(fraction=config.get("train_subset", 1.0), seed=42)
    valid_sdf.sample_subset(fraction=config.get("valid_subset", 1.0), seed=42)

    train_ds = SpectrumDataset(
        train_sdf,
        residue_set,
        config["n_peaks"],
        return_str=False,
        peptide_pad_length=config.get("max_length", 40)
        if config.get("compile_model", False)
        else 0,
        pad_spectrum_max_length=config.get("compile_model", False)
        or config.get("use_flash_attention", False),
        bin_spectra=config.get("conv_peak_encoder", False),
        diffusion=True,
        reverse_peptide=False,  # we do not reverse peptides for diffusion model
    )
    valid_ds = SpectrumDataset(
        valid_sdf,
        residue_set,
        config["n_peaks"],
        return_str=False,
        peptide_pad_length=config.get("max_length", 40),
        pad_spectrum_max_length=config.get("compile_model", False)
        or config.get("use_flash_attention", False),
        bin_spectra=config.get("conv_peak_encoder", False),
        diffusion=True,
        reverse_peptide=False,
    )

    logger.info(
        f"Data loaded: {len(train_ds):,} training samples; {len(valid_ds):,} validation samples"
    )

    train_sequences = pl.Series(list(train_sdf.get_unique_sequences()))
    valid_sequences = pl.Series(list(valid_sdf.get_unique_sequences()))
    if config.get("blacklist", None):
        logger.info("Checking if any training set overlaps with blacklisted sequences...")
        blacklist_df = pd.read_csv(config["blacklist"])
        leakage = any(
            train_sequences.map_elements(remove_modifications, return_dtype=pl.String).is_in(
                blacklist_df["sequence"]
            )
        )
        if leakage:
            raise ValueError(
                "Portion of training set sequences overlaps with blacklisted sequences."
            )
        else:
            logger.info("No blacklisted sequences!")

    if config.get("perform_data_checks", True):
        logger.info("Checking if any validation set overlaps with training set...")
        leakage = any(valid_sequences.is_in(train_sequences))
        if leakage:
            raise ValueError("Portion of validation set sequences overlaps with training set.")
        else:
            logger.info("No data leakage!")

    # Check how many times model will save
    if config.get("save_model", True):
        total_epochs = config.get("epochs", 30)
        epochs_per_save = 1 / (
            len(train_ds) / config.get("train_batch_size", 256) / config.get("ckpt_interval")
        )
        if epochs_per_save > total_epochs:
            logger.warning(
                f"Model checkpoint will never save. Attempting to save every {epochs_per_save:.2f} "
                f"epochs but only training for {total_epochs:d} epochs. "
                "Check ckpt_interval in config."
            )
        else:
            logger.info(f"Model checkpointing every {epochs_per_save:.2f} epochs.")

    # Check warmup
    if config.get("warmup_iters", 100_000) > len(train_ds) / config.get("train_batch_size", 256):
        logger.warning(
            "Model warmup is greater than one epoch of the training set. "
            "Check warmup_iters in config"
        )

    train_dl = DataLoader(
        train_ds,
        batch_size=config["train_batch_size"],
        num_workers=0,  # SDF requirement is 0
        shuffle=False,  # SDF requirement
        collate_fn=collate_batch,
    )

    valid_dl = DataLoader(
        valid_ds,
        batch_size=config["predict_batch_size"],
        num_workers=0,  # SDF requirement is 0
        shuffle=False,
        collate_fn=collate_batch,
    )

    # Update rates based on bs=32
    step_scale = 32 / config["train_batch_size"]
    logger.info(f"Updates per epoch: {len(train_dl):,}, step_scale={step_scale}")

    batch = next(iter(train_dl))
    spectra, precursors, spectra_mask, peptides, peptides_mask = batch
    logger.info("Sample batch:")
    logger.info(f" - spectra.shape={spectra.shape}")
    logger.info(f" - precursors.shape={precursors.shape}")
    logger.info(f" - spectra_mask.shape={spectra_mask.shape}")
    logger.info(f" - peptides.shape={peptides.shape}")
    logger.info(f" - peptides_mask.shape={peptides_mask.shape}")

    if peptides_mask.any():
        raise ValueError(
            "Peptide mask contains True values, attention should apply to the whole sequence when "
            "using diffusion."
        )

    logger.info("Initializing model.")

    transition_model = MassSpectrumTransFusion(
        cfg=config,
        max_transcript_len=config["max_length"],
    )
    diffusion_schedule = cosine_beta_schedule(timesteps=config["time_steps"])
    model = InstaNovoPlus(
        config=config,
        transition_model=transition_model,
        diffusion_schedule=diffusion_schedule,
        residues=residue_set,
    )

    if not config.get("train_from_scratch", True):
        resume_checkpoint_path = config["resume_checkpoint"]
    else:
        resume_checkpoint_path = None

    if resume_checkpoint_path is not None:
        logger.info(f"Loading model checkpoint from '{resume_checkpoint_path}'")

        diffusion_model, _diffusion_config = InstaNovoPlus.load(resume_checkpoint_path)
        model_state = diffusion_model.transition_model.state_dict()

        aa_embed_size = model_state["head.1.weight"].shape[0]
        if aa_embed_size != len(residue_set):
            state_keys = ["head.1.weight", "head.1.bias", "char_embedding.weight"]
            logger.warning(
                f"Model expects vocab size of {len(residue_set)}, checkpoint has {aa_embed_size}."
            )
            logger.warning("Assuming a change was made to the residues in the configuration file.")
            logger.warning(f"Automatically converting {state_keys} to match expected.")

            new_model_state = model.transition_model.state_dict()

            resolution = config.get("residue_conflict_resolution", "delete")

            for k in state_keys:
                # initialise weights to normal distribution with weight 1/sqrt(dim)
                tmp = torch.normal(
                    mean=0,
                    std=1.0 / np.sqrt(config["dim"]),
                    size=new_model_state[k].shape,
                    dtype=new_model_state[k].dtype,
                )
                if "bias" in k:
                    # initialise bias to zeros
                    tmp = torch.zeros_like(tmp)

                if resolution == "delete":
                    del model_state[k]
                elif resolution == "random":
                    model_state[k] = tmp
                elif resolution == "partial":
                    tmp[:aa_embed_size] = model_state[k][: min(tmp.shape[0], aa_embed_size)]
                    model_state[k] = tmp
                else:
                    raise ValueError(f"Unknown residue_conflict_resolution type '{resolution}'")

            logger.warning(
                f"Model checkpoint has {len(state_keys)} weights updated with '{resolution}' "
                "conflict resolution"
            )

        k_missing: int = np.sum(
            [
                x not in list(model_state.keys())
                for x in list(model.transition_model.state_dict().keys())
            ]
        )
        if k_missing > 0:
            logger.warning(f"Model checkpoint is missing {k_missing} keys!")
        k_missing: int = np.sum(
            [
                x not in list(model.transition_model.state_dict().keys())
                for x in list(model_state.keys())
            ]
        )
        if k_missing > 0:
            logger.warning(f"Model state is missing {k_missing} keys!")
        model.transition_model.load_state_dict(model_state, strict=False)

    logger.info(
        f"Model loaded with {np.sum([p.numel() for p in model.parameters()]):,d} parameters"
    )

    # Set device to train on
    device = check_device(config=config)
    logger.info(f"Training InstaNovo+ on device: {device}")
    fp16 = config.get("fp16", True)
    if fp16 and device.lower() == "cpu":
        logger.warning("fp16 is enabled but device type is cpu. fp16 will be disabled.")
        fp16 = False

    model = model.to(device)

    # Initialize metrics
    metrics = Metrics(residue_set, config["isotope_error_range"])

    # Initialize decoder
    decoder = DiffusionDecoder(model=model)

    # Use as an additional data sanity check
    if config.get("validate_precursor_mass", True):
        logger.info("Sanity checking precursor masses for training set...")
        train_sdf.validate_precursor_mass(metrics)
        logger.info("Sanity checking precursor masses for validation set...")
        valid_sdf.validate_precursor_mass(metrics)

    logger.info("Initializing loss.")
    loss_function = DiffusionLoss(model=model)

    logger.info("Initializing optimizer.")
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=config.learning_rate,
        weight_decay=float(config["weight_decay"]),
    )

    warm_up_scheduler = WarmupScheduler(optimizer=optimizer, warmup=config["warmup_iters"])

    # TODO implement forward pass check?

    _temp_directory = None
    if config["save_model"]:
        logger.info("Model saving enabled")
        if s3._s3_enabled():
            _temp_directory = tempfile.mkdtemp()
            logger.info("Temporary directory created.")

        logger.info(f"Saving every {config['ckpt_interval']} training steps.")
    else:
        logger.info("Model saving disabled")

    # Perform training loop
    logger.info("InstaNovo+ training started.")
    global_step = 0
    train_epoch_start_time = None
    running_loss = None

    def valid_epoch(epoch: int, global_step: int) -> None:
        logger.info("Validation loop.")
        model.eval()

        val_losses = []
        targets = []  # type: ignore
        results = []
        all_log_probs = []

        for _, batch in enumerate(valid_dl):
            spectra, precursors, spectra_padding_mask, peptides, peptide_mask = batch
            spectra = spectra.to(device)
            precursors = precursors.to(device)
            spectra_padding_mask = spectra_padding_mask.to(device)
            peptides = peptides.to(device)
            peptide_mask = peptide_mask.to(device)

            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16, enabled=fp16):
                loss = loss_function(
                    peptides,
                    spectra=spectra,
                    spectra_padding_mask=spectra_padding_mask,
                    precursors=precursors,
                    x_padding_mask=peptide_mask,
                )
                val_losses.append(loss.item())

                predictions, log_probs = decoder.decode(
                    spectra=spectra,
                    spectra_padding_mask=spectra_padding_mask,
                    precursors=precursors,
                )  # these predictions have been reversed

            targets.extend(
                residue_set.decode(seq, reverse=False) for seq in peptides
            )  # do not need to reverse peptides
            results.extend(predictions)
            all_log_probs.extend(log_probs)

        val_loss = sum(val_losses) / len(val_losses)
        logger.info(f"Epoch {epoch}, Validation loss: {val_loss:.2f}")
        sw.add_scalar("val/epoch/loss", val_loss, epoch)
        sw.add_scalar("val/step/loss", val_loss, global_step)

        # print("peptides ", targets)
        # print("predictions ", results)
        # print("log_probs ", all_log_probs)

        aa_prec, aa_recall, pep_recall, _ = metrics.compute_precision_recall(targets, results)
        aa_er = metrics.compute_aa_er(targets, results)

        sw.add_scalar("eval/aa_prec", aa_prec, global_step)
        sw.add_scalar("eval/aa_recall", aa_recall, global_step)
        sw.add_scalar("eval/pep_recall", pep_recall, global_step)
        sw.add_scalar("eval/aa_er", aa_er, global_step)

        logger.info(
            "aa_prec: %.4f, aa_recall: %.4f, pep_recall: %.4f, aa_er: %.4f",
            aa_prec,
            aa_recall,
            pep_recall,
            aa_er,
        )

        # print(aa_prec, aa_recall, pep_recall, aa_er)

    num_epochs = config["epochs"]
    for epoch in range(num_epochs):
        logger.info("Training loop")
        model.train()

        if train_epoch_start_time is None:
            train_epoch_start_time = time.time()

        for _, batch in enumerate(train_dl):
            spectra, precursors, spectra_padding_mask, peptides, peptide_mask = batch
            spectra = spectra.to(device)
            precursors = precursors.to(device)
            spectra_padding_mask = spectra_padding_mask.to(device)
            peptides = peptides.to(device)
            peptide_mask = peptide_mask.to(device)

            optimizer.zero_grad()
            loss = loss_function(
                peptides,
                spectra=spectra,
                spectra_padding_mask=spectra_padding_mask,
                precursors=precursors,
                x_padding_mask=peptide_mask,
            )
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=config.get("gradient_clip_val", 10)
            )

            optimizer.step()

            if running_loss is None:
                running_loss = loss.item()
            else:
                running_loss = 0.99 * running_loss + (1 - 0.99) * loss.item()

            if (global_step + 1) % int(
                config.get("tensorboard_logging_steps", 500) * step_scale
            ) == 0:
                lr = warm_up_scheduler.get_last_lr()[0]
                sw.add_scalar("train/step/loss", loss.item(), global_step)
                sw.add_scalar("train/loss_smooth", running_loss, global_step)
                sw.add_scalar("optim/lr", lr, global_step)
                sw.add_scalar("optim/epoch", epoch, global_step)

                logger.info(
                    f"Epoch: {epoch}, Global Step: {global_step}, Training Loss: {loss.item():.2f}"
                )

            if (
                (global_step + 1) % int(config.get("console_logging_steps", 2000) * step_scale)
            ) == 0:
                lr = warm_up_scheduler.get_last_lr()[0]
                delta = time.time() - train_epoch_start_time
                epoch_step = global_step % len(train_dl)
                est_total = delta / (epoch_step + 1) * (len(train_dl) - epoch_step - 1)

                logger.info(
                    f"[TRAIN] [Epoch {epoch:02d}/{num_epochs - 1:02d} Step {global_step:06d}] "
                    f"[Batch {epoch_step + 1:05d}/{len(train_dl):05d}] "
                    f"[{_format_time(delta)}/{_format_time(est_total)}, "
                    f"{(delta / (epoch_step + 1)):.3f}s/it]: "
                    f"train_loss_raw={loss.item():.4f}, "
                    f"running_loss={running_loss:.4f}, LR={lr:.6f}"
                )

            if (
                global_step + 1
            ) % math.ceil(  # TODO for global step 0, model will have trained a step
                config.get("val_check_interval", 1.0) * len(train_dl)
            ) == 0 or global_step == 0:
                valid_epoch(epoch, global_step)  # perform a validation loop
                model.train()

            warm_up_scheduler.step()

            # -- Checkpoint model # TODO add min val ckpt?
            if (
                config.get("save_model", True)
                and (global_step + 1) % config.get("ckpt_interval") == 0
            ):
                ckpt_interval = f"epoch_{epoch}_step_{global_step}"
                logger.info(f"Saving model {ckpt_interval}.")
                model.save(
                    path=config.get("model_save_folder_path", "./checkpoints"),
                    ckpt_details=ckpt_interval,
                    overwrite=True,
                    temp_dir=_temp_directory if s3._s3_enabled() else "",  # type: ignore
                )

            global_step += 1

        train_epoch_start_time = None

    if _temp_directory is not None and os.path.exists(_temp_directory):
        shutil.rmtree(_temp_directory)

    logger.info("InstaNovo+ training finished.")


# TODO remove main function
@hydra.main(config_path=str(CONFIG_PATH), version_base=None, config_name="instanovoplus")
def main(config: DictConfig) -> None:
    """Train the model."""
    logger.info("Initializing training.")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Torch version: {torch.__version__}")
    logger.info(f"CUDA version: {torch.version.cuda}")

    _set_author_neptune_api_token()

    # Unnest hydra configs
    # TODO Use the nested configs by default
    sub_configs_list = ["model", "dataset", "residues"]
    for sub_name in sub_configs_list:
        if sub_name in config:
            with open_dict(config):
                temp = config[sub_name]
                del config[sub_name]
                config.update(temp)

    logger.info(f"Imported hydra config:\n{OmegaConf.to_yaml(config)}")

    if config["n_gpu"] > 1:
        raise ValueError("n_gpu > 1 currently not supported.")

    train(config)
