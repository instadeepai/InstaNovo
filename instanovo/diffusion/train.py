from __future__ import annotations

import logging
import os

import hydra
import polars
import torch
import tqdm
from deepspeed.runtime.lr_schedules import WarmupLR
from dtu_denovo_sequencing.diffusion.data import AnnotatedPolarsSpectrumDataset
from dtu_denovo_sequencing.diffusion.data import collate_batches
from dtu_denovo_sequencing.diffusion.multinomial_diffusion import cosine_beta_schedule
from dtu_denovo_sequencing.diffusion.multinomial_diffusion import DiffusionLoss
from dtu_denovo_sequencing.diffusion.multinomial_diffusion import MassSpectrumTransFusion
from dtu_denovo_sequencing.diffusion.multinomial_diffusion import MultinomialDiffusion
from dtu_denovo_sequencing.utils.residues import ResidueSet
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


@hydra.main(version_base="1.2", config_path="../../config/diffusion")
def main(config: DictConfig) -> None:
    """Main function for training and validating a Multinomial Diffusion model.

    This function performs the following tasks:
    1. Initializes logging and tensorboard logging.
    2. Loads the training and validation data from the specified paths.
    3. Initializes the Multinomial Diffusion model, possibly from a checkpoint.
    4. Initializes the loss function and the optimizer.
    5. Enters the training loop for a specified number of epochs,
       within which it performs the following tasks:
       a. Evaluates the model on the validation dataset and logs the validation loss.
       b. Performs a training step on the training dataset and logs the training loss.
       c. Optionally saves a checkpoint of the model if the validation loss has improved.

    Parameters:
    config (DictConfig): A configuration object containing all the necessary
                         parameters for data loading, model initialization,
                         and training.

    Returns:
    None
    """
    # 1. Initialize logging
    logger = logging.Logger(name=config.model.name, level=logging.INFO)
    logger.addHandler(logging.FileHandler(filename=config.logging.log_file))

    tensorboard_logger = SummaryWriter(
        log_dir=os.path.join(config.logging.tensorboard_dir, config.model.name)
    )

    # 2. Load data
    logger.info("Loading data.")
    residues = ResidueSet(residue_masses=config.residues)
    data_config = config.data

    # -- Load train data
    logger.info("Loading training data.")
    train_data = polars.read_ipc(data_config.train_data_path)
    train_peptides = [sequence[1:-1] for sequence in train_data["Modified sequence"]]
    train_dataset = AnnotatedPolarsSpectrumDataset(data_frame=train_data, peptides=train_peptides)
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=data_config.batch_size,
        collate_fn=collate_batches(
            residues=residues,
            max_length=config.model.max_length,
            time_steps=config.model.time_steps,
            annotated=True,
        ),
    )

    # -- Load validation data
    logger.info("Loading validation data.")
    validation_data = polars.read_ipc(data_config.validation_data_path)
    validation_peptides = [sequence[1:-1] for sequence in validation_data["Modified sequence"]]
    validation_dataset = AnnotatedPolarsSpectrumDataset(
        data_frame=validation_data, peptides=validation_peptides
    )
    validation_data_loader = DataLoader(
        validation_dataset,
        batch_size=data_config.batch_size,
        collate_fn=collate_batches(
            residues=residues,
            max_length=config.model.max_length,
            time_steps=config.model.time_steps,
            annotated=True,
        ),
    )

    # 3. Initialize model
    logger.info("Initializing model.")
    model_config = config.model
    if "checkpoint" in config:
        model = MultinomialDiffusion.load(model_config.checkpoint)
        if model_config.replace_residues:
            model.prepare_fine_tuning(residues=residues)
    else:
        transition_model = MassSpectrumTransFusion(
            cfg=model_config.transition_model, max_transcript_len=model_config.max_length
        )
        diffusion_schedule = cosine_beta_schedule(timesteps=model_config.time_steps)
        model = MultinomialDiffusion(
            config=model_config,
            transition_model=transition_model,
            residues=residues,
            diffusion_schedule=diffusion_schedule,
        )
    model = model.to(model_config.device)

    # 4. Initialize loss
    logger.info("Initializing loss.")
    loss_function = DiffusionLoss(model=model)

    # 4. Initialize optimizer
    logger.info("Initializing optimizer.")
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=config.training.learning_rate, betas=config.training.betas
    )
    warm_up_scheduler = WarmupLR(
        optimizer=optimizer,
        warmup_min_lr=config.training.min_lr,
        warmup_max_lr=config.training.max_lr,
        warmup_num_steps=config.training.num_steps,
        warmup_type=config.training.type,
    )

    # 5. Perform training loop
    logger.info("Starting training.")
    global_step, min_val_loss = 0, float("inf")
    epoch_iterator = tqdm.trange(config.training.num_epochs)
    for epoch in epoch_iterator:
        # -- Calculate validation performance
        val_losses = []
        for spectra, spectra_padding_mask, precursors, peptides, peptide_mask in tqdm.tqdm(
            validation_data_loader, total=len(validation_data_loader)
        ):
            loss = loss_function(
                peptides.to(model_config.device),
                spectra=spectra.to(model_config.device),
                spectra_padding_mask=spectra_padding_mask.to(model_config.device),
                precursors=precursors.to(model_config.device),
                x_padding_mask=peptide_mask.to(model_config.device),
            )
            val_losses.append(loss.item())
        val_loss = sum(val_losses) / len(val_losses)
        logger.info(f"Epoch {epoch}, Validation loss: {val_loss:.2f}")
        tensorboard_logger.add_scalar("val/loss", val_loss, global_step)
        epoch_iterator.set_description(f"Validation loss: {val_loss:.2f}")

        # -- Checkpoint model
        if val_loss < min_val_loss:
            model.save(path=model_config.save_dir, overwrite=True)

        batch_iterator = tqdm.tqdm(train_data_loader, total=len(train_data_loader))
        for spectra, spectra_padding_mask, precursors, peptides, peptide_mask in batch_iterator:
            optimizer.zero_grad()
            loss = loss_function(
                peptides.to(model_config.device),
                spectra=spectra.to(model_config.device),
                spectra_padding_mask=spectra_padding_mask.to(model_config.device),
                precursors=precursors.to(model_config.device),
                x_padding_mask=peptide_mask.to(model_config.device),
            )
            loss.backward()
            optimizer.step()
            warm_up_scheduler.step()
            logger.info(
                f"Epoch: {epoch}, Global Step: {global_step}, Training Loss: {loss.item():.2f}"
            )
            tensorboard_logger.add_scalar("train/loss", loss.item(), global_step)
            batch_iterator.set_description(f"Loss: {loss.item():.2f}")
            global_step += 1


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
