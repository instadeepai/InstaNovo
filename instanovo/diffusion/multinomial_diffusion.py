from __future__ import annotations

import json
import math
import os
import shutil
from importlib import resources
from pathlib import Path
from typing import Tuple
from urllib.parse import urlsplit

import requests
import torch
from jaxtyping import Float, Integer
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.distributions import Categorical
from torch.nn.functional import log_softmax, one_hot
from tqdm import tqdm

from instanovo.__init__ import console
from instanovo.diffusion.model import MassSpectrumTransFusion
from instanovo.types import Peptide, ResidueLogProbabilities, TimeStep
from instanovo.utils.colorlogging import ColorLog
from instanovo.utils.device_handler import check_device
from instanovo.utils.residues import ResidueSet
from instanovo.utils.s3 import S3FileHandler

MODEL_TYPE = "diffusion"

logger = ColorLog(console, __name__).logger


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> Float[torch.Tensor, " time"]:
    """Cosine schedule as proposed in https://arxiv.org/abs/2102.09672 .

    Returns alpha parameters, NOT Beta
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
    alphas = torch.clamp(alphas, 0.001, 1.0)
    return torch.sqrt(alphas)


class InstaNovoPlus(nn.Module):
    r"""This class implements Multinomial Diffusion as described in Hoogeboom et al. 2021.

    Args:
        config (omegaconf.DictConfig):
            The model configuration. This should have keys:
                - 'name': the model name identifier.
                - 'time_steps': the number of time steps in the diffusion process
                - 'max_length': the maximum sequence for the model
                - 'device': the device where the `Pytorch` model should be
                            loaded e.g. `cpu`, `cuda:0` etc.
                - 'vocab_size': the number of residues in the vocabulary
                - 'transition_model': the `DictConfig` for the transition model

            This information is necessary for saving and loading the model.

        transition_model (nn.Module):
            The model that predictions the initial sequence given
            the sequence sampled the current time step and the
            sequence sampled the previous time step. This is
            just a sequence tagging model.

        diffusion_schedule (torch.FloatTensor[time_steps]):
            The sequence of diffusion probabilities. Note
            that `diffusion_schedule[t]` is \alpha_t in
            the paper's terminology, not \beta_t.

        residue_set (ResidueSet):
            The residue vocabulary. This holds a mapping between
            residues and indices and residue masses.
    """

    config_path: str
    schedule_path: str
    checkpoint_path: str

    def __init__(
        self,
        config: DictConfig,
        transition_model: nn.Module,
        diffusion_schedule: Float[torch.Tensor, " time"],
        residue_set: ResidueSet,
    ) -> None:
        super().__init__()
        self.config = config
        self.time_steps = config.time_steps
        self.residue_set = residue_set
        self.transition_model = transition_model
        self.register_buffer("diffusion_schedule", torch.log(diffusion_schedule))
        self.register_buffer("diffusion_schedule_complement", torch.log(1 - diffusion_schedule))
        self.register_buffer("cumulative_schedule", torch.cumsum(self.diffusion_schedule, -1))
        self.register_buffer(
            "cumulative_schedule_complement",
            torch.log(1 - torch.exp(self.cumulative_schedule)),
        )

    def save(
        self,
        path: str,
        ckpt_details: str,
        overwrite: bool = False,
        temp_dir: str | None = None,
        use_legacy_format: bool = False,
    ) -> None:
        """Save the model to a directory.

        Args:
            path (str):
                Path to the base directory where the model is saved.
                The model is saved in a subdirectory with the model's
                name identifier.

            ckpt_details (str):
                Additional checkpoint details to include in model save directory.

            overwrite (bool, optional):
                Whether to overwrite the directory if one already exists
                for the model. Defaults to False.

            temp_dir (str | None, optional):
                Temporary directory to save intermediate files to.
                Defaults to None.

            use_legacy_format (bool, optional):
                Whether to save the model in the legacy folder format.
                If False, saves as a single file. Defaults to False.

        Raises:
            FileExistsError: If `overwrite` is `False` and a directory already exists
                for the model identifier.
        """
        model_dir = os.path.join(path, ckpt_details)

        def save_file_local(filename: str, content: str) -> None:
            """Save a file locally (no upload)."""
            return

        def save_file_s3(filename: str, content: str) -> None:
            """Upload a file to S3."""
            # TODO: fix this
            s3 = S3FileHandler()
            return s3.upload(  # type: ignore
                content, s3.convert_to_s3_output(model_dir + "/" + filename)
            )

        if temp_dir is None:
            if os.path.exists(model_dir) and os.path.isdir(model_dir):
                if overwrite:
                    shutil.rmtree(model_dir)
                else:
                    raise FileExistsError

            if use_legacy_format:
                os.makedirs(model_dir, exist_ok=True)
            elif os.path.dirname(model_dir):
                os.makedirs(os.path.dirname(model_dir), exist_ok=True)

            save_path = model_dir
            save_file = save_file_local

        else:
            save_path = temp_dir
            save_file = save_file_s3

        if use_legacy_format:
            # Save model as a folder
            # Save config
            config_path = os.path.join(save_path, "config.yaml")
            OmegaConf.save(config=self.config, f=config_path)
            save_file("config.yaml", config_path)

            # Save schedule
            diff_schedule_path = os.path.join(save_path, "diffusion_schedule.pt")
            torch.save(torch.exp(self.diffusion_schedule), diff_schedule_path)
            save_file("diffusion_schedule.pt", diff_schedule_path)

            # Save transition model
            self.transition_model.to("cpu")
            transition_model_path = os.path.join(save_path, "transition_model.ckpt")
            torch.save(self.transition_model.state_dict(), transition_model_path)
            save_file("transition_model.ckpt", transition_model_path)

            device = check_device(config=self.config)
            logger.info(f"Moving transition model to device {device}")
            self.transition_model.to(device)
        else:
            # Save model as a single file
            transition_model_state = {
                k: v.cpu() for k, v in self.transition_model.state_dict().items()
            }

            model_data = {
                "config": OmegaConf.to_container(self.config),
                "diffusion_schedule": torch.exp(self.diffusion_schedule).tolist(),
                "transition_model": transition_model_state,
            }

            if temp_dir:
                save_path = os.path.join(save_path, "instanovo_plus.ckpt")
                torch.save(model_data, save_path)
                save_file("instanovo_plus.ckpt", save_path)
            else:
                torch.save(model_data, save_path)
                save_file("instanovo_plus.ckpt", save_path)

    @classmethod
    def load(cls, path: str, device: str = "auto") -> Tuple[InstaNovoPlus, DictConfig]:
        """Load a saved model.

        Args:
            path (str):
                Path to model checkpoint file or directory where model is saved.
            device (str, optional):
                The device to load the model on. Defaults to "auto".

        Returns:
            (InstaNovoPlus, DictConfig): The loaded model and config.

        """
        if os.path.isdir(path):
            # Load config
            cls.config_path = os.path.join(path, "config.yaml")
            config = OmegaConf.load(cls.config_path)

            cls.schedule_path = os.path.join(path, "diffusion_schedule.pt")
            diffusion_schedule = torch.load(
                cls.schedule_path, map_location=torch.device("cpu"), weights_only=True
            )

            cls.checkpoint_path = os.path.join(path, "transition_model.ckpt")
            transition_model_state = torch.load(
                cls.checkpoint_path, map_location=torch.device("cpu"), weights_only=True
            )

            residues = config.get("residues")
            residue_remapping = config.get("residue_remapping", {})
        else:
            # Load model from checkpoint file
            try:
                model_data = torch.load(path, map_location=torch.device("cpu"), weights_only=False)
            except Exception as e:
                raise ValueError(f"Failed to load model from {path}: {str(e)}") from e

            config = OmegaConf.create(model_data["config"])
            diffusion_schedule = torch.tensor(model_data["diffusion_schedule"])

            if "transition_model" in model_data:
                transition_model_state = model_data["transition_model"]
                residues = config.get("residues")
                residue_remapping = config.get("residue_remapping", {})

            elif "state_dict" in model_data:
                transition_model_state = model_data["state_dict"]
                residues = model_data["residues"]
                residue_remapping = model_data.get("residue_remapping", {})

            elif "model" in model_data:
                transition_model_state = model_data["model"]
                residues = model_data["residues"].get("residues", {})
                residue_remapping = model_data["residues"].get("residue_remapping", {})

            else:
                raise ValueError("Model data is missing a key for weights.")

        # Load residues
        residue_set = ResidueSet(
            residue_masses=residues,
            residue_remapping=residue_remapping if residue_remapping else None,
        )

        # Load transition model
        try:
            transition_model = MassSpectrumTransFusion(
                config,
                config.max_length,
            )

            # Remove transition_model prefix from state dict keys if present
            if any(k.startswith("transition_model.") for k in transition_model_state.keys()):
                transition_model_state = {
                    k.replace("transition_model.", ""): v for k, v in transition_model_state.items()
                }

            transition_model.load_state_dict(transition_model_state, strict=False)
        except Exception as e:
            raise ValueError(f"Failed to load transition model: {str(e)}") from e

        device = check_device(device=device)
        logger.info(f"Loading InstaNovoPlus model to device: {device}.")
        transition_model.to(device)
        diffusion_schedule = diffusion_schedule.to(device)

        return cls(
            config=config,
            transition_model=transition_model,
            diffusion_schedule=diffusion_schedule,
            residue_set=residue_set,
        ), config

    @staticmethod
    def get_pretrained() -> list[str]:
        """Get a list of pretrained model ids."""
        # Load the models.json file
        with resources.files("instanovo").joinpath("models.json").open("r", encoding="utf-8") as f:
            models_config = json.load(f)

        if MODEL_TYPE not in models_config:
            return []

        return list(models_config[MODEL_TYPE].keys())

    @classmethod
    def from_pretrained(
        cls, model_id: str, device: str = "auto"
    ) -> Tuple["InstaNovoPlus", "DictConfig"]:
        """Download and load by model id or model path."""
        # Check if model_id is a local dir
        expected_files = ["config.yaml", "diffusion_schedule.pt", "transition_model.ckpt"]
        if os.path.isdir(model_id):
            if all(os.path.exists(os.path.join(model_id, fn)) for fn in expected_files):
                return cls.load(model_id, device=device)
            else:
                missing_files = [
                    fn for fn in expected_files if not os.path.exists(os.path.join(model_id, fn))
                ]
                raise FileNotFoundError(
                    f"InstaNovo+ model directory {model_id} is missing "
                    f"the expected file(s): {', '.join(missing_files)}."
                )
        elif os.path.exists(model_id):
            return cls.load(model_id, device=device)

        # Load the models.json file
        with resources.files("instanovo").joinpath("models.json").open("r", encoding="utf-8") as f:
            models_config = json.load(f)

        # Find the model in the config
        if MODEL_TYPE not in models_config or model_id not in models_config[MODEL_TYPE]:
            raise ValueError(
                f"Model {model_id} not found in models.json, options are "
                f"[{', '.join(models_config[MODEL_TYPE].keys())}]"
            )

        # Create cache directory if it doesn't exist
        cache_dir = Path.home() / ".cache" / "instanovo"
        cache_dir.mkdir(parents=True, exist_ok=True)

        model_info = models_config[MODEL_TYPE][model_id]

        if "remote" in model_info:
            url = model_info["remote"]

            # Generate a filename for the cached model
            file_name = urlsplit(url).path.split("/")[-1]
            cached_file = cache_dir / file_name

            # Check if the file is already cached
            if not cached_file.exists():
                # If not cached, download the file with a progress bar
                response = requests.get(url, stream=True)
                total_size = int(response.headers.get("content-length", 0))
                logger.info(f"Downloading model {model_id} from {url}")

                with (
                    open(cached_file, "wb") as file,
                    tqdm(
                        desc=file_name,
                        total=total_size,
                        unit="iB",
                        unit_scale=True,
                        unit_divisor=1024,
                    ) as progress_bar,
                ):
                    for data in response.iter_content(chunk_size=1024):
                        size = file.write(data)
                        progress_bar.update(size)
                if not os.path.getsize(cached_file) == total_size:
                    raise ValueError(
                        f"Downloaded file is incomplete. Expected size of {total_size} "
                        "bytes does not match downloaded size of "
                        f"{os.path.getsize(cached_file)} bytes."
                    )
            else:
                logger.info(f"Model {model_id} already cached at {cached_file}")

            # Load and return the model
            logger.info(f"Loading model {model_id} (remote)")
            return cls.load(str(cached_file), device=device)

        elif "local" in model_info:
            instanovo_plus_model = model_info["local"]
            if os.path.isdir(instanovo_plus_model):
                if all(
                    os.path.exists(os.path.join(instanovo_plus_model, fn)) for fn in expected_files
                ):
                    logger.info(f"Loading model {model_id} (local)")
                    return cls.load(instanovo_plus_model, device=device)
                else:
                    missing_files = [
                        fn
                        for fn in expected_files
                        if not os.path.exists(os.path.join(instanovo_plus_model, fn))
                    ]
                    raise FileNotFoundError(
                        f"InstaNovo+ model directory {instanovo_plus_model} is missing the "
                        f"expected file(s): {', '.join(missing_files)}."
                    )
            elif os.path.exists(instanovo_plus_model):
                return cls.load(instanovo_plus_model, device=device)
            else:
                raise ValueError(
                    f"Local model path '{instanovo_plus_model}' must exist, be a directory and "
                    f"containing the files {', '.join(expected_files)}."
                )
        else:
            raise ValueError(
                f"Model {model_id} does not have a valid 'remote', 'local' entry in models.json"
            )

    def prepare_fine_tuning(self, residue_set: ResidueSet) -> None:
        """Prepare a model for fine-tuning on a dataset with a new residue vocabulary.

        Args:
            residue_set (ResidueSet): The residue vocabulary for the new dataset.
        """
        # 1. Update residue set
        self.residue_set = residue_set

        num_residues = len(self.residue_set)
        model_dim = self.config.dim

        # 2. Update config
        self.config.vocab_size = num_residues

        # 3. Update modules
        self.transition_model.char_embedding = nn.Embedding(
            num_embeddings=num_residues, embedding_dim=model_dim
        )
        self.transition_model.head[1] = nn.Linear(model_dim, num_residues)

    def mixture_categorical(
        self,
        log_x: Float[ResidueLogProbabilities, "batch token"],
        log_alpha: float,
        log_alpha_complement: float,
    ) -> Float[ResidueLogProbabilities, "batch token"]:
        """A categorical mixture between a base distribution and a uniform distribution.

        Args:
            log_x (torch.FloatTensor[..., num_classes]):
                The base distribution.

            log_alpha (float):
                The log of the mixture weight.

            log_alpha_complement (float):
                The log of 1 minus the mixture weight.

        Returns:
            torch.FloatTensor[..., num_classes]:
                The log-probabilities of the mixture.
        """
        return torch.logaddexp(
            log_x + log_alpha,
            log_alpha_complement - math.log(len(self.residue_set)),
        )

    def forward(
        self,
        log_x_t: Float[ResidueLogProbabilities, "batch token"],
        log_x_0: Float[ResidueLogProbabilities, "batch token"],
        t: Integer[TimeStep, " batch"],
    ) -> Float[ResidueLogProbabilities, "batch token"]:
        """Calculate the log-posterior of `t-1`-th process values given the 0-th and t-th values.

        Args:
            log_x_t (torch.FloatTensor[batch_size, sequence_length, num_classes]):
                The log one-hot representation of the process values at the `t`-th time step.

            log_x_0 (torch.FloatTensor[batch_size, sequence_length, num_classes]):
                The log one-hot representation of the process values at the `t`-th time step.
            t (int):
                The time step.

        Returns:
            torch.FloatTensor[batch_size, sequence_length, num_classes]:
                The log-posterior probabilities of the process values at the `t-1`-th
                time step given the values at the 0-th and `t`-th time step
                i.e. q( x_{t-1} | x_{t}, x_0 ).
        """
        log_prior = self.mixture_categorical(
            log_x=log_x_0,
            log_alpha=self.cumulative_schedule[t - 1].unsqueeze(-1).unsqueeze(-1),
            log_alpha_complement=self.cumulative_schedule_complement[t - 1]
            .unsqueeze(-1)
            .unsqueeze(-1),
        )
        log_likelihood = self.mixture_categorical(
            log_x=log_x_t,
            log_alpha=self.diffusion_schedule[t].unsqueeze(-1).unsqueeze(-1),
            log_alpha_complement=self.diffusion_schedule_complement[t].unsqueeze(-1).unsqueeze(-1),
        )
        t_mask = (t == 0).unsqueeze(-1).unsqueeze(-1).expand_as(log_x_0)
        prior_term = torch.where(t_mask, log_x_0, log_prior)
        logits = log_likelihood + prior_term
        return torch.log_softmax(logits, -1)

    def reverse_distribution(
        self,
        x_t: Integer[Peptide, "batch token"],
        time: Integer[TimeStep, " batch"],
        **kwargs: dict,
    ) -> Float[ResidueLogProbabilities, "batch token"]:
        """Calculate the reverse transition distribution of the diffusion process.

        Args:
            x_t (torch.LongTensor[batch_size, sequence_length]):
                The values at the `t`-th time step of the reverse process.

            time (int):
                The time step.

        Returns:
            torch.FloatTensor[batch_size, sequence_length, num_classes]:
                The log-probabilities of values for the `t-1`-th time step given
                values at the `t`-th time step i.e. `log p( x_{t-1} | x_{t} )`.
        """
        log_x_0 = log_softmax(self.transition_model(x_t, t=time, **kwargs), -1)
        return self.forward(
            log_x_t=torch.log(one_hot(x_t, len(self.residue_set))), log_x_0=log_x_0, t=time
        )


class DiffusionLoss(nn.Module):
    """Holds logic for calculating the diffusion loss.

    Args:
        model (InstaNovoPlus):
            The multinomial diffusion class.
    """

    def __init__(self, model: InstaNovoPlus) -> None:
        super().__init__()
        self.time_steps = model.time_steps
        self.model = model

    @staticmethod
    def kl_divergence(
        log_probs_first: Float[ResidueLogProbabilities, "..."],
        log_probs_second: Float[ResidueLogProbabilities, "..."],
    ) -> Float[torch.Tensor, "..."]:
        """Calculate the Kullback-Liebler divergence between two multinomial distributions.

        Args:
            log_probs_first (torch.FloatTensor[..., num_classes]):
                 The log-probabilities of the base distribution.

            log_probs_second (torch.FloatTensor[..., num_classes]):
                 The log-probabilities of the comparison distribution.

        Returns:
            torch.FloatTensor[1]:
                The KL-divergence averaged over all but the final dimension.
        """
        return (torch.exp(log_probs_first) * (log_probs_first - log_probs_second)).sum(-1).sum(-1)

    def forward(
        self, x_0: Integer[Peptide, "batch token"], **kwargs: dict
    ) -> Float[torch.Tensor, "1"]:
        """Calculate a single Monte Carlo estimate of the multinomial diffusion loss (-ELBO).

        Args:
            x_0 (torch.LongTensor[batch_size, sequence_length]):
                A batch of padded sequences.

        Returns:
            torch.FloatTensor[1]:
                The loss estimate.
        """
        # 1. Sample time step
        t = torch.randint(0, self.time_steps - 1, (x_0.shape[0],)).to(x_0.device)

        # 2. Compute L_t
        loss = self._compute_loss(t=t, x_0=x_0, **kwargs).mean()

        # 3. Calculate prior KL term
        log_x_0 = torch.log(one_hot(x_0, num_classes=len(self.model.residue_set)))
        final_log_probs = self.model.mixture_categorical(
            log_x=log_x_0,
            log_alpha=self.model.cumulative_schedule[self.time_steps - 1]
            .unsqueeze(-1)
            .unsqueeze(-1),
            log_alpha_complement=self.model.cumulative_schedule_complement[self.time_steps - 1]
            .unsqueeze(-1)
            .unsqueeze(-1),
        )
        uniform_log_probs = torch.log(
            torch.ones_like(final_log_probs) / len(self.model.residue_set)
        )
        kl_loss = self.kl_divergence(final_log_probs, uniform_log_probs).mean()
        return loss + kl_loss

    def _compute_loss(
        self,
        x_0: Integer[Peptide, "batch token"],
        t: Integer[TimeStep, " batch"],
        **kwargs: dict,
    ) -> Float[torch.Tensor, " batch"]:
        # 1. sample x_{t+1}
        log_x_0 = torch.log(one_hot(x_0, num_classes=len(self.model.residue_set)))
        log_probs = self.model.mixture_categorical(
            log_x=log_x_0,
            log_alpha=self.model.cumulative_schedule[t].unsqueeze(-1).unsqueeze(-1),
            log_alpha_complement=self.model.cumulative_schedule_complement[t]
            .unsqueeze(-1)
            .unsqueeze(-1),
        )
        x_next = Categorical(logits=log_probs).sample()

        # 2. Calculate loss
        log_dist = self.model.reverse_distribution(x_t=x_next, time=t, **kwargs)

        nll_loss = (
            -(one_hot(x_0, num_classes=len(self.model.residue_set)) * log_dist).sum(-1).sum(-1)
        )

        log_posterior = self.model(
            log_x_0=log_x_0, log_x_t=torch.log(one_hot(x_next, log_probs.size(-1))), t=t
        )
        denoising_loss = self.kl_divergence(log_posterior, log_dist)
        loss = torch.where(t == 0, nll_loss, denoising_loss)
        return loss
