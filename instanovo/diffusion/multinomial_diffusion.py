from __future__ import annotations

import math
import os
import shutil

import torch
from jaxtyping import Float
from jaxtyping import Integer
from omegaconf import DictConfig
from omegaconf import OmegaConf
from torch import nn
from torch.distributions import Categorical
from torch.nn.functional import log_softmax
from torch.nn.functional import one_hot

from instanovo.diffusion.model import MassSpectrumTransFusion
from instanovo.types import Peptide
from instanovo.types import ResidueLogProbabilities
from instanovo.types import TimeStep
from instanovo.utils.residues import ResidueSet


def cosine_beta_schedule(
    timesteps: int, s: float = 0.008
) -> Float[torch.Tensor, " time"]:
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


class MultinomialDiffusion(nn.Module):
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

        residues (ResidueSet):
            The residue vocabulary. This holds a mapping between
            residues and indices and residue masses.
    """

    def __init__(
        self,
        config: DictConfig,
        transition_model: nn.Module,
        diffusion_schedule: Float[torch.Tensor, " time"],
        residues: ResidueSet,
    ) -> None:
        super().__init__()
        self.config = config
        self.time_steps = config.time_steps
        self.residues = residues
        self.transition_model = transition_model
        self.register_buffer("diffusion_schedule", torch.log(diffusion_schedule))
        self.register_buffer(
            "diffusion_schedule_complement", torch.log(1 - diffusion_schedule)
        )
        self.register_buffer(
            "cumulative_schedule", torch.cumsum(self.diffusion_schedule, -1)
        )
        self.register_buffer(
            "cumulative_schedule_complement",
            torch.log(1 - torch.exp(self.cumulative_schedule)),
        )

    def save(self, path: str, overwrite: bool = False) -> None:
        """Save the model to a directory.

        Args:
            path (str):
                Path to the base directory where the model is saved.
                The model is saved in a subdirectory with the model's
                name identifier.

            overwrite (bool, optional):
                Whether to overwrite the directory if one already exists
                for the model. Defaults to False.

        Raises:
            FileExistsError: If `overwrite` is `False` and a directory already exists
                for the model identifier.
        """
        # Make directory
        model_dir = os.path.join(path, self.config.name)
        if os.path.exists(model_dir):
            if overwrite:
                shutil.rmtree(model_dir)
            else:
                raise FileExistsError

        os.mkdir(path=model_dir)

        # Save config
        OmegaConf.save(config=self.config, f=os.path.join(model_dir, "config.yaml"))

        # Save residues
        residues = OmegaConf.create(self.residues.residue_masses)
        OmegaConf.save(config=residues, f=os.path.join(model_dir, "residues.yaml"))

        # Save schedule
        torch.save(
            torch.exp(self.diffusion_schedule),
            os.path.join(model_dir, "diffusion_schedule.pt"),
        )

        # Save transition model
        self.transition_model.to("cpu")
        torch.save(
            self.transition_model.state_dict(),
            os.path.join(model_dir, "transition_model.ckpt"),
        )
        self.transition_model.to(self.config.device)

    @classmethod
    def load(cls, path: str) -> MultinomialDiffusion:
        """Load a saved model.

        Args:
            path (str):
                Path to the directory where the model is saved.

        Returns:
            (MultinomialDiffusion): The loaded model.
        """
        # Load config
        config = OmegaConf.load(os.path.join(path, "config.yaml"))

        # Load residues
        residue_masses = OmegaConf.load(os.path.join(path, "residues.yaml"))
        residues = ResidueSet(residue_masses=residue_masses)

        # Load schedule
        diffusion_schedule = torch.load(os.path.join(path, "diffusion_schedule.pt"))

        # Load transition model
        transition_model = MassSpectrumTransFusion(
            config.transition_model, config.max_length
        )
        transition_model.load_state_dict(
            torch.load(os.path.join(path, "transition_model.ckpt"))
        )

        return cls(
            config=config,
            transition_model=transition_model,
            diffusion_schedule=diffusion_schedule,
            residues=residues,
        )

    def prepare_fine_tuning(self, residues: ResidueSet) -> None:
        """Prepare a model for fine-tuning on a dataset with a new residue vocabulary.

        Args:
            residues (ResidueSet): The residue vocabulary for the new dataset.
        """
        # 1. Update residue set
        self.residues = residues

        num_residues = len(self.residues)
        model_dim = self.config.transition_model.dim

        # 2. Update config
        self.config.transition_model.vocab_size = num_residues

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
            log_alpha_complement - math.log(len(self.residues)),
        )

    def forward(
        self,
        log_x_t: Float[ResidueLogProbabilities, "batch token"],
        log_x_0: Float[ResidueLogProbabilities, "batch token"],
        t: Integer[TimeStep, " batch"],
    ) -> Float[ResidueLogProbabilities, "batch token"]:
        """Calculate the log-posterior of the `t-1`-th process values given the 0-th and t-th values.

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
            log_alpha_complement=self.diffusion_schedule_complement[t]
            .unsqueeze(-1)
            .unsqueeze(-1),
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
            log_x_t=torch.log(one_hot(x_t, len(self.residues))), log_x_0=log_x_0, t=time
        )


class DiffusionLoss(nn.Module):
    """Holds logic for calculating the diffusion loss.

    Args:
        model (MultinomialDiffusion):
            The multinomial diffusion class.
    """

    def __init__(self, model: MultinomialDiffusion) -> None:
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
        return (
            (torch.exp(log_probs_first) * (log_probs_first - log_probs_second))
            .sum(-1)
            .sum(-1)
        )

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
        log_x_0 = torch.log(one_hot(x_0, num_classes=len(self.model.residues)))
        final_log_probs = self.model.mixture_categorical(
            log_x=log_x_0,
            log_alpha=self.model.cumulative_schedule[self.time_steps - 1]
            .unsqueeze(-1)
            .unsqueeze(-1),
            log_alpha_complement=self.model.cumulative_schedule_complement[
                self.time_steps - 1
            ]
            .unsqueeze(-1)
            .unsqueeze(-1),
        )
        uniform_log_probs = torch.log(
            torch.ones_like(final_log_probs) / len(self.model.residues)
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
        log_x_0 = torch.log(one_hot(x_0, num_classes=len(self.model.residues)))
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
            -(one_hot(x_0, num_classes=len(self.model.residues)) * log_dist)
            .sum(-1)
            .sum(-1)
        )

        log_posterior = self.model(
            log_x_0=log_x_0, log_x_t=torch.log(one_hot(x_next, log_probs.size(-1))), t=t
        )
        denoising_loss = self.kl_divergence(log_posterior, log_dist)
        loss = torch.where(t == 0, nll_loss, denoising_loss)
        return loss
