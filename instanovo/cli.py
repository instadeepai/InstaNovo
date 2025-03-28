import glob
import sys
import warnings
from pathlib import Path
from typing import List, Optional

import lightning as L
import numpy as np
import torch
import typer
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf, open_dict
from rich.table import Table
from typing_extensions import Annotated

from instanovo import __version__
from instanovo.__init__ import console
from instanovo.diffusion.multinomial_diffusion import InstaNovoPlus
from instanovo.diffusion.predict import get_preds as diffusion_get_preds
from instanovo.diffusion.train import train as train_diffusion
from instanovo.scripts.convert_to_sdf import app as convert_to_sdf_app
from instanovo.transformer.model import InstaNovo
from instanovo.transformer.predict import get_preds as transformer_get_preds
from instanovo.transformer.train import _set_author_neptune_api_token
from instanovo.transformer.train import train as train_transformer
from instanovo.utils.colorlogging import ColorLog

# Filter out a SyntaxWarning from pubchempy, see:
# https://github.com/mcs07/PubChemPy/pull/53
warnings.filterwarnings(
    "ignore",
    message=r'"is not" with \'int\' literal\. Did you mean "!="\?',
    category=SyntaxWarning,
    module="pubchempy",
)

logger = ColorLog(console, __name__).logger

DEFAULT_TRAIN_CONFIG_PATH = "configs"
DEFAULT_INFERENCE_CONFIG_PATH = "configs/inference"
DEFAULT_INFERENCE_CONFIG_NAME = "default"


def compose_config(
    config_path: Optional[str] = None,
    config_name: Optional[str] = None,
    overrides: Optional[List[str]] = None,
) -> DictConfig:
    """Compose Hydra configuration with given overrides.

    Args:
        config_path: Relative path to config directory
        config_name: Name of the base config file
        overrides: List of Hydra override strings

    Returns:
        DictConfig: Composed configuration
    """
    logger.info(f"Reading config from '{config_path}' with name '{config_name}'.")
    with initialize(config_path=config_path, version_base=None):
        cfg = compose(config_name=config_name, overrides=overrides, return_hydra_config=False)
        return cfg


combined_cli = typer.Typer(rich_markup_mode="rich")  # , pretty_exceptions_enable=False)
instanovo_cli = typer.Typer(rich_markup_mode="rich")  # , pretty_exceptions_enable=False)
instanovo_plus_cli = typer.Typer(rich_markup_mode="rich")  # , pretty_exceptions_enable=False)
combined_cli.add_typer(
    instanovo_cli,
    name="transformer",
    help="Run predictions or train with only the transformer-based InstaNovo model.",
)
combined_cli.add_typer(
    instanovo_plus_cli,
    name="diffusion",
    help="Run predictions or train with only the diffusion-based InstaNovo+ model.",
)
combined_cli.add_typer(convert_to_sdf_app)


@combined_cli.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    """Run predictions with InstaNovo and optionally with InstaNovo+.

    First with the transformer-based InstaNovo model and then optionally refine
    them with the diffusion based InstaNovo+ model.
    """
    # If you just run `instanovo` on the command line, show the help
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())


@instanovo_cli.command("predict")
def transformer_predict(
    data_path: Annotated[
        Optional[str],
        typer.Option(
            "--data-path",
            "-d",
            help="Path to input data file",
        ),
    ] = None,
    output_path: Annotated[
        Optional[Path],
        typer.Option(
            "--output-path",
            "-o",
            help="Path to output file.",
            exists=False,
            file_okay=True,
            dir_okay=False,
        ),
    ] = None,
    instanovo_model: Annotated[
        Optional[str],
        typer.Option(
            "--instanovo-model",
            "-i",
            help=(
                "Either a model ID (currently supported: "
                f"""{", ".join(f"'{model_id}'" for model_id in InstaNovo.get_pretrained())})"""
                " or a path to an Instanovo checkpoint file (.ckpt format)."
            ),
        ),
    ] = None,
    denovo: Annotated[
        Optional[bool],
        typer.Option(
            "--denovo/--evaluation",
            help="Do [i]de novo[/i] predictions or evaluate an annotated file "
            "with peptide sequences?",
        ),
    ] = None,
    config_path: Annotated[
        Optional[str],
        typer.Option(
            "--config-path",
            "-cp",
            help="Relative path to config directory.",
        ),
    ] = None,
    config_name: Annotated[
        Optional[str],
        typer.Option(
            "--config-name",
            "-cn",
            help="The name of the config (usually the file name without the .yaml extension).",
        ),
    ] = None,
    overrides: Optional[List[str]] = typer.Argument(None, hidden=True),
) -> DictConfig:
    """Run predictions with InstaNovo."""
    # Compose config with overrides
    logger.info("Initializing InstaNovo inference.")

    if config_path is None:
        config_path = DEFAULT_INFERENCE_CONFIG_PATH
    if config_name is None:
        config_name = DEFAULT_INFERENCE_CONFIG_NAME

    config = compose_config(
        config_path=config_path,
        config_name=config_name,
        overrides=overrides,
    )

    # Check config inputs
    if data_path is not None:
        if "*" in data_path or "?" in data_path or "[" in data_path:
            # Glob notation: path/to/data/*.parquet
            if not glob.glob(data_path):
                raise ValueError(f"The data_path '{data_path}' doesn't correspond to any file(s).")
        config.data_path = str(data_path)

    if not config.get("data_path", None) and data_path is None:
        raise ValueError(
            "Expected 'data_path' but found None. Please specify it in the "
            "`config/inference/<your_config>.yaml` configuration file or with the cli flag "
            "`--data-path='path/to/data'`. Allows `.mgf`, `.mzml`, `.mzxml`, a directory, or a "
            "`.parquet` file. Glob notation is supported:  eg.: `--data-path='./experiment/*.mgf'`."
        )

    if output_path is not None:
        if output_path.exists():
            logger.info(f"Output path '{output_path}' already exists and will be overwritten.")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        config.output_path = str(output_path)
    if config.get("output_path", None) is None:
        raise ValueError(
            "Expected 'output_path' but found None. Please specify it in the "
            "`config/inference/<your_config>.yaml` configuration file or with the cli flag "
            "`--output-path=path/to/output_file`."
        )

    if denovo is not None:
        config.denovo = denovo

    if instanovo_model is not None:
        if Path(instanovo_model).is_file() and Path(instanovo_model).suffix != ".ckpt":
            raise ValueError(
                f"Checkpoint file '{instanovo_model}' should end with extension '.ckpt'."
            )
        if (
            not Path(instanovo_model).is_file()
            and instanovo_model not in InstaNovo.get_pretrained()
        ):
            raise ValueError(
                f"InstaNovo model ID '{instanovo_model}' is not supported. "
                "Currently supported value(s): "
                f"""{", ".join(f"'{model_id}'" for model_id in InstaNovo.get_pretrained())}"""
            )
        config.instanovo_model = instanovo_model

    if not config.get("instanovo_model", None):
        raise ValueError(
            "Expected 'instanovo_model' but found None. Please specify it in the "
            "`config/inference/<your_config>.yaml` configuration file or with the cli flag "
            "`instanovo transformer predict --instanovo_model=path/to/model.ckpt`."
        )

    logger.info(f"Loading InstaNovo model {config.instanovo_model}")
    if config.instanovo_model in InstaNovo.get_pretrained():
        transformer_model, transformer_config = InstaNovo.from_pretrained(config.instanovo_model)
    else:
        transformer_model, transformer_config = InstaNovo.load(config.instanovo_model)

    logger.info(f"InstaNovo config:\n{OmegaConf.to_yaml(config)}")
    logger.info(
        f"InstaNovo model params: {np.sum([p.numel() for p in transformer_model.parameters()]):,d}"
    )

    if config.get("save_beams", False) and config.get("num_beams", 1) == 1:
        logger.warning(
            "num_beams is 1 and will override save_beams. Only use save_beams in beam search."
        )
        with open_dict(config):
            config["save_beams"] = False

    logger.info(f"Performing search with {config.get('num_beams', 1)} beams")
    transformer_get_preds(config, transformer_model, transformer_config)
    return config


@instanovo_plus_cli.command("predict")
def diffusion_predict(
    data_path: Annotated[
        Optional[str],
        typer.Option(
            "--data-path",
            "-d",
            help="Path to input data file",
        ),
    ] = None,
    output_path: Annotated[
        Optional[Path],
        typer.Option(
            "--output-path",
            "-o",
            help="Path to output file.",
            exists=False,
            file_okay=True,
            dir_okay=False,
        ),
    ] = None,
    instanovo_plus_model: Annotated[
        Optional[str],
        typer.Option(
            "--instanovo-plus-model",
            "-p",
            help="Either a model ID (currently supported: "
            f"""{", ".join(f"'{model_id}'" for model_id in InstaNovoPlus.get_pretrained())})"""
            " or a path to an Instanovo+ checkpoint file (.ckpt format)",
        ),
    ] = None,
    denovo: Annotated[
        Optional[bool],
        typer.Option(
            "--denovo/--evaluation",
            help="Do [i]de novo[/i] predictions or evaluate an annotated file "
            "with peptide sequences?",
        ),
    ] = None,
    refine: Annotated[
        Optional[bool],
        typer.Option(
            "--with-refinement/--no-refinement",
            help="Refine the predictions of the transformer-based InstaNovo model with the "
            "diffusion-based InstaNovo+ model?",
        ),
    ] = None,
    config_path: Annotated[
        Optional[str],
        typer.Option(
            "--config-path",
            "-cp",
            help="Relative path to config directory.",
        ),
    ] = None,
    config_name: Annotated[
        Optional[str],
        typer.Option(
            "--config-name",
            "-cn",
            help="The name of the config (usually the file name without the .yaml extension).",
        ),
    ] = None,
    overrides: Optional[List[str]] = typer.Argument(None, hidden=True),
) -> None:
    """Run predictions with InstaNovo+."""
    logger.info("Initializing InstaNovo+ inference.")

    if config_path is None:
        config_path = DEFAULT_INFERENCE_CONFIG_PATH
    if config_name is None:
        config_name = DEFAULT_INFERENCE_CONFIG_NAME

    config = compose_config(
        config_path=config_path,
        config_name=config_name,
        overrides=overrides,
    )

    # Check config inputs
    if data_path is not None:
        if "*" in data_path or "?" in data_path or "[" in data_path:
            # Glob notation: path/to/data/*.parquet
            if not glob.glob(data_path):
                raise ValueError(f"The data_path '{data_path}' doesn't correspond to any file(s).")
        config.data_path = str(data_path)

    if not config.get("data_path", None) and data_path is None:
        raise ValueError(
            "Expected 'data_path' but found None. Please specify it in the "
            "`config/inference/<your_config>.yaml` configuration file or with the cli flag "
            "`--data-path='path/to/data'`. Allows `.mgf`, `.mzml`, `.mzxml`, a directory, or a "
            "`.parquet` file. Glob notation is supported:  eg.: `--data-path='./experiment/*.mgf'`."
        )

    if denovo is not None:
        config.denovo = denovo

    if output_path is not None:
        if output_path.exists():
            logger.info(f"Output path '{output_path}' already exists and will be overwritten.")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        config.output_path = str(output_path)
    if config.output_path and not Path(config.output_path).parent.exists():
        Path(config.output_path).parent.mkdir(parents=True, exist_ok=True)
    if config.get("output_path", None) is None:
        raise ValueError(
            "Expected 'output_path' but found None. Please specify it in the "
            "`config/inference/<your_config>.yaml` configuration file or with the cli flag "
            "`--output-path=path/to/output_file`."
        )

    if refine is not None:
        config.refine = refine

    if instanovo_plus_model is not None:
        if Path(instanovo_plus_model).is_dir():
            required_files = ["*.ckpt", "*.yaml", "*.pt"]
            missing_files = [
                ext for ext in required_files if not list(Path(instanovo_plus_model).glob(ext))
            ]
            if missing_files:
                raise ValueError(
                    f"The directory '{instanovo_plus_model}' is missing the following "
                    f"required file(s): {', '.join(missing_files)}."
                )
        elif (
            not Path(instanovo_plus_model).is_file()
            and instanovo_plus_model not in InstaNovoPlus.get_pretrained()
        ):
            raise ValueError(
                f"InstaNovo+ model ID '{instanovo_plus_model}' is not supported. Currently "
                "supported value(s): "
                f"""{", ".join(f"'{model_id}'" for model_id in InstaNovoPlus.get_pretrained())}"""
            )
        config.instanovo_plus_model = instanovo_plus_model

    if not config.get("instanovo_plus_model", None):
        raise ValueError(
            "Expected 'instanovo_plus_model' but found None. Please specify it in the "
            "`config/inference/<your_config>.yaml` configuration file or with the cli flag "
            "`instanovo diffusion predict --instanovo-plus-model=path/to/model_dir`."
        )

    logger.info(f"InstaNovo+ config:\n{OmegaConf.to_yaml(config)}")

    # Save transformer predictions before refinement
    if config.get("refine", False) and not config.get("instanovo_predictions_path", ""):
        instanovo_predictions_path = Path(config.output_path).with_stem(
            Path(config.output_path).stem + "_before_refinement"
        )
        instanovo_predictions_path.write_text(Path(config.output_path).read_text())
        config.instanovo_predictions_path = str(instanovo_predictions_path)

    if config.get("refine", False) and config.get("instanovo_predictions_path", None) == config.get(
        "output_path", None
    ):
        raise ValueError(
            "The 'instanovo_predictions_path' should be different from the 'output_path' to avoid "
            "overwriting the transformer predictions."
        )

    logger.info(f"Loading InstaNovo+ model {config.instanovo_plus_model}")

    if config.instanovo_plus_model in InstaNovoPlus.get_pretrained():
        diffusion_model, diffusion_config = InstaNovoPlus.from_pretrained(
            config.instanovo_plus_model
        )
    else:
        diffusion_model, diffusion_config = InstaNovoPlus.load(config.instanovo_plus_model)

    # # assert diffusion_model.residues == transformer_model.residue_set, f"Residue set of the "
    # f"InstaNovo+ diffusion model:\n{diffusion_model.residues.index_to_residue}\ndoes not equal "
    # "the residue set of the InstaNovo transformer model:\n"
    # f"{transformer_model.residue_set.index_to_residue}"
    # # TODO: Align naming in diffusion model from 'residues' to 'residue_set'
    # if diffusion_model.residues != transformer_model.residue_set:
    #     logger.info(
    #         "Residue set of the InstaNovo+ diffusion model:\n"
    # f"{diffusion_model.residues.index_to_residue}\ndoes not equal the residue set of the "
    # f"InstaNovo transformer model:\n{transformer_model.residue_set.index_to_residue}"
    #     )
    #     diffusion_model.residues = transformer_model.residue_set

    logger.info(f"InstaNovo+ config:\n{OmegaConf.to_yaml(config)}")
    logger.info(
        f"InstaNovo+ model params: {np.sum([p.numel() for p in diffusion_model.parameters()]):,d}"
    )

    diffusion_get_preds(config, diffusion_model, diffusion_config)

    if config.get("refine", False) and not config.get("instanovo_predictions_path", ""):
        instanovo_predictions_path.unlink()


@combined_cli.command()
def predict(
    data_path: Annotated[
        Optional[str],
        typer.Option(
            "--data-path",
            "-d",
            help="Path to input data file",
        ),
    ] = None,
    output_path: Annotated[
        Optional[Path],
        typer.Option(
            "--output-path",
            "-o",
            help="Path to output file.",
            exists=False,
            file_okay=True,
            dir_okay=False,
        ),
    ] = None,
    instanovo_model: Annotated[
        Optional[str],
        typer.Option(
            "--instanovo-model",
            "-i",
            help=(
                "Either a model ID (currently supported: "
                f"""{", ".join(f"'{model_id}'" for model_id in InstaNovo.get_pretrained())}) """
                "or a path to an Instanovo checkpoint file (.ckpt format)."
            ),
        ),
    ] = None,
    instanovo_plus_model: Annotated[
        Optional[str],
        typer.Option(
            "--instanovo-plus-model",
            "-p",
            help=(
                "Either a model ID (currently supported: "
                f"""{", ".join(f"'{model_id}'" for model_id in InstaNovoPlus.get_pretrained())}) """
                "or a path to an Instanovo+ checkpoint file (.ckpt format)"
            ),
        ),
    ] = None,
    denovo: Annotated[
        Optional[bool],
        typer.Option(
            "--denovo/--evaluation",
            help="Do [i]de novo[/i] predictions or evaluate an "
            "annotated file with peptide sequences?",
        ),
    ] = None,
    refine: Annotated[
        Optional[bool],
        typer.Option(
            "--with-refinement/--no-refinement",
            help=(
                "Refine the predictions of the transformer-based InstaNovo model with the "
                "diffusion-based InstaNovo+ model?"
            ),
        ),
    ] = True,
    config_path: Annotated[
        Optional[str],
        typer.Option(
            "--config-path",
            "-cp",
            help="Relative path to config directory.",
        ),
    ] = None,
    config_name: Annotated[
        Optional[str],
        typer.Option(
            "--config-name",
            "-cn",
            help="The name of the config (usually the file name without the .yaml extension).",
        ),
    ] = None,
    overrides: Optional[List[str]] = typer.Argument(None, hidden=True),
) -> None:
    """Run predictions with InstaNovo and optionally refine with InstaNovo+.

    First with the transformer-based InstaNovo model and then optionally refine
    them with the diffusion based InstaNovo+ model.
    """
    config = transformer_predict(
        data_path=data_path,
        output_path=output_path,
        instanovo_model=instanovo_model,
        denovo=denovo,
        config_path=config_path,
        config_name=config_name,
        overrides=overrides,
    )

    if refine is not None:
        config.refine = refine

    if config.refine:
        if config.get("instanovo_plus_model", None) is None:
            raise ValueError(
                "Expected 'instanovo_plus_model' when refining, but found None. Please specify it "
                "in the `config/inference/<your_config>.yaml` configuration file or with the cli "
                "flag `instanovo diffusion predict --instanovo-plus-model=path/to/model_dir`."
            )

        diffusion_predict(
            data_path=data_path,
            output_path=output_path,
            instanovo_plus_model=instanovo_plus_model,
            denovo=denovo,
            refine=refine,
            config_path=config_path,
            config_name=config_name,
            overrides=overrides,
        )


@instanovo_cli.command("train")
def transformer_train(
    config_path: Annotated[
        Optional[str],
        typer.Option(
            "--config-path",
            "-cp",
            help="Relative path to config directory.",
        ),
    ] = None,
    config_name: Annotated[
        Optional[str],
        typer.Option(
            "--config-name",
            "-cn",
            help="The name of the config (usually the file name without the .yaml extension).",
        ),
    ] = None,
    overrides: Optional[List[str]] = typer.Argument(None, hidden=True),
) -> None:
    """Train the InstaNovo model."""
    _set_author_neptune_api_token()

    if config_path is None:
        config_path = DEFAULT_TRAIN_CONFIG_PATH
    if config_name is None:
        config_name = "instanovo"

    config = compose_config(
        config_path=config_path,
        config_name=config_name,
        overrides=overrides,
    )

    if config["n_gpu"] > 1:
        raise ValueError("n_gpu > 1 currently not supported.")

    logger.info("Initializing InstaNovo training.")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA version: {torch.version.cuda}")

    # Unnest hydra configs
    # TODO Use the nested configs by default
    sub_configs_list = ["model", "dataset", "residues"]
    for sub_name in sub_configs_list:
        if sub_name in config:
            with open_dict(config):
                temp = config[sub_name]
                del config[sub_name]
                config.update(temp)

    logger.info(f"InstaNovo training config:\n{OmegaConf.to_yaml(config)}")

    train_transformer(config)


@instanovo_plus_cli.command("train")
def diffusion_train(
    config_path: Annotated[
        Optional[str],
        typer.Option(
            "--config-path",
            "-cp",
            help="Relative path to config directory.",
        ),
    ] = None,
    config_name: Annotated[
        Optional[str],
        typer.Option(
            "--config-name",
            "-cn",
            help="The name of the config (usually the file name without the .yaml extension).",
        ),
    ] = None,
    overrides: Optional[List[str]] = typer.Argument(None, hidden=True),
) -> None:
    """Train the InstaNovo+ model."""
    _set_author_neptune_api_token()

    if config_path is None:
        config_path = DEFAULT_TRAIN_CONFIG_PATH
    if config_name is None:
        config_name = "instanovoplus"

    config = compose_config(
        config_path=config_path,
        config_name=config_name,
        overrides=overrides,
    )

    if config["n_gpu"] > 1:
        raise ValueError("n_gpu > 1 currently not supported.")

    logger.info("Initializing InstaNovo+ training.")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA version: {torch.version.cuda}")

    # Unnest hydra configs
    # TODO Use the nested configs by default
    sub_configs_list = ["model", "dataset", "residues"]
    for sub_name in sub_configs_list:
        if sub_name in config:
            with open_dict(config):
                temp = config[sub_name]
                del config[sub_name]
                config.update(temp)

    logger.info(f"InstaNovo+ training config:\n{OmegaConf.to_yaml(config)}")

    train_diffusion(config)


@combined_cli.command()
def version() -> None:
    """Display version information for InstaNovo, Instanovo+ and its dependencies."""
    table = Table("Package", "Version")
    table.add_row("InstaNovo", __version__)
    table.add_row("InstaNovo+", __version__)
    table.add_row("NumPy", np.__version__)
    table.add_row("PyTorch", torch.__version__)
    table.add_row("Lightning", L.__version__)
    console.print(table)


def instanovo_entrypoint() -> None:
    """Main entry point for the InstaNovo CLI application."""
    combined_cli()


if __name__ == "__main__":
    combined_cli()
