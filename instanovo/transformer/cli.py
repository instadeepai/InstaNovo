import glob
from pathlib import Path
from typing import List, Optional

import typer
from omegaconf import DictConfig
from typing_extensions import Annotated

from instanovo.__init__ import console
from instanovo.constants import (
    DEFAULT_INFERENCE_CONFIG_NAME,
    DEFAULT_INFERENCE_CONFIG_PATH,
    DEFAULT_TRAIN_CONFIG_PATH,
)
from instanovo.utils.cli_utils import compose_config
from instanovo.utils.colorlogging import ColorLog

logger = ColorLog(console, __name__).logger

cli = typer.Typer(rich_markup_mode="rich", pretty_exceptions_enable=False)


@cli.command("train")
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
    logger.info("Initializing InstaNovo training.")

    # Defer imports to improve cli performance
    from instanovo.transformer.train import TransformerTrainer

    if config_path is None:
        config_path = DEFAULT_TRAIN_CONFIG_PATH
    if config_name is None:
        config_name = "instanovo"

    config = compose_config(
        config_path=config_path,
        config_name=config_name,
        overrides=overrides,
    )

    logger.info("Initializing InstaNovo training.")
    trainer = TransformerTrainer(config)
    trainer.train()


@cli.command("predict")
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
                "Either a model ID or a path to an Instanovo checkpoint file (.ckpt format)."
                # Removed: expensive in in CLI, TODO: explore re-adding later
                # "Either a model ID (currently supported: "
                # f"""{", ".join(f"'{model_id}'" for model_id in InstaNovo.get_pretrained())})"""
                # " or a path to an Instanovo checkpoint file (.ckpt format)."
            ),
        ),
    ] = None,
    denovo: Annotated[
        Optional[bool],
        typer.Option(
            "--denovo/--evaluation",
            help="Do [i]de novo[/i] predictions or evaluate an annotated file with peptide sequences?",
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

    # Defer imports to improve cli performance
    from instanovo.transformer.model import InstaNovo
    from instanovo.transformer.predict import TransformerPredictor

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
        # Don't compute metrics in denovo mode
        config.denovo = denovo

    if output_path is not None:
        if output_path.exists():
            logger.info(f"Output path '{output_path}' already exists and will be overwritten.")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        config.output_path = str(output_path)
    if config.get("output_path", None) is None and config.get("denovo", False):
        raise ValueError(
            "Expected 'output_path' but found None in denovo mode. Please specify it in the "
            "`config/inference/<your_config>.yaml` configuration file or with the cli flag "
            "`--output-path=path/to/output_file`."
        )

    if instanovo_model is not None:
        if Path(instanovo_model).is_file() and Path(instanovo_model).suffix != ".ckpt":
            raise ValueError(f"Checkpoint file '{instanovo_model}' should end with extension '.ckpt'.")
        if not Path(instanovo_model).is_file() and not instanovo_model.startswith("s3://") and instanovo_model not in InstaNovo.get_pretrained():
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

    logger.info("Initializing InstaNovo inference.")
    predictor = TransformerPredictor(config)
    predictor.predict()
