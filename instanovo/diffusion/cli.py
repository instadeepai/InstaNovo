import glob
from pathlib import Path
from typing import List, Optional

import typer
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
    logger.info("Initializing InstaNovo+ training.")
    from instanovo.diffusion.train import DiffusionTrainer

    if config_path is None:
        config_path = DEFAULT_TRAIN_CONFIG_PATH
    if config_name is None:
        config_name = "instanovoplus"

    config = compose_config(
        config_path=config_path,
        config_name=config_name,
        overrides=overrides,
    )

    logger.info("Initializing diffusion training.")
    trainer = DiffusionTrainer(config)
    trainer.train()


@cli.command("predict")
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
            help="Either a model ID or a path to an Instanovo+ checkpoint file (.ckpt format)",
            # help="Either a model ID (currently supported: "
            # f"""{", ".join(f"'{model_id}'" for model_id in InstaNovoPlus.get_pretrained())})"""
            # " or a path to an Instanovo+ checkpoint file (.ckpt format)",
        ),
    ] = None,
    denovo: Annotated[
        Optional[bool],
        typer.Option(
            "--denovo/--evaluation",
            help="Do [i]de novo[/i] predictions or evaluate an annotated file with peptide sequences?",
        ),
    ] = None,
    refine: Annotated[
        Optional[bool],
        typer.Option(
            "--with-refinement/--no-refinement",
            help="Refine the predictions of the transformer-based InstaNovo model with the diffusion-based InstaNovo+ model?",
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
    # Defer imports to improve cli performance
    from instanovo.diffusion.multinomial_diffusion import InstaNovoPlus
    from instanovo.diffusion.predict import DiffusionPredictor

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
    if refine is not None:
        config.refine = refine

    if output_path is not None:
        if output_path.exists():
            logger.info(f"Output path '{output_path}' already exists and will be overwritten.")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        config.output_path = str(output_path)
    if config.output_path and not Path(config.output_path).parent.exists():
        Path(config.output_path).parent.mkdir(parents=True, exist_ok=True)
    if config.get("output_path", None) is None and (config.get("denovo", False)):
        raise ValueError(
            "Expected 'output_path' but found None in denovo mode. Please specify it "
            "in the `config/inference/<your_config>.yaml` configuration file or with the cli flag "
            "`--output-path=path/to/output_file`."
        )

    if instanovo_plus_model is not None:
        if Path(instanovo_plus_model).is_dir():
            required_files = ["*.ckpt", "*.yaml", "*.pt"]
            missing_files = [ext for ext in required_files if not list(Path(instanovo_plus_model).glob(ext))]
            if missing_files:
                raise ValueError(f"The directory '{instanovo_plus_model}' is missing the following required file(s): {', '.join(missing_files)}.")
        elif (
            not Path(instanovo_plus_model).is_file()
            and not instanovo_plus_model.startswith("s3://")
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

    if (
        config.get("refine", False)
        and config.get("refinement_path", None) == config.get("output_path", None)
        and config.get("refinement_path", None) is not None
    ):
        raise ValueError("The 'refinement_path' should be different from the 'output_path' to avoid overwriting the original predictions.")

    predictor = DiffusionPredictor(config)
    predictor.predict()
