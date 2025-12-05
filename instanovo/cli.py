import glob
import importlib.metadata as metadata
from pathlib import Path
from typing import List, Optional

import typer
from rich.table import Table
from typing_extensions import Annotated

from instanovo import __version__
from instanovo.__init__ import console
from instanovo.constants import DEFAULT_INFERENCE_CONFIG_NAME, DEFAULT_INFERENCE_CONFIG_PATH
from instanovo.diffusion.cli import cli as diffusion_cli
from instanovo.scripts.convert_to_sdf import app as convert_to_sdf_app
from instanovo.transformer.cli import cli as transformer_cli
from instanovo.utils.cli_utils import compose_config
from instanovo.utils.colorlogging import ColorLog

logger = ColorLog(console, __name__).logger

combined_cli = typer.Typer(rich_markup_mode="rich", pretty_exceptions_enable=False)
combined_cli.add_typer(
    transformer_cli,
    name="transformer",
    help="Run predictions or train with only the transformer-based InstaNovo model.",
)
combined_cli.add_typer(
    diffusion_cli,
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
                "Either a model ID or a path to an Instanovo checkpoint file (.ckpt format)"
                # "Either a model ID (currently supported: "
                # f"""{", ".join(f"'{model_id}'" for model_id in InstaNovo.get_pretrained())}) """
                # "or a path to an Instanovo checkpoint file (.ckpt format)."
            ),
        ),
    ] = None,
    instanovo_plus_model: Annotated[
        Optional[str],
        typer.Option(
            "--instanovo-plus-model",
            "-p",
            help=(
                "Either a model ID or a path to an Instanovo+ checkpoint file (.ckpt format)"
                # "Either a model ID (currently supported: "
                # f"""{", ".join(f"'{model_id}'" for model_id in InstaNovoPlus.get_pretrained())}) """
                # "or a path to an Instanovo+ checkpoint file (.ckpt format)"
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
    refine: Annotated[
        Optional[bool],
        typer.Option(
            "--with-refinement/--no-refinement",
            help=("Refine the predictions of the transformer-based InstaNovo model with the diffusion-based InstaNovo+ model?"),
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
    # Defer imports to improve cli performance
    logger.info("Initializing InstaNovo inference.")
    from instanovo.diffusion.multinomial_diffusion import InstaNovoPlus
    from instanovo.diffusion.predict import CombinedPredictor
    from instanovo.transformer.model import InstaNovo

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
            "`instanovo predict --instanovo_model=path/to/model.ckpt`."
        )

    if config.refine:
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

        if config.get("instanovo_plus_model", None) is None:
            raise ValueError(
                "Expected 'instanovo_plus_model' when refining, but found None. Please specify it "
                "in the `config/inference/<your_config>.yaml` configuration file or with the cli "
                "flag `instanovo predict --instanovo-plus-model=path/to/model_dir`."
            )

    predictor = CombinedPredictor(config)
    predictor.predict()


@combined_cli.command()
def version() -> None:
    """Display version information for InstaNovo, Instanovo+ and its dependencies."""
    table = Table("Package", "Version")
    table.add_row("InstaNovo", __version__)
    table.add_row("InstaNovo+", __version__)
    table.add_row("NumPy", metadata.version("numpy"))
    table.add_row("PyTorch", metadata.version("torch"))
    console.print(table)


def instanovo_entrypoint() -> None:
    """Main entry point for the InstaNovo CLI application."""
    combined_cli()


if __name__ == "__main__":
    combined_cli()
