from __future__ import annotations

import logging
import os

import click
import pandas
import polars
import torch
import tqdm
import yaml

from instanovo.diffusion.data import AnnotatedPolarsSpectrumDataset
from instanovo.diffusion.data import collate_batches
from instanovo.diffusion.multinomial_diffusion import MultinomialDiffusion
from instanovo.inference.diffusion import DiffusionDecoder
from instanovo.utils.residues import ResidueSet

# from torch.utils.data import DataLoader


DIFFUSION_START_STEP = 15


@click.command()
@click.option("--input-path", "-i")
@click.option("--start-predictions-path", "-s")
@click.option("--model-path", "-m")
@click.option("--output-path", "-o")
@click.option("--batch-size", "-bs", default=16)
@click.option("--device", "-dv", default="cpu")
def main(
    input_path: str,
    start_predictions_path: str,
    model_path: str,
    output_path: str,
    batch_size: int,
    device: str,
) -> None:
    """Predict peptides from spectra using the diffusion model for iterative refinement.

    Args:
        input_path (str):
            Path to Polars `.ipc` file containing spectra. This should have the columns:
                - "Mass values": the sequence of m/z values as a list or numpy array.
                - "Intensities": the instensities corresponding to the m/z values in "Mass values".
                                (This should be the same length as "Mass values".)
                - "MS/MS m/z": the precursor mass-to-charge ratio.
                - "Charge": the precursor charge

        start_predictions_path (str):
            Path to the csv holding initial predictions to be refined by the diffusion model.
            This should be index-aligned with the Polars `.ipc` file specified in `input_path`
            and the predictions should be in a column called "Predictions" as an already tokenized
            list of strings.
        model_path (str):
            Path to the model checkpoint. This should be a directory (so it should first be unzipped
            if it is a zip file).
        output_path (str):
            Path where the output should be written. This should include the filename which should be
            a `.csv` file.

        batch_size (int):
            Batch size to use during decoding. This will only affect the speed and the memory used.

        device (str):
            Device on which to load the model and data. Any device type supported by `Pytorch` (e.g. "cpu", "cuda", "cuda:0").
            Please note this code has only been tested on CPU and CUDA GPUs.
    """
    logger = logging.Logger(name="diffusion/predict", level=logging.INFO)
    # 1. Load model
    logger.info("Loading model.")
    model = MultinomialDiffusion.load(path=model_path)
    model = model.to(device=device)

    # 2. Initialize decoder
    logger.info("Initializing decoder.")
    decoder = DiffusionDecoder(model=model)

    # 3. Load data
    logger.info("Loading data.")
    logger.info("Loading residues.")
    residue_masses = yaml.safe_load(open(os.path.join(model_path, "residues.yaml")))
    residues = ResidueSet(residue_masses=residue_masses)
    logger.info(f"Loading input data from {input_path}.")
    input_data = polars.read_ipc(input_path)
    logger.info(f"Loading predictions from {start_predictions_path}.")
    start_predictions = pandas.read_csv(start_predictions_path)
    input_dataset = AnnotatedPolarsSpectrumDataset(
        data_frame=input_data, peptides=start_predictions["Predictions"].tolist()
    )
    data_loader = torch.utils.data.DataLoader(
        input_dataset,
        shuffle=False,
        batch_size=batch_size,
        collate_fn=collate_batches(
            residues=residues,
            time_steps=model.time_steps,
            annotated=True,
            max_length=model.config.max_length,
        ),
    )

    # 4. Elicit predictions
    logger.info("Performing decoding.")
    results = []
    with torch.no_grad():
        for spectra, spectra_padding_mask, precursors, peptides, _ in tqdm.tqdm(
            iter(data_loader), total=len(data_loader)
        ):
            predictions = decoder.decode(
                initial_sequence=peptides.to(device),
                spectra=spectra.to(device),
                spectra_padding_mask=spectra_padding_mask.to(device),
                precursors=precursors.to(device),
                start_step=DIFFUSION_START_STEP,
            )
            predictions = [
                prediction if "$" not in prediction else prediction[: prediction.index("$")]
                for prediction in predictions
            ]
            predictions = ["".join(prediction) for prediction in predictions]
            results.extend(predictions)

    # 5. Save predictions
    logger.info("Saving predictions.")
    output = input_data.to_pandas()
    output["Predictions"] = results
    output.to_csv(output_path)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
