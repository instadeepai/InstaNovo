from __future__ import annotations

from typing import Any, Dict

import numpy as np
import spectrum_utils.spectrum as sus
import torch
from jaxtyping import Float
from torch import Tensor

from instanovo.__init__ import console
from instanovo.common import DataProcessor
from instanovo.constants import ANNOTATED_COLUMN, PROTON_MASS_AMU, MSColumns
from instanovo.utils.colorlogging import ColorLog
from instanovo.utils.residues import ResidueSet

logger = ColorLog(console, __name__).logger


class TransformerDataProcessor(DataProcessor):
    """Transformer implementation of theDataProcessor class.

    Includes methods to process spectra and peptides for auto-regressive
    de novo peptide sequencing.
    """

    def __init__(
        self,
        residue_set: ResidueSet,
        n_peaks: int = 200,
        min_mz: float = 50.0,
        max_mz: float = 2500.0,
        min_intensity: float = 0.01,
        remove_precursor_tol: float = 2.0,
        reverse_peptide: bool = True,
        annotated: bool = True,
        return_str: bool = False,
        add_eos: bool = True,
        use_spectrum_utils: bool = True,
        metadata_columns: list[str] | None = None,
    ) -> None:
        """Initialize the data processor.

        Args:
            residue_set (ResidueSet): The residue set to use.
            n_peaks (int): The number of peaks to keep in the spectrum.
            min_mz (float): The minimum m/z to keep in the spectrum.
            max_mz (float): The maximum m/z to keep in the spectrum.
            min_intensity (float): The minimum intensity to keep in the spectrum.
            remove_precursor_tol (float): The tolerance to remove the precursor peak in Da.
            reverse_peptide (bool): Whether to reverse the peptide.
            annotated (bool): Whether the dataset is annotated.
            return_str (bool): Whether to return the peptide as a string.
            add_eos (bool): Whether to add the end of sequence token.
            use_spectrum_utils (bool): Whether to use the spectrum_utils library to process the spectra.
            metadata_columns (list[str] | None): The metadata columns to add to the dataset.
        """
        super().__init__(metadata_columns=metadata_columns)
        self.residue_set = residue_set
        self.n_peaks = n_peaks
        self.min_mz = min_mz
        self.max_mz = max_mz
        self.min_intensity = min_intensity
        self.remove_precursor_tol = remove_precursor_tol
        self.reverse_peptide = reverse_peptide
        self.annotated = annotated
        self.return_str = return_str
        self.add_eos = add_eos
        self.use_spectrum_utils = use_spectrum_utils

    def _process_spectrum(
        self,
        mz_array: Float[Tensor, " peak"],
        int_array: Float[Tensor, " peak"],
        precursor_mz: float,
        precursor_charge: int,
    ) -> torch.tensor:
        """Process a single spectrum.

        Args:
            mz_array (Float[Tensor, " peak"]): The m/z array of the spectrum.
            int_array (Float[Tensor, " peak"]): The intensity array of the spectrum.
            precursor_mz (float): The precursor m/z.
            precursor_charge (int): The precursor charge.

        Returns:
            torch.tensor: The processed spectrum.
        """
        if self.use_spectrum_utils:
            spectrum = sus.MsmsSpectrum(
                "",
                precursor_mz,
                precursor_charge,
                np.asarray(mz_array).astype(np.float32),
                np.asarray(int_array).astype(np.float32),
            )
            try:
                spectrum.set_mz_range(self.min_mz, self.max_mz)
                if len(spectrum.mz) == 0:
                    raise ValueError
                spectrum.remove_precursor_peak(self.remove_precursor_tol, "Da")
                if len(spectrum.mz) == 0:
                    raise ValueError
                spectrum.filter_intensity(self.min_intensity, self.n_peaks)
                if len(spectrum.mz) == 0:
                    raise ValueError
                spectrum.scale_intensity("root", 1)
                intensities = spectrum.intensity / np.linalg.norm(spectrum.intensity)
                return torch.tensor(np.asarray([spectrum.mz, intensities])).T.float()
            except ValueError:
                # Replace invalid spectra by a dummy spectrum.
                return torch.tensor([[0, 1]]).float()

        # Fallback implementation that matches spectrum_utils functionality
        try:
            # 1. Set m/z range
            mask = (mz_array >= self.min_mz) & (mz_array <= self.max_mz)
            mz_array = mz_array[mask]
            int_array = int_array[mask]

            if len(mz_array) == 0:
                raise ValueError

            # 2. Remove precursor peak
            precursor_mask = torch.abs(mz_array - precursor_mz) > self.remove_precursor_tol
            mz_array = mz_array[precursor_mask]
            int_array = int_array[precursor_mask]

            if len(mz_array) == 0:
                raise ValueError

            # 3. Filter by intensity and keep top n_peaks
            intensity_mask = int_array >= self.min_intensity
            mz_array = mz_array[intensity_mask]
            int_array = int_array[intensity_mask]

            if len(mz_array) == 0:
                raise ValueError

            # Get top n_peaks by intensity
            if len(mz_array) > self.n_peaks:
                _, indices = torch.topk(int_array, self.n_peaks)
                mz_array = mz_array[indices]
                int_array = int_array[indices]

            # 4. Scale intensity (root scaling)
            int_array = torch.sqrt(int_array)

            # 5. Normalize intensities
            int_array = int_array / torch.linalg.norm(int_array)

            return torch.stack([mz_array, int_array], dim=1).float()

        except ValueError:
            # Replace invalid spectra by a dummy spectrum
            return torch.tensor([[0.0, 1.0]], dtype=torch.float32)

    def process_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single row of data for auto-regressive de novo peptide sequencing.

        Args:
            row (dict[str, Any]): The row of data to process in dict format.

        Returns:
            dict[str, Any]: The processed row with resulting columns.
        """
        processed = {}

        # Spectra processing
        spectra = self._process_spectrum(
            torch.tensor(row[MSColumns.MZ_ARRAY.value]),
            torch.tensor(row[MSColumns.INTENSITY_ARRAY.value]),
            row[MSColumns.PRECURSOR_MZ.value],
            row[MSColumns.PRECURSOR_CHARGE.value],
        )

        processed["spectra"] = spectra

        # Peptide processing
        if self.annotated:
            if ANNOTATED_COLUMN not in row:
                raise KeyError(f"Annotated column {ANNOTATED_COLUMN} not found in dataset.")
            peptide = row[ANNOTATED_COLUMN]
            if not self.return_str:
                peptide_tokenized = self.residue_set.tokenize(peptide)

                if self.reverse_peptide:
                    peptide_tokenized = peptide_tokenized[::-1]

                peptide_encoding = self.residue_set.encode(peptide_tokenized, add_eos=self.add_eos, return_tensor="pt")

                processed["peptide"] = peptide_encoding
            else:
                processed["peptide"] = peptide

        return processed

    def _get_expected_columns(self) -> list[str]:
        """Get the expected columns.

        These are the columns that will be returned by the `process_row` method.

        Returns:
            list[str]: The expected columns.
        """
        expected_columns = ["spectra", "precursor_mz", "precursor_charge"]
        if self.annotated:
            expected_columns.append("peptide")
        return expected_columns

    def _collate_batch(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor | Any]:
        """Logic for collating a batch.

        Args:
            batch (list[dict[str, Any]]): The batch to collate.

        Returns:
            dict[str, Any]: The collated batch.
        """
        data_batch = [
            (
                row["spectra"],
                row["precursor_mz"],
                row["precursor_charge"],
            )
            for row in batch
        ]

        spectra, precursor_mzs, precursor_charges = zip(*data_batch, strict=True)

        # Pad spectra
        spectra, spectra_mask = DataProcessor._pad_and_mask(spectra)

        precursor_mzs = torch.tensor(precursor_mzs)
        precursor_charges = torch.tensor(precursor_charges)
        precursor_masses = (precursor_mzs - PROTON_MASS_AMU) * precursor_charges
        precursors = torch.vstack([precursor_masses, precursor_charges, precursor_mzs]).T.float()

        # Force all input data to be contiguous
        precursors = precursors.contiguous()
        spectra = spectra.contiguous()

        return_batch = {
            "spectra": spectra,
            "precursors": precursors,
            "spectra_mask": spectra_mask,
        }

        # Add peptide if annotated
        if self.annotated:
            peptides_batch = [row["peptide"] for row in batch]

            # Pad peptide
            if not isinstance(peptides_batch[0], str):
                peptides, peptides_mask = self._pad_and_mask(peptides_batch)
                peptides = peptides.contiguous()
            else:
                peptides = peptides_batch
                peptides_mask = None

            return_batch.update(
                {
                    "peptides": peptides,
                    "peptides_mask": peptides_mask,
                }
            )

        return return_batch
