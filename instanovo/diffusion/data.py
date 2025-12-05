from __future__ import annotations

from typing import Any, Dict

import torch

from instanovo.__init__ import console
from instanovo.constants import ANNOTATED_COLUMN, REFINEMENT_COLUMN, MSColumns
from instanovo.transformer.data import TransformerDataProcessor
from instanovo.utils.colorlogging import ColorLog
from instanovo.utils.residues import ResidueSet

logger = ColorLog(console, __name__).logger


class DiffusionDataProcessor(TransformerDataProcessor):
    """Diffusion implementation of the DataProcessor class.

    Includes methods to process spectra and peptides for diffusion de novo peptide sequencing.
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
        peptide_pad_length: int = 40,
        peptide_pad_value: int = 0,
        truncate_max_length: int | None = 40,
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
            peptide_pad_length (int): The length to pad the peptide to.
            peptide_pad_value (int): The value to pad the peptide with.
            truncate_max_length (int | None): The maximum length to truncate the peptide to.
        """
        self.peptide_pad_length = peptide_pad_length
        self.peptide_pad_value = peptide_pad_value
        self.truncate_max_length = truncate_max_length
        super().__init__(
            residue_set=residue_set,
            n_peaks=n_peaks,
            min_mz=min_mz,
            max_mz=max_mz,
            min_intensity=min_intensity,
            remove_precursor_tol=remove_precursor_tol,
            reverse_peptide=reverse_peptide,
            annotated=annotated,
            return_str=return_str,
            add_eos=add_eos,
            use_spectrum_utils=use_spectrum_utils,
            metadata_columns=metadata_columns,
        )

    def process_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single row of data for diffusion de novo peptide sequencing.

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
            peptide = row[ANNOTATED_COLUMN]
            if not self.return_str:
                peptide_tokenized = self.residue_set.tokenize(peptide)

                if self.reverse_peptide:
                    peptide_tokenized = peptide_tokenized[::-1]

                peptide_encoding = self.residue_set.encode(peptide_tokenized, add_eos=self.add_eos, return_tensor="pt")

                if self.truncate_max_length:
                    peptide_encoding = peptide_encoding[: self.truncate_max_length]

                # Diffusion always padded to fixed length
                peptide_padded = torch.full(
                    (max(self.peptide_pad_length, peptide_encoding.shape[0]),),
                    fill_value=self.peptide_pad_value,
                    dtype=peptide_encoding.dtype,
                    device=peptide_encoding.device,
                )
                peptide_padded[: peptide_encoding.shape[0]] = peptide_encoding

                processed["peptide"] = peptide_padded
            else:
                processed["peptide"] = peptide

        if REFINEMENT_COLUMN in row:
            refine = row[REFINEMENT_COLUMN]

            refine_tokenized = self.residue_set.tokenize(refine)
            if self.reverse_peptide:
                refine_tokenized = refine_tokenized[::-1]

            refine_encoding = self.residue_set.encode(refine_tokenized, add_eos=self.add_eos, return_tensor="pt")

            if self.truncate_max_length:
                refine_encoding = refine_encoding[: self.truncate_max_length]

            # Diffusion always padded to fixed length
            refine_padded = torch.full(
                (max(self.peptide_pad_length, refine_encoding.shape[0]),),
                fill_value=self.peptide_pad_value,
                dtype=refine_encoding.dtype,
                device=refine_encoding.device,
            )
            refine_padded[: refine_encoding.shape[0]] = refine_encoding

            processed[REFINEMENT_COLUMN] = refine_padded

        return processed

    def _collate_batch(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        """Logic for collating a batch.

        Args:
            batch (list[dict[str, Any]]): The batch to collate.

        Returns:
            dict[str, Any] | tuple[Any]: The collated batch.
        """
        return_batch = super()._collate_batch(batch)

        if REFINEMENT_COLUMN not in batch[0]:
            return return_batch  # type: ignore

        refinement_peptide = [row[REFINEMENT_COLUMN] for row in batch]

        refinement_peptide, _ = self._pad_and_mask(refinement_peptide)

        return {
            **return_batch,
            REFINEMENT_COLUMN: refinement_peptide,
        }
