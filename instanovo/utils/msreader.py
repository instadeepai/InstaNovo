from pathlib import Path
from typing import Any

from pyteomics import mgf, mzml, mzxml
from pyteomics.auxiliary import cvquery


# Unused
def read_mgf(file_path: str) -> dict[str, list[Any]]:
    """Read an mgf file and return a data dict."""
    experiment_name = Path(file_path).stem
    data = _initialize_data_dict()

    with mgf.read(file_path, index_by_scans=True) as reader:
        for spectrum in reader:
            data["ms_level"].append(2)  # MGF files contain MS2 spectra only
            data["scan_number"].append(spectrum.get("params", {}).get("title", ""))
            data["experiment_name"].append(experiment_name)
            data["sequence"].append(spectrum.get("params", {}).get("seq", ""))
            data["precursor_mz"].append(spectrum.get("params", {}).get("pepmass", [0])[0])
            data["precursor_charge"].append(spectrum.get("params", {}).get("charge", [0])[0])
            data["retention_time"].append(spectrum.get("params", {}).get("rtinseconds", 0))
            data["mz_array"].append(spectrum.get("m/z array", []))
            data["intensity_array"].append(spectrum.get("intensity array", []))

    return data


def read_mzml(
    file_path: str,
    ms_levels: list[int] | None = None,
) -> dict[str, list[Any]]:
    """Read an mzml file and return a data dict.

    Args:
        file_path: Path to the mzML file.
        ms_levels: List of MS levels to extract. Default [2] for backward compatibility.
            Use [1, 2] to extract both MS1 and MS2 scans.
    """
    if ms_levels is None:
        ms_levels = [2]

    experiment_name = Path(file_path).stem
    data = _initialize_data_dict()

    ms_vocab = {
        "ms_level": "MS:1000511",
        "sequence": "MS:1000889",
        "precursor_mz": ["MS:1000040", "MS:1000827", "MS:1000744"],
        "precursor_charge": "MS:1000041",
        "retention_time": "MS:1000016",
        "mz_array": "MS:1000514",
        "intensity_array": "MS:1000515",
    }

    with mzml.read(file_path) as reader:
        for spectrum in reader:
            spectrum_dict = cvquery(spectrum)
            current_ms_level = spectrum_dict.get(ms_vocab["ms_level"])
            if current_ms_level not in ms_levels:
                continue

            data["ms_level"].append(int(current_ms_level))
            data["scan_number"].append(spectrum.get("id", ""))
            data["experiment_name"].append(experiment_name)
            data["retention_time"].append(spectrum_dict.get(ms_vocab["retention_time"]))
            data["mz_array"].append(list(spectrum_dict.get(ms_vocab["mz_array"])))
            data["intensity_array"].append(list(spectrum_dict.get(ms_vocab["intensity_array"])))

            if current_ms_level == 2:
                data["sequence"].append(spectrum_dict.get(ms_vocab["sequence"], ""))
                pre_mz_key = next(
                    (key for key in ms_vocab["precursor_mz"] if key in spectrum_dict),
                    "",
                )
                data["precursor_mz"].append(spectrum_dict.get(pre_mz_key, 0))
                data["precursor_charge"].append(spectrum_dict.get(ms_vocab["precursor_charge"], 0))
            else:
                # MS1 scans: precursor info not applicable
                data["sequence"].append("")
                data["precursor_mz"].append(None)
                data["precursor_charge"].append(None)

    return data


def read_mzxml(file_path: str, ms_levels: list[int] | None = None) -> dict[str, list[Any]]:
    """Read an mzxml file and return a data dict.

    Args:
        file_path: Path to the mzXML file.
        ms_levels: List of MS levels to extract. Default [2] for backward compatibility.
    """
    if ms_levels is None:
        ms_levels = [2]

    experiment_name = Path(file_path).stem
    data = _initialize_data_dict()

    with mzxml.read(file_path) as reader:
        for spectrum in reader:
            current_ms_level = spectrum.get("msLevel", 0)
            if current_ms_level not in ms_levels:
                continue

            data["ms_level"].append(int(current_ms_level))
            data["scan_number"].append(spectrum.get("num", ""))
            data["experiment_name"].append(experiment_name)
            data["retention_time"].append(spectrum.get("retentionTime"))
            data["mz_array"].append(list(spectrum.get("m/z array")))
            data["intensity_array"].append(list(spectrum.get("intensity array")))

            if current_ms_level == 2:
                data["sequence"].append(spectrum.get("peptide", ""))
                precursor = spectrum.get("precursorMz", [{}])[0]
                data["precursor_mz"].append(precursor.get("precursorMz", 0))
                data["precursor_charge"].append(precursor.get("precursorCharge", 0))
            else:
                data["sequence"].append("")
                data["precursor_mz"].append(None)
                data["precursor_charge"].append(None)

    return data


def _initialize_data_dict() -> dict[str, list[Any]]:
    return {
        "ms_level": [],
        "scan_number": [],
        "experiment_name": [],
        "sequence": [],
        "precursor_mz": [],
        "precursor_charge": [],
        "retention_time": [],
        "mz_array": [],
        "intensity_array": [],
    }
