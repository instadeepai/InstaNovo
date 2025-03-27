from typing import Any

from pyteomics import mgf, mzml, mzxml
from pyteomics.auxiliary import cvquery


# Unused
def read_mgf(file_path: str) -> dict[str, list[Any]]:
    """Read an mgf file and return a data dict."""
    data = _initialize_data_dict()

    with mgf.read(file_path, index_by_scans=True) as reader:
        for spectrum in reader:
            data["scan_number"].append(spectrum.get("params", {}).get("title", ""))
            data["sequence"].append(spectrum.get("params", {}).get("seq", ""))
            data["precursor_mz"].append(spectrum.get("params", {}).get("pepmass", [None])[0])
            data["precursor_charge"].append(spectrum.get("params", {}).get("charge", [None])[0])
            data["retention_time"].append(spectrum.get("params", {}).get("rtinseconds", 0))
            data["mz_array"].append(spectrum.get("m/z array", []))
            data["intensity_array"].append(spectrum.get("intensity array", []))

    return data


def read_mzml(
    file_path: str,
) -> dict[str, list[Any]]:
    """Read an mzml file and return a data dict."""
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
            if spectrum_dict.get(ms_vocab["ms_level"]) == 2:  # Ensure it's an MS2 spectrum
                data["scan_number"].append(spectrum.get("id", ""))

                data["sequence"].append(spectrum_dict.get(ms_vocab["sequence"], ""))

                # Find the first matching precursor mz term
                pre_mz_key = next(
                    (key for key in ms_vocab["precursor_mz"] if key in spectrum_dict),
                    "",
                )
                data["precursor_mz"].append(spectrum_dict.get(pre_mz_key, ""))

                data["precursor_charge"].append(spectrum_dict.get(ms_vocab["precursor_charge"], ""))
                data["retention_time"].append(spectrum_dict.get(ms_vocab["retention_time"]))
                data["mz_array"].append(list(spectrum_dict.get(ms_vocab["mz_array"])))
                data["intensity_array"].append(list(spectrum_dict.get(ms_vocab["intensity_array"])))

    return data


def read_mzxml(file_path: str) -> dict[str, list[Any]]:
    """Read an mzxml file and return a data dict."""
    data = _initialize_data_dict()

    with mzxml.read(file_path) as reader:
        for spectrum in reader:
            if spectrum.get("msLevel", 0) == 2:  # Ensure it's an MS2 spectrum
                data["scan_number"].append(spectrum.get("num", ""))
                data["sequence"].append(spectrum.get("peptide", ""))
                precursor = spectrum.get("precursorMz", [{}])[0]
                data["precursor_mz"].append(precursor.get("precursorMz"))
                data["precursor_charge"].append(precursor.get("precursorCharge"))
                data["retention_time"].append(spectrum.get("retentionTime"))
                data["mz_array"].append(list(spectrum.get("m/z array")))
                data["intensity_array"].append(list(spectrum.get("intensity array")))

    return data


def _initialize_data_dict() -> dict[str, list[Any]]:
    return {
        "scan_number": [],
        "sequence": [],
        "precursor_mz": [],
        "precursor_charge": [],
        "retention_time": [],
        "mz_array": [],
        "intensity_array": [],
    }
