from typing import Any
from pyteomics import mgf, mzml, mzxml


# Unused
def read_mgf(file_path: str) -> dict[str, list[Any]]:
    """Read an mgf file and return a data dict."""
    data = _initialize_data_dict()

    with mgf.read(file_path, index_by_scans=True) as reader:
        for spectrum in reader:
            data["scan_number"].append(spectrum.get("params", {}).get("title", ""))
            data["sequence"].append(spectrum.get("params", {}).get("seq", ""))
            data["precursor_mz"].append(
                spectrum.get("params", {}).get("pepmass", [None])[0]
            )
            data["precursor_charge"].append(
                spectrum.get("params", {}).get("charge", [None])[0]
            )
            data["retention_time"].append(
                spectrum.get("params", {}).get("rtinseconds", 0)
            )
            data["mz_array"].append(spectrum.get("m/z array", []))
            data["intensity_array"].append(spectrum.get("intensity array", []))

    return data


def read_mzml(file_path: str) -> dict[str, list[Any]]:
    """Read an mzml file and return a data dict."""
    data = _initialize_data_dict()

    with mzml.read(file_path) as reader:
        for spectrum in reader:
            if spectrum.get("ms level", 0) == 2:  # Ensure it's an MS2 spectrum
                data["scan_number"].append(spectrum.get("id", ""))
                data["sequence"].append(spectrum.get("peptide", ""))
                precursor = spectrum.get("precursorList", {}).get("precursor", [{}])[0]
                selected_ion = precursor.get("selectedIonList", {}).get(
                    "selectedIon", [{}]
                )[0]
                data["precursor_mz"].append(selected_ion.get("selected ion m/z"))
                data["precursor_charge"].append(selected_ion.get("charge state"))
                data["retention_time"].append(
                    spectrum.get("scanList", {})
                    .get("scan", [{}])[0]
                    .get("scan start time")
                )
                data["mz_array"].append(list(spectrum.get("m/z array")))
                data["intensity_array"].append(list(spectrum.get("intensity array")))

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
