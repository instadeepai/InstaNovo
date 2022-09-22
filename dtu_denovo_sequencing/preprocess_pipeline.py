"""
Created on Fri Sep 16 10:05:46 2022

@author: konka
"""
import os
import zipfile
from pathlib import Path

import pandas as pd
import pyopenms
from tqdm import tqdm

# init variables
# root_dir = r"\\ait-pcifs02.win.dtu.dk\bio$\Shares\Protease-Systems-Biology\Kostas\OtherProjects\De_novo" # noqa: E501
root_dir = Path(__file__).parent.parent
path_raw = root_dir / "ProteomeTools_data/Converted"
path_zip = root_dir / "ProteomeTools_data/Search_results_all"
path_temp = root_dir / "data/temp"
path_out = root_dir / "data/out"
path_log = root_dir / "ProteomeTools_data/File_lists"
failed_log = "Failed.txt"
completed_log = "Completed.txt"
result_file = "evidence.txt"

completed_path = path_log / completed_log
if completed_path.is_file():
    completed_df = pd.read_csv(
        os.path.join(path_log, completed_log), sep="\t", header=None
    )
    if len(completed_df) > 1:
        completed_raw = completed_df.iloc[:, 1].to_list()
    else:
        completed_raw = []
else:
    completed_raw = []


for filename_mzML in tqdm(os.listdir(path_raw)):  # noqa: N816

    if filename_mzML in completed_raw:
        print(f"{filename_mzML} already processed, skipping.")
        continue

    print(f"Starting pipeline for {filename_mzML}")

    experiment_name = filename_mzML.split("-")[1]
    flag_zip = False
    for result_name in os.listdir(path_zip):
        if experiment_name in result_name:
            filename_zip = result_name
            flag_zip = True
            break

    if flag_zip:
        print(f"Located files with pattern {experiment_name}, unpacking search file...")
    else:
        print(f"Could not locate raw or search files with pattern {experiment_name}")

        with open(os.path.join(path_log, failed_log), "a") as fh:

            print(
                experiment_name,
                filename_mzML,
                "File not present",
                sep="\t",
                file=fh,
            )

        continue

    # extract evidence.txt from zip archive
    try:
        source = os.path.join(path_zip, filename_zip)
        archive = zipfile.ZipFile(source)
        for filename in archive.namelist():
            if filename.endswith(result_file):
                print("Found archive result.")
                path_result = archive.extract(filename, path_temp)
                break

        print("Unpacked search file in temp folder. Loading raw file...")

    except:  # noqa: B001, E722 # TODO: use more specific Exception

        print(f"Could not unpack search file for {experiment_name}")

        with open(os.path.join(path_log, failed_log), "a") as fh:

            print(
                experiment_name,
                filename_mzML,
                filename_zip,
                "Archive error",
                sep="\t",
                file=fh,
            )

        continue

    # load mzML file to extract spectra info
    try:
        exp = pyopenms.MSExperiment()
        pyopenms.MzMLFile().load(os.path.join(path_raw, filename_mzML), exp)

        print("Raw file loaded on disc.")

    except:  # noqa: B001, E722 # TODO: use more specific Exception

        print(f"Could not read raw file for {experiment_name}")

        # clean up after error
        os.remove(path_result)

        with open(os.path.join(path_log, failed_log), "a") as fh:
            print(
                experiment_name,
                filename_mzML,
                filename_zip,
                "Raw read error",
                sep="\t",
                file=fh,
            )

        continue

    # read evidence dataframe containing all PSMs
    peptide_df = pd.read_csv(path_result, sep="\t")

    print("Read evidence search table. Extracting data...")

    # init data dict
    header = [
        "Evidence index",
        "MS/MS Scan Number",
        "Sequence",
        "Modified sequence",
        "MS/MS m/z",
        "m/z",
        "Mass",
        "Charge",
        "Retention time",
        "Mass values",
        "Intensity",
    ]
    data = {}

    # extract relevant features
    for index in peptide_df.index:

        # for each PSM
        sequence = peptide_df.loc[index, "Sequence"]
        modified_sequence = peptide_df.loc[index, "Modified sequence"]
        empirical_mz = peptide_df.loc[index, "MS/MS m/z"]
        theoretical_mz = peptide_df.loc[index, "m/z"]
        precursor_mass = peptide_df.loc[index, "Mass"]
        precursor_charge = peptide_df.loc[index, "Charge"]
        retention_time = peptide_df.loc[index, "Retention time"]
        spectrum_index = peptide_df.loc[index, "MS/MS Scan Number"]

        # for each PSM
        try:
            spectrum = exp.getSpectrum(int(spectrum_index))
            mz, intensity = spectrum.get_peaks()

            # write in data structure
            data[index] = [
                index,
                spectrum_index,
                sequence,
                modified_sequence,
                empirical_mz,
                theoretical_mz,
                precursor_mass,
                precursor_charge,
                retention_time,
                mz,
                intensity,
            ]

        except AssertionError as err:

            print(err)
            print(index, spectrum_index, sequence, experiment_name)

            continue

    print("Extracted data, writing to file...")

    # write output for the experiment
    out_df = pd.DataFrame.from_dict(data=data, orient="index", columns=header)
    # out_df.to_excel(f'{experiment_name}.xlsx', index=False)
    out_file = os.path.join(path_out, f"{experiment_name}.pkl")
    out_df.to_pickle(out_file, compression="zip")

    print(f"Wrote to {out_file}. Cleaning up temp folder...")

    # clean up temp dir
    os.remove(path_result)
    for filename in os.listdir(path_temp):
        if filename.endswith("txt"):
            file_path = os.path.join(path_temp, filename)
            if os.path.isdir(file_path):
                os.rmdir(file_path)
            else:
                os.remove(file_path)

    with open(os.path.join(path_log, completed_log), "a") as fh:
        print(experiment_name, filename_mzML, filename_zip, sep="\t", file=fh)

    print(f"\nCleaned up, reported success for {experiment_name}.\n")
