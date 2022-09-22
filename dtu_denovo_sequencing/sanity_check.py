"""
Created on Sun Sep 18 17:37:34 2022

@author: kosta
"""
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pyopenms
import seaborn as sns

# load data

# root_dir = r"\\ait-pcifs02.win.dtu.dk\bio$\Shares\Protease-Systems-Biology\Kostas\OtherProjects\De_novo\" # noqa
root_dir = Path(__file__).parent.parent
path_out = root_dir / "data/out"
# experiment_name = "TUM_first_pool_88_01_01"
experiment_name = "Sample_data"

file_path = os.path.join(path_out, f"{experiment_name}.pkl")
out_df = pd.read_pickle(file_path, compression="zip")

# specify peptide index to draw from
ind = 0
seq = out_df["Sequence"][ind]
charge = out_df["Charge"][ind]
mz = out_df.iloc[ind, -2]
intensity = out_df.iloc[ind, -1]
colors = sns.color_palette("muted")

# plot 1: Raw spectrum for specified index
plt.figure(figsize=(12, 6))
top = 1.1 * max(intensity)
for m, i in zip(mz, intensity):
    ypos = i / top
    plt.axvline(m, 0, ypos, c=colors[0])

plt.ylim([0, 1.1])
plt.title(f"Raw spectrum, peptide {seq} with charge {charge}")
sns.despine()
plt.savefig(f"Raw_spectrum_ind{ind}_{experiment_name}.png", dpi=300, format="png")
plt.show()

# print 10 most intense ions
sorted_mass = [x for _, x in sorted(zip(intensity, mz), reverse=True)]
sorted_intensity = sorted(intensity, reverse=True)
for i in range(10):
    print(sorted_mass[i], sorted_intensity[i])

# fragment peptide in silico
tsg = pyopenms.TheoreticalSpectrumGenerator()
spec1 = pyopenms.MSSpectrum()
peptide = pyopenms.AASequence.fromString(seq)
# standard behavior is adding b- and y-ions of charge 1
p = pyopenms.Param()
p.setValue("add_b_ions", "true")
p.setValue("add_y_ions", "true")
p.setValue("add_losses", "true")
p.setValue("add_metainfo", "true")
tsg.setParameters(p)
if charge > 1:
    tsg.getSpectrum(spec1, peptide, 1, int(charge) - 1)  # charge range 1:precursor-1
else:
    tsg.getSpectrum(spec1, peptide, 1, 1)

# Iterate over annotated ions and their masses
print(f"Spectrum 1 of {peptide} with charge {charge} has {spec1.size()} peaks.")
for ion, peak in zip(spec1.getStringDataArrays()[0], spec1):
    print(ion.decode(), "is generated at m/z", peak.getMZ())

# plot 2: Fragment sequence in silico, annotate fragment ions
plt.figure(figsize=(12, 6))
top = 1.1 * max(intensity)
for m, i in zip(mz, intensity):

    flag_ion = False
    for ion, peak in zip(spec1.getStringDataArrays()[0], spec1):
        ion_name = ion.decode()
        mass = peak.getMZ()
        if -0.02 < mass - m < 0.02:
            flag_ion = True
            break

    ypos = i / top
    if flag_ion:
        plt.axvline(m, 0, ypos, c=colors[1])
        plt.text(
            m, 1.12 * ypos, ion_name, fontsize=9, rotation=90, ha="center", va="bottom"
        )
    else:
        plt.axvline(m, 0, ypos, c=colors[0])

plt.ylim([0, 1.1])
plt.title(f"Annotated spectrum, peptide {seq} with charge {charge}")
sns.despine()
plt.tight_layout()
plt.savefig(f"Annot_spectrum_ind{ind}_{experiment_name}.png", dpi=300, format="png")
plt.show()
