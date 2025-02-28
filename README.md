# _De novo_ peptide sequencing with InstaNovo

[![PyPI version](https://badge.fury.io/py/instanovo.svg)](https://badge.fury.io/py/instanovo)

<!-- [![Tests Status](./reports/junit/tests-badge.svg?dummy=8484744)](./reports/junit/report.html) -->
<!-- [![Coverage Status](./reports/coverage/coverage-badge.svg?dummy=8484744)](./reports/coverage/index.html) -->
<a target="_blank" href="https://colab.research.google.com/github/instadeepai/InstaNovo/blob/main/notebooks/getting_started_with_instanovo.ipynb">
<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a>
<a target="_blank" href="https://kaggle.com/kernels/welcome?src=https://github.com/instadeepai/InstaNovo/blob/main/notebooks/getting_started_with_instanovo.ipynb">
<img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open In Kaggle"/> </a>

The official code repository for InstaNovo. This repo contains the code for training and inference
of InstaNovo and InstaNovo+. InstaNovo is a transformer neural network with the ability to translate
fragment ion peaks into the sequence of amino acids that make up the studied peptide(s). InstaNovo+,
inspired by human intuition, is a multinomial diffusion model that further improves performance by
iterative refinement of predicted sequences.

![Graphical Abstract](https://raw.githubusercontent.com/instadeepai/InstaNovo/main/docs/assets/graphical_abstract.jpeg)

**Links:**

- bioRxiv:
  [https://www.biorxiv.org/content/10.1101/2023.08.30.555055v3](https://www.biorxiv.org/content/10.1101/2023.08.30.555055v3)
- documentation:
  [https://instadeepai.github.io/InstaNovo/](https://instadeepai.github.io/InstaNovo/)

**Developed by:**

- [InstaDeep](https://www.instadeep.com/)
- [The Department of Biotechnology and Biomedicine](https://orbit.dtu.dk/en/organisations/department-of-biotechnology-and-biomedicine) -
  [Technical University of Denmark](https://www.dtu.dk/)

## Usage

### Installation

To use InstaNovo, we need to install the module via `pip`:

```bash
pip install instanovo
```

It is recommended to install InstaNovo in a fresh environment, such as Conda or PyEnv. For example,
if you have
[conda](https://docs.conda.io/en/latest/)/[miniconda](https://docs.conda.io/projects/miniconda/en/latest/)
installed:

```bash
conda env create -f environment.yml
conda activate instanovo
```

Note: InstaNovo is built for Python >= 3.10, <3.12 and tested on Linux.

### Training

To train auto-regressive InstaNovo using Hydra configs (see `--hydra-help` for more information):

```bash
usage: python -m instanovo.transformer.train [--config-name CONFIG_NAME]

Config options:
  config-name       Name of Hydra config in `/configs/`
                    Defaults to `instanovo_acpt`
```

Note: data is expected to be saved as Polars `.ipc` format. See section on data conversion.

To update the InstaNovo model config, modify the config file under
[configs/instanovo/base.yaml](https://github.com/instadeepai/InstaNovo/blob/main/configs/instanovo/base.yaml)

### Prediction

To get _de novo_ predictions from InstaNovo:

```bash
Usage: python -m instanovo.transformer.predict [--config-name CONFIG_NAME] data_path=path/to/data.mgf model_path=path/to/model.ckpt output_path=path/to/output.csv denovo=True

  Predict with the model.

Options:
  data_path         Path to dataset to be evaluated. Must be specified
                    in config or cli. Allows `.mgf`, `.mzxml`, a directory,
                    or an `.ipc` file. Glob notation is supported: eg.:
                    `./experiment/*.mgf`
  model_path        Path to model to be used. Must be specified
                    in config or cli. Model must be a `.ckpt` output by the
                    training script.
  output_path       Path to output csv file.
  config-name       Name of Hydra config in `/configs/inference/`
                    Defaults to `default`
```

To evaluate InstaNovo performance on an annotated dataset:

```bash
Usage: python -m instanovo.transformer.predict [--config-name CONFIG_NAME] data_path=path/to/data.mgf model_path=path/to/model.ckpt denovo=False

  Predict with the model.

Options:
  data_path         Path to dataset to be evaluated. Must be specified
                    in config or cli. Allows `.mgf`, `.mzxml`, a directory,
                    or an `.ipc` file. Glob notation is supported: eg.:
                    `./experiment/*.mgf`
  model_path        Path to model to be used. Must be specified
                    in config or cli. Model must be a `.ckpt` output by the
                    training script.
  config-name       Name of Hydra config in `/configs/inference/`
                    Defaults to `default`
```

The configuration file for inference may be found under
[/configs/inference/default.yaml](./configs/inference/default.yaml)

Note: the `denovo=True/False` flag controls whether metrics will be calculated.

**Output description**

When `output_path` is specified, a CSV file will be generated containing predictions for all the input spectra. The model will attempt to generate a peptide for every MS2 spectrum regardless of confidence. We recommend filtering the output using the **log_probs** and **delta_mass_ppm** columns.

| Column            | Description                                    | Data Type    | Notes                             |
|-----------------|------------------------------------------------|-------------|----------------------------------|
| scan_number     | Scan number of the MS/MS spectrum              | Integer     | Unique identifier from the input file |
| precursor_mz    | Precursor m/z (mass-to-charge ratio)           | Float       | The observed m/z of the precursor ion |
| precursor_charge | Precursor charge state                         | Integer     | Charge state of the precursor ion |
| experiment_name | Experiment name derived from input filename    | String      | Based on the input file name (mgf, mzml, or mzxml) |
| spectrum_id     | Unique spectrum identifier                     | String      | Combination of experiment name and scan number (e.g., `yeast:17738`) |
| targets         | Target peptide sequence                         | String      | Ground truth peptide sequence (if available) |
| preds           | Predicted peptide sequence                     | String      | Model-predicted peptide sequence |
| preds_tokenised | Predicted peptide sequence tokenized by amino acids | List[String] | Each amino acid token separated by commas |
| log_probs       | Log probability of the entire predicted sequence | Float       | Natural logarithm of the sequence confidence, can be converted to probability with np.exp(log_probs). |
| token_log_probs | Log probability of each token in the predicted sequence | List[Float] |  Natural logarithm of the sequence confidence per amino acid |
| delta_mass_ppm  | Mass difference between precursor and predicted peptide in ppm | Float       | Mass deviation in parts per million |


### Models

InstaNovo 1.0.2 includes a new model `instanovo.ckpt` trained on a larger dataset with more PTMs.

> Note: The InstaNovo Extended 1.0.0 training data mis-represented Cysteine as unmodified for the majority of the training data. Please update to the latest version of the model.

**Training Datasets**
- [ProteomeTools](https://www.proteometools.org/) Part [I (PXD004732)](https://www.ebi.ac.uk/pride/archive/projects/PXD004732), [II (PXD010595)](https://www.ebi.ac.uk/pride/archive/projects/PXD010595), and [III (PXD021013)](https://www.ebi.ac.uk/pride/archive/projects/PXD021013) \
(referred to as the  all-confidence ProteomeTools `AC-PT` dataset in our paper)
- Additional PRIDE dataset with more modifications: \
  ([PXD000666](https://www.ebi.ac.uk/pride/archive/projects/PXD000666), [PXD000867](https://www.ebi.ac.uk/pride/archive/projects/PXD000867), [PXD001839](https://www.ebi.ac.uk/pride/archive/projects/PXD001839), [PXD003155](https://www.ebi.ac.uk/pride/archive/projects/PXD003155), [PXD004364](https://www.ebi.ac.uk/pride/archive/projects/PXD004364), [PXD004612](https://www.ebi.ac.uk/pride/archive/projects/PXD004612), [PXD005230](https://www.ebi.ac.uk/pride/archive/projects/PXD005230), [PXD006692](https://www.ebi.ac.uk/pride/archive/projects/PXD006692), [PXD011360](https://www.ebi.ac.uk/pride/archive/projects/PXD011360), [PXD011536](https://www.ebi.ac.uk/pride/archive/projects/PXD011536), [PXD013543](https://www.ebi.ac.uk/pride/archive/projects/PXD013543), [PXD015928](https://www.ebi.ac.uk/pride/archive/projects/PXD015928), [PXD016793](https://www.ebi.ac.uk/pride/archive/projects/PXD016793), [PXD017671](https://www.ebi.ac.uk/pride/archive/projects/PXD017671), [PXD019431](https://www.ebi.ac.uk/pride/archive/projects/PXD019431), [PXD019852](https://www.ebi.ac.uk/pride/archive/projects/PXD019852), [PXD026910](https://www.ebi.ac.uk/pride/archive/projects/PXD026910), [PXD027772](https://www.ebi.ac.uk/pride/archive/projects/PXD027772))
- [Massive-KB v1](https://massive.ucsd.edu/ProteoSAFe/static/massive.jsp)
- Additional phosphorylation dataset \
(not yet publicly released)

**Natively Supported Modifications**
| Amino Acid                 | Single Letter | Modification      | Mass Delta (Da) | Unimod ID   |
|-----------------------------|---------------|------------------|-----------------|-------------|
| Methionine                 | M             | Oxidation       | +15.9949       | [[UNIMOD:35](https://www.unimod.org/modifications_view.php?editid1=35)] |
| Cysteine                   | C             | Carboxyamidomethylation | +57.0215       | [[UNIMOD:4](https://www.unimod.org/modifications_view.php?editid1=4)] |
| Asparagine, Glutamine      | N, Q          | Deamidation     | +0.9840        | [[UNIMOD:7](https://www.unimod.org/modifications_view.php?editid1=7)] |
| Serine, Threonine, Tyrosine | S, T, Y       | Phosphorylation | +79.9663       | [[UNIMOD:21](https://www.unimod.org/modifications_view.php?editid1=21)] |
| N-terminal                 | -             | Ammonia Loss    | -17.0265       | [[UNIMOD:385](https://www.unimod.org/modifications_view.php?editid1=385)] |
| N-terminal                 | -             | Carbamylation   | +43.0058       | [[UNIMOD:5](https://www.unimod.org/modifications_view.php?editid1=5)] |
| N-terminal                 | -             | Acetylation     | +42.0106       | [[UNIMOD:1](https://www.unimod.org/modifications_view.php?editid1=1)] |


See residue configuration under [instanovo/configs/residues/extended.yaml](./instanovo/configs/residues/extended.yaml)

## Additional features

### Spectrum Data Class

InstaNovo introduces a Spectrum Data Class: [SpectrumDataFrame](./instanovo/utils/data_handler.py).
This class acts as an interface between many common formats used for storing mass spectrometry,
including `.mgf`, `.mzml`, `.mzxml`, and `.csv`. This class also supports reading directly from
HuggingFace, Pandas, and Polars.

When using InstaNovo, these formats are natively supported and automatically converted to the
internal SpectrumDataFrame supported by InstaNovo for training and inference. Any data path may be
specified using [glob notation](<https://en.wikipedia.org/wiki/Glob_(programming)>). For example you
could use the following command to get _de novo_ predictions from all the files in the folder
`./experiment`:

```bash
python -m instanovo.transformer.predict data_path=./experiment/*.mgf
```

Alternatively, a list of files may be specified in the
[inference config](./configs/inference/default.yaml).

The SpectrumDataFrame also allows for loading of much larger datasets in a lazy way. To do this, the
data is loaded and stored as [`.parquet`](https://docs.pola.rs/user-guide/io/parquet/) files in a
temporary directory. Alternatively, the data may be saved permanently natively as `.parquet` for
optimal loading.

**Example usage:**

Converting mgf files to the native format:

```python
from instanovo.utils import SpectrumDataFrame

# Convert mgf files native parquet:
sdf = SpectrumDataFrame.load("/path/to/data.mgf", lazy=False, is_annotated=True)
sdf.save("path/to/parquet/folder", partition="train", chunk_size=1e6)
```

Loading the native format in shuffle mode:

```python
# Load a native parquet dataset:
sdf = SpectrumDataFrame.load("path/to/parquet/folder", partition="train", shuffle=True, lazy=True, is_annotated=True)
```

Using the loaded SpectrumDataFrame in a PyTorch DataLoader:

```python
from instanovo.transformer.dataset import SpectrumDataset
from torch.utils.data import DataLoader

ds = SpectrumDataset(sdf)
# Note: Shuffle and workers is handled by the SpectrumDataFrame
dl = DataLoader(
    ds,
    collate_fn=SpectrumDataset.collate_batch,
    shuffle=False,
    num_workers=0,
)
```

Some more examples using the SpectrumDataFrame:

```python
sdf = SpectrumDataFrame.load("/path/to/experiment/*.mzml", lazy=True)

# Remove rows with a charge value > 3:
sdf.filter_rows(lambda row: row["precursor_charge"]<=2)

# Sample a subset of the data:
sdf.sample_subset(fraction=0.5, seed=42)

# Convert to pandas
df = sdf.to_pandas() # Returns a pd.DataFrame

# Convert to polars LazyFrame
lazy_df = sdf.to_polars(return_lazy=True) # Returns a pl.LazyFrame

# Save as an `.mgf` file
sdf.write_mgf("path/to/output.mgf")
```

**SpectrumDataFrame Features:**

- The SpectrumDataFrame supports lazy loading with asynchronous prefetching, mitigating wait times
  between files.
- Filtering and sampling may be performed non-destructively through on file loading
- A two-fold shuffling strategy is introduced to optimise sampling during training (shuffling files
  and shuffling within files).

### Using your own datasets

To use your own datasets, you simply need to tabulate your data in either
[Pandas](https://pandas.pydata.org/) or [Polars](https://www.pola.rs/) with the following schema:

The dataset is tabular, where each row corresponds to a labelled MS2 spectra.

- `sequence (string)` \
  The target peptide sequence including post-translational modifications
- `modified_sequence (string) [legacy]` \
  The target peptide sequence including post-translational modifications
- `precursor_mz (float64)` \
  The mass-to-charge of the precursor (from MS1)
- `charge (int64)` \
  The charge of the precursor (from MS1)
- `mz_array (list[float64])` \
  The mass-to-charge values of the MS2 spectrum
- `intensity_array (list[float32])` \
  The intensity values of the MS2 spectrum

For example, the DataFrame for the
[nine species benchmark](https://huggingface.co/datasets/InstaDeepAI/ms_ninespecies_benchmark)
dataset (introduced in [Tran _et al._ 2017](https://www.pnas.org/doi/full/10.1073/pnas.1705691114))
looks as follows:

|     | sequence                   | precursor_mz | precursor_charge | mz_array                             | intensity_array                     |
| --: | :------------------------- | -----------: | ---------------: | :----------------------------------- | :---------------------------------- |
|   0 | GRVEGMEAR                  |      335.502 |                3 | [102.05527 104.052956 113.07079 ...] | [ 767.38837 2324.8787 598.8512 ...] |
|   1 | IGEYK                      |      305.165 |                2 | [107.07023 110.071236 111.11693 ...] | [ 1055.4957 2251.3171 35508.96 ...] |
|   2 | GVSREEIQR                  |      358.528 |                3 | [103.039444 109.59844 112.08704 ...] | [801.19995 460.65268 808.3431 ...]  |
|   3 | SSYHADEQVNEASK             |      522.234 |                3 | [101.07095 102.0552 110.07163 ...]   | [ 989.45154 2332.653 1170.6191 ...] |
|   4 | DTFNTSSTSN[UNIMOD:7]STSSSSSNSK |      676.282 |                3 | [119.82458 120.08073 120.2038 ...]   | [ 487.86942 4806.1377 516.8846 ...] |

For _de novo_ prediction, the `sequence` column is not required.

We also provide a conversion script for converting to native SpectrumDataFrame (sdf) format:

```bash
usage: python -m instanovo.utils.convert_to_sdf source target [-h] [--is_annotated IS_ANNOTATED] [--name NAME] [--partition {train,valid,test}] [--shard_size SHARD_SIZE] [--max_charge MAX_CHARGE]

positional arguments:
  source                source file(s)
  target                target folder to save data shards

options:
  -h, --help            show this help message and exit
  --is_annotated IS_ANNOTATED
                        whether dataset is annotated
  --name NAME           name of saved dataset
  --partition {train,valid,test}
                        partition of saved dataset
  --shard_size SHARD_SIZE
                        length of saved data shards
  --max_charge MAX_CHARGE
                        maximum charge to filter out
```

_Note: the target path should be a directory/folder._

**ToDo:**

- Multi-GPU support

## License

Code is licensed under the Apache License, Version 2.0 (see [LICENSE](LICENSE.md))

The model checkpoints are licensed under Creative Commons Non-Commercial
([CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/))

## BibTeX entry and citation info

```bibtex
@article{eloff_kalogeropoulos_2024_instanovo,
	title = {De novo peptide sequencing with InstaNovo: Accurate, database-free peptide identification for large scale proteomics experiments},
	author = {Kevin Eloff and Konstantinos Kalogeropoulos and Oliver Morell and Amandla Mabona and Jakob Berg Jespersen and Wesley Williams and Sam van Beljouw and Marcin Skwark and Andreas Hougaard Laustsen and Stan J. J. Brouns and Anne Ljungars and Erwin Marten Schoof and Jeroen Van Goey and Ulrich auf dem Keller and Karim Beguir and Nicolas Lopez Carranza and Timothy Patrick Jenkins},
	year = {2024},
	doi = {10.1101/2023.08.30.555055},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/10.1101/2023.08.30.555055v3},
	journal = {bioRxiv}
}
```

## Acknowledgements

Big thanks to Pathmanaban Ramasamy, Tine Claeys, and Lennart Martens of the [CompOmics](https://www.compomics.com/) research group for providing us with additional phosphorylation training data.
