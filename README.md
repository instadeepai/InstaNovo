<p align="center" width="100%">
     <img width="33%" src="https://raw.githubusercontent.com/instadeepai/InstaNovo/main/docs/assets/instanovo.svg">
</p>

# _De novo_ peptide sequencing with InstaNovo

[![PyPI version](https://badge.fury.io/py/instanovo.svg)](https://badge.fury.io/py/instanovo)
[![DOI](https://zenodo.org/badge/681625644.svg)](https://doi.org/10.5281/zenodo.14712453)
![code coverage](https://raw.githubusercontent.com/instadeepai/InstaNovo/main/docs/assets/coverage.svg?raw=true)

<!-- [![Tests Status](./reports/junit/tests-badge.svg?dummy=8484744)](./reports/junit/report.html) -->
<!-- [![Coverage Status](./reports/coverage/coverage-badge.svg?dummy=8484744)](./reports/coverage/index.html) -->
<a target="_blank" href="https://colab.research.google.com/github/instadeepai/InstaNovo/blob/main/notebooks/getting_started_with_instanovo.ipynb">
<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a>
<!-- <a target="_blank" href="https://kaggle.com/kernels/welcome?src=https://github.com/instadeepai/InstaNovo/blob/main/notebooks/getting_started_with_instanovo.ipynb">
<img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open In Kaggle"/> </a> -->

The official code repository for InstaNovo. This repo contains the code for training and inference
of InstaNovo and InstaNovo+. InstaNovo is a transformer neural network with the ability to translate
fragment ion peaks into the sequence of amino acids that make up the studied peptide(s). InstaNovo+,
inspired by human intuition, is a multinomial diffusion model that further improves performance by
iterative refinement of predicted sequences.

![Graphical Abstract](https://raw.githubusercontent.com/instadeepai/InstaNovo/main/docs/assets/graphical_abstract.jpeg)

**Links:**

- Publication in Nature Machine Intelligence:
  [InstaNovo enables diffusion-powered de novo peptide sequencing in large-scale proteomics experiments](https://www.nature.com/articles/s42256-025-01019-5)
- InstaNovo blog: [https://instanovo.ai/](https://instanovo.ai/)
- Documentation:
  [https://instadeepai.github.io/InstaNovo/](https://instadeepai.github.io/InstaNovo/)

**Developed by:**

- [InstaDeep](https://www.instadeep.com/)
- [The Department of Biotechnology and Biomedicine](https://orbit.dtu.dk/en/organisations/department-of-biotechnology-and-biomedicine) -
  [Technical University of Denmark](https://www.dtu.dk/)

## Usage

### HuggingFace Space

InstaNovo is available as a HuggingFace Space at
[hf.co/spaces/InstaDeepAI/InstaNovo](https://huggingface.co/spaces/InstaDeepAI/InstaNovo) for quick
testing and evaluation. You can upload your own spectra files in `.mgf`, `.mzml`, or `.mzxml` format
and run _de novo_ predictions. The results will be displayed in a table format, and you can download
the predictions as a CSV file. The HuggingFace Space is powered by the InstaNovo model and the
InstaNovo+ model for iterative refinement.

[![HuggingFace Space](https://raw.githubusercontent.com/instadeepai/InstaNovo/main/docs/assets/huggingface_space.png)](https://huggingface.co/spaces/InstaDeepAI/InstaNovo)

### Installation

To use InstaNovo Python package with command line interface, we need to install the module via
`pip`:

```bash
pip install instanovo
```

If you have access to an NVIDIA GPU, you can install InstaNovo with the GPU version of PyTorch
(recommended):

```bash
pip install "instanovo[cu124]"
```

If you are on macOS, you can install the CPU-only version of PyTorch:

```bash
pip install "instanovo[cpu]"
```

### Command line usage

InstaNovo provides a comprehensive command line interface (CLI) for both prediction and training
tasks.

To get help and see the available commands:

```
instanovo --help
```

![`instanovo --help`](https://raw.githubusercontent.com/instadeepai/InstaNovo/main/docs/assets/instanovo_help.svg)

To see the version of InstaNovo, InstaNovo+ and some of the dependencies:

```
instanovo version
```

![`instanovo version`](https://raw.githubusercontent.com/instadeepai/InstaNovo/main/docs/assets/instanovo_version.svg)

### Predicting

To get help about the prediction command line options:

```
instanovo predict --help
```

![`instanovo predict --help`](https://raw.githubusercontent.com/instadeepai/InstaNovo/main/docs/assets/instanovo_predict_help.svg)

### Running predictions with both InstaNovo and InstaNovo+

The default is to run predictions first with the transformer-based InstaNovo model, and then further
improve the performance by iterative refinement of these predicted sequences by the diffusion-based
InstaNov+ model.

```
instanovo predict --data-path ./sample_data/spectra.mgf --output-path predictions.csv
```

Which results in the following output:

```
scan_number,precursor_mz,precursor_charge,experiment_name,spectrum_id,diffusion_predictions_tokenised,diffusion_predictions,diffusion_log_probabilities,transformer_predictions,transformer_predictions_tokenised,transformer_log_probabilities,transformer_token_log_probabilities
0,451.25348,2,spectra,spectra:0,"['A', 'L', 'P', 'Y', 'T', 'P', 'K', 'K']",ALPYTPKK,-0.03160184621810913,LAHYNKK,"L, A, H, Y, N, K, K",-424.5889587402344,"[-0.5959059000015259, -0.0059959776699543, -0.01749008148908615, -0.03598890081048012, -0.48958998918533325, -1.5242897272109985, -0.656516432762146]"
```

To evaluate InstaNovo performance on an annotated dataset:

```bash
instanovo predict --evaluation --data-path ./sample_data/spectra.mgf --output-path predictions.csv
```

Which results in the following output:

```
scan_number,precursor_mz,precursor_charge,experiment_name,spectrum_id,diffusion_predictions_tokenised,diffusion_predictions,diffusion_log_probabilities,targets,transformer_predictions,transformer_predictions_tokenised,transformer_log_probabilities,transformer_token_log_probabilities
0,451.25348,2,spectra,spectra:0,"['L', 'A', 'H', 'Y', 'N', 'K', 'K']",LAHYNKK,-0.06637095659971237,IAHYNKR,LAHYNKK,"L, A, H, Y, N, K, K",-424.5889587402344,"[-0.5959059000015259, -0.0059959776699543, -0.01749008148908615, -0.03598890081048012, -0.48958998918533325, -1.5242897272109985, -0.656516432762146]"
```

Note that the `--evaluation` flag includes the `targets` column in the output, which contains the
ground truth peptide sequence. Metrics will be calculated and displayed in the console.

### Command line arguments and overriding config values

The configuration file for inference may be found under
[instanovo/configs/inference/](instanovo/configs/inference/) folder. By default, the
[`default.yaml`](instanovo/configs/inference/default.yaml) file is used.

InstaNovo uses command line arguments for commonly used parameters:

- `--data-path` - Path to the dataset to be evaluated. Allows `.mgf`, `.mzml`, `.mzxml`, `.ipc` or a
  directory. Glob notation is supported: eg.: `./experiment/*.mgf`
- `--output-path` - Path to output csv file.
- `--instanovo-model` - Model to use for InstaNovo. Either a model ID (currently supported:
  `instanovo-v1.1.0`) or a path to an Instanovo checkpoint file (.ckpt format).
- `--instanovo-plus-model` - Model to use for InstaNovo+. Either a model ID (currently supported:
  `instanovoplus-v1.1.0`) or a path to an Instanovo+ checkpoint file (.ckpt format).
- `--denovo` - Whether to do _de novo_ predictions. If you want to evaluate the model on annotated
  data, use the flag `--evaluation` flag.
- `--with-refinement` - Whether to use InstaNovo+ for iterative refinement of InstaNovo predictions.
  Default is `True`. If you don't want to use refinement,use the flag `--no-refinement`.

To override the configuration values in the config files, you can use command line arguments. For
example, by default beam search with one beam is used. If you want to use beam search with 5 beams,
you can use the following command:

```bash
instanovo predict --data-path ./sample_data/spectra.mgf --output-path predictions.csv num_beams=5
```

Note the lack of prefix `--` before `num_beams` in the command line argument because you are
overriding the value of key defined in the config file.

**Output description**

When `output_path` is specified, a CSV file will be generated containing predictions for all the
input spectra. The model will attempt to generate a peptide for every MS2 spectrum regardless of
confidence. We recommend filtering the output using the **log_probabilities** and **delta_mass_ppm**
columns.

| Column                  | Description                                                    | Data Type    | Notes                                                                                                         |
| ----------------------- | -------------------------------------------------------------- | ------------ | ------------------------------------------------------------------------------------------------------------- |
| scan_number             | Scan number of the MS/MS spectrum                              | Integer      | Unique identifier from the input file                                                                         |
| precursor_mz            | Precursor m/z (mass-to-charge ratio)                           | Float        | The observed m/z of the precursor ion                                                                         |
| precursor_charge        | Precursor charge state                                         | Integer      | Charge state of the precursor ion                                                                             |
| experiment_name         | Experiment name derived from input filename                    | String       | Based on the input file name (mgf, mzml, or mzxml)                                                            |
| spectrum_id             | Unique spectrum identifier                                     | String       | Combination of experiment name and scan number (e.g., `yeast:17738`)                                          |
| targets                 | Target peptide sequence                                        | String       | Ground truth peptide sequence (if available)                                                                  |
| predictions             | Predicted peptide sequences                                    | String       | Model-predicted peptide sequence                                                                              |
| predictions_tokenised   | Predicted peptide sequence tokenized by amino acids            | List[String] | Each amino acid token separated by commas                                                                     |
| log_probabilities       | Log probability of the entire predicted sequence               | Float        | Natural logarithm of the sequence confidence, can be converted to probability with np.exp(log_probabilities). |
| token_log_probabilities | Log probability of each token in the predicted sequence        | List[Float]  | Natural logarithm of the sequence confidence per amino acid                                                   |
| delta_mass_ppm          | Mass difference between precursor and predicted peptide in ppm | Float        | Mass deviation in parts per million                                                                           |

### Models

InstaNovo 1.1.0 includes new models `instanovo-v1.1.0.ckpt`, and `instanovoplus-v1.1.0.ckpt` trained
on a larger dataset with more PTMs.

> Note: The InstaNovo Extended 1.0.0 training data mis-represented Cysteine as unmodified for the
> majority of the training data. Please update to the latest version of the model.

**Training Datasets**

- [ProteomeTools](https://www.proteometools.org/) Part
  [I (PXD004732)](https://www.ebi.ac.uk/pride/archive/projects/PXD004732),
  [II (PXD010595)](https://www.ebi.ac.uk/pride/archive/projects/PXD010595), and
  [III (PXD021013)](https://www.ebi.ac.uk/pride/archive/projects/PXD021013) \
  (referred to as the all-confidence ProteomeTools `AC-PT` dataset in our paper)
- Additional PRIDE dataset with more modifications: \
  ([PXD000666](https://www.ebi.ac.uk/pride/archive/projects/PXD000666), [PXD000867](https://www.ebi.ac.uk/pride/archive/projects/PXD000867),
  [PXD001839](https://www.ebi.ac.uk/pride/archive/projects/PXD001839), [PXD003155](https://www.ebi.ac.uk/pride/archive/projects/PXD003155),
  [PXD004364](https://www.ebi.ac.uk/pride/archive/projects/PXD004364), [PXD004612](https://www.ebi.ac.uk/pride/archive/projects/PXD004612),
  [PXD005230](https://www.ebi.ac.uk/pride/archive/projects/PXD005230), [PXD006692](https://www.ebi.ac.uk/pride/archive/projects/PXD006692),
  [PXD011360](https://www.ebi.ac.uk/pride/archive/projects/PXD011360), [PXD011536](https://www.ebi.ac.uk/pride/archive/projects/PXD011536),
  [PXD013543](https://www.ebi.ac.uk/pride/archive/projects/PXD013543), [PXD015928](https://www.ebi.ac.uk/pride/archive/projects/PXD015928),
  [PXD016793](https://www.ebi.ac.uk/pride/archive/projects/PXD016793), [PXD017671](https://www.ebi.ac.uk/pride/archive/projects/PXD017671),
  [PXD019431](https://www.ebi.ac.uk/pride/archive/projects/PXD019431), [PXD019852](https://www.ebi.ac.uk/pride/archive/projects/PXD019852),
  [PXD026910](https://www.ebi.ac.uk/pride/archive/projects/PXD026910), [PXD027772](https://www.ebi.ac.uk/pride/archive/projects/PXD027772))
- [Massive-KB v1](https://massive.ucsd.edu/ProteoSAFe/static/massive.jsp)
- Additional phosphorylation dataset \
  (not yet publicly released)

**Natively Supported Modifications**

| Amino Acid                  | Single Letter | Modification            | Mass Delta (Da) | Unimod ID                                                                   |
| --------------------------- | ------------- | ----------------------- | --------------- | --------------------------------------------------------------------------- |
| Methionine                  | M             | Oxidation               | +15.9949        | [\[UNIMOD:35\]](https://www.unimod.org/modifications_view.php?editid1=35)   |
| Cysteine                    | C             | Carboxyamidomethylation | +57.0215        | [\[UNIMOD:4\]](https://www.unimod.org/modifications_view.php?editid1=4)     |
| Asparagine, Glutamine       | N, Q          | Deamidation             | +0.9840         | [\[UNIMOD:7\]](https://www.unimod.org/modifications_view.php?editid1=7)     |
| Serine, Threonine, Tyrosine | S, T, Y       | Phosphorylation         | +79.9663        | [\[UNIMOD:21\]](https://www.unimod.org/modifications_view.php?editid1=21)   |
| N-terminal                  | -             | Ammonia Loss            | -17.0265        | [\[UNIMOD:385\]](https://www.unimod.org/modifications_view.php?editid1=385) |
| N-terminal                  | -             | Carbamylation           | +43.0058        | [\[UNIMOD:5\]](https://www.unimod.org/modifications_view.php?editid1=5)     |
| N-terminal                  | -             | Acetylation             | +42.0106        | [\[UNIMOD:1\]](https://www.unimod.org/modifications_view.php?editid1=1)     |

See residue configuration under
[instanovo/configs/residues/extended.yaml](./instanovo/configs/residues/extended.yaml)

### Training

Data to train on may be provided in any format supported by the SpectrumDataHandler. See section on
data conversion for preferred formatting.

#### Training InstaNovo

To train the auto-regressive transformer model InstaNovo using the config file
[instanovo/configs/instanovo.yaml](./instanovo/configs/instanovo.yaml), you can use the following
command:

```bash
instanovo transformer train --help
```

![`instanovo transformer train --help`](https://raw.githubusercontent.com/instadeepai/InstaNovo/main/docs/assets/instanovo_transformer_train_help.svg)

To update the InstaNovo model config, modify the config file under
[instanovo/configs/model/instanovo_base.yaml](instanovo/configs/model/instanovo_base.yaml)

#### Training InstaNovo+

To train the diffusion model InstaNovo+ using the config file
[instanovo/configs/instanovoplus.yaml](instanovo/configs/instanovoplus.yaml), you can use the
following command:

```bash
instanovo diffusion train --help
```

![`instanovo diffusion train --help`](https://raw.githubusercontent.com/instadeepai/InstaNovo/main/docs/assets/instanovo_diffusion_train_help.svg)

To update the InstaNovo+ model config, modify the config file under
[instanovo/configs/model/instanovoplus_base.yaml](instanovo/configs/model/instanovoplus_base.yaml)

### Advanced prediction options

### Run predictions with only InstaNovo

If you want to run predictions with only InstaNovo, you can use the following command:

```bash
instanovo transformer predict --help
```

![`instanovo transformer predict --help`](https://raw.githubusercontent.com/instadeepai/InstaNovo/main/docs/assets/instanovo_transformer_predict_help.svg)

### Run predictions with only InstaNovo+

If you want to run predictions with only InstaNovo+, you can use the following command:

```bash
instanovo diffusion predict --help
```

![`instanovo diffusion predict --help`](https://raw.githubusercontent.com/instadeepai/InstaNovo/main/docs/assets/instanovo_diffusion_predict_help.svg)

### Run predictions with InstaNovo and InstaNovo+ in separate steps

You can first run predictions with InstaNovo

```bash
instanovo transformer predict --data-path ./sample_data/spectra.mgf --output-path instanovo_predictions.csv
```

and then use the predictions as input for InstaNovo+:

```bash
instanovo diffusion predict --data-path ./sample_data/spectra.mgf --output-path instanovo_plus_predictions.csv instanovo_predictions_path=instanovo_predictions.csv
```

## Performance

We have benchmarked our latest models InstaNovo v1.1 and InstaNovo+ v1.1 against our previous
models. For all results below, InstaNovo decoding was performed with knapsack beam search decoding.
InstaNovo+ then refined these results. We present peptide accuracy as the metric of comparison.
Peptide accuracy is a measure of precision at full coverage (no filtering).

### Nine-species dataset

| Dataset  | InstaNovo v0.1 | InstaNovo+ v0.1 | InstaNovo v1.1 | InstaNovo+ v1.1 |
| -------- | -------------- | --------------- | -------------- | --------------- |
| Bacillus | 0.624          | 0.674           | 0.652          | **0.684**       |
| Mouse    | 0.466          | 0.490           | 0.524          | **0.542**       |
| Yeast    | 0.559          | 0.624           | 0.618          | **0.645**       |

InstaNovo and InstaNovo+ v0.1 were fine-tuned on the eight species dataset, excluding the test
species, whereas InstaNovo and InstaNovo+ v1.1 were evaluated zero-shot on these datasets.

### Biological validation datasets

| Dataset                         | InstaNovo v0.1 | InstaNovo+ v0.1 | InstaNovo v1.1 | InstaNovo+ v1.1 |
| ------------------------------- | -------------- | --------------- | -------------- | --------------- |
| HeLa degradome                  | 0.695          | 0.719           | 0.813          | **0.821**       |
| HeLa single-shot                | 0.503          | 0.517           | 0.642          | **0.647**       |
| Herceptin                       | 0.494          | 0.562           | 0.710          | **0.720**       |
| Immunopeptidomics               | 0.581          | 0.697           | 0.707          | **0.748**       |
| _Candidatus_ "Scalindua brodae" | 0.724          | 0.736           | 0.748          | **0.762**       |
| Snake venoms                    | 0.196          | 0.198           | 0.221          | **0.238**       |
| Nanobodies                      | 0.447          | 0.464           | 0.492          | **0.508**       |
| Wound fluids                    | 0.225          | 0.229           | 0.354          | **0.364**       |

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
instanovo predict --data_path=./experiment/*.mgf
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

|     | sequence                       | precursor_mz | precursor_charge | mz_array                             | intensity_array                     |
| --: | :----------------------------- | -----------: | ---------------: | :----------------------------------- | :---------------------------------- |
|   0 | GRVEGMEAR                      |      335.502 |                3 | [102.05527 104.052956 113.07079 ...] | [ 767.38837 2324.8787 598.8512 ...] |
|   1 | IGEYK                          |      305.165 |                2 | [107.07023 110.071236 111.11693 ...] | [ 1055.4957 2251.3171 35508.96 ...] |
|   2 | GVSREEIQR                      |      358.528 |                3 | [103.039444 109.59844 112.08704 ...] | [801.19995 460.65268 808.3431 ...]  |
|   3 | SSYHADEQVNEASK                 |      522.234 |                3 | [101.07095 102.0552 110.07163 ...]   | [ 989.45154 2332.653 1170.6191 ...] |
|   4 | DTFNTSSTSN[UNIMOD:7]STSSSSSNSK |      676.282 |                3 | [119.82458 120.08073 120.2038 ...]   | [ 487.86942 4806.1377 516.8846 ...] |

For _de novo_ prediction, the `sequence` column is not required.

We also provide a conversion script for converting to native SpectrumDataFrame (sdf) format:

```bash
instanovo convert --help
```

![`instanovo convert --help`](https://raw.githubusercontent.com/instadeepai/InstaNovo/main/docs/assets/instanovo_convert_help.svg)

## Development

### `uv` setup

This project is set up to use [uv](https://docs.astral.sh/uv/) to manage Python and dependencies.
First, be sure you [have uv installed](https://docs.astral.sh/uv/getting-started/installation/) on
your system.

On Linux and macOS:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

On Windows:

```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Note: InstaNovo is built for Python >=3.10, <3.13 and tested on Linux.

### Fork and clone the repository

Then [fork](https://github.com/instadeepai/InstaNovo/fork) this repo (having your own fork will make
it easier to contribute) and
[clone it](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository).

```bash
git clone https://github.com/YOUR-USERNAME/InstaNovo.git
cd InstaNovo
```

And install the dependencies. If you do have access to an NVIDIA GPU, you can install the GPU
version of PyTorch (recommended):

```bash
uv sync --extra cu124
uv run pre-commit install
```

If you don't have access to a GPU, you can install the CPU-only version of PyTorch:

```bash
uv sync --extra cpu
uv run pre-commit install
```

Both approaches above also install the development dependencies. If you also want to install the
documentation dependencies, you can do so with:

```bash
uv sync --extra cu124 --group docs
```

Activate the virtual environment:

```bash
source .venv/bin/activate
```

To upgrade all packages to the latest versions, you can run:

```bash
uv lock --upgrade
uv sync --extra cu124
```

### Basic development workflows

#### Testing

InstaNovo uses `pytest` for testing. To run the tests, you can use the following command:

```bash
uv run instanovo/scripts/get_zenodo_record.py # Download the test data
python -m pytest --cov-report=html --cov --random-order --verbose .
```

To see the coverage report, run:

```bash
python -m coverage report -m
```

To view the coverage report in a browser, run:

```bash
python -m http.server --directory ./coverage
```

and navigate to `http://0.0.0.0:8000/` in your browser.

#### Linting

InstaNovo uses [pre-commit hooks](https://pre-commit.com/) to ensure code quality. To run the
linters, you can use the following command:

```bash
pre-commit run --all-files
```

#### Building the documentation

To build the documentation locally, you can use the following commands:

```bash
uv sync --extra cu124 --group docs
git config --global --add safe.directory "$(dirname "$(pwd)")"
rm -rf docs/reference
python ./docs/gen_ref_nav.py
mkdocs build --verbose --site-dir docs_public
mkdocs serve
```

### Generating a requirements.txt file

If you have a `pip` or `conda` based workflow and want to generate a `requirements.txt` file, you
can use the following command:

```bash
uv export --format requirements-txt > requirements.txt
```

### Setting Python interpreter in VSCode

To set the Python interpreter in VSCode, open the Command Palette (`Ctrl+Shift+P`), search for
`Python: Select Interpreter`, and select `./.venv/bin/python`.

## License

Code is licensed under the Apache License, Version 2.0 (see [LICENSE](LICENSE.md))

The model checkpoints are licensed under Creative Commons Non-Commercial
([CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/))

## BibTeX entry and citation info

If you use InstaNovo in your research, please cite the following paper:

```bibtex
@article{eloff_kalogeropoulos_2025_instanovo,
        title        = {InstaNovo enables diffusion-powered de novo peptide sequencing in large-scale
                        proteomics experiments},
        author       = {Eloff, Kevin and Kalogeropoulos, Konstantinos and Mabona, Amandla and Morell,
                        Oliver and Catzel, Rachel and Rivera-de-Torre, Esperanza and Berg Jespersen,
                        Jakob and Williams, Wesley and van Beljouw, Sam P. B. and Skwark, Marcin J.
                        and Laustsen, Andreas Hougaard and Brouns, Stan J. J. and Ljungars,
                        Anne and Schoof, Erwin M. and Van Goey, Jeroen and auf dem Keller, Ulrich and
                        Beguir, Karim and Lopez Carranza, Nicolas and Jenkins, Timothy P.},
        year         = 2025,
        month        = {Mar},
        day          = 31,
        journal      = {Nature Machine Intelligence},
        doi          = {10.1038/s42256-025-01019-5},
        issn         = {2522-5839},
        url          = {https://doi.org/10.1038/s42256-025-01019-5}
}
```

## Acknowledgements

Big thanks to Pathmanaban Ramasamy, Tine Claeys, and Lennart Martens of the
[CompOmics](https://www.compomics.com/) research group for providing us with additional
phosphorylation training data.
