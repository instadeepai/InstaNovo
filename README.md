# _De novo_ peptide sequencing with InstaNovo

[![PyPI version](https://badge.fury.io/py/instanovo.svg)](https://badge.fury.io/py/instanovo)
<a target="_blank" href="https://colab.research.google.com/github/instadeepai/InstaNovo/blob/main/notebooks/getting_started_with_instanovo.ipynb">
<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a>

The official code repository for InstaNovo. This repo contains the code for training and inference
of InstaNovo and InstaNovo+. InstaNovo is a transformer neural network with the ability to translate
fragment ion peaks into the sequence of amino acids that make up the studied peptide(s). InstaNovo+,
inspired by human intuition, is a multinomial diffusion model that further improves performance by
iterative refinement of predicted sequences.

![Graphical Abstract](https://raw.githubusercontent.com/instadeepai/InstaNovo/main/graphical_abstract.jpeg)

**Links:**

- bioRxiv: https://www.biorxiv.org/content/10.1101/2023.08.30.555055v2

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
[anaconda](https://www.anaconda.com/)/[miniconda](https://docs.conda.io/projects/miniconda/en/latest/)
installed:

```bash
conda create -n instanovo python=3.8
conda activate instanovo
```

Note: InstaNovo is built for Python >= 3.8

### Training

To train auto-regressive InstaNovo:

```bash
usage: python -m instanovo.transformer.train train_path valid_path [-h] [--config CONFIG] [--n_gpu N_GPU] [--n_workers N_WORKERS]

required arguments:
  train_path        Training data path
  valid_path        Validation data path

optional arguments:
  --config CONFIG   file in configs folder
  --n_workers N_WORKERS
```

Note: data is expected to be saved as Polars `.ipc` format. See section on data conversion.

To update the InstaNovo model config, modify the config file under
[configs/instanovo/base.yaml](https://github.com/instadeepai/InstaNovo/blob/main/configs/instanovo/base.yaml)

### Prediction

To evaluate InstaNovo:

```bash
usage: python -m instanovo.transformer.predict data_path model_path [-h] [--denovo] [--config CONFIG] [--subset SUBSET] [--knapsack_path KNAPSACK_PATH] [--n_workers N_WORKERS]

required arguments:
  data_path         Evaluation data path
  model_path        Model checkpoint path

optional arguments:
  --denovo          evaluate in de novo mode, will not try to compute metrics
  --output_path OUTPUT_PATH
                    Save predictions to a csv file (required in de novo mode)
  --subset SUBSET
                    portion of set to evaluate
  --knapsack_path KNAPSACK_PATH
                    path to pre-computed knapsack
  --n_workers N_WORKERS
```

### Using your own datasets

To use your own datasets, you simply need to tabulate your data in either
[Pandas](https://pandas.pydata.org/) or [Polars](<(https://www.pola.rs/)>) with the following
schema:

The dataset is tabular, where each row corresponds to a labelled MS2 spectra.

- `sequence (string) [Optional]` \
   The target peptide sequence excluding post-translational modifications
- `modified_sequence (string)` \
  The target peptide sequence including post-translational modifications
- `precursor_mz (float64)` \
  The mass-to-charge of the precursor (from MS1)
- `charge (int64)` \
  The charge of the precursor (from MS1)
- `mz_array (list[float64])` \
  The mass-to-charge values of the MS2 spectrum
- `mz_array (list[float32])` \
  The intensity values of the MS2 spectrum

For example, the DataFrame for the
[Nine-Species excluding Yeast](https://huggingface.co/datasets/InstaDeepAI/instanovo_ninespecies_exclude_yeast)
dataset look as follows:

|     | sequence             | modified_sequence          | precursor_mz | precursor_charge | mz_array                             | intensity_array                     |
| --: | :------------------- | :------------------------- | -----------: | ---------------: | :----------------------------------- | :---------------------------------- |
|   0 | GRVEGMEAR            | GRVEGMEAR                  |      335.502 |                3 | [102.05527 104.052956 113.07079 ...] | [ 767.38837 2324.8787 598.8512 ...] |
|   1 | IGEYK                | IGEYK                      |      305.165 |                2 | [107.07023 110.071236 111.11693 ...] | [ 1055.4957 2251.3171 35508.96 ...] |
|   2 | GVSREEIQR            | GVSREEIQR                  |      358.528 |                3 | [103.039444 109.59844 112.08704 ...] | [801.19995 460.65268 808.3431 ...]  |
|   3 | SSYHADEQVNEASK       | SSYHADEQVNEASK             |      522.234 |                3 | [101.07095 102.0552 110.07163 ...]   | [ 989.45154 2332.653 1170.6191 ...] |
|   4 | DTFNTSSTSNSTSSSSSNSK | DTFNTSSTSN(+.98)STSSSSSNSK |      676.282 |                3 | [119.82458 120.08073 120.2038 ...]   | [ 487.86942 4806.1377 516.8846 ...] |

For _de novo_ prediction, the `modified_sequence` column is not required.

We also provide a conversion script for converting to Polars IPC binary (`.ipc`):

```bash
usage: python -m instanovo.utils.convert_to_ipc source target [-h] [--source_type {mgf,mzml,csv}] [--max_charge MAX_CHARGE] [--verbose]

positional arguments:
  source                source file or folder
  target                target ipc file to be saved

optional arguments:
  -h, --help            show this help message and exit
  --source_type {mgf,mzml,csv}
                        type of input data
  --max_charge MAX_CHARGE
                        maximum charge to filter out
```

_Note: we currently only support `mzml`, `mgf` and `csv` conversions._

If you want to use InstaNovo for evaluating metrics, you will need to manually set the
`modified_sequence` column after conversion.

## Roadmap

This code repo is currently under construction.

**ToDo:**

- Multi-GPU support

## License

Code is licensed under the Apache License, Version 2.0 (see [LICENSE](LICENSE.txt))

The model checkpoints are licensed under Creative Commons Non-Commercial
([CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/))

## BibTeX entry and citation info

```bibtex
@article{eloff_kalogeropoulos_2023_instanovo,
	title = {De novo peptide sequencing with InstaNovo: Accurate, database-free peptide identification for large scale proteomics experiments},
	author = {Kevin Eloff and Konstantinos Kalogeropoulos and Oliver Morell and Amandla Mabona and Jakob Berg Jespersen and Wesley Williams and Sam van Beljouw and Marcin Skwark and Andreas Hougaard Laustsen and Stan J. J. Brouns and Anne Ljungars and Erwin Marten Schoof and Jeroen Van Goey and Ulrich auf dem Keller and Karim Beguir and Nicolas Lopez Carranza and Timothy Patrick Jenkins},
	year = {2023},
	doi = {10.1101/2023.08.30.555055},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/10.1101/2023.08.30.555055v2},
	journal = {bioRxiv}
}
```
