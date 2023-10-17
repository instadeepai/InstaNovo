# De novo peptide sequencing with InstaNovo

The official code repository for InstaNovo. This repo contains the code for training and inference
of InstaNovo and InstaNovo+.

Links:

- bioRxiv: https://www.biorxiv.org/content/10.1101/2023.08.30.555055v1

![Graphical Abstract](./graphical_abstract.jpeg)

## Usage

### Installation

To use InstaNovo, we need to install the module via `pip`. First clone the repository, navigate to
the relevant folder and run the following command:

```bash
pip install instanovo
```

### Training

To train auto-regressive InstaNovo:

```bash
python -m instanovo.transformer.train train_path valid_path [-h] [--config CONFIG] [--n_gpu N_GPU] [--n_workers N_WORKERS]
```

## Roadmap

The code repo is currently under construction.

**ToDo:**

- Add diffusion model code
- Add model checkpoints to releases and HuggingFace
- Add datasets to HuggingFace

## License

Code is licensed under the Apache License, Version 2.0 (see [LICENSE](LICENSE.txt))

The model checkpoints are licensed under Creative Commons Non-Commercial
([CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/))

## Citation

```bibtex
@article{eloff_kalogeropoulos_2023_instanovo,
	title = {De novo peptide sequencing with InstaNovo: Accurate, database-free peptide identification for large scale proteomics experiments},
	author = {Kevin Eloff and Konstantinos Kalogeropoulos and Oliver Morell and Amandla Mabona and Jakob Berg Jespersen and Wesley Williams and Sam van Beljouw and Marcin Skwark and Andreas Hougaard Laustsen and Stan J. J. Brouns and Anne Ljungars and Erwin Marten Schoof and Jeroen Van Goey and Ulrich auf dem Keller and Karim Beguir and Nicolas Lopez Carranza and Timothy Patrick Jenkins},
	year = {2023},
	doi = {10.1101/2023.08.30.555055},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2023/08/31/2023.08.30.555055},
	journal = {bioRxiv}
}
```
