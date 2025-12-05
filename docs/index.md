<p align="center" width="100%">
    <picture>
        <source srcset="assets/instanovo_logo_square.svg" media="(prefers-color-scheme: dark)">
        <img width="8%" src="assets/instanovo_logo_square.svg" alt="InstaNovo Logo">
    </picture>
</p>

# _De novo_ peptide sequencing with InstaNovo

[![PyPI version](https://badge.fury.io/py/instanovo.svg)](https://badge.fury.io/py/instanovo) [![code coverage](./assets/coverage.svg)](./development/coverage.md) [![DOI](https://zenodo.org/badge/681625644.svg)](https://doi.org/10.5281/zenodo.14712453) <a target="_blank" href="https://colab.research.google.com/github/instadeepai/InstaNovo/blob/main/notebooks/getting_started_with_instanovo.ipynb">
<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a>

InstaNovo is a transformer neural network with the ability to translate
fragment ion peaks into the sequence of amino acids that make up the studied peptide(s). InstaNovo+,
inspired by human intuition, is a multinomial diffusion model that further improves performance by
iterative refinement of predicted sequences.

This documentation will help you get started with InstaNovo. It is divided into the following sections:

- **[Tutorials](./tutorials/getting_started.md)**
    - How to [install](./tutorials/getting_started.md#installation) InstaNovo, make your first [prediction](./tutorials/getting_started.md#making-your-first-prediction) and [evaluate](./tutorials/getting_started.md#evaluating-performance) InstaNovo's performance.
    - An end-to-end starter [notebook](./notebooks/getting_started_with_instanovo.ipynb) that you can [run in Google Colab
    ](https://colab.research.google.com/github/instadeepai/InstaNovo/blob/main/notebooks/getting_started_with_instanovo.ipynb).
- **[How-to guides](./how-to/predicting.md)**:
    - How to perform [predictions](./how-to/predicting.md#basic-prediction) with InstaNovo with iterative refinement of InstaNovo+, or how to [use each model separately](./how-to/predicting.md#advanced-prediction-scenarios).
    - Guide for preparing your [own data](./how-to/using_custom_datasets.md) for use with InstaNovo and InstaNovo+.
    - Details how to [train](./how-to/training.md) your own InstaNovo and InstaNovo+ models.
-  **[Reference](./reference/cli.md)**:
    - Overview of the `instanovo` [command-line interface](./reference/cli.md).
    - List of the supported [post translational modifications](./reference/modifications.md).
    - Description of the columns in the [prediction output CSV](./reference/prediction_output.md)
    - Code [reference API](./API/summary.md)
- **[Explanation](./explanation/performance.md)**:
    - Explains our [performance metrics](./explanation/performance.md#performance-metrics) and [benchmarking results](./explanation/performance.md#benchmarks)
    - A detailed explanation of the [`SpectrumDataFrame`](./explanation/spectrum_data_frame.md) class and its features.
- **[Blog](./blog/introducing-the-next-generation-of-instanovo-models.md)**:
    - [Introducing the next generation of InstaNovo models](./blog/introducing-the-next-generation-of-instanovo-models.md)
    - [Introducing InstaNovo-P](./blog/introducing-instanovo-p-a-de-novo-sequencing-model-for-phosphoproteomics.md)
    - [Winnow: calibrated confidence and FDR control for _de novo_ sequencing](./blog/calibrated-confidence-and-fdr-control-for-de-novo-sequencing.md)
- **[For Developers](./development/setup.md)**:
    - How to set up a [development environment](./development/setup.md#setting-up-with-uv).
    - How to run the [tests](./development/setup.md#testing) and [lint](./development/setup.md#linting) the code.
    - View the [test coverage](./development/coverage.md) and [test report](development/allure.md).
- **[How to Cite](./citation.md)**:
    - Bibtex references for our [peer-reviewed publication](./citation.md#instanovo-instanovo) on InstaNovo and InstaNovo+ and our preprints on [InstaNovo-P](citation.md#instanovo-p), [InstaNexus](./citation.md#instanexus) and [Winnow](./citation.md#winnow).
- **[License](./license.md)**:
    - Code is licensed under the [Apache License, Version 2.0](./license.md#apache-license)
    - The model checkpoints are licensed under Creative Commons Non-Commercial ([CC BY-NC-SA 4.0](./license.md#creative-commons-attribution-noncommercial-sharealike-40-international))

**Developed by:**

- [InstaDeep](https://www.instadeep.com/)
- [The Department of Biotechnology and Biomedicine](https://orbit.dtu.dk/en/organisations/department-of-biotechnology-and-biomedicine) -
  [Technical University of Denmark](https://www.dtu.dk/)
