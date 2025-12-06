<p align="center" width="100%">
    <picture>
        <source srcset="https://raw.githubusercontent.com/instadeepai/InstaNovo/main/docs/assets/instanovo_white.svg" media="(prefers-color-scheme: dark)">
        <img width="33%" src="https://raw.githubusercontent.com/instadeepai/InstaNovo/main/docs/assets/instanovo_black.svg" alt="InstaNovo Logo">
    </picture>
</p>

# _De novo_ peptide sequencing with InstaNovo

[![PyPI version](https://badge.fury.io/py/instanovo.svg)](https://badge.fury.io/py/instanovo) [![code coverage](https://raw.githubusercontent.com/instadeepai/InstaNovo/main/docs/assets/coverage.svg?raw=true)](https://instadeepai.github.io/InstaNovo/development/coverage.md) [![DOI](https://zenodo.org/badge/681625644.svg)](https://doi.org/10.5281/zenodo.14712453) <a target="_blank" href="https://colab.research.google.com/github/instadeepai/InstaNovo/blob/main/notebooks/getting_started_with_instanovo.ipynb">
<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a>

The official code repository for InstaNovo. This repo contains the code for training and inference
of InstaNovo and InstaNovo+. InstaNovo is a transformer neural network with the ability to translate
fragment ion peaks into the sequence of amino acids that make up the studied peptide(s). InstaNovo+,
inspired by human intuition, is a multinomial diffusion model that further improves performance by
iterative refinement of predicted sequences.

Publication in Nature Machine Intelligence:
[InstaNovo enables diffusion-powered de novo peptide sequencing in large-scale proteomics experiments](https://www.nature.com/articles/s42256-025-01019-5)

![Graphical Abstract](https://raw.githubusercontent.com/instadeepai/InstaNovo/main/docs/assets/graphical_abstract.jpeg)

The full documentation is available at
[https://instadeepai.github.io/InstaNovo/](https://instadeepai.github.io/InstaNovo/) and consists of
the following sections.

- **[Tutorials](https://instadeepai.github.io/InstaNovo/tutorials/getting_started)**
    - How to [install](https://instadeepai.github.io/InstaNovo/tutorials/getting_started#installation) InstaNovo, make your first [prediction](https://instadeepai.github.io/InstaNovo/tutorials/getting_started#making-your-first-prediction) and [evaluate](https://instadeepai.github.io/InstaNovo/tutorials/getting_started#evaluating-performance) InstaNovo's performance.
    - An end-to-end starter [notebook](https://instadeepai.github.io/InstaNovo/notebooks/getting_started_with_instanovo) that you can [run in Google Colab
    ](https://colab.research.google.com/github/instadeepai/InstaNovo/blob/main/notebooks/getting_started_with_instanovo.ipynb).
- **[How-to guides](https://instadeepai.github.io/InstaNovo/how-to/predicting)**:
    - How to perform [predictions](https://instadeepai.github.io/InstaNovo/how-to/predicting#basic-prediction) with InstaNovo with iterative refinement of InstaNovo+, or how to [use each model separately](https://instadeepai.github.io/InstaNovo/how-to/predicting#advanced-prediction-scenarios).
    - Guide for preparing your [own data](https://instadeepai.github.io/InstaNovo/how-to/using_custom_datasets) for use with InstaNovo and InstaNovo+.
    - Details how to [train](https://instadeepai.github.io/InstaNovo/how-to/training) your own InstaNovo and InstaNovo+ models.
-  **[Reference](https://instadeepai.github.io/InstaNovo/reference/cli)**:
    - Overview of the `instanovo` [command-line interface](https://instadeepai.github.io/InstaNovo/reference/cli).
    - List of the supported [post translational modifications](https://instadeepai.github.io/InstaNovo/reference/modifications).
    - Description of the columns in the [prediction output CSV](https://instadeepai.github.io/InstaNovo/reference/prediction_output)
    - Code [reference API](https://instadeepai.github.io/InstaNovo/API/summary)
- **[Explanation](https://instadeepai.github.io/InstaNovo/explanation/performance)**:
    - Explains our [performance metrics](https://instadeepai.github.io/InstaNovo/explanation/performance#performance-metrics) and [benchmarking results](https://instadeepai.github.io/InstaNovo/explanation/performance#benchmarks)
    - A detailed explanation of the [`SpectrumDataFrame`](https://instadeepai.github.io/InstaNovo/explanation/spectrum_data_frame) class and its features.
- **[Blog](https://instadeepai.github.io/InstaNovo/blog/introducing-the-next-generation-of-instanovo-models)**:
    - [Introducing the next generation of InstaNovo models](https://instadeepai.github.io/InstaNovo/blog/introducing-the-next-generation-of-instanovo-models)
    - [Introducing InstaNovo-P](https://instadeepai.github.io/InstaNovo/blog/introducing-instanovo-p-a-de-novo-sequencing-model-for-phosphoproteomics)
    - [Winnow: calibrated confidence and FDR control for _de novo_ sequencing](https://instadeepai.github.io/InstaNovo/blog/calibrated-confidence-and-fdr-control-for-de-novo-sequencing)
- **[For Developers](https://instadeepai.github.io/InstaNovo/development/setup)**:
    - How to set up a [development environment](https://instadeepai.github.io/InstaNovo/development/setup#setting-up-with-uv).
    - How to run the [tests](https://instadeepai.github.io/InstaNovo/development/setup#testing) and [lint](https://instadeepai.github.io/InstaNovo/development/setup#linting) the code.
    - View the [test coverage](https://instadeepai.github.io/InstaNovo/development/coverage) and [test report](https://instadeepai.github.io/InstaNovo/development/allure).
- **[How to Cite](https://instadeepai.github.io/InstaNovo/citation)**:
    - Bibtex references for our [peer-reviewed publication](https://instadeepai.github.io/InstaNovo/citation#instanovo-instanovo) on InstaNovo and InstaNovo+ and our preprints on [InstaNovo-P](https://instadeepai.github.io/InstaNovo/citation#instanovo-p), [InstaNexus](https://instadeepai.github.io/InstaNovo/citation#instanexus) and [Winnow](https://instadeepai.github.io/InstaNovo/citation#winnow).
- **[License](https://instadeepai.github.io/InstaNovo/license)**:
    - Code is licensed under the [Apache License, Version 2.0](https://instadeepai.github.io/InstaNovo/license#apache-license)
    - The model checkpoints are licensed under Creative Commons Non-Commercial ([CC BY-NC-SA 4.0](https://instadeepai.github.io/InstaNovo/license#creative-commons-attribution-noncommercial-sharealike-40-international))

**Developed by:**

- [InstaDeep](https://www.instadeep.com/)
- [The Department of Biotechnology and Biomedicine](https://orbit.dtu.dk/en/organisations/department-of-biotechnology-and-biomedicine) -
  [Technical University of Denmark](https://www.dtu.dk/)
