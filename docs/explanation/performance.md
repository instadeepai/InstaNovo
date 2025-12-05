# Explanation: InstaNovo Performance

This document provides an overview of InstaNovo's performance on various benchmark datasets.

## Performance Metrics

We evaluate the performance of InstaNovo using **peptide accuracy**. This metric measures the percentage of correctly predicted peptide sequences at full coverage (i.e., without any confidence filtering).

## Benchmarks

We have benchmarked InstaNovo v1.1 and InstaNovo+ v1.1 against our previous models. For all results, InstaNovo decoding was performed with knapsack beam search decoding, and InstaNovo+ was used for refinement.

### Nine-species dataset

This dataset contains spectra from nine different species. The models were evaluated in a zero-shot setting (i.e., without any fine-tuning on the test species).

| Dataset  | InstaNovo v0.1 | InstaNovo+ v0.1 | InstaNovo v1.1 | InstaNovo+ v1.1 |
| -------- | -------------- | --------------- | -------------- | --------------- |
| Bacillus | 0.624          | 0.674           | 0.652          | **0.684**       |
| Mouse    | 0.466          | 0.490           | 0.524          | **0.542**       |
| Yeast    | 0.559          | 0.624           | 0.618          | **0.645**       |

### Biological validation datasets

We also evaluated the models on a variety of challenging biological datasets.

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

As the results show, InstaNovo+ v1.1 consistently outperforms the previous models across all datasets.
