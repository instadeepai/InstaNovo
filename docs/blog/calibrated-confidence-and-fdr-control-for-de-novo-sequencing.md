# Winnow: calibrated confidence and FDR control for _de novo_ sequencing

![Winnow: calibrated confidence and FDR control for de novo sequencing](https://instanovo.ai/wp-content/uploads/2025/11/Winnow-InstaDeep-DTU-1200x675.png)

_Originally published at [instanovo.ai](https://instanovo.ai/calibrated-confidence-and-fdr-control-for-de-novo-sequencing/) on November 4 2025._

De novo peptide sequencing (DNS) models have advanced rapidly in recent years, enabling the translation of mass spectra into peptide sequences without relying on prior databases. This capability enabled by deep learning has opened the door to discovering novel peptides, exploring uncharacterised proteomes, and expanding applications across metaproteomics and immunopeptidomics.

Ensuring the reliability of these discoveries depends on accurate false discovery rate (FDR) estimation, a measure of how many reported peptide identifications are likely incorrect. In traditional proteomics, FDR control is central to trust in the biological interpretation of results and reproducibility. FDR estimation is typically achieved using target–decoy methods, such approaches however are not directly compatible with DNS, where the sequence space is effectively unlimited.

Existing attempts to estimate FDR in DNS often rely on fitting assumed score distributions to separate correct from incorrect peptide–spectrum matches (PSMs), an approach that can bias results and limit generalisability. Others extrapolate confidence thresholds from database-labelled subsets to unlabelled spectra, a brittle solution when score distributions differ.Our new framework **Winnow** brings principled, model-agnostic FDR estimation and calibration to DNS. By grounding error control in statistical theory while avoiding strong distributional assumptions, Winnow offers a more general and robust approach to confidence calibration in de novo peptide predictions.

## **What is Winnow and how does it work?**

**Winnow** is a model-agnostic framework for computing **posterior error probabilities (PEPs)**, **q-values**, and **experiment-wide FDR** without relying on parametric assumptions or database search labels.

![confidence-range](https://instanovo.ai/wp-content/uploads/2025/11/donfidence-range-1.png)

![ms-data-and-searches](https://instanovo.ai/wp-content/uploads/2025/11/winnow-1.png)

<p align="center">
  <img alt="experiment-wide-metrics" src="https://instanovo.ai/wp-content/uploads/2025/11/experiment-wide-metrics-1024x759.png" width="45%">
&nbsp; &nbsp; &nbsp; &nbsp;
  <img alt="spectrum-specific-metrics" src="https://instanovo.ai/wp-content/uploads/2025/11/spectrum-specific-metrics-1024x759.png" width="45%">
</p>


*Figure 1:* ***Overview of the Winnow framework for FDR estimation in DNS***. ***A***) At the core of the Winnow algorithm is a calibrator model that predicts the likelihood of a PSM being correct, based on features derived from both model outputs and experimental spectra. The model used database search labels for training. Score calibration allows us to estimate FDR and other metrics more accurately, retrieve more correct predictions at lower FDR, and generalise the scoring strategy across models and datasets. ***B***) Schematic showing standard usage of Winnow. The tool takes MS/MS scan information (precursor mass, precursor charge, mass-to-charge and intensity values), DNS predictions, and optionally a database search result for the same MS files for calibration. The Winnow framework includes feature calculation, database label matching and calibration. Winnow then applies a neural network to assign probabilities of each PSM being correct. This calibrated confidence is then used to non-parametrically estimate FDR. ***C***) The experiment-wide error metrics, FDR and PEP, and ***D***) spectrum-specific metrics are calculated and reported by Winnow, allowing filtering at both levels. Figure made with [*Biorender.com*](http://biorender.com/).

Designed as a lightweight calibration and rescoring layer, **Winnow** can be applied to any DNS model to improve the reliability of its predictions. The framework follows four main stages:

### **Input and inference**

A DNS tool such as [**InstaNovo**](https://instanovo.ai/) generates PSMs and assigns an initial confidence score to each prediction. These raw scores indicate model confidence but are not always well-calibrated probabilities of correctness.

### **Feature computation**

Winnow complements each candidate PSM with additional information derived from both the **mass spectrometry data** and the **model output** (Figure 2). These supplementary features capture experimental evidence and prediction dynamics that improve calibration accuracy.

Examples include:

- **Precursor mass error**, the deviation between observed and theoretical precursor masses
- **Fragment-ion match counts and intensities** for the top and runner-up predictions
- **Retention-time error**, comparing the iRT value predicted from the peptide sequence using the Prosit model with the iRT value predicted by a neural network trained to map observed retention times to iRT
- **Beam-search statistics** such as margin, median margin, entropy and z-score, which quantify uncertainty within the model's candidate predictions

Together, these features provide a broader representation of each PSM than the DNS score alone.

### **Score calibration**

Next, Winnow applies a feed-forward neural network that learns to map each PSM's raw confidence and features to a calibrated probability that the sequence is correct. By combining evidence from both the DNS model and experimental metadata, this step converts raw, model-specific scores into probabilities that are interpretable and consistent across datasets. The calibrator is trained using reference identifications from database searches but generalises beyond them, enabling accurate calibration even for unlabelled spectra.

### **FDR estimation**

Finally, Winnow estimates error rates using a **non-parametric FDR estimator** that avoids fitting score distributions or assuming prior class ratios. Instead, it integrates directly over calibrated probabilities to calculate PSM-specific and experiment-wide FDR, providing trustworthy, statistically principled control of false discoveries. Winnow also includes a **database-grounded estimator**, which uses database search matches as reference labels but relies on extrapolation to unlabelled data, making it less stable in certain settings.

## **Built for flexibility and trust**

Beyond the core workflow, Winnow was designed around several key principles that make it adaptable across models, datasets, and experimental conditions.

### Flexible calibration modes

Winnow can be applied zero-shot using a pretrained calibrator, fine-tuned on a specific dataset, or retrained from scratch. This flexibility allows adaptation to diverse experimental conditions and instruments. Users can also customise its feature set, disabling less informative inputs or adding new, experiment-specific ones.

### Model-agnostic architecture

Winnow treats the DNS model as a black box. It makes no assumptions about the model's internals or scoring function, meaning it can be seamlessly layered onto any existing DNS tool to improve reliability and confidence calibration.

### Non-parametric, label-free FDR control

Traditional FDR estimation often relies on decoy databases or fitted score distributions. Winnow's estimator takes a different route: it uses calibrated probabilities directly, sidestepping distributional assumptions and removing the need for reference labels. **A new formulation of FDR**

At its core, Winnow introduces a **discriminative decomposition** of FDR, a conceptual advance in proteomics. By directly modelling the probability that a candidate PSM is incorrect, Winnow reframes FDR estimation in a purely discriminative setting, grounding its error control in statistical first principles rather than approximations.

<p align="center">
  <img alt="SHAP-features" src="https://instanovo.ai/wp-content/uploads/2025/11/Fig-SHAP-feature-1024x695.png" width="100%">
</p>

*Figure 2:* ***Feature contributions during calibration***. SHAP feature importance scores for the general model, grouped by feature clusters created with an XGBoost model, with a cutoff for distances less than 0.5. Distance in the clustering is assumed to be scaled roughly between 0 and 1, where a 0 distance means the features are perfectly redundant and 1 means they are completely independent. The most important features are margin between top and runner-up beam predictions, fragment ion match rate and precursor mass error.

## Winnow + InstaNovo

InstaDeep's [**InstaNovo**](https://instanovo.ai/) already delivers high-performing de novo peptide sequencing and provides an ideal testbed for evaluating Winnow's impact.

Applying the **Winnow calibrator** to InstaNovo's raw confidence scores leads to a clear performance gain: **recall increases at fixed FDR thresholds**, meaning more correct PSMs are recovered while maintaining strict error control.

Winnow's **decoy-free FDR estimator** also tracks true false discovery rates when benchmarked against reference proteomes and traditional database search pipelines (Figure 3). This demonstrates that accurate FDR control can be achieved without relying on target–decoy methods or parametric score models.Across diverse datasets, the pairing of Winnow and InstaNovo consistently improves **confidence calibration**, yielding trustworthy error control. Together, they form a powerful combination: **InstaNovo** pushes the boundaries of peptide discovery, and **Winnow** ensures those discoveries can be trusted.

<p align="center">
  <img alt="precision-recall-labelled-data" src="https://instanovo.ai/wp-content/uploads/2025/11/precision-recall-labelled-data-1-1024x911.png" width="45%">
&nbsp; &nbsp; &nbsp; &nbsp;
  <img alt="precision-recall-search-data" src="https://instanovo.ai/wp-content/uploads/2025/11/precision-recall-search-data-2-1024x911.png" width="45%">
</p>

<p align="center">
  <img alt="calibration-curves-labelled-data" src="https://instanovo.ai/wp-content/uploads/2025/11/calibration-curves-labelled-data-1024x911.png" width="45%">
&nbsp; &nbsp; &nbsp; &nbsp;
  <img alt="calibration-curves-search-data" src="https://instanovo.ai/wp-content/uploads/2025/11/calibration-curves-search-data-1024x911.png" width="45%">
</p>

<p align="center">
  <img alt="fdr-accuracy-labelled-data" src="https://instanovo.ai/wp-content/uploads/2025/11/fdr-accuracy-labelled-data-1-1024x911.png" width="45%">
&nbsp; &nbsp; &nbsp; &nbsp;
  <img alt="fdr-accuracy-search-data" src="https://instanovo.ai/wp-content/uploads/2025/11/fdr-accuracy-search-data-1-1024x911.png" width="45%">
</p>

*Figure 3:* ***Performance of Winnow's calibrator and FDR estimation methods on an unseen dataset: C. elegans. A***) Precision-recall curves for the subset of C. elegans that received database search labels, comparing raw DNS model confidence and calibrated confidence. ***B***) Precision-recall curves for the full C. elegans dataset using correct proteome mapping as a proxy for correct PSM identification. ***C***) PSM-specific FDR run plots for Winnow's non-parametric and database-grounded FDR estimation methods on the labelled subset of C. elegans. ***D***) PSM-specific FDR run plots on the full, unlabelled subset of C. elegans. ***E***) Calibration curves for the labelled subset of the C. elegans dataset, comparing calibrated and raw DNS model confidence. ***F***) Calibration curves for the full C. elegans dataset.

## What this means

Winnow introduces a practical and statistically grounded framework for **trustworthy, comparable peptide discovery** in DNS. By enabling calibrated confidence scores and non-parametric FDR estimation, it bridges a long-standing gap between database-driven validation and open-space DNS.

Beyond improving individual model performance, Winnow lays the foundation for a **shared calibration layer** across DNS tools, standardising confidence scales, simplifying model comparison, and enhancing interoperability throughout proteomics workflows.

Looking ahead, we aim to broaden Winnow's generalisability across instruments, fragmentation methods, and sample types, exploring richer feature sets and alternative calibrator architectures. We also envision **multi-model calibration**, where Winnow has the potential to jointly align outputs from several DNS or search engines into unified, statistically consistent confidence estimates.

## How to try Winnow

Winnow is available now as the open-source Python package [winnow-fdr](https://pypi.org/project/winnow-fdr/) complete with a command-line interface and detailed documentation. To use it:

1. Run InstaNovol to generate peptide predictions and confidence scores.
2. Compute PSM-level features with Winnow (mass error, fragment-ion match statistics, etc.).
3. Apply the Winnow calibrator (or train your own) to transform raw scores into calibrated probabilities.
4. Use Winnow's decoy- and label-free estimator to compute PEP, q-values, and experiment-wide FDR.
5. Filter or rank results using these calibrated metrics.

For full details, read our [paper](https://arxiv.org/abs/2509.24952) and explore the [Winnow codebase.](https://github.com/instadeepai/winnow)

***Disclaimer:** All claims made are supported by our research paper: [De novo peptide sequencing rescoring and FDR estimation with Winnow](https://arxiv.org/abs/2509.24952) unless explicitly cited otherwise.*