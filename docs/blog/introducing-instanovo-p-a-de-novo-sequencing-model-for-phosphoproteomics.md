# Introducing InstaNovo-P, a de novo sequencing model for phosphoproteomics

![instanovo-plus-the-de-novo-sequencing-model-for-phosphoproteomics](https://instanovo.ai/wp-content/uploads/2025/05/instanovo-plus-the-de-novo-sequencing-model-for-phosphoproteomics-1200x673.webp)

_Originally published at [instanovo.ai](https://instanovo.ai/introducing-instanovo-p-a-de-novo-sequencing-model-for-phosphoproteomics/) on May 22 2025._

## Announcing InstaNovo-P

We are happy to share our newest model in the InstaNovo family, InstaNovo-P in our latest [bioRxiv preprint](https://www.biorxiv.org/content/10.1101/2025.05.14.654049v1).

InstaNovo-P is a fine-tuned version of our base model, InstaNovo v1.0.0, specifically tailored towards application in phosphoproteomics mass spectrometry experiments. InstaNovo-P was further trained on approx. 2.8 million PSMs from 75 thousand phosphorylated peptides. InstaNovo-P was extended to recognize the residues phospho-tyrosine, -serine and -threonine, achieving high accuracy in detecting phosphorylated peptides while retaining its performance in unmodified peptides.

## Benchmarking InstaNovo-P

InstaNovo-P performs better than the current state of the art de novo sequencing model that supports phosphorylation, PrimeNovo. It also performs better than the base InstaNovo v1.0.0 model in the test dataset of the ProteomeTools dataset used for training the base model, indicating that our gradual unfreezing strategy while fine tuning prevented loss of performance in unmodified peptides (Figure 1A). InstaNovo-P exhibits state-of-the-art performance on detection of phosphorylated peptides (Figure 1B).

![](https://instanovo.ai/wp-content/uploads/2025/05/comparaison-between-instanovo-plus-and-instanovo-1024x649.webp)

## InstaNovo-P performance varies with phosphorylation types

We investigated performance of the model across peptides with different properties. InstaNovo-P performed better in detection of peptides with phosphorylated serines, which reflected the composition of our training dataset (Figure 2A). We also observed performance dependence on peptide length, as well as number of phosphorylation sites present on phosphorylated peptides (Figure 2B). These results indicate that InstaNovo-P presents similar behaviour to our base model, and that additional training data would increase performance on other phosphorylated threonine and tyrosine.

![](https://instanovo.ai/wp-content/uploads/2025/05/instanovo-plus-performance-with-phosphorylation-types-1024x586.webp)

## InstaNovo-P enhances localization confidence and captures biological pathways

We found that InstaNovo-P captures a substantial amount of phosphorylated peptides with low false discovery thresholds (< 5% FDR as assessed by database search result grounding). When validating our model with an external dataset of FGFR2 signaling in breast cancer cells, we find that InstaNovo-P exhibits 66.5% recall overall, with 41.2% recall at 5% FDR (Figure 3A). Notably, we show that we can detect phosphorylation events in crucial proteins that participate in FGFR2 signaling, and our detected events recapitulate the pathways involved. Additionally, we observed high correlation of phosphorylation localization when compared to database search (Figure 3B). This indicates that InstaNovo-P is adept at localizing phosphorylation in peptides that contain more than one possible site, and provides another axis of information that can enhance localization certainly.

![](https://instanovo.ai/wp-content/uploads/2025/05/instanovo-plus-enhances-localization-confidence-and-captures-biological-pathways-1024x606.webp)

## InstaNovo-P provides value to phosphoproteomics experiments

Importantly, InstaNovo detects a considerable number of peptides that go undetected with database approaches (Figure 4A). We could verify the presence of these peptides by targeted proteomics in independent samples, corroborating our predictions (Figure 4B).

![](https://instanovo.ai/wp-content/uploads/2025/05/instanovo-plus-provides-value-to-phosphoproteomics-experiments-1024x476.webp)

Together, our results suggest that InstaNovo-P can provide complementary information to phosphoproteomics database searches, by detecting novel peptides and enhancing localization certainty. The new release of our base model, InstaNovo v1.1 has already been trained with the phosphorylation dataset used to fine tune this model, and now supports phosphorylation, with comparable performance to InstaNovo-P across datasets. We will continue to develop models for important applications of de novo peptide sequencing in proteomics, and expand the supported post translational modifications further. We look forward to researchers using our models for deeper and more robust biological insights in proteomics!

_Source of all images: internal_
