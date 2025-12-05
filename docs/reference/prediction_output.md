# Reference: Prediction Output

When you run predictions with InstaNovo and specify an output path, a CSV file is generated. This document describes the columns in that file.

## Standard Columns

| Column                  | Description                                                    | Data Type    | Notes                                                                                                         |
| ----------------------- | -------------------------------------------------------------- | ------------ | ------------------------------------------------------------------------------------------------------------- |
| experiment_name         | Experiment name derived from input filename                    | String       | Based on the input file name (mgf, mzml, or mzxml)                                                            |
| scan_number             | Scan number of the MS/MS spectrum                              | Integer      | Unique identifier from the input file                                                                         |
| spectrum_id             | Unique spectrum identifier                                     | String       | Combination of experiment name and scan number (e.g., `yeast:17738`)                                          |
| precursor_mz            | Precursor m/z (mass-to-charge ratio)                           | Float        | The observed m/z of the precursor ion                                                                         |
| precursor_charge        | Precursor charge state                                         | Integer      | Charge state of the precursor ion                                                                             |
| prediction_id           | Unique prediction identifier                                   | String       | Internal identifier for the prediction                                                                        |
| group                   | Data group identifier                                          | String       | Used when running predictions on grouped data                                                                 |
| targets                 | Target peptide sequence                                        | String       | Ground truth peptide sequence (only present if running in evaluation mode)                                     |
| predictions             | Best predicted peptide sequence                                | String       | The final predicted peptide sequence (from InstaNovo+ when using refinement)                                 |
| predictions_tokenised   | Best predicted peptide sequence (tokenised)                    | String       | The predicted sequence as comma-separated tokens                                                               |
| log_probs               | Log probability of the best predicted sequence                 | Float        | Natural logarithm of the sequence confidence. Higher is better.                                               |
| token_log_probs         | Log probability of each token in the best prediction           | List[Float]  | Natural logarithm of the confidence for each amino acid in the sequence                                        |
| delta_mass_ppm          | Mass difference between precursor and predicted peptide in ppm | Float        | The mass deviation in parts per million. Lower is better.                                                     |

## InstaNovo (Transformer) Model Columns

These columns are present when using InstaNovo+ (combined transformer + diffusion model).

| Column                                | Description                                                         | Data Type    | Notes                                             |
| ------------------------------------- | ------------------------------------------------------------------- | ------------ | ------------------------------------------------- |
| instanovo_predictions                  | Predicted peptide sequence from InstaNovo                           | String       | The initial peptide sequence from the transformer |
| instanovo_log_probabilities           | Log probability from InstaNovo                                      | Float        | Natural logarithm of the sequence confidence       |
| instanovo_token_log_probabilities     | Token log probabilities from InstaNovo                              | List[Float]  | Natural logarithm of the confidence for each token |
| instanovo_predictions_beam_0-4        | Predicted sequences from each beam                                   | String       | Beam search results when num_beams > 1           |
| instanovo_log_probabilities_beam_0-4  | Log probabilities from each beam                                    | Float        | Confidence scores for each beam                   |
| instanovo_token_log_probabilities_beam_0-4 | Token log probabilities from each beam                           | List[Float]  | Per-token confidence for each beam                |

## InstaNovo+ (Diffusion) Model Columns

These columns are present when using InstaNovo+ (combined transformer + diffusion model).

| Column                                   | Description                                                         | Data Type    | Notes                                             |
| ---------------------------------------- | ------------------------------------------------------------------- | ------------ | ------------------------------------------------- |
| diffusion_predictions                    | Predicted peptide sequence from InstaNovo+                          | String       | The refined peptide sequence from the diffusion   |
| diffusion_log_probabilities              | Log probability from InstaNovo+                                     | Float        | Natural logarithm of the sequence confidence       |
| diffusion_token_log_probabilities        | Token log probabilities from InstaNovo+                             | List[Float]  | Natural logarithm of the confidence for each token |
| diffusion_unrefined_predictions          | Unrefined predictions from InstaNovo+                               | String       | Predictions before refinement                     |
| diffusion_predictions_beam_0-4            | Predicted sequences from each beam                                   | String       | Beam search results when num_beams > 1           |
| diffusion_log_probabilities_beam_0-4     | Log probabilities from each beam                                    | Float        | Confidence scores for each beam                   |

## Usage Notes

- When using InstaNovo+ with refinement, the `predictions` column contains the best prediction from the diffusion model.
- We recommend filtering the output based on the `diffusion_log_probabilities` and `delta_mass_ppm` columns to obtain a set of high-confidence predictions.
- Beam search columns (beam_0 through beam_4) are only present when running with `num_beams > 1`.
- The transformer model columns are prefixed with `instanovo_` and diffusion model columns are prefixed with `diffusion_`.
