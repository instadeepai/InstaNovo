# How-to: Make Predictions with InstaNovo

This guide covers various ways to make predictions with InstaNovo, including using different models and customizing the prediction process.

## Basic Prediction

The most straightforward way to make predictions is to use the `instanovo predict` command. This command runs both the InstaNovo transformer model and the InstaNovo+ diffusion model for refinement.

```bash
instanovo predict --data-path /path/to/your/spectra.mgf --output-path predictions.csv
```

Which results in the following output:

```bash
experiment_name,scan_number,spectrum_id,precursor_mz,precursor_charge,prediction_id,predictions,log_probs,token_log_probs,group,instanovo_predictions,instanovo_log_probabilities,instanovo_token_log_probabilities,instanovo_predictions_beam_0,instanovo_log_probabilities_beam_0,instanovo_token_log_probabilities_beam_0,instanovo_predictions_beam_1,instanovo_log_probabilities_beam_1,instanovo_token_log_probabilities_beam_1,instanovo_predictions_beam_2,instanovo_log_probabilities_beam_2,instanovo_token_log_probabilities_beam_2,instanovo_predictions_beam_3,instanovo_log_probabilities_beam_3,instanovo_token_log_probabilities_beam_3,instanovo_predictions_beam_4,instanovo_log_probabilities_beam_4,instanovo_token_log_probabilities_beam_4,diffusion_predictions,diffusion_log_probabilities,diffusion_token_log_probabilities,diffusion_unrefined_predictions,diffusion_predictions_beam_0,diffusion_log_probabilities_beam_0,diffusion_predictions_beam_1,diffusion_log_probabilities_beam_1,diffusion_predictions_beam_2,diffusion_log_probabilities_beam_2,diffusion_predictions_beam_3,diffusion_log_probabilities_beam_3,diffusion_predictions_beam_4,diffusion_log_probabilities_beam_4,predictions_tokenised,delta_mass_ppm
spectra,0,spectra:0,451.25348,2,0,IAHYNKR,-0.0038739838637411594,,no_group,"['L', 'A', 'H', 'Y', 'N', 'K', 'R']",-1.4490094184875488,[0],LAHYNKR,-1.4490094184875488,[0],LAHYNKR,-1.7640595436096191,[0],LAHYNKR,-1.7640595436096191,[0],LAHYNKR,-1.7640595436096191,[0],LAHYNKR,-1.7640595436096191,[0],"['I', 'A', 'H', 'Y', 'N', 'K', 'R']",-0.0038739838637411594,,"['L', 'A', 'H', 'Y', 'N', 'K', 'R']","['L', 'A', 'H', 'Y', 'N', 'K', 'R']",-0.14114204049110413,"['L', 'A', 'H', 'Y', 'N', 'K', 'R']",-0.055098626762628555,"['I', 'A', 'H', 'Y', 'N', 'K', 'R']",-0.0038739838637411594,"['L', 'A', 'H', 'Y', 'N', 'K', 'R']",-0.06414960324764252,"['I', 'A', 'H', 'Y', 'N', 'K', 'R']",-0.10904442518949509,"I, A, H, Y, N, K, R",0.6781111138830191
```

This output CSV file contains several columns. Here are some of the most important ones:

| Column | Description |
|---|---|
| `scan_number` | The scan number of the spectrum in the input file. |
| `precursor_mz` | The mass-to-charge ratio of the precursor ion. |
| `precursor_charge` | The charge of the precursor ion. |
| `diffusion_predictions` | The peptide sequence predicted by InstaNovo+. |
| `transformer_predictions` | The peptide sequence predicted by the base InstaNovo model. |
| `log_probs` | The log probability of the prediction. Higher (less negative) values indicate greater model confidence in the predicted output. |

For a full description of the output, see the [prediction output reference](../reference/prediction_output.md).

## Evaluation

To evaluate InstaNovo performance on an annotated dataset (a dataset which has a column with the ground truth sequence):

```bash
instanovo predict --evaluation --data-path ./sample_data/spectra.mgf --output-path predictions.csv
```

Which results in the following output:

```bash
experiment_name,scan_number,spectrum_id,precursor_mz,precursor_charge,prediction_id,predictions,targets,log_probs,token_log_probs,group,instanovo_predictions,instanovo_log_probabilities,instanovo_token_log_probabilities,instanovo_predictions_beam_0,instanovo_log_probabilities_beam_0,instanovo_token_log_probabilities_beam_0,instanovo_predictions_beam_1,instanovo_log_probabilities_beam_1,instanovo_token_log_probabilities_beam_1,instanovo_predictions_beam_2,instanovo_log_probabilities_beam_2,instanovo_token_log_probabilities_beam_2,instanovo_predictions_beam_3,instanovo_log_probabilities_beam_3,instanovo_token_log_probabilities_beam_3,instanovo_predictions_beam_4,instanovo_log_probabilities_beam_4,instanovo_token_log_probabilities_beam_4,diffusion_predictions,diffusion_log_probabilities,diffusion_token_log_probabilities,diffusion_unrefined_predictions,diffusion_predictions_beam_0,diffusion_log_probabilities_beam_0,diffusion_predictions_beam_1,diffusion_log_probabilities_beam_1,diffusion_predictions_beam_2,diffusion_log_probabilities_beam_2,diffusion_predictions_beam_3,diffusion_log_probabilities_beam_3,diffusion_predictions_beam_4,diffusion_log_probabilities_beam_4,predictions_tokenised,delta_mass_ppm
spectra,0,spectra:0,451.25348,2,0,LAHYNKR,IAHYNKR,-0.03630233183503151,,no_group,"['L', 'A', 'H', 'Y', 'N', 'K', 'R']",-1.4490094184875488,[0],LAHYNKR,-1.4490094184875488,[0],LAHYNKR,-1.7640595436096191,[0],LAHYNKR,-1.7640595436096191,[0],LAHYNKR,-1.7640595436096191,[0],LAHYNKR,-1.7640595436096191,[0],"['L', 'A', 'H', 'Y', 'N', 'K', 'R']",-0.03630233183503151,,"['L', 'A', 'H', 'Y', 'N', 'K', 'R']","['L', 'A', 'H', 'Y', 'N', 'K', 'R']",-0.3328973650932312,"['L', 'A', 'H', 'Y', 'N', 'K', 'R']",-0.09905184805393219,"['L', 'A', 'H', 'F', 'D', 'K', 'R']",-1.303741455078125,"['L', 'A', 'H', 'Y', 'N', 'K', 'R']",-0.1970413625240326,"['L', 'A', 'H', 'Y', 'N', 'K', 'R']",-0.03630233183503151,"L, A, H, Y, N, K, R",0.6781111138830191
```

Note that the `--evaluation` flag includes the `targets` column in the output, which contains the
ground truth peptide sequence. Metrics will be calculated and displayed in the console.

### Command-line arguments

InstaNovo provides several command-line arguments for common prediction parameters:

- `--data-path`: Path to your spectral data. It can be a single file (`.mgf`, `.mzml`, `.mzxml`, `.ipc`) or a directory. You can also use glob patterns (e.g., `./experiment/*.mgf`).
- `--output-path`: Path to the output CSV file.
- `--instanovo-model`: The InstaNovo model to use. You can specify a model ID (e.g., `instanovo-v1.1.0`) or a path to a checkpoint file (`.ckpt`).
- `--instanovo-plus-model`: The InstaNovo+ model to use. You can specify a model ID (e.g., `instanovoplus-v1.1.0`) or a path to a checkpoint file (`.ckpt`).
- `--denovo`: Use this flag for _de novo_ prediction on unannotated data.
- `--evaluation`: Use this flag to evaluate the model on annotated data.
- `--with-refinement` / `--no-refinement`: Whether to use InstaNovo+ for refinement (default is `True`).

### Overriding configuration values

InstaNovo uses configuration files for more advanced settings, located in `instanovo/configs/inference/`. The default configuration is `default.yaml`.

You can override any value in the configuration file from the command line. For example, to change the number of beams used in beam search to 5, you would run:

```bash
instanovo predict --data-path ./sample_data/spectra.mgf --output-path predictions.csv num_beams=5
```

Note that there is no `--` prefix for configuration overrides.

## Advanced Prediction Scenarios

### Running only InstaNovo (Transformer)

If you want to run predictions using only the InstaNovo transformer model without the diffusion refinement, you can use the `instanovo transformer predict` command:

```bash
instanovo transformer predict --data-path /path/to/your/spectra.mgf --output-path instanovo_predictions.csv
```

To see all available options for this command, run:

```bash
instanovo transformer predict --help
```

### Running only InstaNovo+ (Diffusion)

To run predictions using only the InstaNovo+ diffusion model, you can use the `instanovo diffusion predict` command. This is useful if you already have predictions from another model and want to refine them.

```bash
instanovo diffusion predict --data-path /path/to/your/spectra.mgf --output-path instanovoplus_predictions.csv instanovo_predictions_path=instanovo_predictions.csv
```

Here, `instanovo_predictions.csv` is a CSV file containing the initial predictions.

To see all available options for this command, run:

```bash
instanovo diffusion predict --help
```

### Two-step prediction

You can also run the prediction in two separate steps:

1. **Run InstaNovo:**

    ```bash
    instanovo transformer predict --data-path ./sample_data/spectra.mgf --output-path instanovo_predictions.csv
    ```

2. **Run InstaNovo+ for refinement:**

    ```bash
    instanovo diffusion predict --data-path ./sample_data/spectra.mgf --output-path instanovoplus_predictions.csv instanovo_predictions_path=instanovo_predictions.csv
    ```

This approach gives you more control over the process and allows you to inspect the intermediate predictions from the transformer model.
