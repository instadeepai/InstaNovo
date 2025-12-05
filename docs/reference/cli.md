# Reference: Command-Line Interface

InstaNovo provides a command-line interface (CLI) for accessing its main functionalities.

## Top-level commands

To see the main commands, run:

```
instanovo --help
```
![`instanovo --help`](../assets/screenshots/instanovo_help.svg)

This will show the following top-level commands:

- `predict`: The main command for making predictions.
- `transformer`: Commands related to the InstaNovo transformer model (train, predict, etc.).
- `diffusion`: Commands related to the InstaNovo+ diffusion model (train, predict, etc.).
- `convert`: A command for converting data to InstaNovo's native format.
- `version`: Shows the version of InstaNovo and its dependencies.

## Version info

To see the version of InstaNovo, InstaNovo+ and some of the dependencies, run:

```
instanovo version
```

![`instanovo version`](../assets/screenshots/instanovo_version.svg)

## Prediction commands

### `instanovo predict`

This is the default prediction command that first makes a prediction with the transformer-based InstaNovo model and then iteratively refines the result with the diffusion-based InstaNovo+ model.

```
instanovo predict --help
```
![`instanovo predict --help`](../assets/screenshots/instanovo_predict_help.svg)


### `instanovo transformer predict`

This command runs prediction with only the transformer model.

```
instanovo transformer predict --help
```
![`instanovo transformer predict --help`](../assets/screenshots/instanovo_transformer_predict_help.svg)


### `instanovo diffusion predict`

This command runs prediction with only the diffusion model.

```
instanovo diffusion predict --help
```
![`instanovo diffusion predict --help`](../assets/screenshots/instanovo_diffusion_predict_help.svg)

## Training commands

### `instanovo transformer train`

This command trains the transformer model.

```
instanovo transformer train --help
```
![`instanovo transformer train --help`](../assets/screenshots/instanovo_transformer_train_help.svg)

### `instanovo diffusion train`

This command trains the diffusion model.

```
instanovo diffusion train --help
```
![`instanovo diffusion train --help`](../assets/screenshots/instanovo_diffusion_train_help.svg)

## Data conversion

### `instanovo convert`

This command converts data to InstaNovo's native [`SpectrumDataFrame`](../explanation/spectrum_data_frame.md) format.

```
instanovo convert --help
```
![`instanovo convert --help`](../assets/screenshots/instanovo_convert_help.svg)
