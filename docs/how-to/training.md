# How-to: Train InstaNovo Models

This guide explains how to train your own InstaNovo and InstaNovo+ models.

## Preparing your data

Before you can start training, you need to have your data in a format that InstaNovo can understand. The easiest way to do this is to use the `SpectrumDataFrame`. See the [how-to guide on using custom datasets](./using_custom_datasets.md) for more details.

## Training InstaNovo (Transformer)

The InstaNovo transformer model is the base model that performs the initial _de novo_ sequencing.

To train the transformer model, you use the `instanovo transformer train` command. The training process is configured using a YAML file. The default configuration file is `instanovo/configs/instanovo.yaml`.

To start training with the default configuration, you would run:

```bash
instanovo transformer train
```

You will likely want to customize the training configuration, such as the paths to your training and validation data. You can do this by creating your own YAML file and passing it to the command, or by overriding specific values from the command line.

For example, to specify the training and validation data paths, you could run:

```bash
instanovo transformer train --data.train_data_path /path/to/train/data --data.val_data_path /path/to/val/data
```

To see all the available options for the training command, run:

```bash
instanovo transformer train --help
```

To customize the model architecture, you can modify the model configuration file at [`instanovo/configs/model/instanovo_base.yaml`](https://github.com/instadeepai/InstaNovo/blob/main/instanovo/configs/model/instanovo_base.yaml).

## Training InstaNovo+ (Diffusion)

The InstaNovo+ diffusion model is used to refine the predictions made by the transformer model.

Training the diffusion model is similar to training the transformer model. You use the `instanovo diffusion train` command, and the configuration is managed through a YAML file (default: [`instanovo/configs/instanovoplus.yaml`](https://github.com/instadeepai/InstaNovo/blob/main/instanovo/configs/instanovoplus.yaml)).

To start training the diffusion model, you would run:

```bash
instanovo diffusion train
```

Again, you can customize the configuration by providing your own YAML file or by overriding values from the command line.

To see all the available options, run:

```bash
instanovo diffusion train --help
```

To customize the model architecture, you can modify the model configuration file at `instanovo/configs/model/instanovoplus_base.yaml`.
