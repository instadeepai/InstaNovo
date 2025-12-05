# How-to: Use Your Own Datasets

InstaNovo is designed to be flexible and work with your own mass spectrometry data. This guide explains how to prepare your data for use with InstaNovo.

## The SpectrumDataFrame

InstaNovo uses a custom data structure called [`SpectrumDataFrame`](../explanation/spectrum_data_frame.md) to handle spectral data. This class provides a unified interface for various common mass spectrometry file formats, including `.mgf`, `.mzml`, `.mzxml`, and `.csv`. It also supports reading data directly from Pandas or Polars DataFrames.

One of the key features of the `SpectrumDataFrame` is its ability to handle large datasets that don't fit into memory. It does this through lazy loading, where data is loaded from disk only when it's needed.

## Using standard file formats

The easiest way to use your own data is to provide it in one of the supported file formats. You can then pass the path to your data directly to the `instanovo predict` or `instanovo train` commands.

You can specify a single file, a directory, or use glob patterns to specify multiple files:

```bash
# Predict from a single MGF file
instanovo predict --data-path /path/to/your/data.mgf

# Predict from all MGF files in a directory
instanovo predict --data-path /path/to/your/experiment/*.mgf
```

## Creating a custom dataset from a DataFrame

If your data is not in a standard file format, you can create a `SpectrumDataFrame` from a Pandas or Polars DataFrame.

Your DataFrame must have the following columns:

- `sequence` (string): The target peptide sequence (required for training).
- `precursor_mz` (float): The precursor m/z.
- `precursor_charge` (integer): The precursor charge.
- `mz_array` (list of floats): The m/z values of the MS2 peaks.
- `intensity_array` (list of floats): The intensity values of the MS2 peaks.

Here is an example of how to create a `SpectrumDataFrame` from a Pandas DataFrame:

```python
import pandas as pd
from instanovo.utils.data_handler import SpectrumDataFrame

# Create a sample DataFrame
data = {
    'sequence': ['PEPTIDE'],
    'precursor_mz': [600.0],
    'precursor_charge': [2],
    'mz_array': [[100.1, 200.2, 300.3]],
    'intensity_array': [[1000.0, 2000.0, 1500.0]]
}
df = pd.DataFrame(data)

# Create a SpectrumDataFrame
sdf = SpectrumDataFrame(df)

# You can now use this sdf for training or prediction
```

## Converting to native format

For very large datasets, it can be more efficient to convert your data to InstaNovo's native format, which is based on Apache Parquet. This allows for faster loading and shuffling during training.

You can convert your data using the `SpectrumDataFrame.save` method:

```python
from instanovo.utils import SpectrumDataFrame

# Load your data from MGF files
sdf = SpectrumDataFrame.load("/path/to/data.mgf", lazy=False, is_annotated=True)

# Save it in the native Parquet format
sdf.save("path/to/parquet/folder", partition="train", chunk_size=1e6)
```

You can then load this native dataset for training:

```python
# Load the native Parquet dataset
sdf = SpectrumDataFrame.load("path/to/parquet/folder", partition="train", shuffle=True, lazy=True, is_annotated=True)
```

InstaNovo also provides a command-line script for converting data:

```bash
instanovo convert --help
```

![`instanovo convert --help`](../assets/screenshots/instanovo_convert_help.svg)
