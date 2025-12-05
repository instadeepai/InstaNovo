# Explanation: The SpectrumDataFrame

At the core of InstaNovo's data handling is the `SpectrumDataFrame`. This document provides a more detailed explanation of this class and its features.

## What is it?

The `SpectrumDataFrame` is a specialized data structure designed to provide a unified and efficient interface for working with mass spectrometry data. It acts as a wrapper around various data sources, including:

- Standard mass spectrometry files (`.mgf`, `.mzml`, `.mzxml`)
- Tabular data files (`.csv`, `.parquet`)
- In-memory Pandas and Polars DataFrames

By providing a single API, the `SpectrumDataFrame` simplifies the process of loading, processing, and iterating over spectral data, regardless of its original format.

## Key Features

### Glob Notation Support

The `SpectrumDataFrame` natively supports glob notation when specifying data paths. This allows you to easily reference multiple files at once using wildcard patterns. All supported file formats are automatically detected and converted to the internal SpectrumDataFrame format for training and inference.

For example, you can use glob notation to process all MGF files in a directory:

```bash
instanovo predict --data_path=./experiment/*.mgf
```

You can also specify glob patterns in your configuration files to process multiple files across different directories or with different extensions:

```python
# In your config file
data_path: "./experiment/**/*.mzml"
```

This flexibility makes it easy to work with large datasets organized across multiple files and directories without having to manually list each file.

### Lazy Loading and Asynchronous Prefetching

When working with large datasets, it's often not feasible to load all the data into memory at once. The `SpectrumDataFrame` addresses this with **lazy loading**. When lazy loading is enabled, the `SpectrumDataFrame` only loads the data for the files that are currently being accessed. This allows you to work with datasets that are much larger than your available RAM.

To further improve performance, the `SpectrumDataFrame` uses **asynchronous prefetching**. This means that while you are processing the data from one file, the `SpectrumDataFrame` is already loading the next file in the background. This helps to minimize I/O wait times and keep your GPU busy during training.

### Efficient Shuffling

Effective shuffling of training data is crucial for training robust models. The `SpectrumDataFrame` implements a two-fold shuffling strategy:

1. **File-level shuffling**: The order of the files to be loaded is shuffled.
2. **Within-file shuffling**: The spectra within each file are also shuffled.

This ensures that the model sees a diverse range of data in each training epoch.

### On-the-fly Filtering and Sampling

The `SpectrumDataFrame` allows you to perform filtering and sampling operations without modifying the underlying data on disk. You can apply filters to select specific spectra based on their properties (e.g., precursor charge) or sample a subset of your data for quick experiments.

```python
from instanovo.utils import SpectrumDataFrame

# Load a dataset with lazy loading
sdf = SpectrumDataFrame.load("/path/to/experiment/*.mzml", lazy=True)

# Keep only spectra with a precursor charge of 2 or less
sdf.filter_rows(lambda row: row["precursor_charge"] <= 2)

# Sample 50% of the data
sdf.sample_subset(fraction=0.5, seed=42)
```

### Interoperability

The `SpectrumDataFrame` is designed to be interoperable with other popular data science libraries. You can easily convert a `SpectrumDataFrame` to a Pandas DataFrame or a Polars LazyFrame:

```python
# Convert to a Pandas DataFrame
pandas_df = sdf.to_pandas()

# Convert to a Polars LazyFrame
polars_lazy_df = sdf.to_polars(return_lazy=True)
```

You can also write the contents of a `SpectrumDataFrame` to an `.mgf` file:

```python
sdf.write_mgf("path/to/output.mgf")
```

## Example usage

Converting mgf files to the native parquet format:

```python
from instanovo.utils import SpectrumDataFrame

# Convert mgf files to native parquet:
sdf = SpectrumDataFrame.load("/path/to/data.mgf", lazy=False, is_annotated=True)
sdf.save("path/to/parquet/folder", partition="train", chunk_size=1e6)
```

Loading the native format in shuffle mode:

```python
# Load a native parquet dataset:
sdf = SpectrumDataFrame.load("path/to/parquet/folder", partition="train", shuffle=True, lazy=True, is_annotated=True)
```

Using the loaded SpectrumDataFrame in a PyTorch DataLoader:

```python
from instanovo.transformer.dataset import SpectrumDataset
from torch.utils.data import DataLoader

ds = SpectrumDataset(sdf)
# Note: Shuffle and workers is handled by the SpectrumDataFrame
dl = DataLoader(
    ds,
    collate_fn=SpectrumDataset.collate_batch,
    shuffle=False,
    num_workers=0,
)
```

Some more examples using the SpectrumDataFrame:

```python
sdf = SpectrumDataFrame.load("/path/to/experiment/*.mzml", lazy=True)

# Remove rows with a charge value > 3:
sdf.filter_rows(lambda row: row["precursor_charge"]<=2)

# Sample a subset of the data:
sdf.sample_subset(fraction=0.5, seed=42)

# Convert to pandas
df = sdf.to_pandas() # Returns a pd.DataFrame

# Convert to polars LazyFrame
lazy_df = sdf.to_polars(return_lazy=True) # Returns a pl.LazyFrame

# Save as an `.mgf` file
sdf.write_mgf("path/to/output.mgf")
```
