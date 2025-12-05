# For Developers

This guide is for developers who want to contribute to InstaNovo or set up a development environment.

InstaNovo is built for Python >=3.10, <3.14 and tested on Linux, Windows and macOS.

## Setting up with `uv`

This project uses [uv](https://docs.astral.sh/uv/) to manage Python dependencies.

1.  **Install `uv`**: If you don't have `uv` installed, follow the [official installation instructions](https://docs.astral.sh/uv/getting-started/installation/).

2.  **Fork and clone the repository**:

    ```bash
    git clone https://github.com/YOUR-USERNAME/InstaNovo.git
    cd InstaNovo
    ```

3.  **Install dependencies**:

    If you have an NVIDIA GPU:

    ```bash
    uv sync --extra cu126
    uv run pre-commit install
    ```

    If you are on a CPU-only, or macOS machine:

    ```bash
    uv sync --extra cpu
    uv run pre-commit install
    ```

    To also install the documentation dependencies:

    ```bash
    uv sync --extra cu126 --group docs
    ```

4.  **Activate the virtual environment**:

    ```bash
    source .venv/bin/activate
    ```

### Metal Performance Shaders

InstaNovo now has support for [Metal Performance Shaders](https://developer.apple.com/documentation/metalperformanceshaders) (MPS) for Apple silicon devices. If you would like to use InstaNovo with MPS, please set `mps` to True in the configuration files ([`instanovo/configs/`](https://github.com/instadeepai/InstaNovo/tree/main/instanovo/configs)) and set the environment variable:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1
```

This allows the CPU to be used for functionality not yet supported on MPS.

## Development workflows

### Testing

InstaNovo uses `pytest` for testing.

1.  **Download test data**:

    ```bash
    uv run instanovo/scripts/get_zenodo_record.py
    ```

2.  **Run tests**:

    ```bash
    python -m pytest --cov-report=html --cov --random-order --verbose .
    ```

3.  **View coverage report**:

    ```bash
    python -m coverage report -m
    ```

### Linting

We use `pre-commit` hooks to maintain code quality. To run the linters on all files:

```bash
pre-commit run --all-files
```

### Building the documentation

To build and serve the documentation locally:

```bash
uv sync --extra cu126 --group docs
git config --global --add safe.directory "$(dirname "$(pwd)")"
rm -rf docs/API
python ./docs/gen_ref_nav.py
mkdocs build --verbose --site-dir docs_public
mkdocs serve
```
