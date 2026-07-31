# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.2.2] - 2025-12-05

### Fixed

- Corrected the PyPI publishing workflow's package version.

## [1.2.1] - 2025-12-05

### Fixed

- Corrected release CI and made documentation publication tolerate skipped upstream jobs.

## [1.2.0] - 2025-12-05

### Added

- Added multi-GPU training and prediction through Accelerate.
- Added fine-tuning configuration and support for the smaller InstaNovo+ model.
- Added configurable presets for local and Aichor Accelerate environments.

### Changed

- Reworked transformer and diffusion training and prediction around shared trainer and predictor components.
- Improved beam-search and knapsack decoding performance and added negative-mass support to knapsack decoding.
- Reorganized the command-line interface, configuration hierarchy, documentation, and continuous-integration workflows.

## [1.1.4] - 2025-06-13

### Fixed

- Fixed diffusion sampling and model-checkpoint downloads.

## [1.1.3] - 2025-06-10

### Added

- Added the `instanovoplus-v1.1.0` checkpoint and an InstaNovo-P notebook.
- Added multi-sample diffusion prediction with configurable refinement controls.
- Added named validation-data groups and optional source-file and spectrum-ID columns in spectrum data frames.
- Added configurable S3 upload and temporary-download handling.

### Changed

- Updated inference defaults and model performance documentation for the new checkpoints.

## [1.1.2] - 2025-05-14

### Added

- Added the InstaNovo-P phosphoproteomics checkpoint and inference configuration.
- Added automatic device selection and an InstaNovo v0.1-versus-v1.1 comparison notebook.

### Changed

- Restored documentation builds and test coverage reporting.

## [1.1.1] - 2025-03-31

### Added

- Added sample spectra and updated the getting-started notebook with diffusion usage.

### Fixed

- Automatically select the diffusion device when loading models and use an independent spectrum-data-frame instance for diffusion.

## [1.1.0] - 2025-03-28

### Added

- Added the InstaNovo+ diffusion model, including training, prediction, refinement, and pipeline configurations.
- Added a unified command-line interface and model checkpoint loading and saving support.
- Added dataset presets for Massive-KB, Extended Massive-KB, nine-species, and phosphoproteomics data.
- Added an InstaNovo logo, CLI reference graphics, DOI metadata, and expanded documentation.

### Changed

- Updated the core data handling, decoding, model training, and prediction workflows for the v1.1 model family.

## [1.0.1] - 2025-01-21

### Added

- Added model metadata, charge validation, and updated notebooks and inference configuration.

### Fixed

- Prevented a division-by-zero error when predicting on small sample files.
- Improved scaled-dot-product-attention handling and data conversion coverage.

## [1.0.0] - 2024-10-09

### Added

- Added the InstaNovo v1.0 model code, packaged configuration presets, and model download utilities.
- Added greedy decoding, configurable inference, comprehensive unit and integration tests, and multi-platform notebook testing.
- Added `pyproject.toml`-based packaging and locked dependency sets.

### Changed

- Restructured model training, prediction, spectrum-data processing, and command-line utilities for the v1.0 release.

### Removed

- Removed the earlier diffusion-model implementation from the v1.0 codebase.

## [0.1.7] - 2024-03-06

### Added

- Added integration tests, a documentation site, CodeQL analysis, Dependabot, and multi-version test workflows.
- Added input-file type checks and support for MGF metadata.

### Fixed

- Fixed diffusion and transformer data loading for current schemas, single-example inputs, and de novo error handling.
- Updated the benchmark location and documentation links.

## [0.1.6] - 2023-10-10

### Added

- Added spectrum conversion utilities and a diffusion example to the getting-started notebook.
- Added diffusion-confidence scoring based on loss and returned log-probabilities from the diffusion decoder.

### Changed

- Switched MGF parsing to matchms and renamed the diffusion data module to `dataset`.
- Improved beam search and diffusion inference handling.

### Fixed

- Handled missing charge information, unlabeled MGF input, and source-type edge cases.

## [0.1.5] - 2023-09-27

### Added

- Added the InstaNovo+ diffusion model and its prediction command with logging.

### Fixed

- Corrected package dependencies, release workflow configuration, and notebook installation instructions.

## [0.1.4] - 2023-09-19

### Added

- Added a getting-started notebook and Colab link.
- Added checkpoint configuration support for transformer prediction and training.

## [0.1.3] - 2023-09-02

### Changed

- Updated project URLs and README hyperlinks.

## [0.1.2] - 2023-09-01

### Added

- Added package long-description metadata for PyPI.

## [0.1.1] - 2023-09-01

### Fixed

- Corrected declarative setuptools requirements configuration for package builds.

## [0.1.0] - 2023-09-01

### Added

- Initial public release of InstaNovo with PyPI publishing automation.

[unreleased]: https://github.com/instadeepai/InstaNovo/compare/1.2.2...HEAD
[1.2.2]: https://github.com/instadeepai/InstaNovo/compare/1.2.1...1.2.2
[1.2.1]: https://github.com/instadeepai/InstaNovo/compare/1.2.0...1.2.1
[1.2.0]: https://github.com/instadeepai/InstaNovo/compare/1.1.4...1.2.0
[1.1.4]: https://github.com/instadeepai/InstaNovo/compare/1.1.3...1.1.4
[1.1.3]: https://github.com/instadeepai/InstaNovo/compare/1.1.2...1.1.3
[1.1.2]: https://github.com/instadeepai/InstaNovo/compare/1.1.1...1.1.2
[1.1.1]: https://github.com/instadeepai/InstaNovo/compare/1.1.0...1.1.1
[1.1.0]: https://github.com/instadeepai/InstaNovo/compare/1.0.1...1.1.0
[1.0.1]: https://github.com/instadeepai/InstaNovo/compare/1.0.0...1.0.1
[1.0.0]: https://github.com/instadeepai/InstaNovo/compare/0.1.7...1.0.0
[0.1.7]: https://github.com/instadeepai/InstaNovo/compare/0.1.6...0.1.7
[0.1.6]: https://github.com/instadeepai/InstaNovo/compare/0.1.5...0.1.6
[0.1.5]: https://github.com/instadeepai/InstaNovo/compare/0.1.4...0.1.5
[0.1.4]: https://github.com/instadeepai/InstaNovo/compare/0.1.3...0.1.4
[0.1.3]: https://github.com/instadeepai/InstaNovo/compare/0.1.2...0.1.3
[0.1.2]: https://github.com/instadeepai/InstaNovo/compare/0.1.1...0.1.2
[0.1.1]: https://github.com/instadeepai/InstaNovo/compare/0.1.0...0.1.1
[0.1.0]: https://github.com/instadeepai/InstaNovo/releases/tag/0.1.0
