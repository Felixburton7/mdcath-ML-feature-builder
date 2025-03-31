# 🧪 mdCATH Dataset Processor

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)

A comprehensive Python package for processing the mdCATH protein dynamics dataset. It extracts simulation data, cleans structures, calculates metrics and properties, builds ML-ready feature sets, performs voxelization, and generates high-quality visualizations.

## Features

*   Reads mdCATH HDF5 files
*   Cleans PDB structures (using pdbUtils or internal fallback)
*   Calculates DSSP, SASA, Core/Exterior classification, Phi/Psi angles
*   Extracts and averages RMSF data
*   Generates ML-ready CSV feature files per temperature and overall average
*   Voxelizes cleaned structures using `aposteriori`
*   Creates publishable quality visualizations
*   Configurable pipeline execution
*   Parallel processing support

## Installation

```bash
# 1. Clone the repository (if you haven't already)
# git clone <repository-url>
# cd mdcath-processor

# 2. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install external tools:
#    - DSSP: Install `dssp` or `mkdssp` and ensure it's in your system PATH.
#      (e.g., using conda: `conda install -c salilab dssp`)
#    - Aposteriori: Install `aposteriori`
#      pip install aposteriori

# 5. Install the package itself (editable mode recommended for development)
pip install -e .
```

## Usage

The main entry point is `main.py`, driven by a configuration file (`config/default_config.yaml`).

```bash
# Run the full pipeline using the default configuration
mdprocess

# Specify a different configuration file
mdprocess --config path/to/your/config.yaml

# Specify a different output directory
mdprocess --output-dir /path/to/outputs

# Process only a subset of domains
mdprocess --domains 1a02F00 16pkA02

# Adjust the number of parallel cores
mdprocess --num-cores 8
```

See `config/default_config.yaml` for available configuration options.

## Output Structure

The processed data and visualizations will be saved in the specified output directory (default: `./outputs/`):

```
outputs/
├── logs/
├── ML_features/
├── pdbs/
├── frames/
├── RMSF/
├── voxelized/
└── visualizations/
```
(Refer to the full project specification for details on the content of each folder).

## Contributing

[Details on how to contribute - e.g., reporting issues, submitting pull requests]

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
# mdcath-ML-feature-builder
