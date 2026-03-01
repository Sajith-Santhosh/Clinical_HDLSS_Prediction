# KMD Prediction in High-Dimensional Clinical Data Pipeline

A modular regression pipeline for comparing feature selection methods and data augmentation techniques on imbalanced datasets.

## Features

- Multiple regression models: XGBoost, TabPFN
- Feature selection: GRACES (GNN-based), DeepFS (autoencoder-based)
- Data simulation: Configurable reduction and SMOGN oversampling
- Hybrid approaches combining feature selection with TabPFN
- Comprehensive evaluation metrics: RMSE, MAE, R²

## Requirements

Python 3.8 or higher is required.

Install dependencies:

```bash
pip install -r requirements.txt
```

**Note**: PyTorch Geometric may require additional setup. Refer to the [official installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) if you encounter issues.

## Usage

Basic usage:

```bash
python main.py --data dataset.csv --target target_column
```

### Arguments

**Required:**
- `--data`: Path to CSV dataset
- `--target`: Target column name

**Optional:**
- `--method`: Modeling method (default: `XGBOOST`)
  - Choices: `TABPFN`, `GRACES`, `XGBOOST`, `DEEPFS`, `DEEPFS_TABPFN`, `GRACES_TABPFN`, `ALL`
- `--reduction`: Data reduction percentage (default: `0.0`, range: 0.0-1.0)
- `--reductions`: Comma-separated list of reduction values (e.g., `0.0,0.3,0.6`)
- `--smogn`: Enable SMOGN oversampling for imbalanced regression
- `--features`: Number of features to select (default: `100`)

### Examples

Run XGBoost with full data:

```bash
python main.py --data data.csv --target price
```

Compare multiple methods with data reduction:

```bash
python main.py --data data.csv --target price --method ALL --reductions 0.0,0.3,0.6
```

Run DeepFS + TabPFN hybrid with SMOGN:

```bash
python main.py --data data.csv --target price --method DEEPFS_TABPFN --smogn --features 50
```

## Methods

- **XGBOOST**: Baseline gradient boosting
- **TABPFN**: Transformer-based prior-fitted network
- **GRACES**: GNN-based feature selection + XGBoost
- **DEEPFS**: Supervised autoencoder feature selection + XGBoost
- **DEEPFS_TABPFN**: DeepFS feature selection + TabPFN inference
- **GRACES_TABPFN**: GRACES feature selection + TabPFN inference

## Project Structure

```
OptimisedKMDcode/
├── main.py              # CLI entry point
├── src/
│   ├── pipeline.py      # Main execution paths
│   ├── simulator.py     # Data reduction and SMOGN
│   ├── toolbox.py       # Feature selection methods
│   └── evaluation.py    # Metrics calculation
└── requirements.txt     # Dependencies
```

## Output

Results are printed to console with metrics for each method and configuration. Output includes:

- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² (Coefficient of Determination)

Results are sorted by RMSE (ascending) with best models at the top.

## Notes

- TabPFN has limitations: max 100 features, best on datasets with <2000 samples
- SMOGN is designed for imbalanced regression and may not be necessary for balanced datasets
- Feature selection methods (GRACES, DeepFS) are computationally intensive on large datasets
