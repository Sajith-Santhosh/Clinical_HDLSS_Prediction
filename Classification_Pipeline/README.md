# TabPFN Classification Pipeline

A modular pipeline for binary classification with advanced feature selection methods, following the architecture of OptimisedKMDCode.

## Architecture

```
TabPFN_Classification_Pipeline/
├── main.py           # CLI entry point
├── requirements.txt   # Python dependencies
├── src/
│   ├── __init__.py
│   ├── simulator.py   # Data reduction + SMOTE
│   ├── toolbox.py     # Feature selection (GRACES, DeepFS)
│   ├── evaluation.py  # Classification metrics
│   └── pipeline.py    # Main orchestrator (run_path)
```

## Features

### Modeling Paths

| Method | Description |
|--------|-------------|
| `TABPFN` | Pure TabPFN (no feature selection) |
| `GRACES` | GRACES (GNN) selection → TabPFN |
| `DEEPFS` | DeepFS selection → TabPFN |
| `XGB` | Baseline XGBoost |
| `LR` | Baseline Logistic Regression |
| `GRACES_XGB` | GRACES → XGBoost |
| `DEEPFS_XGB` | DeepFS → XGBoost |
| `GRACES_LR` | GRACES → Logistic Regression |
| `DEEPFS_LR` | DeepFS → Logistic Regression |

### Feature Selection Methods

1. **GRACES** - Graph Convolutional Network-based Feature Selection
   - Paper: Chen et al. "Graph Convolutional Network-based Feature Selection for High-dimensional and Low-sample Size Data" (arXiv:2211.14144)

2. **DeepFS** - Deep Feature Screening
   - Paper: Li et al. "Deep Feature Screening: Feature Selection for Ultra High-Dimensional Data via Deep Neural Networks" (arXiv:2204.01682)

### Data Simulation

- **Data Reduction**: Remove a percentage of training samples
- **SMOTE**: Synthetic Minority Over-sampling Technique for imbalanced classification

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# For HuggingFace login (required for TabPFN)
python -c "from huggingface_hub import login; login()"
```

## Usage

### Basic Usage

```bash
python main.py --data data.csv --target Label --method TABPFN
```

### Run Multiple Methods

```bash
python main.py --data data.csv --target Label --method ALL
```

### With Data Simulation

```bash
# Remove 50% of data + apply SMOTE
python main.py --data data.csv --target Label --method GRACES --reduction 0.5 --smote
```

### Multiple Reduction Levels

```bash
python main.py --data data.csv --target Label --method ALL --reductions 0.0,0.3,0.5,0.7 --smote
```

### Custom Feature Count

```bash
python main.py --data data.csv --target Label --method DEEPFS --features 50
```

## Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--data` | Path to CSV dataset | (required) |
| `--target` | Target column name | (required) |
| `--method` | Modeling method | `TABPFN` |
| `--reduction` | Data reduction % | `0.0` |
| `--reductions` | Comma-separated reductions | None |
| `--smote` | Enable SMOTE variants | False |
| `--features` | Features to select | `100` |
| `--test-size` | Test set size | `0.2` |
| `--random-state` | Random seed | `42` |

## Example Output

```
============================================================
RUNNING PATH: GRACES | 50% Data Removed + SMOTE
============================================================
  [Simulator] Removing 50%: 1000 -> 500 samples
  [SMOTE] Applying oversampling...
    [GRACES] Selecting 100 features...
      Selected 100/100 features
  [Logic] Fitting TabPFN Classifier...
  [Result] Accuracy: 0.9234 | F1: 0.9156 | F1_W: 0.9221
```

## Metrics

- **Accuracy**: Overall correctness
- **F1**: Binary F1 score
- **F1_Weighted**: Weighted F1 score
- **Precision**: Positive predictive value
- **Recall**: Sensitivity
- **AUC**: Area under ROC curve (when applicable)

## Comparison with Original

The `tabpfn_feature.py` (original notebook-style) has been refactored into:

1. **simulator.py** - `balance_dataset()` function from notebook
2. **toolbox.py** - `GRACES` and `DeepFS` classes from notebook
3. **evaluation.py** - `evaluate_model()` function from notebook
4. **pipeline.py** - Orchestrates all components (new)
5. **main.py** - CLI interface (new)

This follows the same modular pattern as `OptimisedKMDCode/src/pipeline.py`.
