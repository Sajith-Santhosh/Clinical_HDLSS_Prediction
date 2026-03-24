# Optimised Machine Learning Pipelines

A comprehensive collection of modular ML pipelines for both regression and classification tasks, featuring advanced feature selection methods and data augmentation techniques.

## Repository Structure

```
OptimisedKMDcode/
├── KMD_Report.pdf            # KMD Prediction Report
├── Regression_Pipeline/      # Pipeline for regression tasks
│   ├── main.py
│   ├── src/
│   ├── requirements.txt
│   └── README.md
│
└── Classification_Pipeline/  # Pipeline for classification tasks
    ├── main.py
    ├── src/
    ├── requirements.txt
    └── README.md
```

## 📖 Documentation

For the complete KMD Prediction in High-Dimensional Clinical Data report, see [KMD Report.pdf](./KMD%20Report.pdf)

## Pipelines

### 🔵 Regression Pipeline

High-dimensional clinical data regression pipeline with:
- **Models**: XGBoost, TabPFN
- **Feature Selection**: GRACES (GNN-based), DeepFS (autoencoder-based)
- **Data Augmentation**: SMOGN oversampling for imbalanced regression
- **Metrics**: RMSE, MAE, R²

[→ View Regression Pipeline Documentation](./Regression_Pipeline/README.md)

### 🟢 Classification Pipeline

Modular classification pipeline with:
- **Models**: TabPFN, XGBoost, Logistic Regression
- **Feature Selection**: GRACES (GNN-based), DeepFS (autoencoder-based)
- **Data Augmentation**: SMOTE oversampling for imbalanced classes
- **Metrics**: Accuracy, F1, F1-Weighted, Precision, Recall, AUC-ROC

[→ View Classification Pipeline Documentation](./Classification_Pipeline/README.md)

## Quick Start

### Regression

```bash
cd Regression_Pipeline
pip install -r requirements.txt
python main.py --data dataset.csv --target target_column
```

### Classification

```bash
cd Classification_Pipeline
pip install -r requirements.txt
python main.py --data dataset.csv --target target_column
```

## Common Features

Both pipelines share:

- **Modular Architecture**: Easy to extend with new methods
- **Multiple Feature Selection Methods**: GRACES and DeepFS implementations
- **Data Simulation**: Configurable data reduction and oversampling
- **Hybrid Approaches**: Combine feature selection with powerful classifiers/regressors
- **Comprehensive Evaluation**: Detailed metrics and performance tracking

## Requirements

- Python 3.8+
- PyTorch & PyTorch Geometric
- scikit-learn
- XGBoost
- TabPFN (optional but recommended)

Refer to individual pipeline `requirements.txt` files for complete dependencies.

## Citation

If you use these pipelines in your research, please cite the original papers for:
- **GRACES**: Chen et al. "Graph Convolutional Network-based Feature Selection" (arXiv:2211.14144)
- **DeepFS**: Li et al. "Deep Feature Screening" (arXiv:2204.01682)
- **TabPFN**: Hollmann et al. "TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second"

## License

MIT License
