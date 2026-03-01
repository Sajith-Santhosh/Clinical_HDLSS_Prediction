"""A modular pipeline for classification with feature selection."""

from .simulator import simulator_module, balance_dataset
from .toolbox import get_toolbox, GRACES_Selector, DeepFS_Selector, TabPFNEmbeddingSelector
from .evaluation import evaluate_model, evaluation_module, format_results
from .pipeline import run_path

__all__ = [
    'simulator_module',
    'balance_dataset',
    'get_toolbox',
    'GRACES_Selector',
    'DeepFS_Selector',
    'TabPFNEmbeddingSelector',
    'evaluate_model',
    'evaluation_module',
    'format_results',
    'run_path',
]
