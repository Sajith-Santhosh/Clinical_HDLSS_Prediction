"""Evaluation metrics: Accuracy, F1, F1_Weighted, Precision, Recall, AUC-ROC."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix
)


def evaluate_model(y_true, y_pred, y_prob=None):
    """
    Calculate classification evaluation metrics.

    Parameters:
    -----------
    y_true : array-like
        True target labels
    y_pred : array-like
        Predicted labels
    y_prob : array-like, optional
        Predicted probabilities for AUC calculation

    Returns:
    --------
    dict : Dictionary containing metrics
    """
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'F1': f1_score(y_true, y_pred, average='binary'),
        'F1_Weighted': f1_score(y_true, y_pred, average='weighted'),
        'Precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
        'Recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
    }

    if y_prob is not None:
        try:
            metrics['AUC'] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics['AUC'] = 0.0

    return metrics


def print_metrics(metrics, method_name):
    """Print evaluation metrics in a formatted way."""
    print(f"  {method_name}:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"    {key}: {value:.4f}")
        else:
            print(f"    {key}: {value}")


def evaluation_module(y_true, y_pred, method_name, variant_info, results_list, y_prob=None):
    """
    Calculate metrics and append to global results list.

    Parameters:
    -----------
    y_true : array-like
        True target labels
    y_pred : array-like
        Predicted labels
    method_name : str
        Name of the technique/method
    variant_info : float or str
        Information about the data variant
    results_list : list
        List to append results to
    y_prob : array-like, optional
        Predicted probabilities for AUC

    Returns:
    --------
    dict : Calculated metrics
    """
    metrics = evaluate_model(y_true, y_pred, y_prob)

    record = {
        "Method": method_name,
        "Data_Variant": variant_info,
        **metrics
    }
    results_list.append(record)

    return metrics


def format_results(results_list, sort_by='F1_Weighted', ascending=False):
    """
    Format results into a sorted DataFrame.

    Parameters:
    -----------
    results_list : list
        List of result dictionaries
    sort_by : str
        Column to sort by
    ascending : bool
        Sort order (False for metrics where higher is better)

    Returns:
    --------
    DataFrame : Formatted and sorted results
    """
    df = pd.DataFrame(results_list)
    df = df.sort_values(by=sort_by, ascending=ascending).reset_index(drop=True)
    return df


def print_results_table(results_df, title="RESULTS"):
    """Print formatted results table."""
    print(f"\n{title}")
    print(results_df.to_string(index=False))


def print_best_model(results_df, metric='F1_Weighted'):
    """Print information about the best performing model."""
    best = results_df.iloc[0]
    print(f"\n BEST MODEL (by {metric}, Higher ↑):")
    print(f"   Method:       {best['Method']}")
    print(f"   Data Variant: {best['Data_Variant']}")
    for col in results_df.columns:
        if col not in ['Method', 'Data_Variant']:
            if isinstance(best[col], float):
                print(f"   {col}: {best[col]:.4f}")
            else:
                print(f"   {col}: {best[col]}")


def generate_report(results_list, sort_by='F1_Weighted'):
    """
    Generate a complete report from results.

    Parameters:
    -----------
    results_list : list
        List of result dictionaries
    sort_by : str
        Metric to sort by

    Returns:
    --------
    DataFrame : Formatted results
    """
    results_df = format_results(results_list, sort_by=sort_by, ascending=False)
    print_results_table(results_df, f"FINAL RESULTS (Sorted by {sort_by})")
    print_best_model(results_df, metric=sort_by)
    return results_df


def print_confusion_matrix(y_true, y_pred, labels=None):
    """Print a formatted confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print("\nConfusion Matrix:")
    print("                Predicted")
    print("             Negative  Positive")
    print(f"Actual Neg   {cm[0,0]:6d}   {cm[0,1]:6d}")
    print(f"Actual Pos   {cm[1,0]:6d}   {cm[1,1]:6d}")
    return cm


def calculate_class_balance(y):
    """Calculate class distribution statistics."""
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)
    distribution = {label: {'count': int(count), 'pct': float(count)/total*100}
                    for label, count in zip(unique, counts)}
    return distribution


def print_class_balance(y, label_name="Target"):
    """Print class distribution."""
    distribution = calculate_class_balance(y)
    print(f"\n{label_name} Distribution:")
    for label, stats in sorted(distribution.items()):
        print(f"  Class {label}: {stats['count']} ({stats['pct']:.1f}%)")
    return distribution
