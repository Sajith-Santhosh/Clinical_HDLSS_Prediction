"""
EVALUATION MODULE (REGRESSION)
==============================
Metrics: RMSE, MAE, R²
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def evaluate_model(y_true, y_pred):
    """
    Calculate regression evaluation metrics.
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
    
    Returns:
    --------
    dict : Dictionary containing RMSE, MAE, and R²
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }


def print_metrics(metrics, method_name):
    """Print evaluation metrics in a formatted way."""
    print(f"  {method_name}:")
    print(f"    RMSE: {metrics['RMSE']:.4f}")
    print(f"    MAE:  {metrics['MAE']:.4f}")
    print(f"    R²:   {metrics['R2']:.4f}")


def evaluation_module(y_true, y_pred, method_name, variant_info, results_list):
    """
    Calculate metrics and append to global results list.
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
    method_name : str
        Name of the technique/method
    variant_info : float or str
        Information about the data variant
    results_list : list
        List to append results to
    
    Returns:
    --------
    dict : Calculated metrics
    """
    metrics = evaluate_model(y_true, y_pred)
    
    record = {
        "Method": method_name,
        "Data_Variant": f"{int(variant_info*100)}%" if isinstance(variant_info, float) else variant_info,
        **metrics
    }
    results_list.append(record)
    
    return metrics


def format_results(results_list, sort_by='RMSE', ascending=True):
    """
    Format results into a sorted DataFrame.
    
    Parameters:
    -----------
    results_list : list
        List of result dictionaries
    sort_by : str
        Column to sort by ('RMSE', 'MAE', or 'R2')
    ascending : bool
        Sort order (True for RMSE/MAE lower=better, False for R² higher=better)
    
    Returns:
    --------
    DataFrame : Formatted and sorted results
    """
    df = pd.DataFrame(results_list)
    df = df.sort_values(by=sort_by, ascending=ascending).reset_index(drop=True)
    return df


def print_results_table(results_df, title="RESULTS"):
    """Print formatted results table."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print(results_df.to_string(index=False))
    print("=" * 80)


def print_best_model(results_df, metric='RMSE'):
    """Print information about the best performing model."""
    best = results_df.iloc[0]
    better = "Lower ↓" if metric in ['RMSE', 'MAE'] else "Higher ↑"
    print(f"\n BEST MODEL (by {metric}, {better}):")
    print(f"   Method:       {best['Method']}")
    print(f"   Data Variant: {best['Data_Variant']}")
    print(f"   RMSE:         {best['RMSE']:.4f}")
    print(f"   MAE:          {best['MAE']:.4f}")
    print(f"   R²:           {best['R2']:.4f}")


def generate_report(results_list, sort_by='RMSE'):
    """
    Generate a complete report from results.
    
    Parameters:
    -----------
    results_list : list
        List of result dictionaries
    sort_by : str
        Metric to sort by ('RMSE', 'MAE', or 'R2')
    
    Returns:
    --------
    DataFrame : Formatted results
    """
    # For RMSE and MAE, lower is better (ascending=True)
    # For R², higher is better (ascending=False)
    ascending = sort_by in ['RMSE', 'MAE']
    
    results_df = format_results(results_list, sort_by=sort_by, ascending=ascending)
    print_results_table(results_df, f"FINAL RESULTS (Sorted by {sort_by})")
    print_best_model(results_df, metric=sort_by)
    return results_df
