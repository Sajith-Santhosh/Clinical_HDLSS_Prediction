"""Data reduction + SMOTE oversampling for imbalanced classification."""

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE


def print_target_balance(y, bins=5):
    """Print class/target imbalance summary."""
    if y.dtype == object or len(np.unique(y)) <= 20:
        counts = pd.Series(y).value_counts().sort_index()
    else:
        counts = pd.cut(y, bins=bins).value_counts().sort_index()
    total = len(y)
    print("\n  TARGET DISTRIBUTION:")
    for label, cnt in counts.items():
        pct = 100 * cnt / total
        print(f"    {label}: {cnt} ({pct:.1f}%)")


def balance_dataset(X, y, reduction_pct=1.0, apply_smote=False,
                    random_state=42, smote_k_neighbors=5):
    """
    Applies data reduction and optional SMOTE for classification.

    Parameters:
    -----------
    X : array-like or DataFrame
        Feature matrix
    y : array-like or Series
        Target labels
    reduction_pct : float
        Percentage of data to REMOVE (0.0 = keep all, 0.5 = remove 50%)
    apply_smote : bool
        Whether to apply SMOTE oversampling
    random_state : int
        Random seed
    smote_k_neighbors : int
        Number of nearest neighbors for SMOTE

    Returns:
    --------
    X_balanced, y_balanced : Processed features and labels
    """
    n_original = len(y)
    retention_pct = 1.0 - reduction_pct

    # Convert to appropriate format
    X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
    y_series = y if isinstance(y, pd.Series) else pd.Series(y)

    # Step 1: Data Reduction
    if retention_pct < 1.0:
        n_keep = int(n_original * retention_pct)
        print(f"  [Simulator] Removing {int(reduction_pct * 100)}%: {n_original} -> {n_keep} samples")
        indices = np.random.RandomState(random_state).choice(
            n_original, n_keep, replace=False
        )
        X_used = X_df.iloc[indices]
        y_used = y_series.iloc[indices]
    else:
        print(f"  [Simulator] Using Full Data (No Reduction). {n_original} samples")
        X_used, y_used = X_df, y_series

    # Print distribution before SMOTE
    print_target_balance(y_used)

    # Step 2: SMOTE Oversampling
    if apply_smote:
        print(f"  [SMOTE] Applying oversampling (k_neighbors={smote_k_neighbors})...")
        smote = SMOTE(
            random_state=random_state,
            k_neighbors=min(smote_k_neighbors, len(y_used) - 1)
        )
        X_balanced, y_balanced = smote.fit_resample(X_used, y_used)
        print_target_balance(y_balanced)
        print(f"  [SMOTE] Final size: {len(y_balanced)} samples")
    else:
        print("  [SMOTE] Skipped.")
        X_balanced, y_balanced = X_used, y_used

    # Return in same format as input
    if isinstance(X, np.ndarray):
        return X_balanced.values, y_balanced.values
    return X_balanced, y_balanced


def simulator_module(X, y, reduction_pct=0.0, apply_smote=False,
                     random_state=42, smote_k_neighbors=5):
    """
    Main simulator entry point for classification.

    Parameters:
    -----------
    X : array-like or DataFrame
        Feature matrix
    y : array-like or Series
        Target labels
    reduction_pct : float
        Percentage of data to REMOVE (0.0 = keep all)
    apply_smote : bool
        Whether to apply SMOTE
    random_state : int
        Random seed
    smote_k_neighbors : int
        SMOTE k_neighbors parameter

    Returns:
    --------
    X_sim, y_sim : Simulated features and labels
    """
    return balance_dataset(X, y, reduction_pct, apply_smote, random_state, smote_k_neighbors)
