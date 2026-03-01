"""
SIMULATOR MODULE (REGRESSION)
=========================================
"""

import numpy as np
import pandas as pd

def smogn_oversample(X, y, random_state=42, low_percentile=5, high_percentile=95, smogn_limit=1.0):
    """
    Fixed SMOGN: Prevents 'Cross-Tail' pollution and caps synthetic generation.
    
    Args:
        smogn_limit (float): The maximum ratio of synthetic samples to add relative 
                             to the original rare count. 
                             1.0 = Max 1 fake per 1 real (Doubling).
                             0.5 = Max 0.5 fake per 1 real (50% increase).
    """
    np.random.seed(random_state)

    # 1. Setup Data
    numeric_indices = []
    if hasattr(X, "dtypes"):
        for i, dtype in enumerate(X.dtypes):
            if pd.api.types.is_numeric_dtype(dtype):
                numeric_indices.append(i)
    else:
        if len(X) > 0:
            sample_row = X[0] if hasattr(X, "iloc") else X[0]
            numeric_indices = [i for i, val in enumerate(sample_row) if isinstance(val, (int, float, np.number))]

    cols = X.columns if hasattr(X, "columns") else None
    y_name = getattr(y, "name", "target")

    X_arr = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
    y_arr = y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)

    # 2. Define Rare Regions Separately
    q_low = np.percentile(y_arr, low_percentile)
    q_high = np.percentile(y_arr, high_percentile)

    # Split into Low-Rare and High-Rare buckets
    mask_low = y_arr <= q_low
    mask_high = y_arr >= q_high
    mask_normal = ~(mask_low | mask_high)

    X_low, y_low = X_arr[mask_low], y_arr[mask_low]
    X_high, y_high = X_arr[mask_high], y_arr[mask_high]
    
    # Just for counting
    n_normal = np.sum(mask_normal)
    n_rare = len(y_low) + len(y_high)

    print(f"  [SMOGN] Definition: Low <= {q_low:.2f}, High >= {q_high:.2f}")
    print(f"  [SMOGN] Pre-Sampling: Low={len(y_low)}, High={len(y_high)} (Total Rare={n_rare}) | Normal={n_normal}")

    if n_rare == 0:
        print("  [SMOGN] No rare samples found.")
        return X, y

    # 3. Calculate Needed Samples (CONSERVATIVE APPROACH)
    # Goal: Try to match n_normal, BUT do not exceed smogn_limit * n_rare
    
    needed_to_balance = n_normal - n_rare
    safety_cap = int(n_rare * smogn_limit) 

    if needed_to_balance <= 0:
        print("  [SMOGN] Rare class is already larger or equal to Normal. No sampling needed.")
        return X, y

    # We take the smaller of the two: don't balance if it requires absurd oversampling
    n_synthetic = min(needed_to_balance, safety_cap)

    print(f"  [SMOGN] Target: {needed_to_balance} to balance.")
    print(f"  [SMOGN] Cap: {safety_cap} (Limit set to {smogn_limit*100}% of rare).")
    print(f"  [SMOGN] Generating: {n_synthetic} synthetic samples.")

    if n_synthetic <= 0:
        return X, y

    synthetic_X = []
    synthetic_y = []
    num_idx_set = set(numeric_indices)
    n_cols = X_arr.shape[1]

    # 4. Smart Generation Loop
    for _ in range(n_synthetic):
        # Decide which tail to boost based on probability (size of tail)
        if len(y_low) > 0 and len(y_high) > 0:
            prob_low = len(y_low) / n_rare
            target_bucket = "low" if np.random.random() < prob_low else "high"
        elif len(y_low) > 0:
            target_bucket = "low"
        else:
            target_bucket = "high"

        # Select parents strictly from the same bucket
        if target_bucket == "low":
            subset_X, subset_y = X_low, y_low
        else:
            subset_X, subset_y = X_high, y_high

        # Pick two parents from this specific subset
        if len(subset_y) < 2:
             # Can't interpolate with 1 sample, just duplicate it
             idx1 = 0
             idx2 = 0
        else:
             idx1, idx2 = np.random.choice(len(subset_y), 2, replace=True)

        alpha = np.random.random()
        new_X = np.empty(n_cols, dtype=object)
        row1, row2 = subset_X[idx1], subset_X[idx2]

        for i in range(n_cols):
            if i in num_idx_set:
                try:
                    val1, val2 = float(row1[i]), float(row2[i])
                    new_X[i] = val1 + alpha * (val2 - val1)
                except:
                    new_X[i] = row1[i]
            else:
                new_X[i] = row1[i]

        new_y_val = subset_y[idx1] + alpha * (subset_y[idx2] - subset_y[idx1])
        
        synthetic_X.append(new_X)
        synthetic_y.append(new_y_val)

    # 5. Stack & Finish
    X_resampled = np.vstack([X_arr, np.array(synthetic_X)])
    y_resampled = np.concatenate([y_arr, np.array(synthetic_y)])

    # Verify Logic
    rare_mask_new = (y_resampled <= q_low) | (y_resampled >= q_high)
    n_rare_new = np.sum(rare_mask_new)
    
    print(f"  [SMOGN] Post-Sampling: Rare={n_rare_new} ({n_rare_new/len(y_resampled):.1%}) | Total={len(y_resampled)}")

    if cols is not None:
        X_resampled = pd.DataFrame(X_resampled, columns=cols).infer_objects()
        y_resampled = pd.Series(y_resampled, name=y_name)

    return X_resampled, y_resampled

def balance_dataset(X, y, reduction_pct=1.0, apply_smogn=False, random_state=42, smogn_limit=1.0):
    """
    Applies data reduction and optional SMOGN.
    """
    n_original = len(y)
    retention_pct = 1.0 - reduction_pct
    
    if retention_pct < 1.0:
        n_keep = int(n_original * retention_pct)
        print(f"  [Simulator] Removing {int(reduction_pct * 100)}% of data: {n_original} -> {n_keep} samples")
        indices = np.random.RandomState(random_state).choice(n_original, n_keep, replace=False)
        X_used = X.iloc[indices] if hasattr(X, 'iloc') else X[indices]
        y_used = y.iloc[indices] if hasattr(y, 'iloc') else y[indices]
    else:
        print(f"  [Simulator] Using Full Data (No Reduction).")
        X_used, y_used = X, y
    
    if apply_smogn:
        # Pass the new smogn_limit parameter
        return smogn_oversample(X_used, y_used, random_state, low_percentile=5, high_percentile=95, smogn_limit=smogn_limit)
    else:
        print("  [Simulator] SMOGN Skipped.")
        return X_used, y_used


def simulator_module(X, y, reduction_pct=0.0, apply_smogn=False, random_state=42, smogn_limit=1.0):
    """
    Main simulator entry point. 
    smogn_limit: 1.0 = Max 100% growth (doubling). 0.5 = Max 50% growth.
    """
    return balance_dataset(X, y, reduction_pct, apply_smogn, random_state, smogn_limit)