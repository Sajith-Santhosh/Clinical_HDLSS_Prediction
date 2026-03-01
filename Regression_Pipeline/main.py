import pandas as pd
import numpy as np
import argparse
import sys
from sklearn.model_selection import train_test_split
from src.pipeline1 import run_path
from src.evaluation1 import format_results




def main():
    # 1. CLI Arguments Configuration
    parser = argparse.ArgumentParser(description="Run Modular Regression Pipeline")
    
    parser.add_argument("--data", type=str, required=True, help="Path to .csv dataset")
    parser.add_argument("--target", type=str, required=True, help="Target column name")
    
    # Method Selection
    parser.add_argument("--method", type=str, default="XGBOOST", 
                        choices=['TABPFN', 'GRACES', 'XGBOOST', 'DEEPFS', 'DEEPFS_TABPFN', 'GRACES_TABPFN', 'ALL'],
                        help="Modeling method to execute (or 'ALL' to run multiple models)")
    
    # Simulator Settings
    parser.add_argument("--reduction", type=float, default=0.0, 
                        help="Single reduction percentage (0.0 = keep all). Ignored if --reductions is set.")
    
    parser.add_argument("--reductions", type=str, default=None,
                        help="Comma-separated list of reduction percentages (e.g., '0.0,0.3,0.6'). Overrides --reduction.")
    
    # SMOGN Flag: The "Multiplier"
    parser.add_argument("--smogn", action='store_true', 
                        help="If set, runs BOTH the standard variant AND the SMOGN variant for every reduction.")
    
    # Feature Selection Settings
    parser.add_argument("--features", type=int, default=100, 
                        help="Number of features to select (for GRACES/DeepFS)")

    args = parser.parse_args()

    # 2. Load Data
    print(f"\nLOADING DATA: {args.data}")
    try:
        df = pd.read_csv(args.data)
    except FileNotFoundError:
        print("Error: File not found.")
        sys.exit(1)

    if args.target not in df.columns:
        print(f"Error: Target '{args.target}' not found in dataset.")
        sys.exit(1)

    X = df.drop(columns=[args.target])
    y = df[args.target]

    # 3. Prepare Split (Hold-out)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Print imbalance before SMOGN
    print_target_balance(y_train)

    # 4. Define Execution Plan
    # A. Reductions: Convert single or list into one standard list
    if args.reductions:
        try:
            reduction_list = [float(r.strip()) for r in args.reductions.split(',')]
        except ValueError:
            print("Error: --reductions must be a comma-separated list of numbers.")
            sys.exit(1)
    else:
        reduction_list = [args.reduction]

    # B. SMOGN Variants: 
    # If --smogn is ON: Run [False, True] (Standard + SMOGN)
    # If --smogn is OFF: Run [False] (Standard only)
    smogn_variants = [False, True] if args.smogn else [False]

    # C. Methods: Allow "ALL" to run the key competitors
    if args.method == "ALL":
        # methods_to_run = ["XGBOOST", "DEEPFS_TABPFN", "GRACES"]
        methods_to_run = ["XGBOOST", "DEEPFS_TABPFN"] # Stick to your top 2 for speed if preferred
    else:
        methods_to_run = [args.method]

    print(f"\n{'='*60}")
    print(f"EXECUTION PLAN:")
    print(f"  Methods:    {methods_to_run}")
    print(f"  Reductions: {reduction_list}")
    print(f"  SMOGN Mode: {'Standard + SMOGN' if args.smogn else 'Standard Only'}")
    print(f"{'='*60}")

    # 5. Execution Loop
    all_results = []
    
    for method in methods_to_run:
        for reduction_pct in reduction_list:
            for apply_smogn in smogn_variants:
                
                # Execute Path
                results_df = run_path(
                    X_train, y_train, 
                    X_test, y_test, 
                    method=method,
                    reduction_pct=reduction_pct,
                    apply_smogn=apply_smogn,
                    n_features=args.features
                )
                
                if not results_df.empty:
                    all_results.append(results_df)

    # 6. Output Final Results
    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)
        print("\n" + "="*60)
        print("FINAL METRICS (ALL VARIANTS):")
        print("="*60)
        # Sort by RMSE (Ascending) so best models are at top
        print(format_results(combined_results.to_dict('records'), sort_by='RMSE').to_string(index=False))
    else:
        print("\nNo results generated.")


# Heloer class for printing target balance for SMOGN
def print_target_balance(y, bins=5):
    """Print class/target imbalance summary."""
    if y.dtype == object or y.nunique() <= 20:
        counts = y.value_counts().sort_index()
    else:
        counts = pd.cut(y, bins=bins).value_counts().sort_index()
    total = len(y)
    print("\nTARGET IMBALANCE (before SMOGN):")
    for label, cnt in counts.items():
        pct = 100 * cnt / total
        print(f"  {label}: {cnt} ({pct:.1f}%)")



if __name__ == "__main__":
    main()
    