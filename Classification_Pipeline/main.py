"""Main entry point for running modular classification experiments."""

import argparse
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

from src.pipeline import run_path
from src.evaluation import format_results, print_class_balance


def main():
    parser = argparse.ArgumentParser(
        description="Run Modular Classification Pipeline with TabPFN"
    )

    # Data arguments
    parser.add_argument("--data", type=str, required=True,
                        help="Path to .csv dataset")
    parser.add_argument("--target", type=str, required=True,
                        help="Target column name")

    # Method Selection
    parser.add_argument("--method", type=str, default="TABPFN",
                        choices=['TABPFN', 'GRACES', 'DEEPFS',
                                 'XGB', 'LR',
                                 'GRACES_XGB', 'DEEPFS_XGB',
                                 'GRACES_LR', 'DEEPFS_LR',
                                 'ALL'],
                        help="Modeling method to execute (or 'ALL' to run multiple models)")

    # Simulator Settings
    parser.add_argument("--reduction", type=float, default=0.0,
                        help="Single reduction percentage (0.0 = keep all). "
                             "Percentage of data to REMOVE.")

    parser.add_argument("--reductions", type=str, default=None,
                        help="Comma-separated list of reduction percentages "
                             "(e.g., '0.0,0.3,0.6'). Overrides --reduction.")

    # SMOTE Flag
    parser.add_argument("--smote", action='store_true',
                        help="If set, runs BOTH the standard variant AND the SMOTE variant "
                             "for every reduction.")

    # Feature Selection Settings
    parser.add_argument("--features", type=int, default=100,
                        help="Number of features to select (for GRACES/DeepFS)")

    # Other Settings
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Test set size for train/test split (default: 0.2)")
    parser.add_argument("--random-state", type=int, default=42,
                        help="Random seed for reproducibility")

    args = parser.parse_args()

    print(f"\nLOADING DATA: {args.data}")
    try:
        df = pd.read_csv(args.data)
    except FileNotFoundError:
        print("Error: File not found.")
        sys.exit(1)

    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns[:10])}{'...' if len(df.columns) > 10 else ''}")

    if args.target not in df.columns:
        print(f"Error: Target '{args.target}' not found in dataset.")
        sys.exit(1)

    X = df.drop(columns=[args.target])
    y = df[args.target]

    print(f"Features: {X.shape[1]}")
    print(f"Samples: {len(y)}")

    print(f"\nTRAIN/TEST SPLIT")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y
    )

    print(f"Train samples: {len(y_train)}")
    print(f"Test samples: {len(y_test)}")

    print_class_balance(y_train, "Training Set")
    print_class_balance(y_test, "Test Set")

    # Parse reductions list
    if args.reductions:
        try:
            reduction_list = [float(r.strip()) for r in args.reductions.split(',')]
        except ValueError:
            print("Error: --reductions must be a comma-separated list of numbers.")
            sys.exit(1)
    else:
        reduction_list = [args.reduction]

    smote_variants = [False, True] if args.smote else [False]
    if args.method == "ALL":
        methods_to_run = [
            "TABPFN",
            "GRACES",
            "DEEPFS",
            "XGB",
            "GRACES_XGB",
            "DEEPFS_XGB"
        ]
    else:
        methods_to_run = [args.method]

    total_experiments = len(methods_to_run) * len(reduction_list) * len(smote_variants)

    print(f"\nEXECUTION PLAN:")
    print(f"  Methods:     {methods_to_run}")
    print(f"  Reductions:  {reduction_list}")
    print(f"  SMOTE Mode:  {'Standard + SMOTE' if args.smote else 'Standard Only'}")
    print(f"  N Features:  {args.features}")
    print(f"  Total Experiments: {total_experiments}")

    all_results = []
    experiment_count = 0

    for method in methods_to_run:
        for reduction_pct in reduction_list:
            for apply_smote in smote_variants:
                experiment_count += 1
                print(f"\n[Experiment {experiment_count}/{total_experiments}]")

                results_df = run_path(
                    X_train, y_train,
                    X_test, y_test,
                    method=method,
                    reduction_pct=reduction_pct,
                    apply_smote=apply_smote,
                    n_features=args.features,
                    random_state=args.random_state
                )

                if not results_df.empty:
                    all_results.append(results_df)

    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)

        # Sort by F1_Weighted (descending - higher is better)
        sorted_results = format_results(
            combined_results.to_dict('records'),
            sort_by='F1_Weighted',
            ascending=False
        )

        print("\nFINAL METRICS (ALL VARIANTS) - Sorted by F1 Weighted")
        print(sorted_results.to_string(index=False))

        best = sorted_results.iloc[0]
        print("\nBEST MODEL SUMMARY:")
        print(f"  Method:       {best['Method']}")
        print(f"  Data Variant: {best['Data_Variant']}")
        print(f"  Accuracy:     {best['Accuracy']:.4f}")
        print(f"  F1:           {best['F1']:.4f}")
        print(f"  F1 Weighted:  {best['F1_Weighted']:.4f}")
        if 'AUC' in best:
            print(f"  AUC:          {best['AUC']:.4f}")

        output_path = "classification_results.csv"
        sorted_results.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")
    else:
        print("\nNo results generated.")


if __name__ == "__main__":
    main()
