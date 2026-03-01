"""Integrates Simulator -> Path Selection -> Evaluation."""

import pandas as pd
import numpy as np
from src.simulator import simulator_module
from src.evaluation import evaluation_module

# Models
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

try:
    from tabpfn import TabPFNClassifier
    TABPFN_AVAILABLE = True
except ImportError:
    TABPFN_AVAILABLE = False
    print("Warning: tabpfn not available. TABPFN paths will be disabled.")


def run_path(X_train, y_train, X_test, y_test,
             method="TABPFN",
             reduction_pct=0.0,
             apply_smote=False,
             n_features=100,
             n_classes=2,
             random_state=42):
    """
    Execute a single specific modeling path with Data Simulation.

    Parameters:
    -----------
    X_train : array-like or DataFrame
        Training features
    y_train : array-like or Series
        Training labels
    X_test : array-like or DataFrame
        Test features
    y_test : array-like or Series
        Test labels
    method : str
        Modeling method to use. Options:
        - 'TABPFN': Pure TabPFN
        - 'GRACES': GRACES selection -> TabPFN
        - 'DEEPFS': DeepFS selection -> TabPFN
        - 'GRACES_XGB': GRACES selection -> XGBoost
        - 'DEEPFS_XGB': DeepFS selection -> XGBoost
        - 'XGB': Baseline XGBoost
        - 'LR': Baseline Logistic Regression
        - 'GRACES_LR': GRACES selection -> Logistic Regression
        - 'DEEPFS_LR': DeepFS selection -> Logistic Regression
    reduction_pct : float
        Percentage of data to REMOVE (0.0 = keep all, 0.5 = remove 50%)
    apply_smote : bool
        Whether to apply SMOTE oversampling
    n_features : int
        Number of features to select (for GRACES/DeepFS)
    n_classes : int
        Number of classes in the target
    random_state : int
        Random seed

    Returns:
    --------
    DataFrame : Results dataframe
    """

    sim_status = f"{int(reduction_pct*100)}% Data Removed"
    if apply_smote:
        sim_status += " + SMOTE"
    else:
        sim_status += " (No SMOTE)"

    print(f"\nRUNNING PATH: {method} | {sim_status}")

    results = []

    try:
        X_sim, y_sim = simulator_module(
            X_train, y_train,
            reduction_pct=reduction_pct,
            apply_smote=apply_smote,
            random_state=random_state
        )

        if not isinstance(X_sim, pd.DataFrame):
            cols = [f"feat_{i}" for i in range(X_sim.shape[1])]
            X_sim_df = pd.DataFrame(X_sim, columns=cols)
            if not isinstance(X_test, pd.DataFrame):
                X_test_df = pd.DataFrame(X_test, columns=cols)
            else:
                X_test_df = X_test.copy()
                X_test_df.columns = cols
        else:
            X_sim_df = X_sim.copy()
            X_test_df = X_test.copy()

        if method == "TABPFN":
            if not TABPFN_AVAILABLE:
                print("  [Error] TabPFN not available. Install with: pip install tabpfn")
                return pd.DataFrame()
            print("  [Logic] Pure TabPFN (Direct usage)")
            model = TabPFNClassifier(device='cpu', n_estimators=32, random_state=random_state)
            model.fit(X_sim_df, y_sim)
            preds = model.predict(X_test_df)
            probs = model.predict_proba(X_test_df)[:, 1] if n_classes == 2 else None

        elif method == "GRACES":
            if not TABPFN_AVAILABLE:
                print("  [Error] TabPFN not available.")
                return pd.DataFrame()
            print(f"  [Logic] GRACES Selection (Target: {n_features} feats) -> TabPFN")

            from src.toolbox import get_toolbox
            toolbox = get_toolbox(n_features=n_features, selection_method='graces')
            selector = toolbox["GRACES"]

            selector.fit(X_sim_df, y_sim)
            X_train_sel = selector.transform(X_sim_df)
            X_test_sel = selector.transform(X_test_df)
            print(f"  [Status] Selected {X_train_sel.shape[1]} features.")

            print("  [Logic] Fitting TabPFN Classifier...")
            model = TabPFNClassifier(device='cpu', n_estimators=32, random_state=random_state)
            model.fit(X_train_sel, y_sim)
            preds = model.predict(X_test_sel)
            probs = model.predict_proba(X_test_sel)[:, 1] if n_classes == 2 else None

        elif method == "DEEPFS":
            if not TABPFN_AVAILABLE:
                print("  [Error] TabPFN not available.")
                return pd.DataFrame()
            print(f"  [Logic] DeepFS Selection (Target: {n_features} feats) -> TabPFN")

            from src.toolbox import get_toolbox
            toolbox = get_toolbox(n_features=n_features, selection_method='deepfs')
            selector = toolbox["DEEPFS"]

            selector.fit(X_sim_df, y_sim)
            X_train_sel = selector.transform(X_sim_df)
            X_test_sel = selector.transform(X_test_df)
            print(f"  [Status] Selected {X_train_sel.shape[1]} features.")

            print("  [Logic] Fitting TabPFN Classifier...")
            model = TabPFNClassifier(device='cpu', n_estimators=32, random_state=random_state)
            model.fit(X_train_sel, y_sim)
            preds = model.predict(X_test_sel)
            probs = model.predict_proba(X_test_sel)[:, 1] if n_classes == 2 else None

        elif method == "XGB":
            print("  [Logic] Baseline XGBoost (No Selection)")

            neg_count = np.sum(y_sim == 0)
            pos_count = np.sum(y_sim == 1)
            scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

            model = XGBClassifier(
                n_estimators=1000,
                max_depth=4,
                learning_rate=0.01,
                subsample=0.8,
                scale_pos_weight=scale_pos_weight,
                eval_metric='logloss',
                random_state=random_state,
                verbosity=0
            )
            model.fit(X_sim_df, y_sim)
            preds = model.predict(X_test_df)
            probs = model.predict_proba(X_test_df)[:, 1] if n_classes == 2 else None

        elif method == "GRACES_XGB":
            print(f"  [Logic] GRACES Selection (Target: {n_features} feats) -> XGBoost")

            from src.toolbox import get_toolbox
            toolbox = get_toolbox(n_features=n_features, selection_method='graces')
            selector = toolbox["GRACES"]

            selector.fit(X_sim_df, y_sim)
            X_train_sel = selector.transform(X_sim_df)
            X_test_sel = selector.transform(X_test_df)
            print(f"  [Status] Selected {X_train_sel.shape[1]} features.")

            neg_count = np.sum(y_sim == 0)
            pos_count = np.sum(y_sim == 1)
            scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

            print("  [Logic] Fitting XGBoost Classifier...")
            model = XGBClassifier(
                n_estimators=1000,
                max_depth=4,
                learning_rate=0.01,
                subsample=0.8,
                scale_pos_weight=scale_pos_weight,
                eval_metric='logloss',
                random_state=random_state,
                verbosity=0
            )
            model.fit(X_train_sel, y_sim)
            preds = model.predict(X_test_sel)
            probs = model.predict_proba(X_test_sel)[:, 1] if n_classes == 2 else None

        elif method == "DEEPFS_XGB":
            print(f"  [Logic] DeepFS Selection (Target: {n_features} feats) -> XGBoost")

            from src.toolbox import get_toolbox
            toolbox = get_toolbox(n_features=n_features, selection_method='deepfs')
            selector = toolbox["DEEPFS"]

            selector.fit(X_sim_df, y_sim)
            X_train_sel = selector.transform(X_sim_df)
            X_test_sel = selector.transform(X_test_df)
            print(f"  [Status] Selected {X_train_sel.shape[1]} features.")

            neg_count = np.sum(y_sim == 0)
            pos_count = np.sum(y_sim == 1)
            scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

            print("  [Logic] Fitting XGBoost Classifier...")
            model = XGBClassifier(
                n_estimators=1000,
                max_depth=4,
                learning_rate=0.01,
                subsample=0.8,
                scale_pos_weight=scale_pos_weight,
                eval_metric='logloss',
                random_state=random_state,
                verbosity=0
            )
            model.fit(X_train_sel, y_sim)
            preds = model.predict(X_test_sel)
            probs = model.predict_proba(X_test_sel)[:, 1] if n_classes == 2 else None

        elif method == "LR":
            print("  [Logic] Baseline Logistic Regression (No Selection)")

            model = LogisticRegression(
                max_iter=3000,
                class_weight='balanced',
                random_state=random_state
            )
            model.fit(X_sim_df, y_sim)
            preds = model.predict(X_test_df)
            probs = model.predict_proba(X_test_df)[:, 1] if n_classes == 2 else None

        elif method == "GRACES_LR":
            print(f"  [Logic] GRACES Selection (Target: {n_features} feats) -> Logistic Regression")

            from src.toolbox import get_toolbox
            toolbox = get_toolbox(n_features=n_features, selection_method='graces')
            selector = toolbox["GRACES"]

            selector.fit(X_sim_df, y_sim)
            X_train_sel = selector.transform(X_sim_df)
            X_test_sel = selector.transform(X_test_sel)
            print(f"  [Status] Selected {X_train_sel.shape[1]} features.")

            print("  [Logic] Fitting Logistic Regression...")
            model = LogisticRegression(
                max_iter=3000,
                class_weight='balanced',
                random_state=random_state
            )
            model.fit(X_train_sel, y_sim)
            preds = model.predict(X_test_sel)
            probs = model.predict_proba(X_test_sel)[:, 1] if n_classes == 2 else None

        elif method == "DEEPFS_LR":
            print(f"  [Logic] DeepFS Selection (Target: {n_features} feats) -> Logistic Regression")

            from src.toolbox import get_toolbox
            toolbox = get_toolbox(n_features=n_features, selection_method='deepfs')
            selector = toolbox["DEEPFS"]

            selector.fit(X_sim_df, y_sim)
            X_train_sel = selector.transform(X_sim_df)
            X_test_sel = selector.transform(X_test_df)
            print(f"  [Status] Selected {X_train_sel.shape[1]} features.")

            print("  [Logic] Fitting Logistic Regression...")
            model = LogisticRegression(
                max_iter=3000,
                class_weight='balanced',
                random_state=random_state
            )
            model.fit(X_train_sel, y_sim)
            preds = model.predict(X_test_sel)
            probs = model.predict_proba(X_test_sel)[:, 1] if n_classes == 2 else None

        else:
            print(f"  [Error] Unknown method: {method}")
            return pd.DataFrame()

        variant_info = f"{int(reduction_pct*100)}%"
        if apply_smote:
            variant_info += "_SMOTE"

        metrics = evaluation_module(
            y_true=y_test,
            y_pred=preds,
            y_prob=probs,
            method_name=method,
            variant_info=variant_info,
            results_list=results
        )

        print(f"  [Result] Accuracy: {metrics['Accuracy']:.4f} | F1: {metrics['F1']:.4f} | F1_W: {metrics['F1_Weighted']:.4f}")

    except Exception as e:
        print(f"  [Error] Path execution failed: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

    return pd.DataFrame(results)
