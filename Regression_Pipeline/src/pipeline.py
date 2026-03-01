"""
PIPELINE MODULE (3-PATH + SIMULATION)
=====================================
Integrates Simulator -> Path Selection -> Evaluation.
"""
import pandas as pd
import numpy as np
from src.toolbox1 import get_toolbox
from src.simulator1 import simulator_module
from src.evaluation1 import evaluation_module

# Models
from xgboost import XGBRegressor
from tabpfn import TabPFNRegressor

def run_path(X_train, y_train, X_test, y_test, 
             method="XGBOOST", 
             reduction_pct=0.0,  
             apply_smogn=False,
             n_features=50):
    """
    Executes a single specific modeling path with Data Simulation.
    
    Args:
        reduction_pct: Percentage of data to REMOVE (0.0 = keep all, 0.7 = remove 70%)
    """
    

    sim_status = f"{int(reduction_pct*100)}% Data Removed"
    if apply_smogn:
        sim_status += " + SMOGN"
    else:
        sim_status += " (No SMOGN)"
    
    print(f"\n{'='*60}")
    print(f"RUNNING PATH: {method} | {sim_status}")
    print(f"{'='*60}")
    
    results = []

    try:

        # 1. SIMULATOR: Reduction + Optional SMOGN (Reduction can be done with or without SMOGN and vice versa)

        X_sim, y_sim = simulator_module(
            X_train, y_train, 
            reduction_pct=reduction_pct, 
            apply_smogn=apply_smogn  # Pass control flag
        )

        # DataFrame safety for TabPFN/XGB
        if not isinstance(X_sim, pd.DataFrame):
            cols = [f"feat_{i}" for i in range(X_sim.shape[1])]
            X_sim_df = pd.DataFrame(X_sim, columns=cols)
            # Align Test columns
            if not isinstance(X_test, pd.DataFrame):
                X_test_df = pd.DataFrame(X_test, columns=cols)
            else:
                X_test_df = X_test.copy()
                X_test_df.columns = cols 
        else:
            X_sim_df = X_sim.copy()
            X_test_df = X_test.copy()


        # 2. Different MODEL EXECUTION PATHS 

        
        # PATH A: TABPFN (Pure)
        if method == "TABPFN":
            print("  [Logic] Pure TabPFN (Direct usage)")
            model = TabPFNRegressor(device='cpu', n_estimators=32, random_state=42)
            model.fit(X_sim_df, y_sim)
            preds = model.predict(X_test_df)


        # PATH B: GRACES (Feature Selection -> XGBOOST)
        elif method == "GRACES":
            print(f"  [Logic] GRACES Selection (Target: {n_features} feats)")
            
            toolbox = get_toolbox(n_features=n_features)
            selector = toolbox["GRACES"]
            
            selector.fit(X_sim_df, y_sim)
            
            X_train_sel = selector.transform(X_sim_df)
            X_test_sel = selector.transform(X_test_df)
            print(f"  [Status] Selected {X_train_sel.shape[1]} features.")
            
            print("  [Logic] Fitting XGBOOST Regressor...")
            model = XGBRegressor(n_estimators=1000, max_depth=4, learning_rate=0.01, random_state=42)
            model.fit(X_train_sel, y_sim)
            preds = model.predict(X_test_sel)

        # PATH C: DeepFS (Feature Selection -> XGBOOST)
        elif method == "DEEPFS":
            print(f"  [Logic] DeepFS Selection (Target: {n_features} feats)")
            
            toolbox = get_toolbox(n_features=n_features)
            selector = toolbox["DEEPFS"]
            
            selector.fit(X_sim_df, y_sim)
            
            X_train_sel = selector.transform(X_sim_df)
            X_test_sel = selector.transform(X_test_df)
            print(f"  [Status] Selected {X_train_sel.shape[1]} features.")
            
            print("  [Logic] Fitting XGBOOST Regressor...")
            model = XGBRegressor(n_estimators=1000, max_depth=4, learning_rate=0.01, subsample=0.8,random_state=42)
            model.fit(X_train_sel, y_sim)
            preds = model.predict(X_test_sel)

        # PATH D: BASELINE XGBOOST
        elif method == "XGBOOST":
            print("  [Logic] Baseline XGBOOST (No Selection)")
            
            model = XGBRegressor(n_estimators=1000, max_depth=4, learning_rate=0.01,subsample=0.8,random_state=42)
            model.fit(X_sim_df, y_sim)
            preds = model.predict(X_test_df)
            
# ...existing code...
# PATH E: HYBRID (DeepFS -> TabPFN)
        elif method == "DEEPFS_TABPFN":
            print(f"  [Logic] Hybrid: DeepFS Selection -> TabPFN Inference")
            
            # 1. Select best 50 features using DeepFS
            toolbox = get_toolbox(n_features=50) # Strict limit for TabPFN
            selector = toolbox["DEEPFS"]
            
            selector.fit(X_sim_df, y_sim)
            X_train_sel = selector.transform(X_sim_df)
            X_test_sel = selector.transform(X_test_df)
            
            print(f"  [Status] DeepFS reduced data to {X_train_sel.shape[1]} features.")
            
            # 2. Feed purified data to TabPFN
            # REMOVED DUPLICATE IMPORT HERE
            
            model = TabPFNRegressor(device='cpu', n_estimators=32, random_state=42)
            model.fit(X_train_sel, y_sim)
            preds = model.predict(X_test_sel)            

# PATH F: HYBRID (GRACES Selection -> TabPFN Inference)
        elif method == "GRACES_TABPFN":
            print(f"  [Logic] Hybrid: GRACES (GNN) Selection -> TabPFN Inference")
            
            # 1. Select best 50 features using GRACES (GNN)
            # TabPFN requires <100 features, so we force the toolbox to 50
            toolbox = get_toolbox(n_features=50) 
            selector = toolbox["GRACES"]
            
            print("  [Status] Training GNN for feature ranking...")
            selector.fit(X_sim_df, y_sim)
            
            X_train_sel = selector.transform(X_sim_df)
            X_test_sel = selector.transform(X_test_df)
            
            print(f"  [Status] GRACES reduced data to {X_train_sel.shape[1]} features.")
            
            # 2. Feed purified data to TabPFN
            # REMOVED DUPLICATE IMPORT HERE
            
            # Note: TabPFN is a Transformer that works best on small data (<2000 rows)
            model = TabPFNRegressor(device='cpu', n_estimators=32, random_state=42)
            model.fit(X_train_sel, y_sim)
            preds = model.predict(X_test_sel)

        # ---------------------------------------------------------
# ...existing code...


        # 3. MODULAR EVALUATION

        variant_info = f"{reduction_pct}"
        if apply_smogn:
            variant_info += "_SMOGN"

        metrics = evaluation_module(
            y_true=y_test, 
            y_pred=preds, 
            method_name=method, 
            variant_info=variant_info, 
            results_list=results
        )
        
        print(f"  [Result] RMSE: {metrics['RMSE']:.4f} | R2: {metrics['R2']:.4f}")

    except Exception as e:
        print(f"  [Error] Path execution failed: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

    return pd.DataFrame(results)