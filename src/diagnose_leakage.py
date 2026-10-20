import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored

# -------------------------------------------------------------
# 1. LOAD YOUR DATA
# -------------------------------------------------------------
# Replace with your actual data loading step
# X: pandas DataFrame of features
# time: numpy array of survival times
# status: numpy array of event indicators (1/0 or True/False)
# patient_ids: numpy array of patient IDs

def audit_features(X_df):
    """Checks if target/metadata variables were accidentally left in X."""
    suspicious_terms = ['time', 'event', 'status', 'days', 'surv', 'vital', 'death', 'fu', 'followup', 'id']
    flagged = [col for col in X_df.columns if any(term in str(col).lower() for term in suspicious_terms)]
    
    print("=== DATA AUDIT ===")
    if flagged:
        print(f"⚠️ WARNING: Found {len(flagged)} potentially leaky metadata columns in feature matrix:")
        print(flagged[:10]) # Show first 10
    else:
        print("✓ No obvious metadata/target terms found in feature column names.")
    print("==================\n")


def run_permutation_test(X, time, status, fit_model_fn):
    """
    Shuffles target labels. 
    If C-Index remains > 0.55 on shuffled data, DATA LEAKAGE IS PRESENT.
    """
    print("=== RUNNING PERMUTATION TEST ===")
    y_shuffled_status = status.copy()
    y_shuffled_time = time.copy()
    
    # Randomly shuffle targets to destroy true biological signal
    np.random.shuffle(y_shuffled_status)
    np.random.shuffle(y_shuffled_time)
    
    c_indices = []
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        t_train, t_test = y_shuffled_time[train_idx], y_shuffled_time[test_idx]
        s_train, s_test = y_shuffled_status[train_idx], y_shuffled_status[test_idx]
        
        # Train and predict using your workflow function
        risk_scores = fit_model_fn(X_train, t_train, s_train, X_test)
        
        c_index = concordance_index_censored(s_test.astype(bool), t_test, risk_scores)[0]
        c_indices.append(c_index)
        
    mean_c_index = np.mean(c_indices)
    print(f"Permuted Target C-Index: {mean_c_index:.4f}")
    
    if mean_c_index > 0.55:
        print("❌ LEAKAGE DETECTED: Model achieves above-random performance on fake, shuffled labels!")
        print("   -> Check feature selection or scaling happening outside cross-validation loops.")
    else:
        print("✓ PASSED: Permuted labels yielded ~0.50 C-Index. Pipeline handles labels safely.")
    print("================================\n")

# Example placeholder execution:
# audit_features(X_df)
# run_permutation_test(X_matrix, survival_times, event_statuses, my_training_function)
