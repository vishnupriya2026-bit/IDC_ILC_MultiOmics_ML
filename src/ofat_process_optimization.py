import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

np.random.seed(42)

# 1. Load Data & Baseline Optimal Hyperparameters
X_train = pd.read_csv("diablo_exports/X_diablo_train_scaled.csv", index_col=0)
y_train_df = pd.read_csv("diablo_exports/y_diablo_train_encoded.csv", index_col=0)

X_test = pd.read_csv("diablo_exports/X_diablo_test_scaled.csv", index_col=0)
y_test_df = pd.read_csv(
    "diablo_exports/y_test_encoded.csv" if "y_test_encoded.csv" in y_train_df.columns 
    else "diablo_exports/y_diablo_test_encoded.csv", index_col=0
)

y_train_df = y_train_df[~y_train_df.index.duplicated(keep='first')]
y_test_df = y_test_df[~y_test_df.index.duplicated(keep='first')]

y_train_raw = y_train_df.reindex(X_train.index).iloc[:, 0].astype(str)
y_test_raw = y_test_df.reindex(X_test.index).iloc[:, 0].astype(str)

X_all = pd.concat([X_train, X_test], axis=0)
y_all_raw = pd.concat([y_train_raw, y_test_raw], axis=0)

le = LabelEncoder()
y_all = le.fit_transform(y_all_raw)

with open("best_bayesian_hyperparameters.json", "r") as f:
    best_params_dict = json.load(f)

baseline_params = best_params_dict.get("Random Forest", {
    "n_estimators": 150,
    "max_depth": 5,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": 0.3
})

# Ensure standard keys exist in baseline dictionary
if 'min_samples_split' not in baseline_params: baseline_params['min_samples_split'] = 2
if 'min_samples_leaf' not in baseline_params: baseline_params['min_samples_leaf'] = 1
if 'max_features' not in baseline_params: baseline_params['max_features'] = 'sqrt'

print(f"Loaded Baseline Champion Parameters: {baseline_params}")

# 2. Setup 10-Fold Stratified Cross-Validation
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

def evaluate_10fold_cv(params):
    train_aucs, val_aucs = [], []
    for train_idx, val_idx in skf.split(X_all, y_all):
        X_tr, X_val = X_all.iloc[train_idx], X_all.iloc[val_idx]
        y_tr, y_val = y_all[train_idx], y_all[val_idx]
        
        clf = RandomForestClassifier(**params, class_weight='balanced', random_state=42)
        clf.fit(X_tr, y_tr)
        
        p_tr = clf.predict_proba(X_tr)[:, 1]
        p_val = clf.predict_proba(X_val)[:, 1]
        
        train_aucs.append(roc_auc_score(y_tr, p_tr))
        val_aucs.append(roc_auc_score(y_val, p_val))
        
    return np.mean(train_aucs), np.mean(val_aucs)

# 3. Define OFAT Factor Grids
ofat_factors = {
    'n_estimators': [25, 50, 75, 100, 150, 200, 250, 300, 400],
    'max_depth': [2, 3, 4, 5, 6, 8, 10, 15, None],
    'max_features': [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.70, 1.00],
    'min_samples_leaf': [1, 2, 3, 5, 8, 10, 15, 20]
}

all_ofat_results = []

print("\nExecuting One-Factor-At-A-Time (OFAT) Sensitivity Analysis across 10 Folds...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
axes = axes.flatten()

for idx, (factor_name, factor_values) in enumerate(ofat_factors.items()):
    factor_records = []
    
    for val in factor_values:
        current_params = baseline_params.copy()
        current_params[factor_name] = val
        
        tr_auc, val_auc = evaluate_10fold_cv(current_params)
        gap = tr_auc - val_auc
        
        val_str = "None" if val is None else str(val)
        
        record = {
            'Factor': factor_name,
            'Value': val_str,
            'Train_AUC': round(tr_auc, 4),
            'Val_AUC': round(val_auc, 4),
            'Overfitting_Gap': round(gap, 4)
        }
        factor_records.append(record)
        all_ofat_results.append(record)
    
    df_factor = pd.DataFrame(factor_records)
    
    # Plot factor sensitivity curve
    ax = axes[idx]
    x_labels = [str(v) for v in factor_values]
    ax.plot(x_labels, df_factor['Train_AUC'], marker='o', color='crimson', lw=2, label='Train ROC-AUC')
    ax.plot(x_labels, df_factor['Val_AUC'], marker='s', color='navy', lw=2, label='10-Fold Val ROC-AUC')
    
    # Fill overfitting gap area
    ax.fill_between(x_labels, df_factor['Val_AUC'], df_factor['Train_AUC'], alpha=0.15, color='orange', label='Overfitting Gap')
    
    ax.set_title(f"OFAT Factor: {factor_name}", fontsize=12, fontweight='bold')
    ax.set_xlabel(f"Parameter Value ({factor_name})", fontsize=10)
    ax.set_ylabel("ROC-AUC Score", fontsize=10)
    ax.set_ylim(0.70, 1.02)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(loc='lower right', fontsize=8)

plt.tight_layout()
plt.savefig("ofat_process_optimization.png")

# Save detailed OFAT results table
df_ofat_summary = pd.DataFrame(all_ofat_results)
df_ofat_summary.to_csv("ofat_process_optimization.csv", index=False)

print("\n" + "="*85)
print("              ONE-FACTOR-AT-A-TIME (OFAT) PROCESS SENSITIVITY SUMMARY           ")
print("="*85)

for factor_name in ofat_factors.keys():
    print(f"\n--- Factor: {factor_name} ---")
    sub_df = df_ofat_summary[df_ofat_summary['Factor'] == factor_name][['Value', 'Train_AUC', 'Val_AUC', 'Overfitting_Gap']]
    print(sub_df.to_string(index=False))

print("="*85 + "\n")
print("OFAT Process Optimization complete. Saved curves to 'ofat_process_optimization.png' and metrics to 'ofat_process_optimization.csv'.")

