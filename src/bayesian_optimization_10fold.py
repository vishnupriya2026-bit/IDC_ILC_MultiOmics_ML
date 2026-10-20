import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import optuna

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

# Suppress Optuna verbose trial logs
optuna.logging.set_verbosity(optuna.logging.WARNING)
np.random.seed(42)

# 1. Load Data
X_train = pd.read_csv("diablo_exports/X_diablo_train_scaled.csv", index_col=0)
y_train_df = pd.read_csv("diablo_exports/y_diablo_train_encoded.csv", index_col=0)

X_test = pd.read_csv("diablo_exports/X_diablo_test_scaled.csv", index_col=0)
y_test_df = pd.read_csv(
    "diablo_exports/y_test_encoded.csv" if "y_test_encoded.csv" in y_train_df.columns 
    else "diablo_exports/y_diablo_test_encoded.csv", index_col=0
)

# Deduplicate indices
y_train_df = y_train_df[~y_train_df.index.duplicated(keep='first')]
y_test_df = y_test_df[~y_test_df.index.duplicated(keep='first')]

y_train_raw = y_train_df.reindex(X_train.index).iloc[:, 0].astype(str)
y_test_raw = y_test_df.reindex(X_test.index).iloc[:, 0].astype(str)

# Combine into total cohort (N=410)
X_all = pd.concat([X_train, X_test], axis=0)
y_all_raw = pd.concat([y_train_raw, y_test_raw], axis=0)

le = LabelEncoder()
y_all = le.fit_transform(y_all_raw)

print(f"Loaded BRCA Cohort: N={X_all.shape[0]} samples, {X_all.shape[1]} features.")

# Setup 10-Fold Stratified Cross-Validation
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scale_pos_weight = (len(y_all) - sum(y_all)) / sum(y_all)

# 2. Objective Functions for Bayesian Optimization
def evaluate_10fold(model_fn):
    train_aucs, val_aucs = [], []
    for train_idx, val_idx in skf.split(X_all, y_all):
        X_tr, X_val = X_all.iloc[train_idx], X_all.iloc[val_idx]
        y_tr, y_val = y_all[train_idx], y_all[val_idx]
        
        clf = model_fn()
        clf.fit(X_tr, y_tr)
        
        p_tr = clf.predict_proba(X_tr)[:, 1]
        p_val = clf.predict_proba(X_val)[:, 1]
        
        train_aucs.append(roc_auc_score(y_tr, p_tr))
        val_aucs.append(roc_auc_score(y_val, p_val))
        
    return np.mean(train_aucs), np.mean(val_aucs)

# Model 1: Random Forest Objective
def objective_rf(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 250, step=25),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
        'max_features': trial.suggest_float('max_features', 0.1, 0.6)
    }
    model_fn = lambda: RandomForestClassifier(**params, class_weight='balanced', random_state=42)
    _, val_auc = evaluate_10fold(model_fn)
    return val_auc

# Model 2: XGBoost Objective
def objective_xgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 200, step=25),
        'max_depth': trial.suggest_int('max_depth', 2, 6),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 0.8),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0, log=True)
    }
    model_fn = lambda: xgb.XGBClassifier(**params, scale_pos_weight=scale_pos_weight, random_state=42, eval_metric='logloss')
    _, val_auc = evaluate_10fold(model_fn)
    return val_auc

# Model 3: LightGBM Objective
def objective_lgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 200, step=25),
        'max_depth': trial.suggest_int('max_depth', 2, 6),
        'num_leaves': trial.suggest_int('num_leaves', 7, 31),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 0.8),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0, log=True)
    }
    model_fn = lambda: lgb.LGBMClassifier(**params, class_weight='balanced', random_state=42, verbose=-1)
    _, val_auc = evaluate_10fold(model_fn)
    return val_auc

# Model 4: CatBoost Objective
def objective_cat(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 50, 200, step=25),
        'depth': trial.suggest_int('depth', 2, 6),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0, log=True)
    }
    model_fn = lambda: CatBoostClassifier(**params, auto_class_weights='Balanced', random_seed=42, verbose=0)
    _, val_auc = evaluate_10fold(model_fn)
    return val_auc

objectives = {
    "Random Forest": (objective_rf, lambda p: RandomForestClassifier(**p, class_weight='balanced', random_state=42)),
    "XGBoost": (objective_xgb, lambda p: xgb.XGBClassifier(**p, scale_pos_weight=scale_pos_weight, random_state=42, eval_metric='logloss')),
    "LightGBM": (objective_lgb, lambda p: lgb.LGBMClassifier(**p, class_weight='balanced', random_state=42, verbose=-1)),
    "CatBoost": (objective_cat, lambda p: CatBoostClassifier(**p, auto_class_weights='Balanced', random_seed=42, verbose=0))
}

results_list = []
best_hyperparams = {}

print("\nRunning Bayesian Optimization (20 Trials per Model across 10 Stratified Folds)...")

for model_name, (obj_fn, builder_fn) in objectives.items():
    print(f" -> Optimizing {model_name}...")
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(obj_fn, n_trials=20)
    
    best_params = study.best_params
    best_hyperparams[model_name] = best_params
    
    # Re-evaluate final model on 10 folds to extract both Train and Val metrics
    final_builder = lambda: builder_fn(best_params)
    train_auc, val_auc = evaluate_10fold(final_builder)
    gap = train_auc - val_auc
    
    fitting_status = "Optimal Fit" if gap <= 0.08 else ("Moderate Overfit" if gap <= 0.15 else "Severe Overfit")
    
    results_list.append({
        "Model": model_name,
        "Best 10-Fold Train AUC": round(train_auc, 4),
        "Best 10-Fold Val AUC": round(val_auc, 4),
        "Overfitting Gap (Train - Val)": round(gap, 4),
        "Fitting Diagnosis": fitting_status
    })

# 3. Save Summary and JSON Configuration
summary_df = pd.DataFrame(results_list).sort_values(by="Best 10-Fold Val AUC", ascending=False)
summary_df.to_csv("bayesian_optimization_results.csv", index=False)

with open("best_bayesian_hyperparameters.json", "w") as f:
    json.dump(best_hyperparams, f, indent=4)

print("\n" + "="*85)
print("             BAYESIAN OPTIMIZATION & 10-FOLD CV MODEL FITTING SUMMARY             ")
print("="*85)
print(summary_df.to_string(index=False))
print("="*85 + "\n")

# 4. Plot Train vs Validation Fitting Comparison
plot_df = summary_df.melt(
    id_vars=['Model'], 
    value_vars=['Best 10-Fold Train AUC', 'Best 10-Fold Val AUC'],
    var_name='Cohort', 
    value_name='ROC-AUC Score'
)

plt.figure(figsize=(9, 5), dpi=300)
sns.barplot(data=plot_df, x='Model', y='ROC-AUC Score', hue='Cohort', palette='Set2')
plt.ylim(0.5, 1.05)
plt.title('10-Fold Stratified Cross-Validation: Train vs Validation ROC-AUC (Bayesian Tuned)')
plt.ylabel('Mean ROC-AUC Score')
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig("bayesian_model_fitting.png")
print("Plot saved to 'bayesian_model_fitting.png' and best parameters saved to 'best_bayesian_hyperparameters.json'.")

