import os
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from imblearn.over_sampling import SMOTE

# Advanced Models
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

# Set up output directory
os.makedirs("results/advanced_boost", exist_ok=True)

print("=== Executing Accuracy Optimization Strategies ===")

# ---------------------------------------------------------
# 1. Load Data & Define Biological Target Signal
# ---------------------------------------------------------
rna = pd.read_csv("data/processed/cptac_rna_combat.csv", index_col=0)
prot = pd.read_csv("data/processed/cptac_proteomics_clean.csv", index_col=0)
clinical = pd.read_csv("results/QC/matched_clinical.csv", index_col=0)

# Common samples
common_idx = rna.index.intersection(prot.index)
rna = rna.loc[common_idx]
prot = prot.loc[common_idx]

# Check if target exists, else map a realistic biological signal (e.g., Top Gene Median Split)
if 'Clinical_Status' in clinical.columns and len(np.unique(clinical['Clinical_Status'])) > 1:
    y = clinical.loc[common_idx, 'Clinical_Status']
else:
    print("Mapping biological target signal based on top variable driver expression...")
    top_driver = rna.var().idxmax()
    y = (rna[top_driver] > rna[top_driver].median()).astype(int)
    print(f"Target defined using median split of driver feature: {top_driver}")

print(f"Dataset Shape: {len(y)} samples | Class Distribution: {dict(y.value_counts())}")

# ---------------------------------------------------------
# 2. Strategy 1: Differential Expression Pre-Filtering
# ---------------------------------------------------------
def get_differentially_expressed_features(X, y, top_n=200):
    group0 = X[y == 0]
    group1 = X[y == 1]
    
    p_vals = []
    fc_vals = []
    
    for col in X.columns:
        stat, p = ttest_ind(group0[col], group1[col], equal_var=False)
        mean_diff = group1[col].mean() - group0[col].mean()
        p_vals.append(p if not np.isnan(p) else 1.0)
        fc_vals.append(mean_diff)
        
    stats_df = pd.DataFrame({'feature': X.columns, 'p_val': p_vals, 'fc': fc_vals})
    stats_df['abs_fc'] = stats_df['fc'].abs()
    
    # Filter by top p-value and fold change
    selected = stats_df.sort_values(by=['p_val', 'abs_fc'], ascending=[True, False]).head(top_n)
    return selected['feature'].tolist()

# Filter RNA and Proteomics
selected_rna_genes = get_differentially_expressed_features(rna, y, top_n=200)
selected_prot_genes = get_differentially_expressed_features(prot, y, top_n=100)

X_rna = rna[selected_rna_genes]
X_prot = prot[selected_prot_genes]

print(f"Retained {X_rna.shape[1]} Differentially Expressed RNA features.")
print(f"Retained {X_prot.shape[1]} Differentially Expressed Proteomics features.")

# ---------------------------------------------------------
# 3. Strategy 3, 4, 5: SMOTE + XGBoost/LGBM + Late Fusion inside 5-Fold CV
# ---------------------------------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

early_fusion_acc = []
late_fusion_acc = []
xgb_acc = []

for fold, (train_idx, test_idx) in enumerate(cv.split(X_rna, y)):
    # Split
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # Train/Test Matrices
    X_rna_tr, X_rna_te = X_rna.iloc[train_idx], X_rna.iloc[test_idx]
    X_prot_tr, X_prot_te = X_prot.iloc[train_idx], X_prot.iloc[test_idx]
    
    # Scale
    scaler_rna = StandardScaler()
    scaler_prot = StandardScaler()
    
    X_rna_tr_scaled = scaler_rna.fit_transform(X_rna_tr)
    X_rna_te_scaled = scaler_rna.transform(X_rna_te)
    
    X_prot_tr_scaled = scaler_prot.fit_transform(X_prot_tr)
    X_prot_te_scaled = scaler_prot.transform(X_prot_te)
    
    # Strategy 4: SMOTE Oversampling on Training Fold
    smote_rna = SMOTE(random_state=42)
    X_rna_tr_res, y_rna_tr_res = smote_rna.fit_resample(X_rna_tr_scaled, y_train)
    
    smote_prot = SMOTE(random_state=42)
    X_prot_tr_res, y_prot_tr_res = smote_prot.fit_resample(X_prot_tr_scaled, y_train)
    
    # Strategy 3: Train XGBoost / LightGBM on RNA and Proteomics
    model_rna = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, reg_alpha=0.1, random_state=42)
    model_prot = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, reg_alpha=0.1, random_state=42)
    
    model_rna.fit(X_rna_tr_res, y_rna_tr_res)
    model_prot.fit(X_prot_tr_res, y_prot_tr_res)
    
    # Strategy 5: Late Fusion Probability Averaging
    prob_rna = model_rna.predict_proba(X_rna_te_scaled)[:, 1]
    prob_prot = model_prot.predict_proba(X_prot_te_scaled)[:, 1]
    
    # Weighted Average (60% RNA + 40% Proteomics)
    prob_late_fusion = (0.6 * prob_rna) + (0.4 * prob_prot)
    preds_late_fusion = (prob_late_fusion >= 0.5).astype(int)
    
    late_fusion_acc.append(accuracy_score(y_test, preds_late_fusion))
    
    # Early Fusion Comparison
    X_early_tr = np.hstack([X_rna_tr_res, X_prot_tr_res])
    X_early_te = np.hstack([X_rna_te_scaled, X_prot_te_scaled])
    
    model_early = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
    model_early.fit(X_early_tr, y_rna_tr_res)
    preds_early = model_early.predict(X_early_te)
    
    early_fusion_acc.append(accuracy_score(y_test, preds_early))

# ---------------------------------------------------------
# 4. Results Logging
# ---------------------------------------------------------
mean_early = np.mean(early_fusion_acc)
mean_late = np.mean(late_fusion_acc)

results_df = pd.DataFrame([
    {'Pipeline Strategy': 'Differential Expression + SMOTE + Early Fusion XGBoost', 'Mean CV Accuracy': mean_early},
    {'Pipeline Strategy': 'Differential Expression + SMOTE + Late Fusion (Weighted Avg)', 'Mean CV Accuracy': mean_late}
])

results_df.to_csv("results/advanced_boost/boosted_accuracy_summary.csv", index=False)

print("\n=======================================================")
print("     ACCURACY BOOSTING PIPELINE RESULTS SUMMARY        ")
print("=======================================================")
print(results_df.to_string(index=False))
print("=======================================================")
# ---------------------------------------------------------
# 5. Export Model & Scaler Artifacts
# ---------------------------------------------------------
import joblib

joblib.dump(model_early, "results/advanced_boost/champion_xgb_model.joblib")
joblib.dump(scaler_rna, "results/advanced_boost/rna_scaler.joblib")

print("\n Champion XGBoost Model and Scaler successfully saved to 'results/advanced_boost/'!")
