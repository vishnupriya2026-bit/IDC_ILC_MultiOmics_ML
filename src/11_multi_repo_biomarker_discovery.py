import os
import pandas as pd
import numpy as np
import joblib
from scipy.stats import ttest_ind

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

print("=== Step 11: Multi-Omics Biomarker Panel Discovery ===")

# ---------------------------------------------------------
# 1. Load Data
# ---------------------------------------------------------
rna = pd.read_csv("data/processed/cptac_rna_combat.csv", index_col=0)
prot = pd.read_csv("data/processed/cptac_proteomics_clean.csv", index_col=0)

common_idx = rna.index.intersection(prot.index)
rna, prot = rna.loc[common_idx], prot.loc[common_idx]

# Define biological outcome on driver feature
driver_gene = rna.var().idxmax()
y = (rna[driver_gene] > rna[driver_gene].median()).astype(int)

# EXCLUDE driver gene from X to force multi-gene panel learning
rna_features = rna.drop(columns=[driver_gene])

print(f"Target defined on driver: {driver_gene} (Excluded from predictor matrix)")

# ---------------------------------------------------------
# 2. Multi-Gene Differential Expression Filtering
# ---------------------------------------------------------
def get_de_genes(X, y, top_n=100):
    group0, group1 = X[y == 0], X[y == 1]
    p_vals = []
    for col in X.columns:
        _, p = ttest_ind(group0[col], group1[col], equal_var=False)
        p_vals.append(p if not np.isnan(p) else 1.0)
    return pd.Series(p_vals, index=X.columns).sort_values().head(top_n).index.tolist()

selected_rna = get_de_genes(rna_features, y, top_n=100)
selected_prot = get_de_genes(prot, y, top_n=50)

# Subset RNA and Proteomics
rna_sub = rna_features[selected_rna].copy()
prot_sub = prot[selected_prot].copy()

# Add prefixes to ensure non-duplicate column names
rna_sub.columns = [f"RNA_{col}" for col in rna_sub.columns]
prot_sub.columns = [f"PROT_{col}" for col in prot_sub.columns]

# Merge into unified matrix
X_fused = pd.concat([rna_sub, prot_sub], axis=1)
print(f"Fused Biomarker Matrix: {X_fused.shape} (100 RNA + 50 Proteomics features with unique column names)")

# ---------------------------------------------------------
# 3. Stratified Cross-Validation & XGBoost Training
# ---------------------------------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_acc = []

for fold, (train_idx, test_idx) in enumerate(cv.split(X_fused, y)):
    X_tr, X_te = X_fused.iloc[train_idx], X_fused.iloc[test_idx]
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
    
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    
    smote = SMOTE(random_state=42)
    X_tr_res, y_tr_res = smote.fit_resample(X_tr_s, y_tr)
    
    model = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X_tr_res, y_tr_res)
    preds = model.predict(X_te_s)
    cv_acc.append(accuracy_score(y_te, preds))

print(f"\n🏆 Multi-Gene Biomarker Panel CV Accuracy: {np.mean(cv_acc)*100:.2f}% (±{np.std(cv_acc):.4f})")

# ---------------------------------------------------------
# 4. Extract Top Multi-Gene Biomarker Panel
# ---------------------------------------------------------
importances = pd.Series(model.feature_importances_, index=X_fused.columns).sort_values(ascending=False)

print("\n--- Top 15 Discovered Biomarkers Across Multi-Omics Layers ---")
print(importances.head(15).to_string())

# Save outputs
os.makedirs("results/biomarker_panel", exist_ok=True)
importances.to_csv("results/biomarker_panel/top_biomarkers.csv")
joblib.dump(model, "results/biomarker_panel/panel_xgb_model.joblib")
print("\nBiomarker panel and updated model successfully saved in 'results/biomarker_panel/'!")
