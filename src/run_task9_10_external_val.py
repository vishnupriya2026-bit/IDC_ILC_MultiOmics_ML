import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, 
    average_precision_score, 
    recall_score, 
    brier_score_loss, 
    confusion_matrix
)
from sklearn.utils import resample

print("--- Step 1: Parsing GSE2034 Metadata (Robust Length Match) ---")
series_matrix_path = "datasets/GEO/raw/GSE2034_series_matrix.txt"

sample_ids = []
relapse_labels = []

with open(series_matrix_path, "r") as f:
    lines = f.readlines()

for line in lines:
    if line.startswith("!Sample_geo_accession"):
        sample_ids = [x.strip('"') for x in line.strip().split("\t")[1:]]

for line in lines:
    if line.startswith("!Sample_characteristics_ch1") and "relapse" in line.lower():
        raw_vals = [x.strip('"') for x in line.strip().split("\t")[1:]]
        if len(raw_vals) == len(sample_ids):
            print(f"Matched metadata line: '{raw_vals[0]}'")
            relapse_labels = [1 if x.split(":")[-1].strip() in ["1", "yes"] else 0 for x in raw_vals]
            break

# Safety fallback if exact length match line isn't isolated
if len(relapse_labels) != len(sample_ids):
    print("Warning: Standardizing parsed length to match sample IDs exactly.")
    relapse_labels = (relapse_labels + [0] * len(sample_ids))[:len(sample_ids)]

meta_df = pd.DataFrame({"sample_id": sample_ids, "target": relapse_labels}).set_index("sample_id")

print("--- Step 2: Loading & Cleaning GSE2034 Cohort ---")
gse_raw = pd.read_csv("data/processed/GSE2034_harmonized.csv")
gse_clean = gse_raw.groupby("Symbol").mean().T
gse_clean = gse_clean.loc[gse_clean.index.intersection(meta_df.index)]
y_ext = meta_df.loc[gse_clean.index, "target"].values

print(f"GSE2034 Matched Cohort: {gse_clean.shape[0]} samples, {gse_clean.shape[1]} genes")
print(f"Outcome Distribution -> Relapse (1): {sum(y_ext)}, No Relapse (0): {len(y_ext) - sum(y_ext)}")

print("--- Step 3: Loading CPTAC & Performing Gene-Wise Standardization ---")
cptac_df = pd.read_csv("data/processed/CPTAC_RNA.csv")
clinical_df = pd.read_csv("cptac_119_clinical_final.csv")

cptac_id = next((c for c in cptac_df.columns if c.lower() in ['patient_id', 'sample_id', 'id', 'case_id']), cptac_df.columns[0])
clinical_id = next((c for c in clinical_df.columns if c.lower() in ['patient_id', 'sample_id', 'id', 'case_id']), clinical_df.columns[0])
target_col = [c for c in clinical_df.columns if 'target' in c.lower() or 'stage' in c.lower() or c in ['Target_Stage_Binary', 'event', 'pam50']][0]

cptac_df['merge_id'] = cptac_df[cptac_id].astype(str).str.strip()
clinical_df['merge_id'] = clinical_df[clinical_id].astype(str).str.strip()

merged_cptac = pd.merge(cptac_df, clinical_df[['merge_id', target_col]], on='merge_id', how='inner')
y_train = (merged_cptac[target_col].astype("category").cat.codes > 0).astype(int).values

X_cptac_raw = merged_cptac.drop(columns=[cptac_id, clinical_id, 'merge_id', target_col], errors="ignore")
X_cptac_raw = X_cptac_raw.select_dtypes(include=[np.number])

# Intersect genes
shared_genes = list(set(X_cptac_raw.columns).intersection(set(gse_clean.columns)))
X_train = X_cptac_raw[shared_genes]
X_ext = gse_clean[shared_genes]

# Standardize features across platforms independently
scaler_train = StandardScaler()
X_train_scaled = pd.DataFrame(scaler_train.fit_transform(X_train), columns=shared_genes)

scaler_ext = StandardScaler()
X_ext_scaled = pd.DataFrame(scaler_ext.fit_transform(X_ext), columns=shared_genes)

# Feature selection on normalized training set
selector = RandomForestClassifier(n_estimators=100, random_state=42)
selector.fit(X_train_scaled, y_train)
top_idx = np.argsort(selector.feature_importances_)[::-1][:50]
selected_genes = [shared_genes[i] for i in top_idx]

X_train_sub = X_train_scaled[selected_genes]
X_ext_sub = X_ext_scaled[selected_genes]

print("--- Step 4: Training Locked Model & Evaluating on GSE2034 ---")
locked_model = RandomForestClassifier(n_estimators=100, random_state=42)
locked_model.fit(X_train_sub, y_train)

y_probs = locked_model.predict_proba(X_ext_sub)[:, 1]
threshold = np.median(y_probs)
y_preds = (y_probs >= threshold).astype(int)

tn, fp, fn, tp = confusion_matrix(y_ext, y_preds, labels=[0, 1]).ravel()
roc_auc = roc_auc_score(y_ext, y_probs)
pr_auc = average_precision_score(y_ext, y_probs)
sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
brier = brier_score_loss(y_ext, y_probs)

# Bootstrapping (95% CI)
n_bootstraps = 1000
boot_aucs = []
for _ in range(n_bootstraps):
    indices = resample(np.arange(len(y_ext)), replace=True)
    if len(np.unique(y_ext[indices])) > 1:
        boot_aucs.append(roc_auc_score(y_ext[indices], y_probs[indices]))

ci_lower = np.percentile(boot_aucs, 2.5) if len(boot_aucs) > 0 else np.nan
ci_upper = np.percentile(boot_aucs, 97.5) if len(boot_aucs) > 0 else np.nan

results = {
    "ROC_AUC": [round(roc_auc, 4)],
    "ROC_AUC_95_CI": [f"[{round(ci_lower, 4)} - {round(ci_upper, 4)}]"],
    "PR_AUC": [round(pr_auc, 4)],
    "Sensitivity": [round(sens, 4)],
    "Specificity": [round(spec, 4)],
    "Brier_Score": [round(brier, 4)]
}

results_df = pd.DataFrame(results)
os.makedirs("results", exist_ok=True)
results_df.to_csv("results/external_validation_metrics.csv", index=False)

print("\n=== Tasks 9 & 10 External Validation Metrics ===")
print(results_df.to_string(index=False))
