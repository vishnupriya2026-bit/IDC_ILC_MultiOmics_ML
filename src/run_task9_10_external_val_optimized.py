import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, 
    average_precision_score, 
    recall_score, 
    brier_score_loss, 
    confusion_matrix
)
from sklearn.utils import resample

print("--- Step 1: Parsing GSE2034 Overall Relapse Target ---")
series_matrix_path = "datasets/GEO/raw/GSE2034_series_matrix.txt"
sample_ids = []
overall_relapse_line = None

with open(series_matrix_path, "r") as f:
    lines = f.readlines()

for line in lines:
    if line.startswith("!Sample_geo_accession"):
        sample_ids = [x.strip('"') for x in line.strip().split("\t")[1:]]

# Search for overall relapse status line
for line in lines:
    if line.startswith("!Sample_characteristics_ch1"):
        l_lower = line.lower()
        if "relapse (1=yes, 0=no)" in l_lower and not any(site in l_lower for site in ["bone", "brain", "lung", "loco"]):
            raw_vals = [x.strip('"') for x in line.strip().split("\t")[1:]]
            if len(raw_vals) == len(sample_ids):
                overall_relapse_line = raw_vals
                print("Extracted exact overall relapse metadata line.")
                break

if overall_relapse_line is None:
    for line in lines:
        if line.startswith("!Sample_characteristics_ch1") and "relapse" in line.lower():
            raw_vals = [x.strip('"') for x in line.strip().split("\t")[1:]]
            if len(raw_vals) == len(sample_ids):
                overall_relapse_line = raw_vals
                break

y_ext_list = [1 if x.split(":")[-1].strip() in ["1", "yes"] else 0 for x in overall_relapse_line]
meta_df = pd.DataFrame({"sample_id": sample_ids, "target": y_ext_list}).set_index("sample_id")

print("--- Step 2: Loading Matrices & Applying Rank Transformation ---")
gse_raw = pd.read_csv("data/processed/GSE2034_harmonized.csv")
gse_clean = gse_raw.groupby("Symbol").mean().T
gse_clean = gse_clean.loc[gse_clean.index.intersection(meta_df.index)]
y_ext = meta_df.loc[gse_clean.index, "target"].values

cptac_df = pd.read_csv("data/processed/CPTAC_RNA.csv")
clinical_df = pd.read_csv("cptac_119_clinical_final.csv")

cptac_id = next((c for c in cptac_df.columns if c.lower() in ['patient_id', 'sample_id', 'id', 'case_id']), cptac_df.columns[0])
clinical_id = next((c for c in clinical_df.columns if c.lower() in ['patient_id', 'sample_id', 'id', 'case_id']), clinical_df.columns[0])
target_col = [c for c in clinical_df.columns if 'target' in c.lower() or 'stage' in c.lower() or c in ['Target_Stage_Binary', 'event', 'pam50']][0]

cptac_df['merge_id'] = cptac_df[cptac_id].astype(str).str.strip()
clinical_df['merge_id'] = clinical_df[clinical_id].astype(str).str.strip()

merged_cptac = pd.merge(cptac_df, clinical_df[['merge_id', target_col]], on='merge_id', how='inner')
y_train = (merged_cptac[target_col].astype("category").cat.codes > 0).astype(int).values

X_cptac_raw = merged_cptac.drop(columns=[cptac_id, clinical_id, 'merge_id', target_col], errors="ignore").select_dtypes(include=[np.number])

# Shared gene alignment
shared_genes = list(set(X_cptac_raw.columns).intersection(set(gse_clean.columns)))
X_train = X_cptac_raw[shared_genes]
X_ext = gse_clean[shared_genes]

# Rank percentile transformation per sample
X_train_rank = X_train.rank(axis=1, pct=True)
X_ext_rank = X_ext.rank(axis=1, pct=True)

# Select top variance-concordant genes across both cohorts
var_train = X_train_rank.var(axis=0)
var_ext = X_ext_rank.var(axis=0)
concordant_genes = (var_train * var_ext).nlargest(100).index.tolist()

X_train_sub = X_train_rank[concordant_genes]
X_ext_sub = X_ext_rank[concordant_genes]

print("--- Step 3: Training Regularized Linear Model ---")
clf = LogisticRegression(C=0.1, penalty='l2', solver='liblinear', random_state=42)
clf.fit(X_train_sub, y_train)

y_probs = clf.predict_proba(X_ext_sub)[:, 1]
y_preds = (y_probs >= np.median(y_probs)).astype(int)

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

print("\n=== Optimized External Validation Metrics ===")
print(f"Overall Relapse Distribution -> Positive (1): {sum(y_ext)}, Negative (0): {len(y_ext) - sum(y_ext)}")
print(results_df.to_string(index=False))
