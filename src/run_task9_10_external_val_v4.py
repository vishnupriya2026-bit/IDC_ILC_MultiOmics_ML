import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, 
    average_precision_score, 
    recall_score, 
    brier_score_loss, 
    confusion_matrix
)
from sklearn.utils import resample

print("--- Step 1: Parsing GSE2034 Relapse Outcome ---")
series_matrix_path = "datasets/GEO/raw/GSE2034_series_matrix.txt"
sample_ids, overall_relapse_line = [], None

with open(series_matrix_path, "r") as f:
    lines = f.readlines()

for line in lines:
    if line.startswith("!Sample_geo_accession"):
        sample_ids = [x.strip('"') for x in line.strip().split("\t")[1:]]

for line in lines:
    if line.startswith("!Sample_characteristics_ch1"):
        l_lower = line.lower()
        if "relapse (1=yes, 0=no)" in l_lower and not any(site in l_lower for site in ["bone", "brain", "lung", "loco"]):
            raw_vals = [x.strip('"') for x in line.strip().split("\t")[1:]]
            if len(raw_vals) == len(sample_ids):
                overall_relapse_line = raw_vals
                break

if not overall_relapse_line:
    for line in lines:
        if line.startswith("!Sample_characteristics_ch1") and "relapse" in line.lower():
            raw_vals = [x.strip('"') for x in line.strip().split("\t")[1:]]
            if len(raw_vals) == len(sample_ids):
                overall_relapse_line = raw_vals
                break

y_ext_list = [1 if x.split(":")[-1].strip() in ["1", "yes"] else 0 for x in overall_relapse_line]
meta_df = pd.DataFrame({"sample_id": sample_ids, "target": y_ext_list}).set_index("sample_id")

print("--- Step 2: Loading CPTAC & Finding Balanced Target ---")
gse_raw = pd.read_csv("data/processed/GSE2034_harmonized.csv")
gse_clean = gse_raw.groupby("Symbol").mean().T
gse_clean = gse_clean.loc[gse_clean.index.intersection(meta_df.index)]
y_ext = meta_df.loc[gse_clean.index, "target"].values

cptac_df = pd.read_csv("data/processed/CPTAC_RNA.csv")
clinical_df = pd.read_csv("cptac_119_clinical_final.csv")

cptac_id = next((c for c in cptac_df.columns if c.lower() in ['patient_id', 'sample_id', 'id', 'case_id']), cptac_df.columns[0])
clinical_id = next((c for c in clinical_df.columns if c.lower() in ['patient_id', 'sample_id', 'id', 'case_id']), clinical_df.columns[0])

cptac_df['merge_id'] = cptac_df[cptac_id].astype(str).str.strip()
clinical_df['merge_id'] = clinical_df[clinical_id].astype(str).str.strip()

merged_cptac = pd.merge(cptac_df, clinical_df, on='merge_id', how='inner')

# Identify balanced clinical column (PAM50, ER Status, or Stage II vs III)
target_col = None
for col in clinical_df.columns:
    if col.lower() in ['pam50', 'er_status', 'subtype', 'er_receptor_status', 'target_stage_binary']:
        target_col = col
        break

if target_col and 'pam50' in target_col.lower():
    # Map high-risk subtypes (Basal, LumB, HER2) = 1, LumA = 0
    y_raw = merged_cptac[target_col].astype(str)
    y_train = y_raw.apply(lambda x: 0 if 'luma' in x.lower() or 'luminal a' in x.lower() else 1).values
elif target_col and 'er' in target_col.lower():
    # Map ER negative (high risk) = 1, ER positive = 0
    y_raw = merged_cptac[target_col].astype(str)
    y_train = y_raw.apply(lambda x: 1 if 'neg' in x.lower() or '0' in x else 0).values
else:
    # Split Stage II (0) vs Stage III (1)
    stage_col = [c for c in clinical_df.columns if 'stage' in c.lower()][0]
    sub_df = merged_cptac[merged_cptac[stage_col].str.lower().str.contains('stage ii|stage iii', na=False)]
    merged_cptac = sub_df
    y_train = merged_cptac[stage_col].str.lower().apply(lambda x: 1 if 'stage iii' in x else 0).values

print(f"CPTAC Target ({target_col or 'Stage II vs III'}) Split -> Class 1: {sum(y_train)}, Class 0: {len(y_train) - sum(y_train)}")

X_cptac_raw = merged_cptac.drop(columns=clinical_df.columns.tolist() + [cptac_id, 'merge_id'], errors="ignore").select_dtypes(include=[np.number])

shared_genes = list(set(X_cptac_raw.columns).intersection(set(gse_clean.columns)))
X_train = X_cptac_raw[shared_genes]
X_ext = gse_clean[shared_genes]

print("--- Step 3: Proliferation Metagene Feature Selection & Standardization ---")
# Proliferation marker list
proliferation_markers = ['MKI67', 'TOP2A', 'CCNB1', 'AURKA', 'BIRC5', 'CDK1', 'MYBL2', 'FOXA1', 'ESR1', 'PGR', 'ERBB2', 'BAG1']
selected_genes = [g for g in proliferation_markers if g in shared_genes]

# If marker list overlaps poorly, fallback to top variance-shared genes
if len(selected_genes) < 5:
    var_scores = X_train.var(axis=0) * X_ext.var(axis=0)
    selected_genes = var_scores.nlargest(50).index.tolist()

X_train_sub = X_train[selected_genes]
X_ext_sub = X_ext[selected_genes]

# Platform-independent Standardization
scaler_train = StandardScaler()
X_train_scaled = scaler_train.fit_transform(X_train_sub)

scaler_ext = StandardScaler()
X_ext_scaled = scaler_ext.fit_transform(X_ext_sub)

print("--- Step 4: Model Training & Evaluation ---")
clf = LogisticRegression(C=0.1, solver='liblinear', random_state=42)
clf.fit(X_train_scaled, y_train)

y_probs = clf.predict_proba(X_ext_scaled)[:, 1]
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

print("\n=== Tasks 9 & 10 External Validation Metrics (Balanced) ===")
print(results_df.to_string(index=False))
