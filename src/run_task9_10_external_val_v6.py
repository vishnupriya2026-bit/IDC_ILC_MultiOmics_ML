import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_curve,
    roc_auc_score, 
    precision_recall_curve,
    average_precision_score, 
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

print("--- Step 2: Loading Data & Cross-Platform Standardization ---")
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

prolif_genes = ['MKI67', 'AURKA', 'CCNB1', 'TOP2A', 'BIRC5', 'CDK1', 'MYBL2', 'BUB1B', 'MCM2']
luminal_genes = ['ESR1', 'PGR', 'FOXA1', 'GATA3', 'BCL2', 'SCUBE2', 'BAG1']

X_cptac_raw = merged_cptac.drop(columns=clinical_df.columns.tolist() + [cptac_id, 'merge_id'], errors="ignore").select_dtypes(include=[np.number])

shared_genes = list(set(X_cptac_raw.columns).intersection(set(gse_clean.columns)))
p_genes = [g for g in prolif_genes if g in shared_genes]
l_genes = [g for g in luminal_genes if g in shared_genes]

scaler_cptac = StandardScaler()
scaler_gse = StandardScaler()

X_cptac_std = pd.DataFrame(scaler_cptac.fit_transform(X_cptac_raw[shared_genes]), columns=shared_genes)
X_gse_std = pd.DataFrame(scaler_gse.fit_transform(gse_clean[shared_genes]), columns=shared_genes)

risk_index_cptac = X_cptac_std[p_genes].mean(axis=1) - X_cptac_std[l_genes].mean(axis=1)
y_train = (risk_index_cptac >= risk_index_cptac.median()).astype(int).values

feature_panel = p_genes + l_genes
X_train_sub = X_cptac_std[feature_panel]
X_ext_sub = X_gse_std[feature_panel]

print("--- Step 3: Model Training (Cleaned Regularization) ---")
clf = LogisticRegression(C=1.0, solver='liblinear', random_state=42)
clf.fit(X_train_sub, y_train)

y_probs = clf.predict_proba(X_ext_sub)[:, 1]
y_preds = (y_probs >= 0.50).astype(int)

tn, fp, fn, tp = confusion_matrix(y_ext, y_preds, labels=[0, 1]).ravel()
roc_auc = roc_auc_score(y_ext, y_probs)
pr_auc = average_precision_score(y_ext, y_probs)
sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
brier = brier_score_loss(y_ext, y_probs)

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

print("--- Step 4: Generating High-Resolution ROC/PR Visualizations ---")
fig, axes = plt.subplots(1, 2, figsize=(11, 5), dpi=300)

fpr, tpr, _ = roc_curve(y_ext, y_probs)
axes[0].plot(fpr, tpr, color='#1f77b4', lw=2.5, label=f'External Validation (AUC = {roc_auc:.4f})')
axes[0].plot([0, 1], [0, 1], color='#7f7f7f', linestyle='--', lw=1.5, label='Chance Baseline (0.50)')
axes[0].set_xlabel('False Positive Rate (1 - Specificity)', fontsize=11, fontweight='bold')
axes[0].set_ylabel('True Positive Rate (Sensitivity)', fontsize=11, fontweight='bold')
axes[0].set_title('Receiver Operating Characteristic (ROC)', fontsize=12, fontweight='bold')
axes[0].legend(loc='lower right', frameon=True)
axes[0].grid(True, linestyle=':', alpha=0.6)

precision, recall, _ = precision_recall_curve(y_ext, y_probs)
baseline_pr = sum(y_ext) / len(y_ext)
axes[1].plot(recall, precision, color='#d62728', lw=2.5, label=f'External Validation (PR-AUC = {pr_auc:.4f})')
axes[1].axhline(y=baseline_pr, color='#7f7f7f', linestyle='--', lw=1.5, label=f'Prevalence Baseline ({baseline_pr:.2f})')
axes[1].set_xlabel('Recall (Sensitivity)', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Precision (Positive Predictive Value)', fontsize=11, fontweight='bold')
axes[1].set_title('Precision-Recall (PR) Curve', fontsize=12, fontweight='bold')
axes[1].legend(loc='upper right', frameon=True)
axes[1].grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plot_path = "results/external_val_roc_curve.png"
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Publication-ready curves successfully saved to {plot_path}")
print("\n=== Tasks 9 & 10 External Validation Metrics ===")
print(results_df.to_string(index=False))
