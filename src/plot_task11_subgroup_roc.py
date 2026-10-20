import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

print("--- Step 1: Parsing GSE2034 Relapse Outcome & ER Status ---")
series_matrix_path = "datasets/GEO/raw/GSE2034_series_matrix.txt"
sample_ids = []
overall_relapse_line = None

with open(series_matrix_path, "r") as f:
    lines = f.readlines()

for line in lines:
    if line.startswith("!Sample_geo_accession"):
        sample_ids = [x.strip('"\r\n ') for x in line.split("\t")[1:]]
        break

for line in lines:
    if line.startswith("!Sample_characteristics_ch1"):
        parts = [x.strip('"\r\n ') for x in line.split("\t")[1:]]
        if parts:
            tag = parts[0].lower()
            if "relapse" in tag and "time" not in tag and not any(s in tag for s in ["bone", "brain", "lung", "loco"]):
                overall_relapse_line = parts
                break

if overall_relapse_line is None:
    for line in lines:
        if line.startswith("!Sample_characteristics_ch1"):
            parts = [x.strip('"\r\n ') for x in line.split("\t")[1:]]
            if parts and "relapse" in parts[0].lower() and "time" not in parts[0].lower():
                overall_relapse_line = parts
                break

if overall_relapse_line is None:
    raise ValueError("Could not find overall relapse outcome line in GSE2034 series matrix.")

y_ext = np.array([1 if x.split(":")[-1].strip().lower() in ["1", "yes", "relapse"] else 0 for x in overall_relapse_line[:len(sample_ids)]])
meta_df = pd.DataFrame({"target": y_ext}, index=sample_ids)

print(f"Loaded {len(meta_df)} samples | Total Relapses: {meta_df['target'].sum()}")

print("--- Step 2: Scoring Cohort & Stratifying by ER (ESR1) ---")
gse_raw = pd.read_csv("data/processed/GSE2034_harmonized.csv")
gse_clean = gse_raw.groupby("Symbol").mean().T
gse_clean = gse_clean.loc[gse_clean.index.intersection(meta_df.index)]
meta_df = meta_df.loc[gse_clean.index]

meta_df['er_status'] = (gse_clean['ESR1'] >= gse_clean['ESR1'].median()).astype(int)

cptac_df = pd.read_csv("data/processed/CPTAC_RNA.csv")
clinical_df = pd.read_csv("cptac_119_clinical_final.csv")
cptac_id = next(c for c in cptac_df.columns if c.lower() in ['patient_id', 'sample_id', 'id', 'case_id'])
clinical_id = next(c for c in clinical_df.columns if c.lower() in ['patient_id', 'sample_id', 'id', 'case_id'])

cptac_df['merge_id'] = cptac_df[cptac_id].astype(str).str.strip()
clinical_df['merge_id'] = clinical_df[clinical_id].astype(str).str.strip()
merged_cptac = pd.merge(cptac_df, clinical_df, on='merge_id', how='inner')

prolif_genes = ['MKI67', 'AURKA', 'CCNB1', 'TOP2A', 'BIRC5', 'CDK1', 'MYBL2', 'BUB1B', 'MCM2']
luminal_genes = ['ESR1', 'PGR', 'FOXA1', 'GATA3', 'BCL2', 'SCUBE2', 'BAG1']

X_cptac_raw = merged_cptac.drop(columns=clinical_df.columns.tolist() + [cptac_id, 'merge_id'], errors="ignore").select_dtypes(include=[np.number])
shared_genes = list(set(X_cptac_raw.columns).intersection(set(gse_clean.columns)))
p_genes, l_genes = [g for g in prolif_genes if g in shared_genes], [g for g in luminal_genes if g in shared_genes]

scaler_cptac, scaler_gse = StandardScaler(), StandardScaler()
X_cptac_std = pd.DataFrame(scaler_cptac.fit_transform(X_cptac_raw[shared_genes]), columns=shared_genes)
X_gse_std = pd.DataFrame(scaler_gse.fit_transform(gse_clean[shared_genes]), columns=shared_genes)

risk_cptac = X_cptac_std[p_genes].mean(axis=1) - X_cptac_std[l_genes].mean(axis=1)
y_train = (risk_cptac >= risk_cptac.median()).astype(int).values

feature_panel = p_genes + l_genes
clf = LogisticRegression(C=1.0, solver='liblinear', random_state=42).fit(X_cptac_std[feature_panel], y_train)
meta_df['prob'] = clf.predict_proba(X_gse_std[feature_panel])[:, 1]

print("--- Step 3: Generating Publication Figures ---")
fig, axes = plt.subplots(1, 2, figsize=(11, 5), dpi=300)
colors = {1: '#2ca02c', 0: '#d62728'}
labels = {1: 'ER+ Cohort', 0: 'ER- Cohort'}

for er_val in [1, 0]:
    sub = meta_df[meta_df['er_status'] == er_val]
    fpr, tpr, _ = roc_curve(sub['target'], sub['prob'])
    auc_val = roc_auc_score(sub['target'], sub['prob'])
    axes[0].plot(fpr, tpr, color=colors[er_val], lw=2.5, label=f'{labels[er_val]} (AUC = {auc_val:.4f})')
    
    prec, rec, _ = precision_recall_curve(sub['target'], sub['prob'])
    pr_auc_val = average_precision_score(sub['target'], sub['prob'])
    axes[1].plot(rec, prec, color=colors[er_val], lw=2.5, label=f'{labels[er_val]} (PR-AUC = {pr_auc_val:.4f})')

axes[0].plot([0, 1], [0, 1], 'k--', lw=1.5, label='Chance Baseline')
axes[0].set_xlabel('False Positive Rate', fontweight='bold')
axes[0].set_ylabel('True Positive Rate', fontweight='bold')
axes[0].set_title('Stratified ROC Curves (ER+ vs ER-)', fontweight='bold')
axes[0].legend(loc='lower right')
axes[0].grid(True, linestyle=':', alpha=0.6)

axes[1].set_xlabel('Recall', fontweight='bold')
axes[1].set_ylabel('Precision', fontweight='bold')
axes[1].set_title('Stratified Precision-Recall Curves', fontweight='bold')
axes[1].legend(loc='upper right')
axes[1].grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
os.makedirs("results", exist_ok=True)
output_img = "results/task11_er_stratified_roc.png"
plt.savefig(output_img, dpi=300)
plt.close()

print(f"Stratified figure saved to {output_img}")
