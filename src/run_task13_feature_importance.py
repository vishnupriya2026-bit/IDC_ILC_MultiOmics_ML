import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

print("--- Step 1: Loading & Preprocessing Multi-Omic Panel ---")
rna_path = "data/processed/CPTAC_RNA.csv"
prot_path = "data/processed/CPTAC_Proteomics.csv" if os.path.exists("data/processed/CPTAC_Proteomics.csv") else "data/processed/CPTAC_Prot.csv"
clinical_path = "cptac_119_clinical_final.csv"

rna_df = pd.read_csv(rna_path)
clinical_df = pd.read_csv(clinical_path)

rna_id = next(c for c in rna_df.columns if c.lower() in ['patient_id', 'sample_id', 'id', 'case_id'])
clinical_id = next(c for c in clinical_df.columns if c.lower() in ['patient_id', 'sample_id', 'id', 'case_id'])

rna_df['merge_id'] = rna_df[rna_id].astype(str).str.strip()
clinical_df['merge_id'] = clinical_df[clinical_id].astype(str).str.strip()

if os.path.exists(prot_path):
    prot_df = pd.read_csv(prot_path)
    prot_id = next(c for c in prot_df.columns if c.lower() in ['patient_id', 'sample_id', 'id', 'case_id'])
    prot_df['merge_id'] = prot_df[prot_id].astype(str).str.strip()
    merged_df = pd.merge(rna_df, prot_df, on='merge_id', suffixes=('_rna', '_prot'))
    merged_df = pd.merge(merged_df, clinical_df, on='merge_id')
else:
    merged_df = pd.merge(rna_df, clinical_df, on='merge_id')

prolif_genes = ['MKI67', 'AURKA', 'CCNB1', 'TOP2A', 'BIRC5', 'CDK1', 'MYBL2', 'BUB1B', 'MCM2']
luminal_genes = ['ESR1', 'PGR', 'FOXA1', 'GATA3', 'BCL2', 'SCUBE2', 'BAG1']
target_genes = prolif_genes + luminal_genes

rna_cols = [c for c in merged_df.columns if any(c == g or c == f"{g}_rna" for g in target_genes)]
prot_cols = [c for c in merged_df.columns if any(c == f"{g}_prot" for g in target_genes)]
all_feature_cols = rna_cols + prot_cols

esr1_col = next(c for c in rna_cols if 'ESR1' in c)
y = (merged_df[esr1_col] >= merged_df[esr1_col].median()).astype(int).values

scaler = StandardScaler()
X_std = pd.DataFrame(scaler.fit_transform(merged_df[all_feature_cols].select_dtypes(include=[np.number])), columns=all_feature_cols)

print("--- Step 2: Extracting 5-Fold Averaged Feature Weights ---")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
coef_matrix = []

for train_idx, test_idx in skf.split(X_std, y):
    X_tr, y_tr = X_std.iloc[train_idx], y[train_idx]
    clf = LogisticRegression(C=0.5, solver='lbfgs', max_iter=1000, random_state=42)
    clf.fit(X_tr, y_tr)
    coef_matrix.append(clf.coef_[0])

mean_coefs = np.mean(coef_matrix, axis=0)
std_coefs = np.std(coef_matrix, axis=0)

importance_df = pd.DataFrame({
    'Feature': all_feature_cols,
    'Modality': ['RNA' if f in rna_cols else 'Proteomics' for f in all_feature_cols],
    'Mean_Coefficient': mean_coefs,
    'Std_Coefficient': std_coefs
}).sort_values(by='Mean_Coefficient', ascending=True)

os.makedirs("results", exist_ok=True)
importance_df.to_csv("results/task13_feature_importance.csv", index=False)

print("--- Step 3: Plotting Multi-Omics Signature Drivers ---")
plt.figure(figsize=(9, 7), dpi=300)
colors = ['#1f77b4' if m == 'RNA' else '#ff7f0e' for m in importance_df['Modality']]

plt.barh(importance_df['Feature'], importance_df['Mean_Coefficient'], xerr=importance_df['Std_Coefficient'], color=colors, alpha=0.85, edgecolor='black', capsize=3)
plt.axvline(0, color='black', linestyle='--', linewidth=1)
plt.xlabel('Log-Odds Coefficient Weight (Mean +/- Std)', fontweight='bold')
plt.title('Multi-Omic Feature Importance (CPTAC RNA vs Proteomics)', fontweight='bold')
plt.grid(axis='x', linestyle=':', alpha=0.6)

rna_patch = mpatches.Patch(color='#1f77b4', label='RNA Expression')
prot_patch = mpatches.Patch(color='#ff7f0e', label='Proteomic Abundance')
plt.legend(handles=[rna_patch, prot_patch], loc='lower right')

plt.tight_layout()
plt.savefig("results/task13_feature_importance.png", dpi=300)
plt.close()

print("Task 13 completed. Saved CSV and figure to 'results/task13_feature_importance.png'")
