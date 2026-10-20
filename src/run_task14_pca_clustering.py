import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

print("--- Step 1: Loading & Aligning Multi-Omic Datasets ---")
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

esr1_col = next(c for c in rna_cols if 'ESR1' in c)
y = (merged_df[esr1_col] >= merged_df[esr1_col].median()).astype(int).values

scaler_rna = StandardScaler()
X_rna_std = scaler_rna.fit_transform(merged_df[rna_cols].select_dtypes(include=[np.number]))

if len(prot_cols) > 0:
    scaler_prot = StandardScaler()
    X_prot_std = scaler_prot.fit_transform(merged_df[prot_cols].select_dtypes(include=[np.number]))
    X_multi_std = np.hstack([X_rna_std, X_prot_std])
    datasets = {
        "RNA Transcriptomics": X_rna_std,
        "Proteomics Abundance": X_prot_std,
        "Integrated Multi-Omics": X_multi_std
    }
else:
    datasets = {"RNA Transcriptomics": X_rna_std}

print("--- Step 2: Running 2D Principal Component Decomposition ---")
fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4.5), dpi=300)
if len(datasets) == 1:
    axes = [axes]

colors = {1: '#2ca02c', 0: '#d62728'}
labels_map = {1: 'ER+ High', 0: 'ER- Low'}

for ax, (title, X_mat) in zip(axes, datasets.items()):
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_mat)
    var_exp = pca.explained_variance_ratio_ * 100
    
    for cls in [1, 0]:
        idx = y == cls
        ax.scatter(X_pca[idx, 0], X_pca[idx, 1], c=colors[cls], label=labels_map[cls], alpha=0.75, edgecolors='k', s=50)
        
    ax.set_title(f"{title}\n(PC1: {var_exp[0]:.1f}%, PC2: {var_exp[1]:.1f}%)", fontweight='bold')
    ax.set_xlabel('Principal Component 1', fontweight='bold')
    ax.set_ylabel('Principal Component 2', fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper right')

plt.tight_layout()
os.makedirs("results", exist_ok=True)
output_path = "results/task14_multiomics_pca.png"
plt.savefig(output_path, dpi=300)
plt.close()

print(f"Task 14 completed. Saved PCA projection figure to '{output_path}'")
