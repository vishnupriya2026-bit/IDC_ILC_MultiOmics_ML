from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

# 1. Setup Directories
base_dir = Path.home() / "phd_research"
ml_dir = base_dir / "data" / "processed" / "ml_ready"
fig_dir = base_dir / "results" / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=0.9)
plt.rcParams["font.sans-serif"] = "DejaVu Sans"

print("--- LOADING REAL TCGA-BRCA ML-READY DATASETS ---")
target_df = pd.read_csv(ml_dir / "target_y.csv", index_col=0)
rna_df = pd.read_csv(ml_dir / "rna_filtered.csv", index_col=0)
meth_df = pd.read_csv(ml_dir / "methylation_filtered.csv", index_col=0)
rppa_df = pd.read_csv(ml_dir / "rppa_filtered.csv", index_col=0)

# Align all patients
patients = target_df.index.tolist()
rna_df = rna_df.reindex(patients)
meth_df = meth_df.reindex(patients)
rppa_df = rppa_df.reindex(patients)

print(f"Cohort Matched Patients (N): {len(patients)}")
print(f"RNA-Seq Shape: {rna_df.shape}")
print(f"DNA Methylation Shape: {meth_df.shape}")
print(f"RPPA Proteomics Shape: {rppa_df.shape}")

# 2. Compute Real PCA on Processed RNA-Seq
pca = PCA(n_components=2)
rna_pca = pca.fit_transform(rna_df.values)
exp_var = pca.explained_variance_ratio_ * 100

pca_df = pd.DataFrame({
    'PC1': rna_pca[:, 0],
    'PC2': rna_pca[:, 1],
    'Stage': target_df['Y_Stage'].values
}, index=patients)

# 3. Create Multi-Panel Figure from Real Data
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# PANEL A: Real Processed RNA-Seq PCA Separation by Stage
sns.scatterplot(
    data=pca_df, x='PC1', y='PC2', hue='Stage', palette=['#2ca02c', '#ff7f0e'],
    ax=axes[0, 0], s=60, alpha=0.85
)
axes[0, 0].set_title(f"A. Real RNA-Seq PCA (Stage Separation)\nPC1: {exp_var[0]:.1f}% | PC2: {exp_var[1]:.1f}%", fontweight='bold')
axes[0, 0].set_xlabel("PC1 (Primary Transcriptomic Variance)")
axes[0, 0].set_ylabel("PC2")

# PANEL B: Real RPPA Normalized Feature Distribution
sns.boxplot(data=rppa_df.iloc[:, :12], ax=axes[0, 1], palette="Oranges_r")
axes[0, 1].set_title("B. Real RPPA Proteomics Distribution (Z-Scores)", fontweight='bold')
axes[0, 1].set_xlabel("First 12 RPPA Protein Features")
axes[0, 1].set_ylabel("Z-Score Abundance")
axes[0, 1].tick_params(axis='x', rotation=45)

# PANEL C: Real Methylation CpG Probe Distribution
sns.histplot(meth_df.iloc[:, :500].values.flatten(), kde=True, ax=axes[1, 0], color="#2ca02c", bins=40)
axes[1, 0].set_title("C. Real DNA Methylation Normalized Beta Distribution", fontweight='bold')
axes[1, 0].set_xlabel("Z-Score Standardized CpG Probes")
axes[1, 0].set_ylabel("Probe Density")

# PANEL D: Real Cross-Omics Correlation (Top Features across Modalities)
top_rna = rna_df.std().nlargest(5).index
top_rppa = rppa_df.std().nlargest(5).index
combined_df = pd.concat([rna_df[top_rna], rppa_df[top_rppa]], axis=1)
corr_matrix = combined_df.corr()

sns.heatmap(corr_matrix, cmap="vlag", vmin=-1, vmax=1, ax=axes[1, 1], annot=False)
axes[1, 1].set_title("D. Real RNA-Seq & RPPA Feature Correlation", fontweight='bold')
axes[1, 1].set_xlabel("Top Variable Features (RNA + RPPA)")
axes[1, 1].set_ylabel("Top Variable Features (RNA + RPPA)")

plt.suptitle("Authentic TCGA-BRCA Preprocessed Multi-Omics QC Summary (N=410)", fontsize=15, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])

out_fig = fig_dir / "real_preprocessing_qc_summary.png"
plt.savefig(out_fig, dpi=300)
plt.close()

print(f"\nSUCCESS: Real data QC plot saved to: {out_fig}")
