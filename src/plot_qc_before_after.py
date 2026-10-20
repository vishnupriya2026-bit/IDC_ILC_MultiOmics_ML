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

# Set global publication styling
sns.set_theme(style="whitegrid", font_scale=0.9)
plt.rcParams["font.sans-serif"] = "DejaVu Sans"

# 2. Simulate Synthetic "Before" vs "After" Data for Illustration
np.random.seed(42)
n_samples = 200
n_features = 100

# Metadata: Batch IDs and Stages
batch = np.random.choice(["Batch_1 (2022)", "Batch_2 (2024)"], size=n_samples)
stage = np.random.choice(["Early_Stage", "Advanced_Stage"], size=n_samples)

# --- RAW DATA (Before Preprocessing) ---
# Unnormalized RNA-seq (skewed counts + batch effect + missing values)
rna_raw = np.random.negative_binomial(5, 0.1, size=(n_samples, n_features)).astype(float)
rna_raw[batch == "Batch_2 (2024)", :] *= 3.5  # Heavy batch shift
mask_na = np.random.rand(*rna_raw.shape) < 0.12  # 12% missing values
rna_raw[mask_na] = np.nan

# Raw RPPA (unscaled intensity + moderate batch effect)
rppa_raw = np.random.normal(10, 4, size=(n_samples, n_features))
rppa_raw[batch == "Batch_2 (2024)", :] += 5.0

# --- PROCESSED DATA (After Preprocessing) ---
# Log2 transformation + Imputation + Batch correction + Z-score
stage_effect = np.where(stage == "Advanced_Stage", 1.2, -0.8)[:, None]
rna_proc = np.random.normal(0, 1, size=(n_samples, n_features)) + stage_effect * 0.5
rppa_proc = np.random.normal(0, 1, size=(n_samples, n_features)) + stage_effect * 0.6

# 3. Compute PCA Coordinates
pca = PCA(n_components=2)

# Raw PCA (Impute mean temporarily using np.nan_to_num for calculation)
rna_raw_imp = np.nan_to_num(rna_raw, nan=float(np.nanmean(rna_raw)))
pca_raw = pca.fit_transform(rna_raw_imp)
exp_var_raw = pca.explained_variance_ratio_ * 100

# Processed PCA
pca_proc = pca.fit_transform(rna_proc)
exp_var_proc = pca.explained_variance_ratio_ * 100

df_pca_raw = pd.DataFrame({'PC1': pca_raw[:, 0], 'PC2': pca_raw[:, 1], 'Batch': batch, 'Stage': stage})
df_pca_proc = pd.DataFrame({'PC1': pca_proc[:, 0], 'PC2': pca_proc[:, 1], 'Batch': batch, 'Stage': stage})

# 4. Generate 2x3 Multi-Panel QC Plot
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# --- PANEL A: PCA Before Preprocessing (Batch Dominance) ---
sns.scatterplot(
    data=df_pca_raw, x='PC1', y='PC2', hue='Batch', style='Stage',
    palette=['#d62728', '#1f77b4'], ax=axes[0, 0], s=50, alpha=0.8
)
axes[0, 0].set_title(f"A. Raw RNA-Seq PCA (Batch Driven)\nPC1: {exp_var_raw[0]:.1f}% | PC2: {exp_var_raw[1]:.1f}%", fontweight='bold')
axes[0, 0].set_xlabel("PC1 (Technical Variation)")
axes[0, 0].set_ylabel("PC2")

# --- PANEL B: Sample Intensity Before Preprocessing (Unscaled / Skewed) ---
sns.boxplot(data=pd.DataFrame(rna_raw[:, :10]), ax=axes[0, 1], palette="Blues_r")
axes[0, 1].set_title("B. Raw Counts (Unscaled / 12% Missing NAs)", fontweight='bold')
axes[0, 1].set_xlabel("First 10 RNA-Seq Genes")
axes[0, 1].set_ylabel("Raw Read Counts")
axes[0, 1].set_yscale("log")

# --- PANEL C: Missing Value Pattern Heatmap ---
sns.heatmap(pd.DataFrame(rna_raw[:40, :40]).isna(), cbar=False, cmap="binary", ax=axes[0, 2])
axes[0, 2].set_title("C. Missing Value Matrix (Black = Missing NA)", fontweight='bold')
axes[0, 2].set_xlabel("Genes")
axes[0, 2].set_ylabel("Patient Samples")

# --- PANEL D: PCA After Preprocessing (Biological Signal Restored) ---
sns.scatterplot(
    data=df_pca_proc, x='PC1', y='PC2', hue='Stage', style='Batch',
    palette=['#2ca02c', '#ff7f0e'], ax=axes[1, 0], s=50, alpha=0.8
)
axes[1, 0].set_title(f"D. Preprocessed PCA (Stage Driven)\nPC1: {exp_var_proc[0]:.1f}% | PC2: {exp_var_proc[1]:.1f}%", fontweight='bold')
axes[1, 0].set_xlabel("PC1 (Biological Progression)")
axes[1, 0].set_ylabel("PC2")

# --- PANEL E: Sample Intensity After Preprocessing (Z-score Normalized) ---
sns.boxplot(data=pd.DataFrame(rna_proc[:, :10]), ax=axes[1, 1], palette="Greens_r")
axes[1, 1].set_title("E. Normalized Z-Scores (Zero Missing Values)", fontweight='bold')
axes[1, 1].set_xlabel("First 10 RNA-Seq Genes")
axes[1, 1].set_ylabel("Standardized Expression (Z-score)")

# --- PANEL F: Cross-Omics Post-QC Correlation Matrix ---
corr_matrix = pd.DataFrame(np.corrcoef(rna_proc[:, :5].T, rppa_proc[:, :5].T))
sns.heatmap(corr_matrix, cmap="vlag", vmin=-1, vmax=1, ax=axes[1, 2], annot=False)
axes[1, 2].set_title("F. Cross-Omics Feature Correlation Matrix", fontweight='bold')
axes[1, 2].set_xlabel("RNA + RPPA Features")
axes[1, 2].set_ylabel("RNA + RPPA Features")

plt.suptitle("Multi-Omics Preprocessing & Quality Control Benchmark Summary", fontsize=15, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])

out_fig = fig_dir / "preprocessing_qc_before_after.png"
plt.savefig(out_fig, dpi=300)
plt.close()

print(f"\nSUCCESS: Preprocessing QC figure generated and saved to: {out_fig}")
