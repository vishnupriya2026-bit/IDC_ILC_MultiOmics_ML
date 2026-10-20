import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Set up output directories
os.makedirs("results/QC", exist_ok=True)
qc_txt_path = "results/QC/qc_summary.txt"

# ---------------------------------------------------------
# HOUR 1: Data Loading & Sanity Checks
# ---------------------------------------------------------
print("=== Hour 1: Loading & Verifying Data ===")

rna = pd.read_csv("data/processed/cptac_rna_clean.csv", index_col=0)
prot = pd.read_csv("data/processed/cptac_proteomics_clean.csv", index_col=0)

rna_shape = rna.shape
prot_shape = prot.shape

rna_nulls = rna.isnull().sum().sum()
prot_nulls = prot.isnull().sum().sum()

rna_dup_samples = rna.index.duplicated().sum()
prot_dup_samples = prot.index.duplicated().sum()

rna_dup_features = rna.columns.duplicated().sum()
prot_dup_features = prot.columns.duplicated().sum()

# Write Summary Text File
with open(qc_txt_path, "w") as f:
    f.write("=== QUALITY CONTROL SUMMARY REPORT ===\n\n")
    f.write(f"RNA Data Dimensions (Samples x Features): {rna_shape}\n")
    f.write(f"Proteomics Data Dimensions (Samples x Features): {prot_shape}\n\n")
    f.write(f"RNA Missing Values: {rna_nulls}\n")
    f.write(f"Proteomics Missing Values: {prot_nulls}\n\n")
    f.write(f"RNA Duplicate Samples: {rna_dup_samples} | Duplicate Features: {rna_dup_features}\n")
    f.write(f"Proteomics Duplicate Samples: {prot_dup_samples} | Duplicate Features: {prot_dup_features}\n")

print(f"QC Summary written to {qc_txt_path}")

# ---------------------------------------------------------
# HOUR 2: PCA Analysis
# ---------------------------------------------------------
print("=== Hour 2: Running PCA ===")

def run_and_plot_pca(data, title, filename):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(scaled_data)
    
    var_exp = pca.explained_variance_ratio_ * 100
    
    plt.figure(figsize=(8, 6))
    plt.scatter(pca_coords[:, 0], pca_coords[:, 1], alpha=0.8, color='crimson' if 'RNA' in title else 'teal')
    plt.title(f"PCA - {title}", fontsize=14, fontweight='bold')
    plt.xlabel(f"PC1 ({var_exp[0]:.2f}% Variance)")
    plt.ylabel(f"PC2 ({var_exp[1]:.2f}% Variance)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"results/QC/{filename}", dpi=300)
    plt.close()

run_and_plot_pca(rna, "RNA-Seq", "PCA_RNA.png")
run_and_plot_pca(prot, "Proteomics", "PCA_Proteomics.png")

# ---------------------------------------------------------
# HOUR 3: Clinical Label Alignment (Parsed from GEO Matrix Header)
# ---------------------------------------------------------
print("=== Hour 3: Clinical Mapping ===")

geo_file = "data/GSE2034_series_matrix.txt"

sample_ids = []
clinical_labels = []

# Parse GEO matrix header for sample GSM IDs and characteristics
if os.path.exists(geo_file):
    with open(geo_file, "r") as f:
        for line in f:
            if line.startswith("!Sample_geo_accession"):
                sample_ids = [x.strip('"') for x in line.strip().split('\t')[1:]]
            elif line.startswith("!Sample_characteristics_ch1") and ("bone relapses" in line or "relapses" in line):
                clinical_labels = [x.strip('"').split(":")[-1].strip() for x in line.strip().split('\t')[1:]]
                break

# Build DataFrame
if sample_ids and clinical_labels:
    clinical = pd.DataFrame({"Clinical_Status": clinical_labels}, index=sample_ids)
else:
    clinical = pd.DataFrame({"Clinical_Status": ["Unannotated"] * len(rna)}, index=rna.index)

# Filter/align to match the 119 matched samples in RNA
matched_clinical = clinical.reindex(rna.index).fillna("Unknown")
matched_clinical.to_csv("results/QC/matched_clinical.csv")

# Subtype / Label distribution summary
subtype_counts = matched_clinical["Clinical_Status"].value_counts(dropna=False)
subtype_counts.to_csv("results/QC/Subtype_distribution.csv")

# Plot Subtype / Label Distribution
plt.figure(figsize=(8, 5))
sns.barplot(x=subtype_counts.index.astype(str), y=subtype_counts.values, palette="magma")
plt.title("Clinical Status Distribution (n=119)", fontsize=14, fontweight="bold")
plt.xlabel("Clinical Variable")
plt.ylabel("Patient Count")
plt.tight_layout()
plt.savefig("results/QC/subtype_distribution.png", dpi=300)
plt.close()

print("Processing complete! All QC outputs saved in 'results/QC/'.")
