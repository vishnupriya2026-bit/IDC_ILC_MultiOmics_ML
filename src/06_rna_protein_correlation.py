import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

print("Loading CPTAC RNA and Proteomics datasets...")
rna_df = pd.read_csv("data/processed/CPTAC_RNA.csv", index_col=0)
prot_df = pd.read_csv("data/processed/CPTAC_Proteomics.csv", index_col=0)

# Align common genes and samples
common_genes = list(set(rna_df.index).intersection(set(prot_df.index)))
common_samples = list(set(rna_df.columns).intersection(set(prot_df.columns)))

print(f"Found {len(common_genes)} common genes across {len(common_samples)} overlapping samples.")

if len(common_genes) > 0 and len(common_samples) > 0:
    rna_sub = rna_df.loc[common_genes, common_samples]
    prot_sub = prot_df.loc[common_genes, common_samples]
    
    # Compute per-gene correlation across samples
    correlations = rna_sub.corrwith(prot_sub, axis=1, method="spearman").dropna()
    
    # Plot Correlation Distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(correlations, kde=True, color="teal", bins=30)
    plt.axvline(correlations.mean(), color="red", linestyle="--", label=f"Mean Correlation: {correlations.mean():.2f}")
    plt.title("RNA vs. Protein Expression Correlation (CPTAC BRCA)")
    plt.xlabel("Spearman Correlation (r)")
    plt.ylabel("Gene Count")
    plt.legend()
    plt.tight_layout()
    
    os.makedirs("results/figures", exist_ok=True)
    out_path = "results/figures/RNA_Protein_Correlation.png"
    plt.savefig(out_path, dpi=300)
    print(f"✅ Saved correlation plot to {out_path}")
