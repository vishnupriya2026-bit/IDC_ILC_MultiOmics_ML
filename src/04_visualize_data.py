import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure output directory exists
os.makedirs("results/figures", exist_ok=True)

# 1. Load Processed Datasets
datasets = {
    "GEO_Microarray": "data/processed/GSE2034_harmonized.csv",
    "CPTAC_RNA": "data/processed/CPTAC_RNA.csv",
    "CPTAC_Proteomics": "data/processed/CPTAC_Proteomics.csv",
    "Metabolomics": "data/processed/Metabolomics_harmonized.csv"
}

# Color palettes for each layer
color_maps = {
    "GEO_Microarray": "vlag",
    "CPTAC_RNA": "Blues",
    "CPTAC_Proteomics": "mako",
    "Metabolomics": "rocket"
}

for name, path in datasets.items():
    if not os.path.exists(path):
        print(f"Skipping {name}: File not found at {path}")
        continue
    
    print(f"Generating heatmap for {name}...")
    df = pd.read_csv(path, index_col=0)

    # Automatically transpose if loaded as (Samples x Features)
    if df.shape[0] < df.shape[1] and df.shape[0] < 500:
        df = df.T

    # Convert all values to numeric
    df = df.apply(pd.to_numeric, errors="coerce").dropna(how="all")

    # Select the Top 50 Most Variable Features across samples
    variances = df.std(axis=1)
    top_features = variances.nlargest(min(50, len(variances))).index
    df_top = df.loc[top_features]

    if df_top.empty or df_top.shape[1] < 2:
        print(f"⚠️ Not enough numeric sample data to generate heatmap for {name}.")
        continue

    # Z-Score Standardisation across rows (Genes/Metabolites)
    df_zscore = df_top.sub(df_top.mean(axis=1), axis=0).div(df_top.std(axis=1), axis=0).fillna(0)

    # Create Clustermap (Hierarchical Clustering)
    g = sns.clustermap(
        df_zscore,
        cmap=color_maps.get(name, "viridis"),
        figsize=(10, 8),
        xticklabels=False,  # Hide dense sample labels
        yticklabels=True,   # Show top feature names
        cbar_kws={'label': 'Z-Score Expression'},
        dendrogram_ratio=(0.15, 0.15)
    )
    
    g.fig.suptitle(f"Top 50 Most Variable Features - {name}", fontsize=14, fontweight="bold", y=1.02)
    
    out_img = f"results/figures/{name}_heatmap.png"
    plt.savefig(out_img, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved heatmap to {out_img}")

print("\n--- All heatmaps generated successfully in results/figures/ ---")
