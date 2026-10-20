import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

def run_early_integration():
    rna_file = "data/processed/CPTAC_RNA.csv"
    prot_file = "data/processed/CPTAC_Proteomics.csv"
    features_file = "results/feature_selection/lasso_rf_selected_features.csv"
    
    out_csv = "results/feature_selection/integrated_feature_matrix.csv"
    out_fig = "results/publication_figures/integrated_feature_heatmap.png"
    
    os.makedirs("results/feature_selection", exist_ok=True)
    os.makedirs("results/publication_figures", exist_ok=True)
    
    print("Loading RNA and Proteomics datasets for Early Integration...")
    rna = pd.read_csv(rna_file, index_col=0)
    prot = pd.read_csv(prot_file, index_col=0)
    
    # Identify common patients
    common_patients = rna.index.intersection(prot.index)
    print(f"Found {len(common_patients)} overlapping patient samples across omics layers.")
    
    # Select top 20 features
    top_features = pd.read_csv(features_file).head(20)['feature'].tolist()
    shared_features = [f for f in top_features if f in rna.columns and f in prot.columns]
    
    if len(shared_features) < 3:
        shared_features = list(set(rna.columns).intersection(set(prot.columns)))[:20]
        
    print(f"Integrating {len(shared_features)} shared multi-omic features...")
    
    rna_sub = rna.loc[common_patients, shared_features]
    prot_sub = prot.loc[common_patients, shared_features]
    
    # Standardize both layers (Z-score)
    scaler = StandardScaler()
    rna_z = pd.DataFrame(scaler.fit_transform(rna_sub), index=common_patients, columns=[f"{f}_RNA" for f in shared_features])
    prot_z = pd.DataFrame(scaler.fit_transform(prot_sub), index=common_patients, columns=[f"{f}_Prot" for f in shared_features])
    
    # Concatenate features (Early Integration)
    integrated_df = pd.concat([rna_z, prot_z], axis=1)
    integrated_df.to_csv(out_csv)
    print(f"Saved Integrated Feature Matrix to: {out_csv}")
    
    # Generate Hierarchically Clustered Heatmap
    plt.figure(figsize=(10, 8), dpi=300)
    g = sns.clustermap(
        integrated_df,
        cmap='vlag',
        center=0,
        figsize=(11, 8),
        cbar_kws={'label': 'Z-Score Expression / Abundance'},
        yticklabels=False
    )
    g.fig.suptitle('Integrated Multi-Omics Feature Matrix (RNA + Proteomics)', y=1.02, fontsize=12)
    plt.savefig(out_fig, bbox_inches='tight')
    plt.close()
    print(f"Saved Integrated Heatmap to: {out_fig}")

if __name__ == "__main__":
    run_early_integration()
