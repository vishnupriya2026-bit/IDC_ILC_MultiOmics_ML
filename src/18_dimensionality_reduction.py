import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

try:
    import umap
    HAS_UMAP = True
except ImportError:
    from sklearn.manifold import TSNE
    HAS_UMAP = False

def run_dim_reduction():
    rna_file = "data/processed/CPTAC_RNA.csv"
    features_file = "results/feature_selection/lasso_rf_selected_features.csv"
    output_figure = "results/publication_figures/pca_umap_reduction.png"
    
    df = pd.read_csv(rna_file)
    if 'Patient_ID' in df.columns:
        df.set_index('Patient_ID', inplace=True)

    top_features = pd.read_csv(features_file).head(50)['feature'].tolist()
    
    target_gene = "ESR1" if "ESR1" in df.columns else df.columns[0]
    labels = np.where(df[target_gene] >= df[target_gene].median(), 'ESR1 High', 'ESR1 Low')
    
    X = df[top_features]
    X_scaled = StandardScaler().fit_transform(X)
    
    pca = PCA(n_components=2)
    pca_res = pca.fit_transform(X_scaled)
    var_exp = pca.explained_variance_ratio_ * 100
    
    if HAS_UMAP:
        reducer = umap.UMAP(n_components=2, random_state=42)
        dim2_res = reducer.fit_transform(X_scaled)
        dim2_title = 'UMAP'
    else:
        dim2_res = TSNE(n_components=2, random_state=42).fit_transform(X_scaled)
        dim2_title = 't-SNE'
        
    plot_df = pd.DataFrame({
        'PCA1': pca_res[:, 0], 'PCA2': pca_res[:, 1],
        'Dim1': dim2_res[:, 0], 'Dim2': dim2_res[:, 1],
        'Group': labels
    })
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    sns.set_theme(style='ticks')
    
    sns.scatterplot(data=plot_df, x='PCA1', y='PCA2', hue='Group', palette='Set1', ax=axes[0], s=50)
    axes[0].set_title(f'PCA (PC1: {var_exp[0]:.1f}%, PC2: {var_exp[1]:.1f}%)')
    axes[0].set_xlabel('Principal Component 1')
    axes[0].set_ylabel('Principal Component 2')
    
    sns.scatterplot(data=plot_df, x='Dim1', y='Dim2', hue='Group', palette='Set1', ax=axes[1], s=50)
    axes[1].set_title(f'{dim2_title} Projection')
    axes[1].set_xlabel(f'{dim2_title} 1')
    axes[1].set_ylabel(f'{dim2_title} 2')
    
    plt.tight_layout()
    plt.savefig(output_figure)
    plt.close()
    print(f"Saved Dimensionality Reduction Plot to: {output_figure}")

if __name__ == "__main__":
    run_dim_reduction()
