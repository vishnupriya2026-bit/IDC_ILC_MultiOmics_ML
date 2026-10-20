import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def plot_multiomics_correlation():
    rna_file = "results/feature_selection/differential_expression_results.csv"
    prot_file = "results/feature_selection/proteomics_de_results.csv"
    output_figure = "results/publication_figures/rna_vs_protein_correlation.png"
    
    rna = pd.read_csv(rna_file)
    prot = pd.read_csv(prot_file)
    
    merged = pd.merge(rna, prot, on='feature', suffixes=('_rna', '_protein'))
    merged = merged.dropna(subset=['log2FC_rna', 'log2FC_protein'])
    
    # Calculate Spearman Correlation
    r_val, p_val = stats.spearmanr(merged['log2FC_rna'], merged['log2FC_protein'])
    
    plt.figure(figsize=(7, 6), dpi=300)
    sns.set_theme(style='ticks')
    
    # Highlight concordant DEGs
    concordant_mask = (merged['status_rna'] == merged['status_protein']) & (merged['status_rna'] != 'Not Significant')
    
    plt.scatter(
        merged.loc[~concordant_mask, 'log2FC_rna'],
        merged.loc[~concordant_mask, 'log2FC_protein'],
        color='grey', alpha=0.3, s=15, label='Discordant / Non-sig'
    )
    plt.scatter(
        merged.loc[concordant_mask, 'log2FC_rna'],
        merged.loc[concordant_mask, 'log2FC_protein'],
        color='#d95f02', alpha=0.8, s=30, label=f'Concordant DEGs (n={concordant_mask.sum()})'
    )
    
    # Add trend line
    sns.regplot(
        data=merged, x='log2FC_rna', y='log2FC_protein',
        scatter=False, color='black', line_kws={'linestyle': '--', 'linewidth': 1}
    )
    
    # Annotate Top Multi-Omic Markers
    top_markers = ['ESR1', 'PGR', 'MAPT', 'AGR3', 'PSAT1', 'PPP1R14C']
    for marker in top_markers:
        row = merged[merged['feature'] == marker]
        if not row.empty:
            x, y = row['log2FC_rna'].values[0], row['log2FC_protein'].values[0]
            plt.text(x + 0.05, y + 0.05, marker, fontsize=9, weight='bold')
            
    plt.axhline(0, color='grey', linestyle=':', alpha=0.5)
    plt.axvline(0, color='grey', linestyle=':', alpha=0.5)
    
    plt.xlabel(r'RNA $\log_2(\text{Fold Change})$', fontsize=11)
    plt.ylabel(r'Protein $\log_2(\text{Fold Change})$', fontsize=11)
    plt.title(f'Multi-Omic Concordance (Spearman r = {r_val:.2f}, p = {p_val:.2e})', fontsize=11)
    plt.legend(loc='upper left')
    plt.tight_layout()
    
    plt.savefig(output_figure)
    plt.close()
    print(f"Saved Multi-Omics Correlation Plot to: {output_figure}")

if __name__ == "__main__":
    plot_multiomics_correlation()