import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests

def run_differential_expression():
    # 1. Define File Paths
    input_file = "data/processed/CPTAC_RNA.csv"
    output_csv = "results/feature_selection/differential_expression_results.csv"
    output_figure = "results/publication_figures/volcano_plot.png"
    
    os.makedirs("results/feature_selection", exist_ok=True)
    os.makedirs("results/publication_figures", exist_ok=True)
    
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    
    if 'Patient_ID' in df.columns:
        df.set_index('Patient_ID', inplace=True)
        
    # ------------------------------------------------------------------
    # Target gene used for Median Split (e.g., 'ESR1', 'ERBB2', or any marker)
    # ------------------------------------------------------------------
    target_gene = "ESR1" 
    
    if target_gene not in df.columns:
        target_gene = df.columns[0]
        print(f"Warning: Target gene not found. Falling back to '{target_gene}'.")
        
    print(f"Splitting samples based on median expression of '{target_gene}'...")
    marker_expr = df[target_gene]
    median_val = marker_expr.median()
    
    high_samples = marker_expr[marker_expr >= median_val].index
    low_samples = marker_expr[marker_expr < median_val].index
    
    print(f"High group ({target_gene} >= {median_val:.2f}): {len(high_samples)} samples")
    print(f"Low group  ({target_gene} <  {median_val:.2f}): {len(low_samples)} samples")
    
    # Extract numerical features
    feature_cols = df.select_dtypes(include=[np.number]).columns
    
    group_a = df.loc[high_samples, feature_cols]
    group_b = df.loc[low_samples, feature_cols]
    
    # 2. Compute Statistics per Feature
    results = []
    for gene in feature_cols:
        vals_a = group_a[gene].dropna()
        vals_b = group_b[gene].dropna()
        
        if len(vals_a) == 0 or len(vals_b) == 0:
            continue
            
        log2_fc = np.mean(vals_a) - np.mean(vals_b)
        t_stat, p_val = stats.ttest_ind(vals_a, vals_b, equal_var=False)
        
        if np.isnan(p_val):
            p_val = 1.0
            
        results.append({
            'feature': gene,
            'log2FC': log2_fc,
            'p_value': p_val
        })
        
    res_df = pd.DataFrame(results)
    
    # 3. Multiple Testing Correction (Benjamini-Hochberg FDR)
    res_df['fdr'] = multipletests(res_df['p_value'], method='fdr_bh')[1]
    res_df['-log10_p'] = -np.log10(res_df['p_value'].replace(0, 1e-300))
    
    # Define Significance Cutoffs
    fc_cutoff = 0.5
    fdr_cutoff = 0.05
    
    res_df['status'] = 'Not Significant'
    res_df.loc[(res_df['log2FC'] >= fc_cutoff) & (res_df['fdr'] < fdr_cutoff), 'status'] = 'Upregulated'
    res_df.loc[(res_df['log2FC'] <= -fc_cutoff) & (res_df['fdr'] < fdr_cutoff), 'status'] = 'Downregulated'
    
    res_df.sort_values(by='fdr', inplace=True)
    res_df.to_csv(output_csv, index=False)
    print(f"Saved DEG results to: {output_csv}")
    
    # 4. Generate Volcano Plot
    plt.figure(figsize=(8, 6), dpi=300)
    sns.set_theme(style='ticks')
    colors = {'Not Significant': 'grey', 'Upregulated': '#e41a1c', 'Downregulated': '#377eb8'}
    
    sns.scatterplot(
        data=res_df,
        x='log2FC',
        y='-log10_p',
        hue='status',
        palette=colors,
        alpha=0.7,
        edgecolor=None,
        s=25
    )
    
    plt.axvline(x=fc_cutoff, color='black', linestyle='--', linewidth=0.8, alpha=0.7)
    plt.axvline(x=-fc_cutoff, color='black', linestyle='--', linewidth=0.8, alpha=0.7)
    plt.axhline(y=-np.log10(fdr_cutoff), color='black', linestyle='--', linewidth=0.8, alpha=0.7)
    
    # Annotate Top Hits
    top_hits = res_df[res_df['status'] != 'Not Significant'].head(5)
    for _, row in top_hits.iterrows():
        plt.text(row['log2FC'], row['-log10_p'], f" {row['feature']}", fontsize=8, weight='semibold')
        
    plt.title(f'Volcano Plot ({target_gene} High vs Low)', fontsize=12, pad=10)
    plt.xlabel(r'$\log_2(\text{Fold Change})$', fontsize=10)
    plt.ylabel(r'$-\log_{10}(p\text{-value})$', fontsize=10)
    plt.legend(title='Status', loc='upper right')
    plt.tight_layout()
    
    plt.savefig(output_figure)
    plt.close()
    print(f"Saved Volcano Plot to: {output_figure}")

if __name__ == "__main__":
    run_differential_expression()