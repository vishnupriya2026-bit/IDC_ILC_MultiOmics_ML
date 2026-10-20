import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests

def run_proteomics_de():
    input_file = "data/processed/CPTAC_Proteomics.csv"
    output_csv = "results/feature_selection/proteomics_de_results.csv"
    output_figure = "results/publication_figures/proteomics_volcano_plot.png"
    
    os.makedirs("results/feature_selection", exist_ok=True)
    os.makedirs("results/publication_figures", exist_ok=True)
    
    print(f"Loading Proteomics data from {input_file}...")
    df = pd.read_csv(input_file)
    if 'Patient_ID' in df.columns:
        df.set_index('Patient_ID', inplace=True)
        
    target_protein = "ESR1"
    if target_protein not in df.columns:
        target_protein = df.columns[0]
        
    marker_expr = df[target_protein]
    median_val = marker_expr.median()
    
    high_samples = marker_expr[marker_expr >= median_val].index
    low_samples = marker_expr[marker_expr < median_val].index
    
    feature_cols = df.select_dtypes(include=[np.number]).columns
    group_a = df.loc[high_samples, feature_cols]
    group_b = df.loc[low_samples, feature_cols]
    
    results = []
    for protein in feature_cols:
        vals_a = group_a[protein].dropna()
        vals_b = group_b[protein].dropna()
        if len(vals_a) < 3 or len(vals_b) < 3:
            continue
        log2_fc = np.mean(vals_a) - np.mean(vals_b)
        _, p_val = stats.ttest_ind(vals_a, vals_b, equal_var=False)
        if np.isnan(p_val): p_val = 1.0
        results.append({'feature': protein, 'log2FC': log2_fc, 'p_value': p_val})
        
    res_df = pd.DataFrame(results)
    res_df['fdr'] = multipletests(res_df['p_value'], method='fdr_bh')[1]
    res_df['-log10_p'] = -np.log10(res_df['p_value'].replace(0, 1e-300))
    
    res_df['status'] = 'Not Significant'
    res_df.loc[(res_df['log2FC'] >= 0.5) & (res_df['fdr'] < 0.05), 'status'] = 'Upregulated'
    res_df.loc[(res_df['log2FC'] <= -0.5) & (res_df['fdr'] < 0.05), 'status'] = 'Downregulated'
    
    res_df.to_csv(output_csv, index=False)
    print(f"Proteomics Differential Expression complete! Saved to {output_csv}")

if __name__ == "__main__":
    run_proteomics_de()