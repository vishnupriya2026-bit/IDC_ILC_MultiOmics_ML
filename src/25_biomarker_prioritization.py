import os
import numpy as np
import pandas as pd

def run_biomarker_prioritization():
    out_dir = "results/biomarker_discovery"
    os.makedirs(out_dir, exist_ok=True)
    
    deg_file = "results/feature_selection/differential_expression_results.csv"
    lasso_rf_file = "results/feature_selection/lasso_rf_selected_features.csv"
    ppi_file = "results/feature_selection/ppi_hub_genes.csv"
    
    print("Consolidating multi-omic features across statistical, ML, and network layers...")
    deg_df = pd.read_csv(deg_file)
    lasso_rf_df = pd.read_csv(lasso_rf_file)
    ppi_df = pd.read_csv(ppi_file)
    
    # Merge criteria
    merged = pd.merge(lasso_rf_df, deg_df[['feature', 'p_value', 'fdr', 'log2FC']], on='feature', how='left')
    merged = pd.merge(merged, ppi_df[['gene', 'degree_centrality']], left_on='feature', right_on='gene', how='left')
    merged['degree_centrality'] = merged['degree_centrality'].fillna(0)
    
    # Calculate Multi-Criteria Score
    max_lasso = merged['lasso_coef'].max() if merged['lasso_coef'].max() > 0 else 1
    max_rf = merged['rf_importance'].max() if merged['rf_importance'].max() > 0 else 1
    max_cent = merged['degree_centrality'].max() if merged['degree_centrality'].max() > 0 else 1
    
    merged['p_score'] = -np.log10(merged['p_value'].clip(lower=1e-10))
    max_p = merged['p_score'].max() if merged['p_score'].max() > 0 else 1
    
    merged['prioritization_score'] = (
        (merged['lasso_coef'] / max_lasso) * 0.25 +
        (merged['rf_importance'] / max_rf) * 0.25 +
        (merged['p_score'] / max_p) * 0.25 +
        (merged['degree_centrality'] / max_cent) * 0.25
    )
    
    merged.sort_values(by='prioritization_score', ascending=False, inplace=True)
    
    out_csv = f"{out_dir}/prioritized_biomarker_panel.csv"
    merged.to_csv(out_csv, index=False)
    
    print("\n=== TOP 10 PRIORITIZED MULTI-OMICS BIOMARKERS ===")
    cols_to_print = ['feature', 'log2FC', 'fdr', 'composite_score', 'prioritization_score']
    print(merged[cols_to_print].head(10).to_string(index=False))
    print(f"\nSaved prioritized biomarker panel to: {out_csv}")

if __name__ == "__main__":
    run_biomarker_prioritization()
