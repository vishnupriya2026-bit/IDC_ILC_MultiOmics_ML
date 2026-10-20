import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

def run_shap_explanation():
    int_matrix_file = "results/feature_selection/integrated_feature_matrix.csv"
    rna_file = "data/processed/CPTAC_RNA.csv"
    fig_dir = "results/publication_figures"
    
    X_df = pd.read_csv(int_matrix_file, index_col=0)
    rna_df = pd.read_csv(rna_file, index_col=0)
    
    target_gene = "ESR1" if "ESR1" in rna_df.columns else rna_df.columns[0]
    common_ids = X_df.index.intersection(rna_df.index)
    
    X = X_df.loc[common_ids]
    y = (rna_df.loc[common_ids, target_gene] >= rna_df.loc[common_ids, target_gene].median()).astype(int).values
    
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    print("Training Random Forest model for SHAP interpretability...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    
    print("Calculating SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled)
    
    # Generate SHAP Summary Plot
    plt.figure(figsize=(10, 6), dpi=300)
    if isinstance(shap_values, list):
        shap.summary_plot(shap_values[1], X_scaled, show=False)
    else:
        shap.summary_plot(shap_values, X_scaled, show=False)
        
    plt.title("SHAP Global Feature Importance (Integrated Features)", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/shap_summary_plot.png", bbox_inches='tight')
    plt.close()
    print(f"Saved SHAP Summary Plot to: {fig_dir}/shap_summary_plot.png")

if __name__ == "__main__":
    run_shap_explanation()
