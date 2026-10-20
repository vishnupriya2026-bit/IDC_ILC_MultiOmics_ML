import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

def run_final_polish():
    out_dir = "results/ml_models"
    fig_dir = "results/publication_figures"
    
    # 1. Multi-Metric Comparison Bar Plot
    print("Generating Multi-Metric Performance Comparison plot...")
    df_metrics = pd.read_csv(f"{out_dir}/model_performance_summary.csv")
    
    df_melted = df_metrics.melt(
        id_vars=['Model'], 
        value_vars=['Test Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'MCC'],
        var_name='Metric', value_name='Score'
    )
    
    plt.figure(figsize=(12, 6), dpi=300)
    sns.barplot(data=df_melted, x='Metric', y='Score', hue='Model', palette='Set2')
    plt.ylim(0, 1.1)
    plt.title('Comprehensive Model Evaluation Across Performance Metrics', fontsize=12)
    plt.ylabel('Metric Score (0 - 1)')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/model_metric_comparison.png")
    plt.close()
    print(f"Saved Metric Comparison Plot to: {fig_dir}/model_metric_comparison.png")
    
    # 2. Local Interpretability (SHAP Waterfall / Force Plot for sample #0)
    print("Generating Local Interpretability (SHAP Waterfall Plot for Patient #1)...")
    int_matrix_file = "results/feature_selection/integrated_feature_matrix.csv"
    rna_file = "data/processed/CPTAC_RNA.csv"
    
    X_df = pd.read_csv(int_matrix_file, index_col=0)
    rna_df = pd.read_csv(rna_file, index_col=0)
    
    target_gene = "ESR1" if "ESR1" in rna_df.columns else rna_df.columns[0]
    common_ids = X_df.index.intersection(rna_df.index)
    
    X = X_df.loc[common_ids]
    y = (rna_df.loc[common_ids, target_gene] >= rna_df.loc[common_ids, target_gene].median()).astype(int).values
    
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_scaled, y)
    
    explainer = shap.Explainer(rf.predict, X_scaled)
    shap_values_sample = explainer(X_scaled.iloc[:1])
    
    plt.figure(figsize=(8, 4), dpi=300)
    shap.plots.waterfall(shap_values_sample[0], show=False)
    plt.title("Local Interpretability: Patient #1 SHAP Explanation", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/shap_local_waterfall.png", bbox_inches='tight')
    plt.close()
    print(f"Saved Local Interpretability Plot to: {fig_dir}/shap_local_waterfall.png")

if __name__ == "__main__":
    run_final_polish()
