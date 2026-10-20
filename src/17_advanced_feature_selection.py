import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

def run_feature_selection():
    rna_file = "data/processed/CPTAC_RNA.csv"
    output_csv = "results/feature_selection/lasso_rf_selected_features.csv"
    output_figure = "results/publication_figures/feature_importance.png"
    
    os.makedirs("results/feature_selection", exist_ok=True)
    os.makedirs("results/publication_figures", exist_ok=True)
    
    print("Loading RNA expression dataset...")
    df = pd.read_csv(rna_file)
    if 'Patient_ID' in df.columns:
        df.set_index('Patient_ID', inplace=True)
        
    target_gene = "ESR1" if "ESR1" in df.columns else df.columns[0]
    y = (df[target_gene] >= df[target_gene].median()).astype(int)
    
    X = df.select_dtypes(include=[np.number]).drop(columns=[target_gene], errors='ignore')
    X_scaled = StandardScaler().fit_transform(X)
    
    print("Running LASSO (L1 Regularization)...")
    lasso = LogisticRegression(penalty='l1', solver='liblinear', C=0.1, random_state=42)
    lasso.fit(X_scaled, y)
    lasso_coefs = np.abs(lasso.coef_[0])
    
    print("Running Random Forest Classifier...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_scaled, y)
    rf_importances = rf.feature_importances_
    
    feature_df = pd.DataFrame({
        'feature': X.columns,
        'lasso_coef': lasso_coefs,
        'rf_importance': rf_importances
    })
    
    max_lasso = feature_df['lasso_coef'].max()
    max_rf = feature_df['rf_importance'].max()
    
    feature_df['composite_score'] = (
        (feature_df['lasso_coef'] / max_lasso if max_lasso > 0 else 0) + 
        (feature_df['rf_importance'] / max_rf if max_rf > 0 else 0)
    ) / 2
    
    feature_df.sort_values(by='composite_score', ascending=False, inplace=True)
    feature_df.to_csv(output_csv, index=False)
    print(f"Saved selected features to: {output_csv}")
    
    top15 = feature_df.head(15)
    plt.figure(figsize=(8, 5), dpi=300)
    sns.barplot(data=top15, x='composite_score', y='feature', palette='viridis')
    plt.title('Top 15 Biomarkers (LASSO + Random Forest Composite Score)')
    plt.xlabel('Normalized Importance Score')
    plt.tight_layout()
    plt.savefig(output_figure)
    plt.close()
    print(f"Saved Feature Importance plot to: {output_figure}")

if __name__ == "__main__":
    run_feature_selection()
