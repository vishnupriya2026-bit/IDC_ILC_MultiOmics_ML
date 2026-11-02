import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
import os

def preprocess_rppa(input_path='data/processed/rppa_aligned.csv', output_path='data/processed/rppa_aligned.csv'):
    if not os.path.exists(input_path):
        print(f"File {input_path} not found.")
        return
    
    df = pd.read_csv(input_path, index_col=0)
    print(f"Initial RPPA dimensions: {df.shape[0]} samples x {df.shape[1]} features")
    
    # 1. Filter features with > 20% missing values
    missing_pct = df.isnull().mean(axis=0)
    valid_features = missing_pct[missing_pct <= 0.20].index
    df_filtered = df[valid_features].copy()
    
    # 2. Filter near-zero variance features
    variances = df_filtered.var(axis=0)
    non_zero_var = variances[variances > 1e-5].index
    df_filtered = df_filtered[non_zero_var]
    
    # 3. k-NN Imputation for missing values (< 20%)
    if df_filtered.isnull().sum().sum() > 0:
        imputer = KNNImputer(n_neighbors=5)
        imputed_data = imputer.fit_transform(df_filtered)
        df_imputed = pd.DataFrame(imputed_data, index=df_filtered.index, columns=df_filtered.columns)
    else:
        df_imputed = df_filtered
        
    # 4. Standardize (z-score normalization across samples)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_imputed)
    df_processed = pd.DataFrame(scaled_data, index=df_imputed.index, columns=df_imputed.columns)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_processed.to_csv(output_path)
    print(f"Successfully processed RPPA dataset: {df_processed.shape[0]} samples x {df_processed.shape[1]} features saved to {output_path}")

if __name__ == '__main__':
    preprocess_rppa()
