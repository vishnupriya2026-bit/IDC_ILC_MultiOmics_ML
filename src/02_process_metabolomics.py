import os
import pandas as pd

# Define paths
RAW_FILE = os.path.expanduser("data/raw/metabolomics/ST000915_raw_data.txt")
OUT_FILE = os.path.expanduser("data/processed/Metabolomics_harmonized.csv")

print("Processing Metabolomics Workbench dataset...")

# Parse raw text matrix from Metabolomics Workbench
if os.path.exists(RAW_FILE):
    # Load tab-delimited raw file
    df_metab = pd.read_csv(RAW_FILE, sep="\t")
    
    # Check column names (usually 'metabolite_name' or 'refmet_name')
    if 'metabolite_name' in df_metab.columns:
        df_metab = df_metab.dropna(subset=['metabolite_name'])
        df_metab = df_metab.groupby('metabolite_name').mean(numeric_only=True)
    elif 'refmet_name' in df_metab.columns:
        df_metab = df_metab.dropna(subset=['refmet_name'])
        df_metab = df_metab.groupby('refmet_name').mean(numeric_only=True)
    
    # Save formatted matrix (Metabolites as rows, Samples as columns)
    df_metab.to_csv(OUT_FILE)
    print(f"Successfully processed Metabolomics Matrix: {df_metab.shape} (Metabolites x Samples)")
    print(f"Saved to: {OUT_FILE}")
else:
    print(f"File not found: {RAW_FILE}")
