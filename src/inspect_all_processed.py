import os
from pathlib import Path
import pandas as pd

processed_dir = Path("data/processed")

# Locate all CSV, TSV, and Parquet files under data/processed/
files = list(processed_dir.rglob("*.csv")) + list(processed_dir.rglob("*.tsv")) + list(processed_dir.rglob("*.parquet"))

print("\n" + "="*80)
print(f" FOUND {len(files)} PREPROCESSED DATASET FILES IN {processed_dir.resolve()}")
print("="*80 + "\n")

for filepath in sorted(files):
    rel_path = filepath.relative_to(processed_dir.parent)
    print("="*80)
    print(f" FILE: {rel_path}")
    print("="*80)
    
    try:
        if filepath.suffix == '.parquet':
            df = pd.read_parquet(filepath)
        elif filepath.suffix == '.tsv':
            df = pd.read_csv(filepath, sep='\t', index_col=0)
        else:
            df = pd.read_csv(filepath, index_col=0)
            
        null_count = df.isnull().sum().sum()
        
        print(f"• Shape (Rows x Columns) : {df.shape[0]} Patients/Samples x {df.shape[1]} Features")
        print(f"• Total Missing Values   : {null_count}")
        print(f"• Data Types Summary     : {df.dtypes.value_counts().to_dict()}")
        
        # Check Z-score standardization if numeric
        numeric_cols = df.select_dtypes(include=['float64', 'float32', 'int64', 'int32'])
        if not numeric_cols.empty and df.shape[1] > 1:
            mean_val = numeric_cols.mean().mean()
            std_val = numeric_cols.std().mean()
            print(f"• Global Mean / Std      : Mean = {mean_val:.4f} | Std = {std_val:.4f}")

        print("\n--- Snippet (First 3 Samples x First 5 Features) ---")
        print(df.iloc[:3, :min(5, df.shape[1])])
        print("\n")
        
    except Exception as e:
        print(f" Could not read file: {e}\n")

