import pandas as pd
import numpy as np
from pathlib import Path

def inspect_tcga_matrix(file_path: str, dataset_name: str = "Dataset"):
    path = Path(file_path)
    if not path.exists():
        print(f"File not found: {file_path}")
        return
        
    sep = '\t' if path.suffix in ['.tsv', '.txt'] else ','
    print(f"\nLoading {dataset_name} from: {file_path} ...")
    df = pd.read_csv(file_path, sep=sep, index_col=0, nrows=100) # Previews top 100 rows safely
    
    print(f"--- {dataset_name.upper()} PREVIEW ---")
    print(f"Shape (Preview): {df.shape[0]} rows x {df.shape[1]} columns")
    print("\nFirst 5 rows & 4 samples:")
    print(df.iloc[:5, :4])

if __name__ == "__main__":
    # Point directly to your target file
    inspect_tcga_matrix("/home/priya_26phd0306/phd_research/data/target_y.csv", "Target Labels")
