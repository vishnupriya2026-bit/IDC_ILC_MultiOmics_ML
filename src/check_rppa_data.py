from pathlib import Path
import pandas as pd

base_dir = Path.home() / 'phd_research'
ml_dir = base_dir / 'data' / 'processed' / 'ml_ready'
raw_dir = base_dir / 'data' / 'raw'

print("Searching for existing RPPA files across project directories...")
possible_files = list(ml_dir.glob('*rppa*')) + list(raw_dir.glob('*rppa*')) + list(base_dir.glob('*rppa*'))

if not possible_files:
    print("No RPPA files found in project directories.")
else:
    for file_path in set(possible_files):
        print(f"\n--- File: {file_path.name} ---")
        try:
            df = pd.read_excel(file_path, index_col=0) if file_path.suffix in ['.xlsx', '.xls'] else pd.read_csv(file_path, index_col=0)
            print(f"Shape: {df.shape} (Rows x Columns)")
            num_df = df.select_dtypes(include=['number'])
            
            if num_df.empty:
                print("Result: No numeric features detected.")
                continue
                
            stds = num_df.std(axis=0)
            valid_cols = (stds > 0.01).sum()
            print(f"Proteins with real variance (std > 0.01): {valid_cols} / {num_df.shape[1]}")
            
            if valid_cols == 0:
                print("Result: UNUSABLE (All patient rows are identical / zero variance)")
            else:
                print("Result: VALID DATASET (Contains varying patient measurements)")
        except Exception as e:
            print(f"Error checking {file_path.name}: {e}")
