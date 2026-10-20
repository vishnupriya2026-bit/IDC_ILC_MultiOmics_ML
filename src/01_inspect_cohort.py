import os
import pandas as pd
from pathlib import Path

base_dir = Path.home() / "phd_research" / "data"

# 1. Load Target Labels
target_path = base_dir / "target_y.csv"
target_df = pd.read_csv(target_path)
print("=== TARGET DATASET INFO ===")
print(f"Total Rows in target_y.csv: {len(target_df)}")
print("\nTarget Columns & Missing Values:")
print(target_df.isnull().sum())

# Extract 12-character Patient IDs (e.g., TCGA-A7-A0CE)
patient_col = [c for c in target_df.columns if 'patient' in c.lower() or 'cases' in c.lower() or 'submitter' in c.lower()][0]
target_df['patient_id_12'] = target_df[patient_col].astype(str).str.slice(0, 12)
valid_target = target_df.dropna(subset=['Y_Stage']) # Change to Y_Histology if preferred
valid_patients = set(valid_target['patient_id_12'])

print(f"\nPatients with non-null Y_Stage: {len(valid_patients)}")
