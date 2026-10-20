import pandas as pd
import glob
import os

drug_files = glob.glob("**/*drug*.csv", recursive=True)
drug_files = [f for f in drug_files if "prioritized_repurposed" not in f and "targets_for" not in f]

print("==========================================================")
print("  INSPECTING WORKSPACE DRUG CANDIDATE FILES")
print("==========================================================")

for f in drug_files:
    print(f"\nFile: '{f}'")
    try:
        df = pd.read_csv(f)
        print(f" -> Shape: {df.shape[0]} rows x {df.shape[1]} columns")
        print(f" -> Columns: {list(df.columns)}")
        print(" -> Sample Rows:")
        print(df.head(3).to_string(index=False))
    except Exception as e:
        print(f" -> Could not read file: {e}")

