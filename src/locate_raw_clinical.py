import os
import pandas as pd

BASE_DIR = "/home/priya_26phd0306/BreastCancer_MultiOmics"

print("=" * 65)
print("LOCATING RAW CLINICAL & PHENOTYPE METADATA FILES")
print("=" * 65)

found_files = []
for root, dirs, files in os.walk(BASE_DIR):
    for f in files:
        if any(term in f.lower() for term in ['clinical', 'pheno', 'series_matrix', 'meta', 'survival', 'patient']):
            full_path = os.path.join(root, f)
            found_files.append(full_path)
            print(f"• Found file: {full_path}")

print("\n" + "-" * 65)
print("INSPECTING HEADERS OF FOUND CLINICAL FILES")
print("-" * 65)

for fp in found_files:
    if fp.endswith(('.csv', '.tsv', '.txt', '.xlsx')):
        try:
            sep = '\t' if fp.endswith(('.tsv', '.txt')) else ','
            df = pd.read_csv(fp, sep=sep, nrows=3, comment='!')
            print(f"\nFile: {os.path.basename(fp)}")
            print(f"Path: {fp}")
            print(f"Columns ({len(df.columns)} total): {list(df.columns[:12])}")
        except Exception as e:
            print(f"Could not parse {os.path.basename(fp)}: {e}")

