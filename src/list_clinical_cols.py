import os, pandas as pd

base_dir = os.path.expanduser("~/phd_research/data")
clin_path = os.path.join(base_dir, "clinical/extracted/clinical.tsv")
clin_df = pd.read_csv(clin_path, sep="\t", low_memory=False, nrows=5)

print("--- Available Clinical Columns ---")
for c in sorted(clin_df.columns):
    print(f" • {c}")
