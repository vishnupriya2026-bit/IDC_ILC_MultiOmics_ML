import os
import pandas as pd
from io import StringIO

RAW_FILE = os.path.expanduser("data/raw/metabolomics/ST000355_raw_data.txt")
OUT_FILE = os.path.expanduser("data/processed/Metabolomics_harmonized.csv")

print("Processing ST000355 Metabolomics Workbench dataset...")

if not os.path.exists(RAW_FILE):
    print(f"Error: {RAW_FILE} not found!")
    exit()

# Extract matrix lines (skipping comment blocks)
lines = []
with open(RAW_FILE, "r") as f:
    for line in f:
        if not line.startswith("#") and "\t" in line:
            lines.append(line)

# Load into pandas
df_raw = pd.read_csv(StringIO("".join(lines)), sep="\t")

# Clean out top-level metadata rows if present
first_col = df_raw.columns[0]
df_clean = df_raw[~df_raw[first_col].astype(str).str.contains("study_id|analysis_id|subject_type|unit|factor", case=False, na=False)].copy()

# Set feature name as index
if "metabolite_name" in df_clean.columns:
    df_metab = df_clean.dropna(subset=["metabolite_name"]).set_index("metabolite_name")
elif "refmet_name" in df_clean.columns:
    df_metab = df_clean.dropna(subset=["refmet_name"]).set_index("refmet_name")
else:
    df_metab = df_clean.set_index(first_col)

# Convert intensity columns to numeric numbers
df_numeric = df_metab.apply(pd.to_numeric, errors="coerce")

# Drop empty rows and columns
df_final = df_numeric.dropna(how="all", axis=1).dropna(how="all", axis=0)

# Save harmonized matrix
os.makedirs("data/processed", exist_ok=True)
df_final.to_csv(OUT_FILE)

print(f"✅ Success! Metabolomics Matrix Shape: {df_final.shape} (Metabolites x Samples)")
