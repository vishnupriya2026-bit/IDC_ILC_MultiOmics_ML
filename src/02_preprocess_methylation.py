import json
import urllib.request
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# 1. Setup Directories & Load RNA-Seq Cohort Alignment
base_dir = Path.home() / "phd_research"
meth_dir = base_dir / "data" / "methylation"
ml_dir = base_dir / "data" / "processed" / "ml_ready"

print("--- Step 1: Loading Cohort Patient Barcodes ---")
y_train = pd.read_csv(ml_dir / "y_train.csv", index_col=0)
y_test = pd.read_csv(ml_dir / "y_test.csv", index_col=0)

train_patients = y_train.index.tolist()
test_patients = y_test.index.tolist()
matched_patients = train_patients + test_patients

print(f"Target Matched Cohort: N = {len(matched_patients)} Patients ({len(train_patients)} Train / {len(test_patients)} Test)")

# 2. Locate Methylation Files & Query NIH GDC API
print("\n--- Step 2: Querying NIH GDC API for Methylation Mapping ---")
meth_files = list(meth_dir.glob("**/*.txt")) + list(meth_dir.glob("**/*.tsv"))
file_names = [f.name for f in meth_files]

print(f"Found {len(meth_files)} DNA Methylation candidate files in {meth_dir}")

file_to_patient = {}
chunk_size = 400

for i in range(0, len(file_names), chunk_size):
    chunk = file_names[i:i + chunk_size]
    payload = {
        "filters": {
            "op": "in",
            "content": {"field": "file_name", "value": chunk}
        },
        "fields": "file_name,cases.submitter_id",
        "format": "JSON",
        "size": len(chunk)
    }
    
    req = urllib.request.Request(
        "https://api.gdc.cancer.gov/files",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    
    try:
        with urllib.request.urlopen(req) as resp:
            res_data = json.loads(resp.read().decode("utf-8"))
            hits = res_data.get("data", {}).get("hits", [])
            for hit in hits:
                fname = hit.get("file_name")
                cases = hit.get("cases", [])
                if cases:
                    pid = cases[0].get("submitter_id")[:12]
                    if pid in matched_patients:
                        file_to_patient[fname] = pid
    except Exception as e:
        print(f"API Error: {e}")

print(f"Mapped {len(file_to_patient)} / {len(matched_patients)} target patients to Methylation files.")

# 3. Read and Construct Beta Value Matrix
print("\n--- Step 3: Parsing CpG Probe Beta Values ---")
meth_dict = {}

for fpath in meth_files:
    fname = fpath.name
    patient_id = file_to_patient.get(fname)
    if not patient_id:
        continue

    try:
        df = pd.read_csv(fpath, sep="\t", comment="#", header=None)
        if df.shape[1] >= 2:
            probe_col = df.columns[0]
            val_col = df.columns[1]
            
            # Filter rows with valid cg probe IDs
            df = df[df[probe_col].astype(str).str.startswith("cg")]
            
            series = pd.to_numeric(df[val_col], errors='coerce')
            series.index = df[probe_col]
            
            meth_dict[patient_id] = series
    except Exception:
        continue

meth_raw_matrix = pd.DataFrame(meth_dict).T  # Patients x Probes
print(f"Raw Integrated Methylation Matrix: {meth_raw_matrix.shape[0]} Patients x {meth_raw_matrix.shape[1]} Probes")

# Reindex to exact 410 patient cohort
meth_matched = meth_raw_matrix.reindex(matched_patients)

# 4. Impute Missing Values & Filter Top 20,000 CpG Probes
print("\n--- Step 4: Median Imputation & Top 20,000 Feature Filtering ---")
# Drop probes missing in > 20% samples
meth_cleaned = meth_matched.dropna(axis=1, thresh=int(0.80 * len(matched_patients)))

# Impute remaining missing NAs with median
meth_imputed = meth_cleaned.fillna(meth_cleaned.median())

# Variance filtering: Top 20,000 CpG probes
probe_vars = meth_imputed.var(axis=0)
top_20k_probes = probe_vars.nlargest(min(20000, meth_imputed.shape[1])).index
meth_filtered = meth_imputed[top_20k_probes]

# 5. Standardize Z-Scores & Split
scaler = StandardScaler()
meth_scaled_vals = scaler.fit_transform(meth_filtered)
meth_scaled = pd.DataFrame(meth_scaled_vals, index=meth_filtered.index, columns=meth_filtered.columns)

# Split into Train and Test matching exact RNA patient ordering
X_train_meth = meth_scaled.loc[train_patients]
X_test_meth = meth_scaled.loc[test_patients]

# 6. Save Processed Datasets
meth_scaled.to_csv(ml_dir / "methylation_filtered.csv")
X_train_meth.to_csv(ml_dir / "methylation_train.csv")
X_test_meth.to_csv(ml_dir / "methylation_test.csv")

print("\n=========================================================")
print("  ✅ SUCCESS: DNA METHYLATION PREPROCESSING COMPLETE!   ")
print("=========================================================")
print(f"Full Methylation Matrix: {meth_scaled.shape[0]} Patients x {meth_scaled.shape[1]} CpG Probes")
print(f"Train Set Shape: {X_train_meth.shape[0]} Patients x {X_train_meth.shape[1]} Probes")
print(f"Test Set Shape:  {X_test_meth.shape[0]} Patients x {X_test_meth.shape[1]} Probes")
print(f"Saved to: {ml_dir}")
