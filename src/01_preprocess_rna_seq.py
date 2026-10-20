import os
import glob
import json
import urllib.request
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Setup Directories
base_dir = Path.home() / "phd_research"
rna_dir = base_dir / "data" / "rna_seq"
manifest_dir = base_dir / "manifests"
ml_dir = base_dir / "data" / "processed" / "ml_ready"
ml_dir.mkdir(parents=True, exist_ok=True)

print("--- Step 1: Locating Raw GDC RNA-Seq Files ---")
rna_files = list(rna_dir.glob("**/*.rna_seq.augmented_star_gene_counts.tsv")) + \
            list(rna_dir.glob("**/*.tsv"))

# Filter out non-counts TSV files if any
rna_files = [f for f in rna_files if "augmented_star_gene_counts" in f.name or f.name.endswith(".tsv")]
file_names = [f.name for f in rna_files]
print(f"Found {len(rna_files)} candidate RNA-Seq files in {rna_dir}")

# 2. Query NIH GDC API to map File Names -> TCGA Patient Barcodes
print("\n--- Step 2: Querying NIH GDC API for File-to-Patient Barcode Mapping ---")
file_to_patient = {}

# Batch API requests (max 500 files per batch)
chunk_size = 400
for i in range(0, len(file_names), chunk_size):
    chunk = file_names[i:i + chunk_size]
    
    payload = {
        "filters": {
            "op": "in",
            "content": {
                "field": "file_name",
                "value": chunk
            }
        },
        "fields": "file_name,cases.submitter_id,cases.samples.submitter_id",
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
                    patient_barcode = cases[0].get("submitter_id")[:12] # e.g. TCGA-A1-A0SF
                    if patient_barcode.startswith("TCGA-"):
                        file_to_patient[fname] = patient_barcode
    except Exception as e:
        print(f"API Batch query error: {e}")

print(f"Successfully mapped {len(file_to_patient)} / {len(rna_files)} files to TCGA Patient Barcodes!")

# 3. Read and Aggregate Gene Expression Counts
print("\n--- Step 3: Parsing Gene Expression Matrices ---")
rna_dict = {}

for fpath in rna_files:
    fname = fpath.name
    patient_id = file_to_patient.get(fname)
    
    if not patient_id:
        continue

    try:
        df = pd.read_csv(fpath, sep="\t", comment="#")
        
        # Identify ENSEMBL ID column
        gene_col = "gene_id" if "gene_id" in df.columns else df.columns[0]
        
        # Select expression metric (TPM unstranded preferred, or unstranded counts)
        val_col = None
        for col in ["tpm_unstranded", "unstranded", "gene_expression"]:
            if col in df.columns:
                val_col = col
                break
        if not val_col:
            val_col = df.columns[3] if len(df.columns) > 3 else df.columns[1]

        # Filter out GDC summary meta-rows (N_unmapped, N_multimapping, etc.)
        df = df[~df[gene_col].str.startswith("N_")]
        
        series = df.set_index(gene_col)[val_col].astype(float)
        
        # If multiple files exist for the same patient, take average
        if patient_id in rna_dict:
            rna_dict[patient_id] = (rna_dict[patient_id] + series) / 2.0
        else:
            rna_dict[patient_id] = series
            
    except Exception as e:
        continue

rna_raw_matrix = pd.DataFrame(rna_dict).T  # Rows = Patients, Cols = Genes
print(f"Integrated Raw RNA-Seq Matrix: {rna_raw_matrix.shape[0]} Patients x {rna_raw_matrix.shape[1]} Genes")

# 4. Load Clinical Target Y (Stage Metadata)
print("\n--- Step 4: Aligning with Target Stage Metadata ---")
target_y = None
target_y_path = ml_dir / "target_y.csv"

if target_y_path.exists():
    target_y = pd.read_csv(target_y_path, index_col=0)
else:
    # Try searching for clinical TSVs in data folder
    clin_files = list(base_dir.glob("**/*clinical*.tsv"))
    for cfile in clin_files:
        try:
            tdf = pd.read_csv(cfile, sep="\t", low_memory=False)
            barcode_col = [c for c in tdf.columns if "barcode" in c.lower() or "submitter" in c.lower()]
            stage_col = [c for c in tdf.columns if "stage" in c.lower()]
            if barcode_col and stage_col:
                tdf = tdf[[barcode_col[0], stage_col[0]]].dropna()
                tdf[barcode_col[0]] = tdf[barcode_col[0]].str[:12]
                tdf = tdf[tdf[barcode_col[0]].str.startswith("TCGA-")].drop_duplicates(subset=[barcode_col[0]])
                tdf = tdf.set_index(barcode_col[0])
                
                # Map stage to Early_Stage vs Advanced_Stage
                tdf["Y_Stage"] = tdf[stage_col[0]].apply(
                    lambda x: "Early_Stage" if any(s in str(x).lower() for s in ["stage i", "stage ii", "stage ia", "stage ib", "stage iia", "stage iib"]) else "Advanced_Stage"
                )
                target_y = tdf[["Y_Stage"]]
                break
        except Exception:
            continue

if target_y is None or target_y.empty:
    raise RuntimeError("Could not locate clinical stage data. Ensure target_y.csv or clinical TSV is present.")

# Intersect patients
matched_patients = rna_raw_matrix.index.intersection(target_y.index)
rna_matched = rna_raw_matrix.loc[matched_patients]
y_matched = target_y.loc[matched_patients]

print(f"Matched Cohort: N = {len(matched_patients)} Patients")
print(f"Class Distribution:\n{y_matched['Y_Stage'].value_counts()}")

# 5. Log2 Transformation & Top 10,000 Feature Selection
print("\n--- Step 5: Log2 Transformation & Feature Filtering ---")
rna_log2 = np.log2(rna_matched + 1.0)

# Select top 10,000 variable genes
gene_variances = rna_log2.var(axis=0)
top_10k_genes = gene_variances.nlargest(min(10000, rna_log2.shape[1])).index
rna_filtered = rna_log2[top_10k_genes]

# 6. Standardize Z-Scores
scaler = StandardScaler()
rna_scaled_vals = scaler.fit_transform(rna_filtered)
rna_scaled = pd.DataFrame(rna_scaled_vals, index=rna_filtered.index, columns=rna_filtered.columns)

# 7. Perform Stratified Train/Test Split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    rna_scaled, y_matched, test_size=0.20, random_state=42, stratify=y_matched["Y_Stage"]
)

# Save ML-Ready Data
rna_scaled.to_csv(ml_dir / "rna_filtered.csv")
y_matched.to_csv(ml_dir / "target_y.csv")
X_train.to_csv(ml_dir / "rna_train.csv")
X_test.to_csv(ml_dir / "rna_test.csv")
y_train.to_csv(ml_dir / "y_train.csv")
y_test.to_csv(ml_dir / "y_test.csv")

print("\n=========================================================")
print("  ✅ SUCCESS: RNA-SEQ PREPROCESSING & SPLIT COMPLETE!   ")
print("=========================================================")
print(f"Full RNA Matrix: {rna_scaled.shape[0]} Patients x {rna_scaled.shape[1]} Genes")
print(f"Train Set Shape: {X_train.shape[0]} Patients x {X_train.shape[1]} Genes")
print(f"Test Set Shape:  {X_test.shape[0]} Patients x {X_test.shape[1]} Genes")
print(f"Saved to: {ml_dir}")
