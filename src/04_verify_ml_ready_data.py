from pathlib import Path
import pandas as pd
import numpy as np

# 1. Setup Directories
base_dir = Path.home() / "phd_research"
ml_dir = base_dir / "data" / "processed" / "ml_ready"

print("=========================================================")
print("   MULTI-OMICS PREPROCESSING INTEGRITY AUDIT REPORT     ")
print("=========================================================\n")

# Load Target Y
target_path = ml_dir / "target_y.csv"
if not target_path.exists():
    raise FileNotFoundError(f"Missing target file at {target_path}")

target_df = pd.read_csv(target_path, index_col=0)
patients = target_df.index.tolist()
print(f"[TARGET Y] Total Matched Cohort: {len(patients)} Patients")
print(f"  - Class Breakdown: {target_df['Y_Stage'].value_counts().to_dict()}\n")

# Define files to inspect
files = {
    "RNA-Seq": ml_dir / "rna_filtered.csv",
    "DNA Methylation": ml_dir / "methylation_filtered.csv",
    "RPPA Proteomics": ml_dir / "rppa_filtered.csv"
}

summary = []

for name, path in files.items():
    if not path.exists():
        print(f"❌ [MISSING]: {name} not found at {path}")
        continue
    
    df = pd.read_csv(path, index_col=0)
    
    # Check 1: Patient alignment
    aligned_patients = list(df.index) == patients
    
    # Check 2: Missing values
    na_count = df.isna().sum().sum()
    
    # Check 3: Zero variance features
    stds = df.std(axis=0)
    zero_var_count = (stds == 0).sum()
    
    # Check 4: Scale check (Mean ~ 0, Std ~ 1)
    mean_val = df.values.mean()
    std_val = df.values.std()
    
    summary.append({
        "Modality": name,
        "Patients (N)": df.shape[0],
        "Features (P)": df.shape[1],
        "Aligned to Y?": "✅ YES" if aligned_patients else "❌ NO",
        "Missing NAs": na_count,
        "Zero-Var Features": zero_var_count,
        "Mean (μ)": f"{mean_val:.3f}",
        "Std Dev (σ)": f"{std_val:.3f}"
    })

# Render Summary Table
summary_df = pd.DataFrame(summary)
print(summary_df.to_string(index=False))
print("\n=========================================================")

if all(s["Aligned to Y?"] == "✅ YES" and s["Missing NAs"] == 0 and s["Zero-Var Features"] == 0 for s in summary):
    print("STATUS: ✅ ALL MODALITIES FULLY PREPROCESSED & READY FOR DIABLO!")
else:
    print("STATUS: ⚠️ PREPROCESSING ISSUES DETECTED. REVIEW TABLE ABOVE.")
print("=========================================================")
