import os
import glob
import pandas as pd

home_dir = os.path.expanduser('~')
print("\n" + "="*65)
print(" 1. SEARCHING FOR METABRIC REFERENCES IN YOUR PYTHON CODE")
print("="*65)

py_files = []
for root, dirs, files in os.walk(home_dir):
    if any(skip in root for skip in ['ml_env', '.cache', '.local', '.git', 'node_modules']):
        continue
    for f in files:
        if f.endswith('.py'):
            py_files.append(os.path.join(root, f))

found_code_refs = False
for py_file in py_files:
    try:
        with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            for line_no, line in enumerate(lines, 1):
                if 'metabric' in line.lower():
                    found_code_refs = True
                    print(f"Script: {py_file} (Line {line_no})")
                    print(f"   --> {line.strip()}\n")
    except Exception:
        pass

if not found_code_refs:
    print("No METABRIC code references found in Python scripts.\n")

print("="*65)
print(" 2. SEARCHING FOR METABRIC DATASET FILES ON DISK")
print("="*65)

data_files = []
for root, dirs, files in os.walk(home_dir):
    if any(skip in root for skip in ['ml_env', '.cache', '.local', '.git', 'node_modules']):
        continue
    for f in files:
        if 'metabric' in f.lower() and f.endswith(('.csv', '.tsv', '.txt', '.parquet', '.pkl', '.feather')):
            data_files.append(os.path.join(root, f))

data_files = list(set(data_files))

if not data_files:
    print("No offline METABRIC data files found on disk.")
    print("Your project likely downloads the 'brca_metabric' cohort dynamically via cBioPortal API.")
else:
    for filepath in data_files:
        print(f"\nFound Dataset: {filepath}")
        try:
            sep = '\t' if filepath.endswith(('.tsv', '.txt')) else ','
            if filepath.endswith('.pkl'):
                df = pd.read_pickle(filepath)
            elif filepath.endswith('.parquet'):
                df = pd.read_parquet(filepath)
            else:
                df = pd.read_csv(filepath, sep=sep, low_memory=False)
            
            print(f"  Shape: {df.shape[0]} rows (patients) x {df.shape[1]} columns (features)")
            
            sub_cols = [c for c in df.columns if any(k in c.lower() for k in ['histolog', 'subtype', 'cancer_type', 'type', 'label'])]
            if sub_cols:
                print(f"  Histology / Subtype Column(s): {sub_cols}")
                for col in sub_cols[:2]:
                    print(f"  Class breakdown for [{col}]:")
                    print(df[col].value_counts().dropna().head(5))
        except Exception as e:
            print(f"  Could not load dataset details: {e}")

