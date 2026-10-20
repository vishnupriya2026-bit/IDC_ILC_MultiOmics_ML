import os
import pandas as pd

DATA_DIR = "/home/priya_26phd0306/BreastCancer_MultiOmics/data/processed"

print("=" * 65)
print("CHECKING FOR AUTHENTIC CLINICAL SURVIVAL ENDPOINTS")
print("=" * 65)

# 1. Check GSE2034 Harmonized
gse_path = os.path.join(DATA_DIR, "GSE2034_harmonized.csv")
if os.path.exists(gse_path):
    df_gse = pd.read_csv(gse_path, nrows=10)
    print("\n[+] GSE2034 Column Sample:")
    print(list(df_gse.columns[:10]))

# 2. Search all CSV files in processed folder for survival columns
for f in os.listdir(DATA_DIR):
    if f.endswith('.csv'):
        fp = os.path.join(DATA_DIR, f)
        try:
            df_head = pd.read_csv(fp, nrows=2)
            cols = [c.lower() for c in df_head.columns]
            survival_matches = [c for c in df_head.columns if any(k in c.lower() for k in ['rfs', 'relapse', 'dfs', 'os', 'time', 'status', 'event', 'months', 'days'])]
            if survival_matches and f != "synthetic_clinical_survival.csv":
                print(f"\n★ FOUND POTENTIAL CLINICAL METADATA IN: {f}")
                print(f"   Matching Columns: {survival_matches}")
        except Exception:
            pass

print("\n" + "=" * 65)
