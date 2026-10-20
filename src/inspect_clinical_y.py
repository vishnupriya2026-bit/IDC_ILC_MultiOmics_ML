import os
import pandas as pd

base_dir = os.path.expanduser("~/phd_research/data")

# 1. Load Master Cohort and Clinical Data
master_path = os.path.join(base_dir, "master_cohort_metadata.csv")
clin_path = os.path.join(base_dir, "clinical/extracted/clinical.tsv")

if not os.path.exists(master_path):
    raise FileNotFoundError(f"Master metadata missing at {master_path}. Run cohort intersection first.")

master_df = pd.read_csv(master_path)
clin_df = pd.read_csv(clin_path, sep="\t", low_memory=False)

cohort_pts = set(master_df['patient_id'])
print(f"Master Cohort Cohort Size: {len(cohort_pts)} patients")

# 2. Match Clinical Records to Master Cohort
clin_id_col = next((c for c in ['submitter_id', 'bcr_patient_barcode', 'case_submitter_id'] if c in clin_df.columns), clin_df.columns[0])
cohort_clin = clin_df[clin_df[clin_id_col].isin(cohort_pts)].copy()

# 3. Search Key Phenotypic Variables
keywords = ['estrogen', 'progesterone', 'her2', 'receptor', 'stage', 'subtype', 'pam50', 'histologic', 'vital_status']
candidate_cols = [c for c in cohort_clin.columns if any(k in c.lower() for k in keywords)]

print("\n" + "="*60)
print(" CLINICAL TARGET (Y) CANDIDATE EVALUATION ")
print("="*60)

for col in candidate_cols:
    s = cohort_clin[col].fillna('Missing').astype(str).str.strip()
    s = s.replace({'nan': 'Missing', 'not reported': 'Missing', '[Not Evaluated]': 'Missing', "'--": 'Missing', '--': 'Missing'})
    
    counts = s.value_counts()
    missing_count = counts.get('Missing', 0)
    valid_count = len(s) - missing_count
    
    # Filter out columns with no valid reported values
    if valid_count > 0:
        print(f"\n[Variable]: {col}")
        print(f"Coverage: {valid_count}/{len(s)} patients ({valid_count/len(s)*100:.1f}% complete)")
        for val, cnt in counts.items():
            print(f"  • {val}: {cnt}")

