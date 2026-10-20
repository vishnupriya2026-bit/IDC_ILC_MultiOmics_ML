import pandas as pd
import glob
import os

# 1. Automatically find the 119 IDs CSV file
id_files = glob.glob("**/*119*.csv", recursive=True)
if not id_files:
    # Default fallback check
    id_files = [f for f in os.listdir('.') if 'matched' in f.lower() or '119' in f.lower()]

if id_files:
    target_id_file = id_files[0]
    print(f"Found patient ID file: '{target_id_file}'")
    matched_ids_df = pd.read_csv(target_id_file)
    id_col = matched_ids_df.columns[0]
    my_119_ids = set(matched_ids_df[id_col].astype(str).str.strip().tolist())
else:
    print("Could not automatically locate the 119 IDs file. Please specify path manually.")
    exit()

# 2. Load the full retrieved clinical metadata
clinical_df = pd.read_csv("cptac_119_clinical_matched.csv", index_col=0)
clinical_df.index = clinical_df.index.astype(str).str.strip()

# 3. Filter down to 119 matched patients
matched_clinical = clinical_df[clinical_df.index.isin(my_119_ids)]

print(f"\nSuccessfully matched: {len(matched_clinical)} / {len(my_119_ids)} patient records.")

# 4. Check Key Machine Learning Target Variables
print("\n--- Target Variable Distribution ---")
if 'tumor_stage_pathological' in matched_clinical.columns:
    print("\nTumor Stage Breakdown:")
    print(matched_clinical['tumor_stage_pathological'].value_counts(dropna=False))

if 'Survivalstatus (1, dead; 0, alive)' in matched_clinical.columns:
    print("\nOverall Survival Status Breakdown:")
    print(matched_clinical['Survivalstatus (1, dead; 0, alive)'].value_counts(dropna=False))

# 5. Overwrite file with exact 119 matched patient rows
matched_clinical.to_csv("cptac_119_clinical_final.csv")
print("\nSaved final 119 matched clinical metadata to 'cptac_119_clinical_final.csv'.")
