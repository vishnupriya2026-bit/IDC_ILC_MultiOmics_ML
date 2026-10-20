import os
import requests
import pandas as pd

OUT_FILE = "data/processed/Metabolomics_harmonized.csv"
print("Downloading clean tabular ST000915 data via Metabolomics Workbench API...")

# Fetch study data as JSON
url = "https://www.metabolomicsworkbench.org/rest/study/study_id/ST000915/data/json"
res = requests.get(url)

if res.status_code == 200 and res.json():
    data = res.json()
    
    # Extract metabolite intensity measurements
    records = []
    for key, val in data.items():
        if isinstance(val, dict) and "Data" in val:
            records.extend(val["Data"])
            
    if records:
        df_metab = pd.DataFrame(records)
        
        # Determine feature names (metabolite_name or refmet_name)
        feature_col = "metabolite_name" if "metabolite_name" in df_metab.columns else df_metab.columns[0]
        
        # Set feature names as index and drop non-sample metadata columns
        metadata_cols = ["study_id", "analysis_id", "subject_type", "units", "factors"]
        sample_cols = [c for c in df_metab.columns if c not in metadata_cols and c != feature_col]
        
        df_clean = df_metab.set_index(feature_col)[sample_cols]
        df_clean = df_clean.apply(pd.to_numeric, errors="coerce").dropna(how="all")
        
        # Save clean matrix (Features x Samples)
        os.makedirs("data/processed", exist_ok=True)
        df_clean.to_csv(OUT_FILE)
        print(f"✅ Successfully saved Metabolomics Matrix! Shape: {df_clean.shape} (Metabolites x Samples)")
    else:
        print("⚠️ No data records found in JSON response.")
else:
    print(f"❌ Failed to download study ST000915. Status code: {res.status_code}")
