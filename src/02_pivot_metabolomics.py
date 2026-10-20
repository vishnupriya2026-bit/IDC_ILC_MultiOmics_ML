import os
import requests
import pandas as pd
from io import StringIO

OUT_FILE = "data/processed/Metabolomics_harmonized.csv"
print("Fetching clean datatable for ST000355 (Analysis AN000582)...")

# Metabolomics Workbench REST API datatable endpoint
url = "https://www.metabolomicsworkbench.org/rest/study/analysis_id/AN000582/datatable"
res = requests.get(url)

if res.status_code == 200 and len(res.text) > 100:
    # Read tab-delimited datatable directly into Pandas
    df = pd.read_csv(StringIO(res.text), sep="\t")
    
    # Identify feature column (metabolite name)
    first_col = df.columns[0]
    if "metabolite_name" in df.columns:
        df_clean = df.dropna(subset=["metabolite_name"]).set_index("metabolite_name")
    elif "refmet_name" in df.columns:
        df_clean = df.dropna(subset=["refmet_name"]).set_index("refmet_name")
    else:
        df_clean = df.set_index(first_col)
        
    # Convert sample intensity values to numeric floats
    df_numeric = df_clean.apply(pd.to_numeric, errors="coerce")
    
    # Drop all-NaN rows/columns
    df_final = df_numeric.dropna(how="all", axis=0).dropna(how="all", axis=1)
    
    os.makedirs("data/processed", exist_ok=True)
    df_final.to_csv(OUT_FILE)
    
    print(f"✅ SUCCESS! Created Metabolomics Matrix: {df_final.shape} (Metabolites x Samples)")

else:
    print(f"❌ Failed to query datatable API. Status: {res.status_code}")
