import pandas as pd
import numpy as np
import glob
import os

print("==========================================================")
print("  MULTI-OMICS SHAP BIOMARKER TO DRUG REPURPOSING MATCHING")
print("==========================================================")

# 1. Load Top SHAP Biomarkers
shap_path = "prioritized_multiomics_biomarkers_shap.csv"
if not os.path.exists(shap_path):
    print(f"Error: {shap_path} not found.")
    exit()

shap_df = pd.read_csv(shap_path)
shap_df['Gene_Symbol'] = shap_df['Biomarker'].str.replace('RNA_', '', regex=False).str.replace('PROT_', '', regex=False)
print(f"Loaded {len(shap_df)} prioritized SHAP biomarkers.")

# 2. Locate Drug Reference Files in Workspace
drug_files = glob.glob("**/*drug*.csv", recursive=True)
drug_files = [f for f in drug_files if "prioritized_multiomics" not in f and "repurposed_drug_candidates" not in f]

print(f"\nDiscovered Drug Reference Datasets:")
for f in drug_files:
    print(f" -> {f}")

# 3. Match SHAP Biomarkers Against Drug Target Files
matched_results = []

for dfile in drug_files:
    try:
        df_drug = pd.read_csv(dfile)
        df_string = df_drug.astype(str)
        
        for idx, row in shap_df.iterrows():
            gene = row['Gene_Symbol']
            shap_val = row['SHAP_Importance']
            modality = "Proteomics" if "PROT_" in row['Biomarker'] else "Transcriptomics (RNA)"
            
            # Search for gene symbol matches in the drug dataset
            matches = df_drug[df_string.apply(lambda r: r.str.contains(gene, case=False, regex=False).any(), axis=1)]
            
            if not matches.empty:
                for _, mrow in matches.iterrows():
                    match_dict = {
                        "SHAP_Biomarker": row['Biomarker'],
                        "Gene_Symbol": gene,
                        "Modality": modality,
                        "SHAP_Importance": shap_val,
                        "Source_Dataset": os.path.basename(dfile)
                    }
                    for col in mrow.index[:5]:
                        match_dict[col] = mrow[col]
                    matched_results.append(match_dict)
    except Exception as e:
        print(f"Note skipping {dfile}: {e}")

# 4. Save and Display Output
if matched_results:
    res_df = pd.DataFrame(matched_results).sort_values("SHAP_Importance", ascending=False)
    res_df.to_csv("prioritized_repurposed_drug_candidates.csv", index=False)
    print("\n" + "="*62)
    print("  MATCHED CANDIDATE DRUGS FOR TOP SHAP BIOMARKERS")
    print("="*62)
    print(res_df.head(10).to_string(index=False))
    print(f"\nSaved matched candidate drugs to 'prioritized_repurposed_drug_candidates.csv'")
else:
    print("\nNo direct string matches found in local drug files.")
    print("Exporting target SHAP biomarkers for external DGIdb/ChEMBL lookup...")
    shap_df[['Gene_Symbol', 'Biomarker', 'SHAP_Importance']].to_csv("targets_for_dgidb_lookup.csv", index=False)
    print("Saved target list to 'targets_for_dgidb_lookup.csv'")

