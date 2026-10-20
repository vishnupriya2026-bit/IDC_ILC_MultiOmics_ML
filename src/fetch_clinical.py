import pandas as pd
import requests

print("=== CPTAC BRCA Clinical Metadata Retrieval ===")

# 1. Load patient IDs
matched_ids_file = "CPTAC_119_matched_patient_ids.csv"
try:
    matched_ids_df = pd.read_csv(matched_ids_file)
    id_col = matched_ids_df.columns[0]
    my_119_ids = set(matched_ids_df[id_col].astype(str).str.strip().tolist())
    print(f"Loaded {len(my_119_ids)} target patient IDs from {matched_ids_file}.")
except Exception as e:
    print(f"Warning: Could not read {matched_ids_file} ({e}). Proceeding with all retrieved clinical records.")
    my_119_ids = None

clinical_df = None

# 2. Try fetching via cptac package (specifying source explicitly)
try:
    import cptac
    brca = cptac.Brca()
    clinical_df = brca.get_clinical(source="mssm")
    print("Successfully fetched clinical data via CPTAC package (MSSM source).")
except Exception as err:
    print(f"CPTAC package call failed ({err}).")
    print("Switching to direct cBioPortal CPTAC 2020 REST API retrieval...")

# 3. Fallback: Query cBioPortal API directly for CPTAC 2020 Cohort
if clinical_df is None:
    url = "https://www.cbioportal.org/api/studies/brca_cptac_2020/clinical-data?clinicalDataType=PATIENT"
    resp = requests.get(url)
    if resp.status_code == 200:
        raw_df = pd.DataFrame(resp.json())
        # Pivot table: Patients x Attributes
        clinical_df = raw_df.pivot(index="patientId", columns="clinicalAttributeId", values="value")
        print("Successfully fetched CPTAC 2020 Breast Cancer metadata from cBioPortal API!")
    else:
        print(f"Error fetching from cBioPortal API (Status: {resp.status_code}).")

# 4. Filter to 119 matched IDs and Save
if clinical_df is not None:
    if my_119_ids:
        # Filter matching rows
        clinical_df.index = clinical_df.index.astype(str).str.strip()
        matched_clinical = clinical_df[clinical_df.index.isin(my_119_ids)]
        print(f"\nSuccessfully matched {len(matched_clinical)} / {len(my_119_ids)} patient IDs!")
    else:
        matched_clinical = clinical_df

    print("\nSummary of Clinical Columns Available:")
    print(matched_clinical.columns.tolist())
    
    output_filename = "cptac_119_clinical_matched.csv"
    matched_clinical.to_csv(output_filename)
    print(f"\nSaved matched clinical data to '{output_filename}'.")
else:
    print("\nFailed to retrieve clinical data. Please check network connectivity.")

