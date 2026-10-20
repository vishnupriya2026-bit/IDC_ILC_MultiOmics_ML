from typing import Dict, Optional
import pandas as pd

def align_patient_cohort(
    datasets: Dict[str, Optional[pd.DataFrame]], 
    matched_ids: list, 
    id_col: str = "Patient_ID"
) -> Dict[str, pd.DataFrame]:
    """
    Filters and aligns all input omics/clinical datasets to contain 
    only the intersected cohort of Patient_IDs, sorted in identical order.
    """
    aligned_datasets = {}
    
    for name, df in datasets.items():
        if df is None:
            continue
        
        # Ensure ID column is string type for exact matching
        df_copy = df.copy()
        df_copy[id_col] = df_copy[id_col].astype(str)
        
        # Filter to matched IDs and sort by ID to ensure row-order alignment
        filtered_df = df_copy[df_copy[id_col].isin(matched_ids)].sort_values(by=id_col).reset_index(drop=True)
        aligned_datasets[name] = filtered_df

    return aligned_datasets

if __name__ == "__main__":
    rna_mock = pd.DataFrame({"Patient_ID": ["P1", "P2", "P3"], "BRCA1": [2.3, 4.1, 1.2]})
    prot_mock = pd.DataFrame({"Patient_ID": ["P3", "P2", "P4"], "TP53": [0.9, 0.1, 0.4]})
    
    matched = ["P2", "P3"]
    aligned = align_patient_cohort({"rna": rna_mock, "proteomics": prot_mock}, matched)
    
    print("Aligned RNA Patients:", aligned["rna"]["Patient_ID"].tolist())
    print("Aligned Prot Patients:", aligned["proteomics"]["Patient_ID"].tolist())