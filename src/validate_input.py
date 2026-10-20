from dataclasses import dataclass
from typing import List, Dict, Optional
import pandas as pd

@dataclass
class ValidationReport:
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    matched_ids: List[str]
    metrics: Dict[str, int]

def validate_input_contract(
    datasets: Dict[str, Optional[pd.DataFrame]], 
    id_col: str = "Patient_ID"
) -> ValidationReport:
    """
    Validates input datasets against schema standards, checks duplicate keys,
    and identifies common patient cohort overlap across omics modalities.
    """
    errors = []
    warnings = []
    matched_ids_set = None
    metrics = {}

    for name, df in datasets.items():
        if df is None:
            continue
        
        # 1. Check ID column existence
        if id_col not in df.columns:
            errors.append(f"Dataset '{name}' lacks required key column '{id_col}'.")
            continue
            
        # 2. Check for duplicate Patient IDs
        if df[id_col].duplicated().any():
            errors.append(f"Dataset '{name}' contains duplicate {id_col} entries.")

        # 3. Collect Patient IDs and calculate intersection
        current_ids = set(df[id_col].dropna().astype(str))
        metrics[f"{name}_samples"] = len(current_ids)
        
        if matched_ids_set is None:
            matched_ids_set = current_ids
        else:
            matched_ids_set = matched_ids_set.intersection(current_ids)

    matched_ids = list(matched_ids_set) if matched_ids_set else []
    metrics["matched_patients"] = len(matched_ids)

    # Threshold checks
    if len(matched_ids) == 0:
        errors.append("Zero overlapping Patient_IDs found across provided datasets.")
    elif len(matched_ids) < 15:
        warnings.append(f"Low matched cohort size ({len(matched_ids)} patients). Cross-validation may be unstable.")

    return ValidationReport(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        matched_ids=matched_ids,
        metrics=metrics
    )

if __name__ == "__main__":
    # Test execution with dummy data
    rna_mock = pd.DataFrame({"Patient_ID": ["P1", "P2", "P3"], "BRCA1": [2.3, 4.1, 1.2]})
    prot_mock = pd.DataFrame({"Patient_ID": ["P2", "P3", "P4"], "TP53_prot": [0.1, 0.9, 0.4]})
    
    rep = validate_input_contract({"rna": rna_mock, "proteomics": prot_mock})
    print("Is Valid:", rep.is_valid)
    print("Matched Cohort Size:", rep.metrics["matched_patients"])
    print("Matched IDs:", rep.matched_ids)