import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Dict, Tuple
import pandas as pd
import json

def run_quality_control(
    aligned_datasets: Dict[str, pd.DataFrame], 
    run_dir: Path, 
    id_col: str = "Patient_ID",
    max_missing_pct: float = 0.3
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, dict]]:
    """
    Evaluates feature missingness, removes features exceeding max_missing_pct threshold,
    and exports QC metrics to the session's qc/ directory.
    """
    qc_summary = {}
    cleaned_datasets = {}
    qc_dir = run_dir / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)

    for name, df in aligned_datasets.items():
        if df is None or df.empty:
            continue
            
        numeric_cols = df.select_dtypes(include=["number"]).columns
        missing_pcts = df[numeric_cols].isnull().mean()
        
        # High missingness drop threshold
        failed_features = missing_pcts[missing_pcts > max_missing_pct].index.tolist()
        passed_cols = [c for c in df.columns if c not in failed_features]
        
        cleaned_df = df[passed_cols].copy()
        # Basic mean imputation for numeric features passing QC threshold
        for col in cleaned_df.select_dtypes(include=["number"]).columns:
            if cleaned_df[col].isnull().any():
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
                
        cleaned_datasets[name] = cleaned_df
        
        qc_summary[name] = {
            "initial_samples": df.shape[0],
            "initial_features": df.shape[1] - 1, # Excluding ID column
            "dropped_high_missing_features": len(failed_features),
            "retained_features": cleaned_df.shape[1] - 1,
            "overall_missingness_pct": round(float(df[numeric_cols].isnull().sum().sum()) / (df.shape[0] * max(len(numeric_cols), 1)) * 100, 2)
        }

    # Save QC metrics JSON to current session directory
    with open(qc_dir / "qc_summary.json", "w") as f:
        json.dump(qc_summary, f, indent=4)

    return cleaned_datasets, qc_summary

if __name__ == "__main__":
    import numpy as np
    
    rna_mock = pd.DataFrame({
        "Patient_ID": ["P1", "P2", "P3"], 
        "BRCA1": [2.3, np.nan, 1.2], 
        "BAD_GENE": [np.nan, np.nan, 1.0]
    })
    
    mock_run_dir = Path("results/test_qc_run")
    cleaned, stats = run_quality_control({"rna": rna_mock}, mock_run_dir)
    print("QC Execution Complete. Stats:")
    print(json.dumps(stats, indent=2))