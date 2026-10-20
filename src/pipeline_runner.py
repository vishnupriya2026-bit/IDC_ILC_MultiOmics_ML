import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Dict, Optional
import pandas as pd

from src.utils.session_manager import RunSession
from src.ingestion.validate_input import validate_input_contract
from src.integration.patient_matching import align_patient_cohort
from src.preprocessing.qc import run_quality_control
from src.modeling.benchmark import run_benchmarking
from src.explainability.shap_analysis import run_explainability

class OncoPharmEngine:
    def __init__(self, disease: str):
        self.session = RunSession(disease=disease)
        
    def execute(
        self, 
        datasets: Dict[str, Optional[pd.DataFrame]], 
        target_col: str = "Outcome"
    ) -> dict:
        """
        Executes full end-to-end OncoPharmAI pipeline.
        """
        print(f"[OPA] Initializing Pipeline Run: {self.session.run_id}")
        
        # 1. Validation
        val_report = validate_input_contract(datasets)
        if not val_report.is_valid:
            return {"status": "FAILED", "errors": val_report.errors}

        # Save raw inputs
        for name, df in datasets.items():
            if df is not None:
                df.to_csv(self.session.root_dir / "input" / f"{name}_raw.csv", index=False)

        # 2. Patient Cohort Alignment
        aligned_data = align_patient_cohort(datasets, val_report.matched_ids)
        
        # 3. Quality Control
        cleaned_data, qc_stats = run_quality_control(aligned_data, self.session.root_dir)
        for name, df in cleaned_data.items():
            df.to_csv(self.session.root_dir / "processed" / f"{name}_cleaned.csv", index=False)

        # 4. Feature Matrix Preparation
        clinical_df = cleaned_data.get("clinical")
        if clinical_df is None or target_col not in clinical_df.columns:
            return {
                "status": "PARTIAL_SUCCESS",
                "run_id": self.session.run_id,
                "message": f"Target column '{target_col}' missing from clinical data. ML phase skipped.",
                "qc_summary": qc_stats
            }

        y = clinical_df[target_col]
        
        # Merge omics feature matrices
        feature_dfs = []
        for name, df in cleaned_data.items():
            if name != "clinical":
                feature_dfs.append(df.drop(columns=["Patient_ID"]))
        
        if not feature_dfs:
            return {"status": "FAILED", "errors": ["No omics feature matrices available for modeling."]}

        X = pd.concat(feature_dfs, axis=1)

        # 5. Machine Learning Benchmarking
        ml_results = run_benchmarking(X, y, self.session.root_dir)

        # 6. SHAP / Feature Explainability
        shap_results = run_explainability(
            ml_results["fitted_model"], X, self.session.root_dir
        )

        print(f"[OPA] Workflow Complete. Selected Model: {ml_results['model_name']}")
        
        return {
            "status": "SUCCESS",
            "run_id": self.session.run_id,
            "results_path": str(self.session.root_dir),
            "matched_patients": len(val_report.matched_ids),
            "qc_summary": qc_stats,
            "ml_summary": ml_results["summary"],
            "top_features": shap_results["top_features"]
        }

if __name__ == "__main__":
    # Test end-to-end execution
    rna_mock = pd.DataFrame({
        "Patient_ID": ["P1", "P2", "P3", "P4"], 
        "BRCA1": [2.3, 4.1, 1.2, 3.8],
        "ESR1": [0.1, 1.2, 0.4, 1.1]
    })
    clin_mock = pd.DataFrame({
        "Patient_ID": ["P1", "P2", "P3", "P4"],
        "Outcome": [0, 1, 0, 1]
    })
    
    engine = OncoPharmEngine(disease="Breast Cancer")
    res = engine.execute({"rna": rna_mock, "clinical": clin_mock})
    print("Full Pipeline Execution Output:")
    print(res)