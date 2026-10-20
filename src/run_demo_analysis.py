import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.load_demo import load_cptac_demo_datasets
from src.pipeline_runner import OncoPharmEngine

def main():
    print("=== Starting OncoPharmAI Real Dataset Execution ===")
    
    # 1. Load CPTAC Data
    datasets = load_cptac_demo_datasets()
    
    if "clinical" not in datasets or datasets["clinical"] is None:
        print("[OPA] Error: Clinical dataset missing.")
        return

    # Select target outcome column
    target_column = "PAM50" if "PAM50" in datasets["clinical"].columns else "Outcome"
    if target_column not in datasets["clinical"].columns:
        cols = [c for c in datasets["clinical"].columns if c != "Patient_ID"]
        target_column = cols[0] if cols else "Outcome"
        
    print(f"[OPA] Target Outcome Column selected: '{target_column}'")

    # 2. Run Engine Pipeline
    engine = OncoPharmEngine(disease="Breast Cancer (CPTAC)")
    results = engine.execute(datasets, target_col=target_column)

    print("\n=== Execution Summary ===")
    print(f"Status:          {results.get('status')}")
    print(f"Run ID:          {results.get('run_id')}")
    print(f"Matched Cohort:  {results.get('matched_patients')} patients")
    print(f"Results Path:    {results.get('results_path')}")
    
    if "ml_summary" in results:
        print(f"Selected Model:  {results['ml_summary'].get('selected_model')}")
        print(f"Best F1 Score:   {results['ml_summary'].get('best_f1_score')}")

if __name__ == "__main__":
    main()