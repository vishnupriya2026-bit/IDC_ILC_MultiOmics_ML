import pandas as pd

files_to_check = [
    "data/processed/survival_scorecard_results.csv",
    "results/task23_patient_risk_scorecard.csv",
    "results/task16_cox_regression.csv",
    "results/task15_survival_summary.csv",
    "results/external_validation_metrics.csv",
    "results/OPA-2026-06383/processed/clinical_cleaned.csv",
    "results/OPA-2026-55101/processed/clinical_cleaned.csv"
]

for filepath in files_to_check:
    print("="*60)
    print(f"FILE: {filepath}")
    try:
        df = pd.read_csv(filepath)
        print("Shape:", df.shape)
        print("Columns:", df.columns.tolist()[:10])
        print("Head:")
        print(df.head(2))
    except Exception as e:
        print("Could not read file:", e)
