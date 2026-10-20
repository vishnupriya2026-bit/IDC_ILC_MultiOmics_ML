import pandas as pd
import os

os.makedirs("results/tables", exist_ok=True)
excel_path = "results/tables/Supplementary_Tables.xlsx"

with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
    # Table S1: Cohort Quality Control & Sample Dimensions
    if os.path.exists("results/QC/final_cohort_summary.csv"):
        df_s1 = pd.read_csv("results/QC/final_cohort_summary.csv")
        df_s1.to_excel(writer, sheet_name="Table S1 - QC Summary", index=False)

    # Table S2: Top Biomarker Panel & Feature Importances
    if os.path.exists("results/feature_selection/top_20_features_importance.csv"):
        df_s2 = pd.read_csv("results/feature_selection/top_20_features_importance.csv")
        df_s2.to_excel(writer, sheet_name="Table S2 - Top Biomarkers", index=False)

    # Table S3: Benchmark Model Summary
    if os.path.exists("results/tables/final_benchmark_summary.csv"):
        df_s3 = pd.read_csv("results/tables/final_benchmark_summary.csv")
        df_s3.to_excel(writer, sheet_name="Table S3 - Benchmark", index=False)

print(f"[+] Consolidated workbook exported successfully to: {excel_path}")
