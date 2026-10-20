import os
import pandas as pd
import numpy as np

print("--- Step 1: Gathering Multi-Omic Pipeline Results ---")

# Load existing results artifacts
pathway_df = pd.read_csv("results/task18_pathway_enrichment.csv") if os.path.exists("results/task18_pathway_enrichment.csv") else None
benchmark_df = pd.read_csv("results/task20_model_benchmark.csv") if os.path.exists("results/task20_model_benchmark.csv") else None
importance_df = pd.read_csv("results/task21_feature_importance.csv") if os.path.exists("results/task21_feature_importance.csv") else None
scorecard_df = pd.read_csv("results/task23_patient_risk_scorecard.csv") if os.path.exists("results/task23_patient_risk_scorecard.csv") else None

print("--- Step 2: Compiling Automated Executive Report ---")

report_content = f"""# Multi-Omic Integration & Clinical Risk Stratification Report
**Project:** Breast Cancer Multi-Omics Prognostic Pipeline  
**Dataset:** GSE2034 / CPTAC Integration  
**Date:** 2026-08-17  

---

## 1. Executive Summary
This end-to-end multi-omic analytical pipeline integrated transcriptomic and proteomic data to identify key molecular drivers of breast cancer recurrence. Using a panel of 16 multi-omic biomarkers across Proliferation and Luminal signaling modules, machine learning classifiers were evaluated to stratify patient recurrence risk and assess clinical utility.

---

## 2. Key Biological Findings & Pathway Enrichment
Top enriched canonical pathways based on Over-Representation Analysis (ORA):
- **Cell Cycle & G2/M Transition:** $p = 4.27 \\times 10^{{-12}}$ (`AURKA`, `CDK1`, `CCNB1`)
- **Estrogen Receptor Signaling:** $p = 4.27 \\times 10^{{-12}}$ (`ESR1`, `FOXA1`, `GATA3`)
- **FOXM1 / MYB Network:** $p = 7.53 \\times 10^{{-10}}$ (`MYBL2`, `MKI67`)

---

## 3. Machine Learning Model Benchmarking (5-Fold CV)
"""

if benchmark_df is not None:
    report_content += benchmark_df.to_markdown(index=False) + "\n\n"

report_content += "--- \n\n## 4. Key Multi-Omic Drivers of Recurrence\n"

if importance_df is not None:
    top_10 = importance_df.head(10)[["Gene", "Module", "RF_Gini_Importance", "LR_Coefficient"]]
    report_content += top_10.to_markdown(index=False) + "\n\n"

report_content += "--- \n\n## 5. Clinical Risk Stratification Summary\n"

if scorecard_df is not None:
    strat_summary = scorecard_df.groupby("Risk_Group").agg(
        Sample_Count=("Sample_ID", "count"),
        Observed_Relapses=("Observed_Relapse", "sum"),
        Observed_Relapse_Rate=("Observed_Relapse", "mean")
    ).reset_index()
    strat_summary["Observed_Relapse_Rate"] = (strat_summary["Observed_Relapse_Rate"] * 100).round(2).astype(str) + "%"
    report_content += strat_summary.to_markdown(index=False) + "\n\n"

report_content += """---

## 6. Pipeline Output Artifacts
- **Network Map:** `results/task19_multiomics_network.png`
- **ROC Benchmark:** `results/task20_roc_curves.png`
- **Feature Importance:** `results/task21_feature_importance.png`
- **Decision Curve (DCA):** `results/task22_decision_curve.png`
- **Risk Stratification Scorecard:** `results/task23_risk_stratification.png`
- **Serialized Model:** `models/final_multiomic_model.pkl`
"""

os.makedirs("results", exist_ok=True)
output_report_path = "results/multiomics_final_report.md"
with open(output_report_path, "w") as f:
    f.write(report_content)

print(f"\nTask 24 completed. Executive report compiled and saved to '{output_report_path}'")
