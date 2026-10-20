import os
import pandas as pd
import numpy as np

# Set up output directory
os.makedirs("results/final_report", exist_ok=True)

print("=== Step 8: Consolidating Final Multi-Omics Pipeline Summary ===")

# ---------------------------------------------------------
# 1. Load Step Outputs
# ---------------------------------------------------------
try:
    batch_df = pd.read_csv("results/batch_benchmark/batch_correction_comparison.csv")
except Exception:
    batch_df = None

try:
    ml_df = pd.read_csv("results/ml_benchmark/model_accuracy_comparison.csv")
except Exception:
    ml_df = None

try:
    opt_df = pd.read_csv("results/advanced_ml/optimized_model_comparison.csv")
except Exception:
    opt_df = None

# ---------------------------------------------------------
# 2. Compile Pipeline Milestones
# ---------------------------------------------------------
summary_records = []

# Step 1-3: QC & Matched Samples
summary_records.append({
    'Pipeline Phase': 'Phase 1: Preprocessing & Alignment',
    'Metric / Step': 'Matched Multi-Omics Cohort Size',
    'Value / Result': '119 Samples (RNA + Proteomics)'
})

# Step 4: Batch Correction Benchmark
if batch_df is not None:
    best_batch_method = batch_df.sort_values(by='Batch_Silhouette (Lower=Better)').iloc[0]['Method']
    best_batch_score = batch_df.sort_values(by='Batch_Silhouette (Lower=Better)').iloc[0]['Batch_Silhouette (Lower=Better)']
    summary_records.append({
        'Pipeline Phase': 'Phase 2: Batch Effect Correction',
        'Metric / Step': f'Best Mixing Algorithm ({best_batch_method})',
        'Value / Result': f'Silhouette Score: {best_batch_score}'
    })

# Step 5: Feature Selection
summary_records.append({
    'Pipeline Phase': 'Phase 3: High-Dimensional Filtering',
    'Metric / Step': 'Selected Top Biomarkers',
    'Value / Result': '50 Refined Multi-Omics Features'
})

# Step 6: Baseline ML Benchmarks
if ml_df is not None:
    top_baseline_model = ml_df.iloc[0]['Model']
    top_baseline_acc = ml_df.iloc[0]['Mean Accuracy']
    summary_records.append({
        'Pipeline Phase': 'Phase 4: Baseline ML Benchmarking',
        'Metric / Step': f'Top Single Classifier ({top_baseline_model})',
        'Value / Result': f'{top_baseline_acc*100:.2f}% CV Accuracy'
    })

# Step 7: Advanced Multi-Omics Optimization
if opt_df is not None:
    top_opt_model = opt_df.iloc[0]['Model']
    top_opt_acc = opt_df.iloc[0]['Mean CV Accuracy']
    summary_records.append({
        'Pipeline Phase': 'Phase 5: Multi-Omics Stacking Ensemble',
        'Metric / Step': f'Champion Model ({top_opt_model})',
        'Value / Result': f'{top_opt_acc*100:.2f}% CV Accuracy'
    })

# ---------------------------------------------------------
# 3. Export Final Report
# ---------------------------------------------------------
master_report = pd.DataFrame(summary_records)
master_report.to_csv("results/final_report/master_pipeline_summary.csv", index=False)

print("\n=======================================================")
print("          MASTER PIPELINE MANUSCRIPT SUMMARY           ")
print("=======================================================")
print(master_report.to_string(index=False))
print("=======================================================")
print("\nFinal report successfully saved to 'results/final_report/master_pipeline_summary.csv'!")
