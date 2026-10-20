import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig_dpi = 300

# 1. Plot Multi-Omics SHAP Importance Bar Chart
print("1. Generating SHAP Biomarker Importance Figure...")
shap_df = pd.read_csv("prioritized_multiomics_biomarkers_shap.csv").head(15)

plt.figure(figsize=(10, 6))
palette = ['#2b5c8f' if x.startswith('PROT_') else '#d95f02' for x in shap_df['Biomarker']]
ax = sns.barplot(x='SHAP_Importance', y='Biomarker', data=shap_df, palette=palette)

plt.title("Top 15 Multi-Omics Biomarkers Ranked by SHAP Importance", fontsize=14, fontweight='bold', pad=15)
plt.xlabel("Mean |SHAP Value| (Impact on Model Output)", fontsize=12)
plt.ylabel("Biomarker (RNA / Proteomics)", fontsize=12)

# Custom legend for Omics modalities
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#d95f02', label='RNA Transcriptomics'),
    Patch(facecolor='#2b5c8f', label='Proteomics Feature')
]
plt.legend(handles=legend_elements, loc='lower right', frameon=True)

plt.tight_layout()
plt.savefig("fig_shap_multiomics_biomarkers.png", dpi=fig_dpi)
plt.close()
print("Saved 'fig_shap_multiomics_biomarkers.png'")

# 2. Plot Model Performance Comparison Chart
print("2. Generating Model Benchmark Comparison Figure...")
bench_df = pd.read_csv("ml_multiomics_benchmarks.csv")

bench_melted = bench_df.melt(id_vars=['Model'], value_vars=['ROC_AUC', 'F1_Score', 'Accuracy'],
                             var_name='Metric', value_name='Score')

plt.figure(figsize=(10, 5))
sns.barplot(x='Model', y='Score', hue='Metric', data=bench_melted, palette='viridis')
plt.title("Multi-Omics Machine Learning Model Performance Comparison", fontsize=14, fontweight='bold', pad=15)
plt.ylim(0, 1.0)
plt.ylabel("Score", fontsize=12)
plt.xlabel("Classifier Model", fontsize=12)
plt.legend(loc='upper right')
plt.xticks(rotation=15)

plt.tight_layout()
plt.savefig("fig_multiomics_model_benchmarks.png", dpi=fig_dpi)
plt.close()
print("Saved 'fig_multiomics_model_benchmarks.png'")

print("\nAll figures generated successfully!")
