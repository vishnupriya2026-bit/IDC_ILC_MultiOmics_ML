import os
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/task12_multiomics_validation.csv")

fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

modalities = df['Modality']
auc_means = df['ROC_AUC_Mean']
auc_stds = df['ROC_AUC_Std']

colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
bars = ax.bar(modalities, auc_means, yerr=auc_stds, capsize=6, color=colors, alpha=0.85, edgecolor='black', linewidth=1.2)

ax.set_ylim(0.92, 1.00)
ax.set_ylabel('Mean 5-Fold ROC-AUC', fontweight='bold')
ax.set_title('CPTAC Multi-Omics Cross-Validation Performance', fontweight='bold')
ax.grid(axis='y', linestyle=':', alpha=0.6)

for bar, mean, std in zip(bars, auc_means, auc_stds):
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + std + 0.0015, f"{mean:.4f} ± {std:.4f}", ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.xticks(fontweight='bold')
plt.tight_layout()
os.makedirs("results", exist_ok=True)
output_path = "results/task12_multiomics_benchmark.png"
plt.savefig(output_path, dpi=300)
plt.close()

print(f"Benchmark plot saved to {output_path}")
