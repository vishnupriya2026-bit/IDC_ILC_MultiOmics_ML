import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

ml_dir = Path("data/processed/ml_ready")

# Load train feature counts
rna_count = pd.read_csv(ml_dir / "rna_train.csv", index_col=0).shape[1]
meth_count = pd.read_csv(ml_dir / "methylation_train.csv", index_col=0).shape[1]
rppa_count = pd.read_csv(ml_dir / "rppa_train.csv", index_col=0).shape[1]

# Plot Setup
fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
modalities = ["RNA-Seq", "DNA Methylation", "RPPA Proteomics"]
counts = [rna_count, meth_count, rppa_count]
colors = ["#3B528B", "#21908C", "#5DC863"]

bars = ax.bar(modalities, counts, color=colors, width=0.5)

# Add exact feature count labels on top of bars
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + (max(counts)*0.015), f"{yval:,}", ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_title("ML-Ready Feature Scale per Modality (Zero Missing Values)", fontsize=13, fontweight='bold', pad=15)
ax.set_ylabel("Filtered & Normalized Features", fontsize=11)
ax.set_ylim(0, max(counts) * 1.15)
ax.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
out_path = Path("results/figures/plot_b_updated_feature_scale.png")
out_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_path, dpi=300)
print(f"Updated Plot B saved to {out_path}")
