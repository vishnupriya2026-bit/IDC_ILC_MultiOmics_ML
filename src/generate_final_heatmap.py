import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path

base = Path.home() / 'phd_research'
ml = base / 'data' / 'processed' / 'ml_ready'
fig_dir = base / 'results' / 'figures'
fig_dir.mkdir(parents=True, exist_ok=True)

# 1. Load and prepare data
print("[1/3] Loading and Preprocessing RNA-Seq Data...")
rna = pd.read_csv(ml / 'rna_seq_filtered.csv', index_col=0)
target = pd.read_csv(ml / 'target_y.csv', index_col=0)

common_samples = rna.index.intersection(target.index)
rna = rna.loc[common_samples]
target = target.loc[common_samples]
rna_log = np.log2(rna + 1)

# Select top 50 high-variance genes for readability
top_50_genes = rna_log.std(axis=0).nlargest(50).index
rna_top50 = rna_log[top_50_genes]

# Setup clinical stage annotation colors
stages = target['Y_Stage']
color_map = {'Early_Stage': '#2ca02c', 'Advanced_Stage': '#ff7f0e'}
row_colors = stages.map(color_map)

# 2. Generate optimized, clean clustered heatmap
print("[2/3] Generating Uncluttered Publication-Grade Clustermap...")

# Set global theme for publication
sns.set_theme(style="white", font_scale=0.9)

g = sns.clustermap(
    rna_top50,
    z_score=0,          # Z-score standardization across samples for each gene
    cmap='vlag',        # Diverging palette (Blue-White-Red)
    row_colors=row_colors,
    figsize=(11, 9),    # Large size for high resolution
    dendrogram_ratio=(0.12, 0.12), # Compact dendrograms
    cbar_pos=(0.02, 0.82, 0.03, 0.13), # Floating Z-Score colorbar top-left
    yticklabels=False,  # Hide sample IDs for clean presentation
    xticklabels=False   # Hide complex gene IDs
)

# Fix Colorbar label
g.ax_cbar.set_title("Z-Score\nExpression", fontsize=9, fontweight='bold', pad=10)

# 3. Add clean dedicated legends without overlaps
legend_elements = [
    Patch(facecolor='#2ca02c', label='Early Stage'),
    Patch(facecolor='#ff7f0e', label='Advanced Stage')
]

g.ax_heatmap.legend(
    handles=legend_elements,
    title="Tumor Stage",
    loc="lower left",
    bbox_to_anchor=(1.02, 0.0), # Position entirely outside the heatmap area
    frameon=True,
    fontsize=9
)

# Add clean titles/labels
plt.suptitle('RNA-Seq Gene Expression Signature (Top 50 Varied Genes)', fontsize=14, fontweight='bold', y=1.01)
g.ax_heatmap.set_xlabel("Genes (N=50)", fontsize=11, fontweight='bold')
g.ax_heatmap.set_ylabel("Patient Samples (N=410)", fontsize=11, fontweight='bold')

# Save as 300 DPI high-resolution image
out_path = fig_dir / 'rna_seq_manuscript_heatmap.png'
plt.savefig(out_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"[3/3] Final heatmap generated successfully!")
print(f"      Saved to: {out_path}")
