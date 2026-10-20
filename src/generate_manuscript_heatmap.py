import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path
from sklearn.preprocessing import StandardScaler

base = Path.home() / 'phd_research'
ml = base / 'data' / 'processed' / 'ml_ready'
fig_dir = base / 'results' / 'figures'
fig_dir.mkdir(parents=True, exist_ok=True)

# 1. Load and prep data
print("[1/3] Loading and Preprocessing RNA-Seq Data...")
rna = pd.read_csv(ml / 'rna_seq_filtered.csv', index_col=0)
target = pd.read_csv(ml / 'target_y.csv', index_col=0)

common_samples = rna.index.intersection(target.index)
rna = rna.loc[common_samples]
target = target.loc[common_samples]
rna_log = np.log2(rna + 1)

# Select a subset (e.g., top 50 highly variable genes) to make the plot readable
top_50_genes = rna_log.std(axis=0).nlargest(50).index
rna_top50 = rna_log[top_50_genes]

# Compute Z-score standardization across patient samples for each gene
scaler = StandardScaler()
rna_top50_z = scaler.fit_transform(rna_top50)
rna_top50_z_df = pd.DataFrame(rna_top50_z, index=rna_top50.index, columns=rna_top50.columns)

# Setup clinical stage annotation colors (Green, Orange)
stages = target['Y_Stage']
color_map = {'Early_Stage': '#2ca02c', 'Advanced_Stage': '#ff7f0e'} 
row_colors = stages.map(color_map)

# 2. Generate optimized, clean clustered heatmap with high DPI and proper arrangement
print("[2/3] Generating High-Resolution Publication-Grade Clustermap...")

# Set global theme for publication
sns.set_theme(style="white", font_scale=1.1)

# Control dendrogram proportions and make the figure wide enough to accommodate rotated labels
g = sns.clustermap(
    rna_top50_z_df,
    cmap='vlag',        # Diverging palette (Blue-White-Red)
    row_colors=row_colors,
    figsize=(18, 12),   # Large wide size for high resolution and clean margins
    dendrogram_ratio=(0.12, 0.15), # Dendrogram proportions
    cbar_pos=None,      # !!! TURN OFF DEFAULT COLORBAR (Prevents overwriting top dendrogram)
    yticklabels=False,  # Hide 410 individual sample IDs for a clean presentation
    xticklabels=True   # Keep gene Ensembl IDs
)

# 3. Fix Layout: Re-introduce legends entirely outside the plotting area
print("[3/3] Arranging Layout, Fixing Legend Overwrites, and Saving 300 DPI Image...")

# Fix Z-score legend (move it from default overwrite position to bottom left of main plot)
# Create a dedicated axis for the colorbar legend entirely outside the heatmap area.
# [left, bottom, width, height]
cax = g.fig.add_axes([0.02, 0.05, 0.03, 0.15]) 
g.ax_heatmap.figure.colorbar(g.ax_heatmap.collections[0], cax=cax, orientation='vertical')
cax.set_title("Z-Score\nExpression", fontsize=11, fontweight='bold', pad=15)

# Add a dedicated vertical title for the main heatmap row section
g.ax_heatmap.set_ylabel("TCGA Patient Samples (N=410)", fontsize=13, fontweight='bold', labelpad=20)

# Create a dedicated, clean legend for Metadata (Early/Advanced Stage) on the right side
legend_elements = [
    Patch(facecolor='#2ca02c', label='Early Stage'),
    Patch(facecolor='#ff7f0e', label='Advanced Stage')
]
g.ax_heatmap.legend(
    handles=legend_elements,
    title="Tumor Stage\n(Metadata)",
    loc="lower left",
    bbox_to_anchor=(1.02, 0.0), # Place entirely outside the heatmap area
    frameon=True,
    fontsize=11
)

# Arrange X-axis properly: Ensembl IDs must be rotated and made smaller to fit
plt.setp(g.ax_heatmap.xaxis.get_majorticklabels(), rotation=90, fontsize=8, family='monospace')
g.ax_heatmap.set_xlabel("Ensembl Gene IDs (N=50)", fontsize=13, fontweight='bold', labelpad=10)

# Remove any other default messy titles and add a clean, single high-level integrated title
plt.suptitle('RNA-Seq Clustered Gene Expression Signature (Top 50 Varied Genes)', fontsize=16, fontweight='bold', y=1.02)

# Save as 300 DPI high-resolution image with proper tight bounding box to avoid cutoffs
out_path = fig_dir / 'rna_seq_manuscript_heatmap.png'
plt.savefig(out_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Publication heatmap generated successfully and saved to: {out_path}")
