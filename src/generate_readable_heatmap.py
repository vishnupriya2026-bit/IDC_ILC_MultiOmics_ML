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

# Setup clinical stage annotation colors to match the example exactly
stages = target['Y_Stage']
color_map = {'Early_Stage': '#2ca02c', 'Advanced_Stage': '#ff7f0e'} # Green, Orange
row_colors = stages.map(color_map)

# 2. Generate optimized, clean clustered heatmap with high DPI and proper arrangement
print("[2/3] Generating High-Resolution Publication-Grade Clustermap...")

# Set global theme for publication
sns.set_theme(style="white", font_scale=1.1)

# Control dendrogram proportions and make the figure wide enough
g = sns.clustermap(
    rna_top50_z_df,
    cmap='vlag',        # Diverging palette (Blue-White-Red)
    row_colors=row_colors,
    figsize=(18, 12),   # Large wide size for high resolution and clean margins
    dendrogram_ratio=(0.12, 0.15), # Dendrogram proportions
    cbar_pos=(0.02, 0.85, 0.03, 0.15), # Floating Z-Score colorbar at top-left
    yticklabels=False,  # Hide 410 individual sample IDs for a clean presentation
    xticklabels=True   # Keep gene Ensembl IDs and make them readable
)

# 3. Refine layout, move legends to avoid overwriting, and arrange labels
print("[3/3] Arranging Layout and Saving 300 DPI Image...")

# Fix Z-score title and position
g.ax_cbar.set_title("Z-Score\nExpression", fontsize=11, fontweight='bold', pad=15)

# Add a dedicated vertical title for the main heatmap row section
g.ax_heatmap.set_ylabel("TCGA Patients", fontsize=13, fontweight='bold', labelpad=20)

# Position the Metadata colorbar and give it a clean title
g.ax_row_colors.set_title("Metadata\n(Tumor Stage)", fontsize=11, fontweight='bold', pad=15, rotation=90)
g.ax_row_colors.set_ylabel("Tumor Stage", fontsize=10, fontweight='bold')

# Create a dedicated, clean legend for Metadata (Early/Advanced Stage) on the right side
legend_elements = [
    Patch(facecolor='#2ca02c', label='Early Stage'),
    Patch(facecolor='#ff7f0e', label='Advanced Stage')
]
g.ax_heatmap.legend(
    handles=legend_elements,
    title="Metadata Legend",
    loc="lower left",
    bbox_to_anchor=(1.02, 0.0), # Place entirely outside the heatmap area
    frameon=True,
    fontsize=11
)

# Arrange X-axis properly: Ensembl IDs must be rotated and made smaller to fit
plt.setp(g.ax_heatmap.xaxis.get_majorticklabels(), rotation=90, fontsize=8, family='monospace')
g.ax_heatmap.set_xlabel("Ensembl Gene IDs (N=50)", fontsize=13, fontweight='bold', labelpad=10)

# Remove the default clutter on top and add a clean, integrated title
plt.suptitle('Hierarchical Cluster and Expression Map: RNA-Seq Data', fontsize=16, fontweight='bold', y=1.02)

# Add the main 'Metadata Legend' top title separately above the legend area
plt.figtext(0.96, 0.95, "Metadata Legend", fontsize=13, fontweight='bold', horizontalalignment='center')

# Save as 300 DPI high-resolution image with proper tight bounding box to avoid cutoffs
out_path = fig_dir / 'rna_seq_readable_heatmap.png'
plt.savefig(out_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Readable heatmap generated successfully and saved to: {out_path}")
