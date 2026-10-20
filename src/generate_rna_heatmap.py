import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

base = Path.home() / 'phd_research'
ml = base / 'data' / 'processed' / 'ml_ready'
fig_dir = base / 'results' / 'figures'
fig_dir.mkdir(parents=True, exist_ok=True)

print("[1/3] Loading RNA-Seq & Clinical Stage Data...")
rna = pd.read_csv(ml / 'rna_seq_filtered.csv', index_col=0)
target = pd.read_csv(ml / 'target_y.csv', index_col=0)

# Align common samples
common_samples = rna.index.intersection(target.index)
rna = rna.loc[common_samples]
target = target.loc[common_samples]

# Log2 transformation log2(x + 1)
rna_log = np.log2(rna + 1)

# Select top 50 most variable genes across samples
top_50_genes = rna_log.std(axis=0).nlargest(50).index
rna_top50 = rna_log[top_50_genes]

# Map tumor stages to color palette for sample annotations
stages = target['Y_Stage']
unique_stages = stages.unique()
palette = sns.color_palette('Set2', n_colors=len(unique_stages))
color_map = dict(zip(unique_stages, palette))
row_colors = stages.map(color_map)

print("[2/3] Generating Hierarchical Clustered Heatmap...")
sns.set_theme(style="white", font_scale=0.9)

g = sns.clustermap(
    rna_top50,
    z_score=0,          # Standardize across samples for each gene (row z-score)
    cmap='vlag',        # Blue (low expression) to Red (high expression)
    row_colors=row_colors,
    figsize=(12, 10),
    dendrogram_ratio=(0.15, 0.15),
    cbar_kws={'label': 'Z-Score Expression'},
    yticklabels=False,  # Hide 410 individual sample IDs for clean presentation
    xticklabels=True
)

# Add Legend for Clinical Tumor Stages
for stage, color in color_map.items():
    g.ax_col_dendrogram.bar(0, 0, color=color, label=stage, linewidth=0)

g.ax_col_dendrogram.legend(
    title='Tumor Stage', 
    loc='center', 
    bbox_to_anchor=(0.5, 1.25), 
    ncol=len(unique_stages),
    frameon=True
)

g.fig.suptitle('RNA-Seq Gene Expression Heatmap (Top 50 High-Variance Genes)', y=1.02, fontsize=14, fontweight='bold')

out_path = fig_dir / 'rna_seq_heatmap.png'
plt.savefig(out_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"[3/3] Heatmap generated successfully!")
print(f"      Saved to: {out_path}")
