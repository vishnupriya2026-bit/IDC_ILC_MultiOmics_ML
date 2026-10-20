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

print("[1/4] Loading and Inspecting RPPA Data...")
rppa = pd.read_csv(ml / 'rppa_filtered.csv', index_col=0)
target = pd.read_csv(ml / 'target_y.csv', index_col=0)

# Align sample IDs cleanly
rppa.index = rppa.index.astype(str).str.strip()
target.index = target.index.astype(str).str.strip()

common_samples = rppa.index.intersection(target.index)
if len(common_samples) == 0:
    print("      Re-aligning sample index by row position...")
    min_len = min(len(rppa), len(target))
    rppa = rppa.iloc[:min_len]
    target = target.iloc[:min_len]
    rppa.index = target.index
    common_samples = target.index
else:
    rppa = rppa.loc[common_samples]
    target = target.loc[common_samples]

print(f"      Matched samples: {len(common_samples)}")

print("[2/4] Parsing Clean Numeric Matrix & Removing Zero Variance Columns...")
# Clean strings without coercing everything to NaN
for col in rppa.columns:
    if rppa[col].dtype == object:
        rppa[col] = rppa[col].astype(str).str.replace('"', '').str.replace("'", "").str.strip()
        rppa[col] = pd.to_numeric(rppa[col], errors='coerce')

# Fill true missing values with column median rather than 0
rppa = rppa.fillna(rppa.median(numeric_only=True))

# Identify columns with actual biological variance (> 0.001)
col_stds = rppa.std(axis=0)
valid_cols = col_stds[col_stds > 1e-3].index
print(f"      Valid protein features with non-zero variance: {len(valid_cols)} / {rppa.shape[1]}")

# Select top 50 highest variance valid protein features
top_50_proteins = col_stds.loc[valid_cols].nlargest(50).index
rppa_top50 = rppa[top_50_proteins]

# Compute Z-score manually per column
rppa_z = (rppa_top50 - rppa_top50.mean(axis=0)) / rppa_top50.std(axis=0)

print(f"      Z-Score Range Check: Min = {rppa_z.min().min():.2f}, Max = {rppa_z.max().max():.2f}")

print("[3/4] Plotting High-Contrast Publication Heatmap...")
color_map = {'Early_Stage': '#2ca02c', 'Advanced_Stage': '#ff7f0e'}
row_colors = target['Y_Stage'].map(color_map)

sns.set_theme(style="white", font_scale=1.0)

g = sns.clustermap(
    rppa_z,
    cmap='vlag',
    vmin=-2.5, vmax=2.5,  # Force sharp color distribution contrast
    row_colors=row_colors,
    figsize=(18, 12),
    dendrogram_ratio=(0.12, 0.15),
    cbar_pos=None,
    yticklabels=False,
    xticklabels=True
)

# Colorbar positioning
cax = g.fig.add_axes([0.02, 0.05, 0.03, 0.15])
g.ax_heatmap.figure.colorbar(g.ax_heatmap.collections[0], cax=cax, orientation='vertical')
cax.set_title("Z-Score\nExpression", fontsize=11, fontweight='bold', pad=15)

g.ax_heatmap.set_ylabel(f"TCGA Patient Samples (N={len(common_samples)})", fontsize=13, fontweight='bold', labelpad=20)

legend_elements = [
    Patch(facecolor='#2ca02c', label='Early Stage'),
    Patch(facecolor='#ff7f0e', label='Advanced Stage')
]
g.ax_heatmap.legend(
    handles=legend_elements,
    title="Tumor Stage\n(Metadata)",
    loc="lower left",
    bbox_to_anchor=(1.02, 0.0),
    frameon=True,
    fontsize=11
)

plt.setp(g.ax_heatmap.xaxis.get_majorticklabels(), rotation=90, fontsize=8)
g.ax_heatmap.set_xlabel("RPPA Protein Markers (Top 50 Varied)", fontsize=13, fontweight='bold', labelpad=10)
plt.suptitle('RPPA Proteomics Clustered Expression Signature', fontsize=16, fontweight='bold', y=1.02)

out_path = fig_dir / 'rppa_manuscript_heatmap.png'
plt.savefig(out_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"[4/4] Successfully generated populated RPPA heatmap at: {out_path}")
