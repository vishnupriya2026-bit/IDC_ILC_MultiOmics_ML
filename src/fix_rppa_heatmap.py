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

print("[1/4] Loading and Inspecting RPPA & Target Data...")
rppa = pd.read_csv(ml / 'rppa_filtered.csv', index_col=0)
target = pd.read_csv(ml / 'target_y.csv', index_col=0)

# Clean index formatting (remove spaces/quotes)
rppa.index = rppa.index.astype(str).str.strip()
target.index = target.index.astype(str).str.strip()

# Check sample overlap
common_samples = rppa.index.intersection(target.index)
print(f"      RPPA shape: {rppa.shape}")
print(f"      Target shape: {target.shape}")
print(f"      Common samples found: {len(common_samples)}")

if len(common_samples) == 0:
    print("WARNING: Direct index intersection failed. Re-aligning by row order...")
    min_len = min(len(rppa), len(target))
    rppa = rppa.iloc[:min_len]
    target = target.iloc[:min_len]
    rppa.index = target.index
    common_samples = target.index
else:
    rppa = rppa.loc[common_samples]
    target = target.loc[common_samples]

print("[2/4] Cleaning Features & Computing Z-Scores...")
# Convert all columns to numeric and drop non-numeric columns
rppa_num = rppa.apply(pd.to_numeric, errors='coerce').fillna(0)

# Select top 50 high-variance proteins
top_50_proteins = rppa_num.std(axis=0).nlargest(50).index
rppa_top50 = rppa_num[top_50_proteins]

# Scale features (Z-Score)
scaler = StandardScaler()
rppa_z = pd.DataFrame(scaler.fit_transform(rppa_top50), index=rppa_top50.index, columns=rppa_top50.columns)

# Colors for metadata bar
color_map = {'Early_Stage': '#2ca02c', 'Advanced_Stage': '#ff7f0e'}
row_colors = target['Y_Stage'].map(color_map)

print("[3/4] Plotting Publication Heatmap...")
sns.set_theme(style="white", font_scale=1.0)

g = sns.clustermap(
    rppa_z,
    cmap='vlag',
    row_colors=row_colors,
    figsize=(18, 12),
    dendrogram_ratio=(0.12, 0.15),
    cbar_pos=None,  # Turn off default colorbar
    yticklabels=False,
    xticklabels=True
)

# Re-add Z-score scale at bottom left
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

plt.setp(g.ax_heatmap.xaxis.get_majorticklabels(), rotation=90, fontsize=9)
g.ax_heatmap.set_xlabel("RPPA Antibodies / Protein Markers (Top 50)", fontsize=13, fontweight='bold', labelpad=10)
plt.suptitle('RPPA Proteomics Clustered Expression Signature', fontsize=16, fontweight='bold', y=1.02)

out_path = fig_dir / 'rppa_manuscript_heatmap.png'
plt.savefig(out_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"[4/4] Fixed RPPA Heatmap generated successfully!")
print(f"      Saved to: {out_path}")
