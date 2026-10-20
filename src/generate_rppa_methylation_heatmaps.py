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

target = pd.read_csv(ml / 'target_y.csv', index_col=0)
color_map = {'Early_Stage': '#2ca02c', 'Advanced_Stage': '#ff7f0e'}

# ==========================================
# 1. RPPA (PROTEOMICS) HEATMAP
# ==========================================
print("[1/2] Generating RPPA (Proteomics) Manuscript Heatmap...")
rppa = pd.read_csv(ml / 'rppa_filtered.csv', index_col=0)

common_rppa = rppa.index.intersection(target.index)
rppa = rppa.loc[common_rppa]
target_rppa = target.loc[common_rppa]

# Select top 50 highest variance RPPA protein features
top_50_rppa = rppa.std(axis=0).nlargest(50).index
rppa_top50 = rppa[top_50_rppa]

# Z-score standardization across patient samples
scaler = StandardScaler()
rppa_z = pd.DataFrame(scaler.fit_transform(rppa_top50), index=rppa_top50.index, columns=rppa_top50.columns)
row_colors_rppa = target_rppa['Y_Stage'].map(color_map)

sns.set_theme(style="white", font_scale=1.0)
g1 = sns.clustermap(
    rppa_z,
    cmap='vlag',
    row_colors=row_colors_rppa,
    figsize=(18, 12),
    dendrogram_ratio=(0.12, 0.15),
    cbar_pos=None,  # Turn off default colorbar to prevent dendrogram overlap
    yticklabels=False,
    xticklabels=True
)

# Dedicated Z-Score colorbar at bottom-left
cax1 = g1.fig.add_axes([0.02, 0.05, 0.03, 0.15])
g1.ax_heatmap.figure.colorbar(g1.ax_heatmap.collections[0], cax=cax1, orientation='vertical')
cax1.set_title("Z-Score\nExpression", fontsize=11, fontweight='bold', pad=15)

g1.ax_heatmap.set_ylabel("TCGA Patient Samples (N=410)", fontsize=13, fontweight='bold', labelpad=20)

legend_elements = [
    Patch(facecolor='#2ca02c', label='Early Stage'),
    Patch(facecolor='#ff7f0e', label='Advanced Stage')
]
g1.ax_heatmap.legend(
    handles=legend_elements,
    title="Tumor Stage\n(Metadata)",
    loc="lower left",
    bbox_to_anchor=(1.02, 0.0),
    frameon=True,
    fontsize=11
)

plt.setp(g1.ax_heatmap.xaxis.get_majorticklabels(), rotation=90, fontsize=9)
g1.ax_heatmap.set_xlabel("RPPA Antibodies / Protein Markers (Top 50)", fontsize=13, fontweight='bold', labelpad=10)
plt.suptitle('RPPA Proteomics Clustered Expression Signature', fontsize=16, fontweight='bold', y=1.02)

out_rppa = fig_dir / 'rppa_manuscript_heatmap.png'
plt.savefig(out_rppa, dpi=300, bbox_inches='tight')
plt.close()
print(f"      Saved RPPA Heatmap to: {out_rppa}")

# ==========================================
# 2. DNA METHYLATION HEATMAP
# ==========================================
print("[2/2] Generating DNA Methylation Manuscript Heatmap...")
meth = pd.read_csv(ml / 'methylation_filtered.csv', index_col=0)

common_meth = meth.index.intersection(target.index)
meth = meth.loc[common_meth]
target_meth = target.loc[common_meth]

# Select top 50 highest variance CpG methylation probes
top_50_meth = meth.std(axis=0).nlargest(50).index
meth_top50 = meth[top_50_meth]

# Z-score standardization across patient samples
meth_z = pd.DataFrame(scaler.fit_transform(meth_top50), index=meth_top50.index, columns=meth_top50.columns)
row_colors_meth = target_meth['Y_Stage'].map(color_map)

g2 = sns.clustermap(
    meth_z,
    cmap='vlag',
    row_colors=row_colors_meth,
    figsize=(18, 12),
    dendrogram_ratio=(0.12, 0.15),
    cbar_pos=None,  # Turn off default colorbar to prevent dendrogram overlap
    yticklabels=False,
    xticklabels=True
)

# Dedicated Z-Score colorbar at bottom-left
cax2 = g2.fig.add_axes([0.02, 0.05, 0.03, 0.15])
g2.ax_heatmap.figure.colorbar(g2.ax_heatmap.collections[0], cax=cax2, orientation='vertical')
cax2.set_title("Z-Score\nBeta Value", fontsize=11, fontweight='bold', pad=15)

g2.ax_heatmap.set_ylabel("TCGA Patient Samples (N=410)", fontsize=13, fontweight='bold', labelpad=20)

g2.ax_heatmap.legend(
    handles=legend_elements,
    title="Tumor Stage\n(Metadata)",
    loc="lower left",
    bbox_to_anchor=(1.02, 0.0),
    frameon=True,
    fontsize=11
)

plt.setp(g2.ax_heatmap.xaxis.get_majorticklabels(), rotation=90, fontsize=8, family='monospace')
g2.ax_heatmap.set_xlabel("CpG Probe Identifiers (Top 50)", fontsize=13, fontweight='bold', labelpad=10)
plt.suptitle('DNA Methylation Clustered Profile (Top 50 High-Variance CpG Sites)', fontsize=16, fontweight='bold', y=1.02)

out_meth = fig_dir / 'methylation_manuscript_heatmap.png'
plt.savefig(out_meth, dpi=300, bbox_inches='tight')
plt.close()
print(f"      Saved Methylation Heatmap to: {out_meth}")

print("\n=== BOTH SINGLE-OMICS HEATMAPS SUCCESSFULLY CREATED ===")
