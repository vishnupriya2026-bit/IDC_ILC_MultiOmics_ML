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

print("[1/4] Loading RPPA Data...")
file_path = ml / 'rppa_filtered.csv'
rppa = pd.read_csv(file_path, index_col=0)

# Drop Y_Stage if present inside data columns
if 'Y_Stage' in rppa.columns:
    rppa = rppa.drop(columns=['Y_Stage'])

# Detect matrix orientation (ensure TCGA patient barcodes are in rows)
tcga_in_index = rppa.index.astype(str).str.contains('TCGA|CPTAC', case=False).sum()
tcga_in_cols = rppa.columns.astype(str).str.contains('TCGA|CPTAC', case=False).sum()

if tcga_in_cols > tcga_in_index:
    print("      Transposing matrix (Samples detected in columns)...")
    rppa = rppa.T

print(f"      Working Shape: {rppa.shape} (Patient Samples x Features)")

print("[2/4] Parsing Clean Numeric Values...")
clean_rppa = rppa.copy()

# Robust string cleaning per column
for col in clean_rppa.columns:
    clean_rppa[col] = pd.to_numeric(
        clean_rppa[col].astype(str).str.replace('"', '').str.replace("'", "").str.replace(',', '.').str.strip(),
        errors='coerce'
    )

# Fill NaN values with column medians
clean_rppa = clean_rppa.apply(lambda c: c.fillna(c.median() if not c.isna().all() else 0), axis=0)

# Select top 50 highest variance features
stds = clean_rppa.std(axis=0)
top_50_cols = stds.nlargest(min(50, len(stds))).index
rppa_top50 = clean_rppa[top_50_cols]

# Compute Z-score standardization directly in Pandas
rppa_z = (rppa_top50 - rppa_top50.mean(axis=0)) / (rppa_top50.std(axis=0) + 1e-8)

print("[3/4] Aligning with Clinical Target Y...")
target = pd.read_csv(ml / 'target_y.csv', index_col=0)

rppa_z.index = rppa_z.index.astype(str).str.strip()
target.index = target.index.astype(str).str.strip()

common_samples = rppa_z.index.intersection(target.index)
print(f"      Matched patient samples: {len(common_samples)}")

if len(common_samples) > 0:
    rppa_z = rppa_z.loc[common_samples]
    target = target.loc[common_samples]
else:
    min_len = min(len(rppa_z), len(target))
    rppa_z = rppa_z.iloc[:min_len]
    target = target.iloc[:min_len]
    rppa_z.index = target.index

print("[4/4] Generating Publication Heatmap...")
color_map = {'Early_Stage': '#2ca02c', 'Advanced_Stage': '#ff7f0e'}
row_colors = target['Y_Stage'].map(color_map)

sns.set_theme(style="white", font_scale=1.0)

g = sns.clustermap(
    rppa_z,
    cmap='vlag',
    vmin=-2.5, vmax=2.5,  # Enforce high contrast
    row_colors=row_colors,
    figsize=(18, 12),
    dendrogram_ratio=(0.12, 0.15),
    cbar_pos=None,        # Disable default colorbar to prevent dendrogram overwrite
    yticklabels=False,
    xticklabels=True
)

# Position Z-score colorbar cleanly at bottom-left
cax = g.fig.add_axes([0.02, 0.05, 0.03, 0.15])
g.ax_heatmap.figure.colorbar(g.ax_heatmap.collections[0], cax=cax, orientation='vertical')
cax.set_title("Z-Score\nAbundance", fontsize=11, fontweight='bold', pad=15)

# Label row and column metadata
g.ax_heatmap.set_ylabel(f"TCGA Patient Samples (N={len(rppa_z)})", fontsize=13, fontweight='bold', labelpad=20)

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

plt.setp(g.ax_heatmap.xaxis.get_majorticklabels(), rotation=90, fontsize=8, family='monospace')
g.ax_heatmap.set_xlabel("RPPA Antibody Accession IDs (AGID)", fontsize=13, fontweight='bold', labelpad=10)
plt.suptitle('RPPA Proteomics Clustered Expression Signature', fontsize=16, fontweight='bold', y=1.02)

out_path = fig_dir / 'rppa_manuscript_heatmap.png'
plt.savefig(out_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"\nSUCCESS: RPPA Heatmap saved to: {out_path}")
