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

print("[1/4] Loading RPPA Data...")
file_path = ml / 'rppa_filtered.csv'

# Load CSV
df = pd.read_csv(file_path, index_col=0)

# Drop Y_Stage if present inside the dataset columns
if 'Y_Stage' in df.columns:
    df = df.drop(columns=['Y_Stage'])

# Detect matrix orientation (ensure patients in rows, features in columns)
tcga_in_index = df.index.astype(str).str.contains('TCGA|CPTAC', case=False).sum()
tcga_in_cols = df.columns.astype(str).str.contains('TCGA|CPTAC', case=False).sum()

if tcga_in_cols > tcga_in_index:
    print("      Transposing matrix (Samples detected in columns)...")
    df = df.T

print(f"      Working Shape: {df.shape} (Patient Samples x Features)")

print("[2/4] Cleaning & Converting Feature Values to Numeric...")
df_numeric = pd.DataFrame(index=df.index)

for col in df.columns:
    s = df[col].astype(str).str.replace('"', '').str.replace("'", "").str.replace(',', '.').str.strip()
    converted = pd.to_numeric(s, errors='coerce')
    if not converted.isna().all():
        df_numeric[col] = converted

# Impute missing values with column median
df_numeric = df_numeric.apply(lambda c: c.fillna(c.median()), axis=0)

# Filter out zero-variance features
stds = df_numeric.std(axis=0)
valid_cols = stds[stds > 1e-6].index
print(f"      Valid protein features with non-zero variance: {len(valid_cols)} / {df_numeric.shape[1]}")

df_valid = df_numeric[valid_cols]

# Select top 50 highest-variance protein features (AGID accession IDs)
top50_cols = stds.loc[valid_cols].nlargest(min(50, len(valid_cols))).index
rppa_top50 = df_valid[top50_cols]

# Compute Z-score standardization across patient samples
scaler = StandardScaler()
rppa_z = pd.DataFrame(scaler.fit_transform(rppa_top50), index=rppa_top50.index, columns=rppa_top50.columns)

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

print("[4/4] Generating High-Contrast Publication Heatmap...")
color_map = {'Early_Stage': '#2ca02c', 'Advanced_Stage': '#ff7f0e'}
row_colors = target['Y_Stage'].map(color_map)

sns.set_theme(style="white", font_scale=1.0)

g = sns.clustermap(
    rppa_z,
    cmap='vlag',
    vmin=-2.5, vmax=2.5,  # Force high contrast range
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
