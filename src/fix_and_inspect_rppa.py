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

file_path = ml / 'rppa_filtered.csv'
print(f"[1/4] Inspecting Raw RPPA File: {file_path}")

# Load raw without index assumption to safely check headers
df_raw = pd.read_csv(file_path)
print(f"      Raw Shape: {df_raw.shape}")
print(f"      First 3 Columns: {df_raw.columns[:3].tolist()}")

# Determine index column (Sample IDs)
if df_raw.columns[0].lower() in ['unnamed: 0', 'sample', 'sample_id', 'tcga_id', 'barcode', 'id', '']:
    df_raw = df_raw.set_index(df_raw.columns[0])
else:
    first_col_vals = df_raw.iloc[:, 0].astype(str)
    if first_col_vals.str.contains('TCGA|CPTAC', case=False, regex=True).any():
        df_raw = df_raw.set_index(df_raw.columns[0])
    else:
        df_raw = pd.read_csv(file_path, index_col=0)

print(f"      Matrix shape after indexing: {df_raw.shape}")

print("[2/4] Robust Cleaning & Element-wise Numeric Conversion...")

# Helper function to strip bad text characters and convert to float
def safe_float(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip().replace('"', '').replace("'", "").replace(',', '.')
    try:
        return float(s)
    except ValueError:
        return np.nan

# Vectorized conversion across entire matrix
if hasattr(df_raw, 'map'):
    df_numeric = df_raw.map(safe_float)
else:
    df_numeric = df_raw.applymap(safe_float)

# Fill missing values with column medians
df_numeric = df_numeric.apply(lambda col: col.fillna(col.median()), axis=0)
# Drop columns that are completely NaN
df_numeric = df_numeric.dropna(axis=1, how='all')

# Variance Check
col_stds = df_numeric.std(axis=0)
valid_features = (col_stds > 1e-5).sum()
print(f"      Valid protein features identified: {valid_features} / {df_numeric.shape[1]}")

if valid_features == 0:
    raise ValueError("ERROR: Still 0 valid numeric features found. Check if the CSV file is corrupted or empty.")

# Select top 50 highest variance features
top_50_cols = col_stds.nlargest(min(50, df_numeric.shape[1])).index
rppa_top50 = df_numeric[top_50_cols]

# Compute Z-scores manually
rppa_z = (rppa_top50 - rppa_top50.mean(axis=0)) / (rppa_top50.std(axis=0) + 1e-8)

print("[3/4] Aligning with Clinical Target Y...")
target = pd.read_csv(ml / 'target_y.csv', index_col=0)

rppa_z.index = rppa_z.index.astype(str).str.strip()
target.index = target.index.astype(str).str.strip()

common_samples = rppa_z.index.intersection(target.index)
print(f"      Overlapping patient samples: {len(common_samples)}")

if len(common_samples) > 0:
    rppa_z = rppa_z.loc[common_samples]
    target = target.loc[common_samples]
else:
    print("      Re-aligning by row position...")
    min_len = min(len(rppa_z), len(target))
    rppa_z = rppa_z.iloc[:min_len]
    target = target.iloc[:min_len]
    rppa_z.index = target.index

print("[4/4] Plotting High-Contrast Publication Heatmap...")
color_map = {'Early_Stage': '#2ca02c', 'Advanced_Stage': '#ff7f0e'}
row_colors = target['Y_Stage'].map(color_map)

sns.set_theme(style="white", font_scale=1.0)

g = sns.clustermap(
    rppa_z,
    cmap='vlag',
    vmin=-2.5, vmax=2.5,  # Force strong color contrast (Blue/Red)
    row_colors=row_colors,
    figsize=(18, 12),
    dendrogram_ratio=(0.12, 0.15),
    cbar_pos=None,        # Prevents overwriting dendrogram
    yticklabels=False,
    xticklabels=True
)

# Position Z-Score legend at bottom-left
cax = g.fig.add_axes([0.02, 0.05, 0.03, 0.15])
g.ax_heatmap.figure.colorbar(g.ax_heatmap.collections[0], cax=cax, orientation='vertical')
cax.set_title("Z-Score\nExpression", fontsize=11, fontweight='bold', pad=15)

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

plt.setp(g.ax_heatmap.xaxis.get_majorticklabels(), rotation=90, fontsize=8)
g.ax_heatmap.set_xlabel("RPPA Protein Markers (Top 50 Varied)", fontsize=13, fontweight='bold', labelpad=10)
plt.suptitle('RPPA Proteomics Clustered Expression Signature', fontsize=16, fontweight='bold', y=1.02)

out_path = fig_dir / 'rppa_manuscript_heatmap.png'
plt.savefig(out_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"\nSUCCESS: RPPA Heatmap generated and saved to: {out_path}")
