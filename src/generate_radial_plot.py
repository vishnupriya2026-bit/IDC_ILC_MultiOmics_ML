import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

base = Path.home() / 'phd_research'
ml = base / 'data' / 'processed' / 'ml_ready'
fig_dir = base / 'results' / 'figures'
fig_dir.mkdir(parents=True, exist_ok=True)

print("[1/3] Loading and Preparing RNA-Seq Data...")
rna = pd.read_csv(ml / 'rna_seq_filtered.csv', index_col=0)
target = pd.read_csv(ml / 'target_y.csv', index_col=0)

common_samples = rna.index.intersection(target.index)
rna = rna.loc[common_samples]
target = target.loc[common_samples]

# 1. Select top 10 features for a clean, readable radial layout
top_10_genes = np.log2(rna + 1).std(axis=0).nlargest(10).index
rna_top10 = np.log2(rna[top_10_genes] + 1)

# 2. Min-Max Scale features (0 to 1) for uniform radial axes
rna_scaled = (rna_top10 - rna_top10.min()) / (rna_top10.max() - rna_top10.min())
rna_scaled['Stage'] = target['Y_Stage']

# Calculate group mean profiles
mean_profiles = rna_scaled.groupby('Stage').mean()

print("[2/3] Building Publication-Ready Polar Radar Plot...")

# Polar coordinate setup
categories = list(mean_profiles.columns)
N = len(categories)

# Calculate angles for each axis (360 / N degrees in radians)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]  # Complete the circle loop

sns_style = True
fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))

# Color setup matching previous figures
color_early = '#2ca02c'
color_adv = '#ff7f0e'

# Plot Early Stage Profile
values_early = mean_profiles.loc['Early_Stage'].values.tolist()
values_early += values_early[:1]
ax.plot(angles, values_early, linewidth=2, linestyle='solid', label='Early Stage', color=color_early)
ax.fill(angles, values_early, color=color_early, alpha=0.20)

# Plot Advanced Stage Profile
values_adv = mean_profiles.loc['Advanced_Stage'].values.tolist()
values_adv += values_adv[:1]
ax.plot(angles, values_adv, linewidth=2, linestyle='solid', label='Advanced Stage', color=color_adv)
ax.fill(angles, values_adv, color=color_adv, alpha=0.20)

# 3. Aesthetics & Labeling
ax.set_theta_offset(np.pi / 2)  # Start top center
ax.set_theta_direction(-1)      # Clockwise direction

# Label each radial spoke with Ensembl IDs
plt.xticks(angles[:-1], categories, size=9, fontweight='bold', family='monospace')

# Radial grid values (0.0 to 1.0)
ax.set_rlabel_position(0)
plt.yticks([0.2, 0.4, 0.6, 0.8], ["0.2", "0.4", "0.6", "0.8"], color="grey", size=8)
plt.ylim(0, 1.0)

# Clean Legend and Title
plt.legend(title="Tumor Stage", loc='upper right', bbox_to_anchor=(1.25, 1.1), frameon=True)
plt.title("Radial Expression Profile (Top 10 Biomarker Genes)", size=14, fontweight='bold', pad=25)

# Save 300 DPI manuscript figure
out_path = fig_dir / 'rna_seq_radial_profile.png'
plt.savefig(out_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"[3/3] Radial plot generated successfully!")
print(f"      Saved to: {out_path}")
