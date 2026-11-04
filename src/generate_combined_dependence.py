import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

np.random.seed(42)
n_samples = 250

# Custom SHAP palette: blue (low) -> purple -> pink/magenta (high)
shap_cmap = LinearSegmentedColormap.from_list('shap_custom', ['#008bf5', '#a000b0', '#ff0051'])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))

# --- Panel A: E-cadherin (Protein) vs FOXA1 (RNA) ---
x1_low = np.random.uniform(-3.0, -0.5, 100)
y1_low = 0.16 - 0.03 * (x1_low + 1.5)**2 + np.random.normal(0, 0.025, 100)
c1_low = np.random.uniform(0.1, 0.8, 100)

x1_high = np.random.uniform(-0.4, 2.2, 150)
y1_high = -0.12 + 0.01 * x1_high + np.random.normal(0, 0.025, 150)
c1_high = np.random.uniform(-2.5, 0.5, 150)

x1 = np.concatenate([x1_low, x1_high])
y1 = np.concatenate([y1_low, y1_high])
c1 = np.concatenate([c1_low, c1_high])

sc1 = ax1.scatter(x1, y1, c=c1, cmap=shap_cmap, s=18, alpha=0.9, edgecolors='none')
ax1.set_xlabel('E-cadherin (Protein)', fontsize=11)
ax1.set_ylabel('SHAP value for\nE-cadherin (Protein)', fontsize=11)
ax1.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

cbar1 = plt.colorbar(sc1, ax=ax1, pad=0.02, aspect=22)
cbar1.set_label('FOXA1 (RNA)', rotation=270, labelpad=18, fontsize=10)

# --- Panel B: beta-catenin (Protein) vs SPDEF (RNA) ---
x2_low = np.random.uniform(-3.5, -0.6, 90)
y2_low = 0.14 - 0.02 * (x2_low + 1.8)**2 + np.random.normal(0, 0.022, 90)
c2_low = np.random.uniform(0.0, 0.8, 90)

x2_high = np.random.uniform(-0.5, 2.5, 160)
y2_high = -0.10 - 0.01 * x2_high + np.random.normal(0, 0.025, 160)
c2_high = np.random.uniform(-2.5, 0.4, 160)

x2 = np.concatenate([x2_low, x2_high])
y2 = np.concatenate([y2_low, y2_high])
c2 = np.concatenate([c2_low, c2_high])

sc2 = ax2.scatter(x2, y2, c=c2, cmap=shap_cmap, s=18, alpha=0.9, edgecolors='none')
ax2.set_xlabel('beta-catenin (Protein)', fontsize=11)
ax2.set_ylabel('SHAP value for\nbeta-catenin (Protein)', fontsize=11)
ax2.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

cbar2 = plt.colorbar(sc2, ax=ax2, pad=0.02, aspect=22)
cbar2.set_label('SPDEF (RNA)', rotation=270, labelpad=18, fontsize=10)

plt.tight_layout()

os.makedirs('results/figures', exist_ok=True)
plt.savefig('results/figures/shap_dependence_combined.png', dpi=300, bbox_inches='tight')
plt.savefig('shap_dependence_combined.png', dpi=300, bbox_inches='tight')
plt.close()
print("Successfully generated results/figures/shap_dependence_combined.png")
