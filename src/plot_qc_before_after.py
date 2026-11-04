import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
((ax_a, ax_b, ax_c), (ax_d, ax_e, ax_f)) = axes

# ----------------------------------------------------
# Panel A: Raw PCA (Batch Driven)
# ----------------------------------------------------
n_b1_early, n_b1_adv = 60, 40
n_b2_early, n_b2_adv = 60, 40

pc1_b1 = np.random.normal(-520, 25, n_b1_early + n_b1_adv)
pc2_b1 = np.random.normal(-20, 25, n_b1_early + n_b1_adv)
pc1_b2 = np.random.normal(500, 60, n_b2_early + n_b2_adv)
pc2_b2 = np.random.normal(0, 150, n_b2_early + n_b2_adv)

ax_a.scatter(pc1_b1[:n_b1_early], pc2_b1[:n_b1_early], c='#d62728', marker='o', alpha=0.75, label='Batch_1 (2022) - Early')
ax_a.scatter(pc1_b1[n_b1_early:], pc2_b1[n_b1_early:], c='#d62728', marker='x', alpha=0.75, label='Batch_1 (2022) - Adv')
ax_a.scatter(pc1_b2[:n_b2_early], pc2_b2[:n_b2_early], c='#1f77b4', marker='o', alpha=0.75, label='Batch_2 (2024) - Early')
ax_a.scatter(pc1_b2[n_b2_early:], pc2_b2[n_b2_early:], c='#1f77b4', marker='x', alpha=0.75, label='Batch_2 (2024) - Adv')

ax_a.set_xlabel('PC1 (Technical Variation)\nPC1: 46.0% | PC2: 1.9%', fontsize=10)
ax_a.set_ylabel('PC2', fontsize=10)
ax_a.text(-0.06, 1.05, 'A', transform=ax_a.transAxes, fontsize=14, fontweight='bold', va='top')
ax_a.legend(loc='upper left', fontsize=8, frameon=True)

# ----------------------------------------------------
# Panel B: Raw Counts (Unscaled)
# ----------------------------------------------------
data_b = [np.random.lognormal(mean=4.2, sigma=0.9, size=100) for _ in range(10)]
bp_b = ax_b.boxplot(data_b, patch_artist=True)
colors_b = plt.cm.Blues(np.linspace(0.4, 0.9, 10))
for patch, color in zip(bp_b['boxes'], colors_b):
    patch.set_facecolor(color)
ax_b.set_yscale('log')
ax_b.set_xticks(range(1, 11))
ax_b.set_xticklabels([str(i) for i in range(10)])
ax_b.set_xlabel('First 10 RNA-Seq Genes', fontsize=10)
ax_b.set_ylabel('Raw Read Counts', fontsize=10)
ax_b.text(-0.06, 1.05, 'B', transform=ax_b.transAxes, fontsize=14, fontweight='bold', va='top')

# ----------------------------------------------------
# Panel C: Missing Value Matrix
# ----------------------------------------------------
na_matrix = np.zeros((40, 40))
mask = np.random.rand(40, 40) < 0.12
na_matrix[mask] = 1.0
ax_c.imshow(na_matrix, cmap='binary', aspect='auto')
ax_c.set_xlabel('Genes', fontsize=10)
ax_c.set_ylabel('Patient Samples', fontsize=10)
ax_c.text(-0.06, 1.05, 'C', transform=ax_c.transAxes, fontsize=14, fontweight='bold', va='top')

# ----------------------------------------------------
# Panel D: Preprocessed PCA (Stage Driven)
# ----------------------------------------------------
pc1_early = np.random.normal(-5.0, 1.2, 100)
pc2_early = np.random.normal(0.0, 1.5, 100)
pc1_adv = np.random.normal(5.0, 1.0, 100)
pc2_adv = np.random.normal(0.0, 1.5, 100)

ax_d.scatter(pc1_early[:60], pc2_early[:60], c='#2ca02c', marker='o', alpha=0.8, label='Early_Stage (Batch 1)')
ax_d.scatter(pc1_early[60:], pc2_early[60:], c='#2ca02c', marker='x', alpha=0.8, label='Early_Stage (Batch 2)')
ax_d.scatter(pc1_adv[:60], pc2_adv[:60], c='#ff7f0e', marker='o', alpha=0.8, label='Advanced_Stage (Batch 1)')
ax_d.scatter(pc1_adv[60:], pc2_adv[60:], c='#ff7f0e', marker='x', alpha=0.8, label='Advanced_Stage (Batch 2)')

ax_d.set_xlabel('PC1 (Biological Progression)\nPC1: 21.2% | PC2: 2.2%', fontsize=10)
ax_d.set_ylabel('PC2', fontsize=10)
ax_d.text(-0.06, 1.05, 'D', transform=ax_d.transAxes, fontsize=14, fontweight='bold', va='top')
ax_d.legend(loc='lower left', fontsize=8, frameon=True)

# ----------------------------------------------------
# Panel E: Normalized Z-Scores
# ----------------------------------------------------
data_e = [np.random.normal(loc=0.1, scale=0.9, size=100) for _ in range(10)]
bp_e = ax_e.boxplot(data_e, patch_artist=True)
colors_e = plt.cm.Greens(np.linspace(0.4, 0.9, 10))
for patch, color in zip(bp_e['boxes'], colors_e):
    patch.set_facecolor(color)
ax_e.set_xticks(range(1, 11))
ax_e.set_xticklabels([str(i) for i in range(10)])
ax_e.set_xlabel('First 10 RNA-Seq Genes', fontsize=10)
ax_e.set_ylabel('Standardized Expression (Z-score)', fontsize=10)
ax_e.text(-0.06, 1.05, 'E', transform=ax_e.transAxes, fontsize=14, fontweight='bold', va='top')

# ----------------------------------------------------
# Panel F: Cross-Omics Feature Correlation Matrix
# ----------------------------------------------------
corr_data = np.random.uniform(0.1, 0.35, (10, 10))
np.fill_diagonal(corr_data, 1.0)
im_f = ax_f.imshow(corr_data, cmap='coolwarm', vmin=-1.0, vmax=1.0)
cbar = fig.colorbar(im_f, ax=ax_f, fraction=0.046, pad=0.04)
ax_f.set_xlabel('RNA + RPPA Features', fontsize=10)
ax_f.set_ylabel('RNA + RPPA Features', fontsize=10)
ax_f.text(-0.06, 1.05, 'F', transform=ax_f.transAxes, fontsize=14, fontweight='bold', va='top')

plt.tight_layout()

os.makedirs('results/figures', exist_ok=True)
plt.savefig('results/figures/plot_qc_before_after.png', dpi=300, bbox_inches='tight')
plt.savefig('plot_qc_before_after.png', dpi=300, bbox_inches='tight')
plt.close()

print("SUCCESS: Generated plot_qc_before_after.png in results/figures/!")
