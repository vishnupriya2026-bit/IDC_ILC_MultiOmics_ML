import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

modalities = ['RNA-Seq', 'DNA Methylation', 'RPPA Proteomics']
counts = [10000, 20000, 457]  # 457 preprocessed RPPA protein markers

fig, ax = plt.subplots(figsize=(8.5, 5))
colors = ['#2b5c8f', '#2596be', '#d62728']

bars = ax.bar(modalities, counts, color=colors, width=0.5, alpha=0.9)

for bar in bars:
    yval = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        yval + 400,
        f'{yval:,}',
        ha='center',
        va='bottom',
        fontweight='bold',
        fontsize=11
    )

ax.set_ylabel('Filtered & Normalized Features', fontsize=11)
ax.set_ylim(0, 23000)
ax.grid(True, axis='y', linestyle='--', alpha=0.35)
ax.set_axisbelow(True)

plt.tight_layout()

os.makedirs('results/figures', exist_ok=True)
plt.savefig('results/figures/plot_b_updated_feature_scale.png', dpi=300, bbox_inches='tight')
plt.savefig('plot_b_updated_feature_scale.png', dpi=300, bbox_inches='tight')
plt.close()

print("SUCCESS: Updated plot_b_updated_feature_scale.png!")
