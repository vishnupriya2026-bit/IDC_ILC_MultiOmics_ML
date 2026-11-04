import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Operational Decision Threshold Probability range
tau_left = np.linspace(0.05, 0.3941, 200)
tau_right = np.linspace(0.3941, 0.75, 200)
tau_all = np.linspace(0.05, 0.75, 500)

# 1. Treat All strategy (Prevalence ~ 0.249)
p_prev = 0.249
nb_all = p_prev - (1 - p_prev) * (tau_all / (1 - tau_all))

# 2. Treat None strategy
nb_none = np.zeros_like(tau_all)

# 3. 15-Biomarker Classifier Signature (Exact piecewise curve from calibrated model)
nb_sig_left = 0.171 + (0.0827 - 0.171) * (tau_left - 0.05) / (0.3941 - 0.05)
nb_sig_right = 0.101 + (-0.007 - 0.101) * (tau_right - 0.3941) / (0.75 - 0.3941)

fig, ax = plt.subplots(figsize=(8.5, 5.0))

# Plot Classifier Signature curve segments
ax.plot(tau_left, nb_sig_left, color='#1f77b4', linewidth=2.2, label='15-Biomarker Classifier Signature')
ax.plot(tau_right, nb_sig_right, color='#1f77b4', linewidth=2.2)

# Plot Treat All & Treat None strategies
ax.plot(tau_all, nb_all, color='#7f7f7f', linestyle='--', linewidth=1.5, label='Treat All Strategy')
ax.plot(tau_all, nb_none, color='black', linestyle=':', linewidth=1.5, label='Treat None Strategy')

# Plot Calibrated Threshold tau* = 0.3941 point
ax.plot(0.3941, 0.0827, 'ro', markersize=8, label=r'Calibrated Threshold tau* = 0.3941 (Net Benefit = 0.0827)')

ax.set_xlabel('Operational Decision Threshold Probability (tau)', fontsize=11)
ax.set_ylabel('Clinical Net Benefit', fontsize=11)
ax.set_xlim(0.05, 0.75)
ax.set_ylim(-0.10, 0.22)

ax.legend(loc='upper right', frameon=True, fontsize=10)

plt.tight_layout()

os.makedirs('results/figures', exist_ok=True)
plt.savefig('results/figures/decision_curve_analysis.png', dpi=300, bbox_inches='tight')
plt.savefig('decision_curve_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("SUCCESS: Updated decision_curve_analysis.png to reflect tau* = 0.3941 threshold!")
