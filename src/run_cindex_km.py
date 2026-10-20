import pandas as pd
import numpy as np
from lifelines.utils import concordance_index
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt

# Load CPTAC merged data
df = pd.read_csv("cptac_119_clinical_final.csv")
time_col = 'Overall survival, days'
event_col = 'Survival status (1, dead; 0, alive)'

# Filter valid survival rows
df = df.dropna(subset=[time_col, event_col])
df[time_col] = pd.to_numeric(df[time_col], errors='coerce')
df[event_col] = pd.to_numeric(df[event_col], errors='coerce')

# Mock or load your multi-omics prediction score
np.random.seed(42)
df['multiomics_score'] = np.random.uniform(0, 1, size=len(df))

# Calculate Harrell's Concordance Index (C-Index)
c_idx = concordance_index(df[time_col], -df['multiomics_score'], df[event_col])
print(f"[+] Multi-Omics Concordance Index (C-index): {c_idx:.3f}")

# Plot Kaplan-Meier High vs Low Risk Groups
df['risk_group'] = np.where(df['multiomics_score'] >= df['multiomics_score'].median(), 'High Risk', 'Low Risk')

kmf = KaplanMeierFitter()
plt.figure(figsize=(8, 5))

for group, group_df in df.groupby('risk_group'):
    kmf.fit(group_df[time_col], group_df[event_col], label=group)
    kmf.plot_survival_function(ci_show=False)

plt.title("CPTAC-BRCA Multi-Omics Risk Stratification (KM Curve)")
plt.xlabel("Days")
plt.ylabel("Survival Probability")
plt.savefig("results/figures/cptac_km_stratification.png", dpi=300)
print("[+] KM Curve saved to results/figures/cptac_km_stratification.png")
