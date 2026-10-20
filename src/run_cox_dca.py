import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test
from dcurves import dca
from dcurves.plot_graphs import plot_graphs

# 1. Load clinical dataset
df_clin = pd.read_csv("cptac_119_clinical_final.csv")
score_df = pd.read_csv("data/processed/survival_scorecard_results.csv")

# Merge on ID
id_col = 'Patient_ID'
df = df_clin.merge(score_df, on=id_col, how='inner')

# Target Recurrence-Free Survival (RFS) if available, otherwise Overall Survival (OS)
rfs_time = 'Recurrence-free survival, days'
rfs_event = 'Recurrence status (1, yes; 0, no)'

time_col = rfs_time if rfs_time in df.columns else 'Overall survival, days'
event_col = rfs_event if rfs_event in df.columns else 'Survival status (1, dead; 0, alive)'

score_col = 'risk_score' if 'risk_score' in df.columns else score_df.columns[1]

# Clean data
df[time_col] = pd.to_numeric(df[time_col], errors='coerce')
df[event_col] = pd.to_numeric(df[event_col], errors='coerce')
cox_df = df.dropna(subset=[time_col, event_col, score_col])
cox_df = cox_df[cox_df[time_col] > 0]

# Z-score standardization
cox_df['risk_score_z'] = (cox_df[score_col] - cox_df[score_col].mean()) / cox_df[score_col].std()

print(f"[+] Endpoint: {time_col}")
print(f"[+] Total Samples: {len(cox_df)} | Total Events Observed: {int(cox_df[event_col].sum())}")

# 2. Fit Cox Model
cph = CoxPHFitter()
cph.fit(cox_df[[time_col, event_col, 'risk_score_z']], duration_col=time_col, event_col=event_col)
print("\n==========================================")
print("       COX PROPORTIONAL HAZARDS MODEL     ")
print("==========================================")
cph.print_summary()

# 3. Decision Curve Analysis (DCA)
cox_df['pred_prob'] = (cox_df[score_col] - cox_df[score_col].min()) / (cox_df[score_col].max() - cox_df[score_col].min())

dca_results = dca(
    data=cox_df[[event_col, 'pred_prob']],
    outcome=event_col,
    modelnames=['pred_prob'],
    thresholds=np.arange(0.01, 0.90, 0.01)
)

plot_graphs(dca_results, graph_type='net_benefit')
plt.title(f"DCA Net Benefit ({time_col})")
plt.savefig("results/figures/dca_net_benefit.png", dpi=300, bbox_inches='tight')
print("\n[+] Saved DCA net benefit plot to results/figures/dca_net_benefit.png")
