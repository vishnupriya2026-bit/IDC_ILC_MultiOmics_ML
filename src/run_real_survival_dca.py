import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from lifelines import KaplanMeierFitter
from lifelines.utils import concordance_index

try:
    from dcurves import dca
    from dcurves.plot_graphs import plot_graphs
    HAS_DCURVES = True
except ImportError:
    HAS_DCURVES = False

os.makedirs("results/figures", exist_ok=True)

print("==========================================================")
print("  PART 1: REAL CPTAC-BRCA C-INDEX & KM STRATIFICATION")
print("==========================================================")

cptac_path = "data/processed/survival_scorecard_results.csv"
cptac_df = pd.read_csv(cptac_path)

# Drop missing survival values
cptac_clean = cptac_df.dropna(subset=['time', 'event', 'Risk_Score']).copy()
cptac_clean['time'] = pd.to_numeric(cptac_clean['time'], errors='coerce')
cptac_clean['event'] = pd.to_numeric(cptac_clean['event'], errors='coerce')
cptac_clean['Risk_Score'] = pd.to_numeric(cptac_clean['Risk_Score'], errors='coerce')
cptac_clean = cptac_clean[cptac_clean['time'] > 0]

# Real Harrell's C-index
c_idx_cptac = concordance_index(cptac_clean['time'], -cptac_clean['Risk_Score'], cptac_clean['event'])
print(f"[+] CPTAC N={len(cptac_clean)} | Events={int(cptac_clean['event'].sum())}")
print(f"[+] Real CPTAC Harrell's C-index: {c_idx_cptac:.4f}")

# Plot KM Curve by Risk Tier / Median Split
kmf = KaplanMeierFitter()
plt.figure(figsize=(8, 5))

if 'Risk_Tier' in cptac_clean.columns:
    group_col = 'Risk_Tier'
else:
    cptac_clean['Risk_Tier'] = np.where(cptac_clean['Risk_Score'] >= cptac_clean['Risk_Score'].median(), 'High Risk', 'Low Risk')
    group_col = 'Risk_Tier'

for group, g_df in cptac_clean.groupby(group_col):
    kmf.fit(g_df['time'], g_df['event'], label=f"{group} (N={len(g_df)})")
    kmf.plot_survival_function(ci_show=False)

plt.title("CPTAC-BRCA Multi-Omics Risk Stratification (KM Curve)")
plt.xlabel("Follow-up Time (Days)")
plt.ylabel("Overall Survival Probability")
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig("results/figures/cptac_real_km_stratification.png", dpi=300, bbox_inches='tight')
plt.close()
print("[+] CPTAC KM Plot saved: results/figures/cptac_real_km_stratification.png")


print("\n==========================================================")
print("  PART 2: EXTERNAL VALIDATION COHORT (N=286) DCA & METRICS")
print("==========================================================")

ext_path = "results/task23_patient_risk_scorecard.csv"
ext_df = pd.read_csv(ext_path)

prob_col = 'Recurrence_Probability'
outcome_col = 'Observed_Relapse'

ext_clean = ext_df.dropna(subset=[prob_col, outcome_col]).copy()
ext_clean[prob_col] = pd.to_numeric(ext_clean[prob_col], errors='coerce')
ext_clean[outcome_col] = pd.to_numeric(ext_clean[outcome_col], errors='coerce')

total_n = len(ext_clean)
events_n = int(ext_clean[outcome_col].sum())
print(f"[+] External Validation Cohort N={total_n} | Relapse Events={events_n}")

# Decision Curve Analysis
if HAS_DCURVES:
    dca_df = dca(
        data=ext_clean,
        outcome=outcome_col,
        modelnames=[prob_col],
        thresholds=np.arange(0.01, 0.99, 0.01)
    )
    
    plot_graphs(dca_df, graph_type='net_benefit')
    plt.title("External Validation Decision Curve Analysis (DCA)")
    plt.savefig("results/figures/external_dca_net_benefit.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("[+] DCA Plot saved: results/figures/external_dca_net_benefit.png")
else:
    # Manual Net Benefit calculation if dcurves package is unavailable
    thresholds = np.linspace(0.01, 0.99, 99)
    net_benefit_model = []
    net_benefit_all = []
    
    n = len(ext_clean)
    p_event = events_n / n
    
    for p_t in thresholds:
        # Model Net Benefit
        tp = ((ext_clean[prob_col] >= p_t) & (ext_clean[outcome_col] == 1)).sum()
        fp = ((ext_clean[prob_col] >= p_t) & (ext_clean[outcome_col] == 0)).sum()
        nb = (tp / n) - (fp / n) * (p_t / (1 - p_t))
        net_benefit_model.append(nb)
        
        # Treat All Net Benefit
        tp_all = events_n
        fp_all = n - events_n
        nb_all = (tp_all / n) - (fp_all / n) * (p_t / (1 - p_t))
        net_benefit_all.append(nb_all)

    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, net_benefit_model, label='Multi-Omics Model', color='blue', lw=2)
    plt.plot(thresholds, net_benefit_all, label='Treat All', color='grey', linestyle='--')
    plt.axhline(0, label='Treat None', color='black', linestyle=':')
    plt.ylim(-0.05, max(net_benefit_model) + 0.1)
    plt.xlabel('Threshold Probability')
    plt.ylabel('Net Benefit')
    plt.title('External Cohort Decision Curve Analysis (Net Benefit)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig("results/figures/external_dca_net_benefit.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("[+] Manual DCA Plot saved: results/figures/external_dca_net_benefit.png")

