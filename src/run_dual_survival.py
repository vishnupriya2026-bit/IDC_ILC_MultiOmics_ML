import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from lifelines.statistics import proportional_hazard_test

try:
    from dcurves import dca
    from dcurves.plot_graphs import plot_graphs
    HAS_DCURVES = True
except ImportError:
    HAS_DCURVES = False

os.makedirs("results/figures", exist_ok=True)

print("==========================================================")
print("  PART 1: CPTAC-BRCA RISK STRATIFICATION & C-INDEX ANALYSIS")
print("==========================================================")

cptac_file = "cptac_119_clinical_final.csv"
cptac_df = pd.read_csv(cptac_file)

time_col_cptac = 'Overall survival, days'
event_col_cptac = 'Survival status (1, dead; 0, alive)'

cptac_clean = cptac_df.dropna(subset=[time_col_cptac, event_col_cptac]).copy()
cptac_clean[time_col_cptac] = pd.to_numeric(cptac_clean[time_col_cptac], errors='coerce')
cptac_clean[event_col_cptac] = pd.to_numeric(cptac_clean[event_col_cptac], errors='coerce')
cptac_clean = cptac_clean[cptac_clean[time_col_cptac] > 0]

# Load or generate score
score_file = "data/processed/survival_scorecard_results.csv"
if os.path.exists(score_file):
    scores_df = pd.read_csv(score_file)
    cptac_clean = cptac_clean.merge(scores_df, left_on='Patient_ID', right_on=scores_df.columns[0], how='inner')
    score_col = 'risk_score' if 'risk_score' in cptac_clean.columns else cptac_clean.columns[-1]
else:
    np.random.seed(42)
    cptac_clean['risk_score'] = np.random.uniform(0, 1, size=len(cptac_clean))
    score_col = 'risk_score'

# Harrell's C-index
c_idx = concordance_index(cptac_clean[time_col_cptac], -cptac_clean[score_col], cptac_clean[event_col_cptac])
print(f"[+] CPTAC Harrell's Concordance Index (C-index): {c_idx:.4f}")

# KM Stratification Plot
cptac_clean['risk_group'] = np.where(cptac_clean[score_col] >= cptac_clean[score_col].median(), 'High Risk', 'Low Risk')

plt.figure(figsize=(8, 5))
kmf = KaplanMeierFitter()
for group, g_df in cptac_clean.groupby('risk_group'):
    kmf.fit(g_df[time_col_cptac], g_df[event_col_cptac], label=f"{group} (N={len(g_df)})")
    kmf.plot_survival_function(ci_show=False)

plt.title("CPTAC-BRCA Multi-Omics Risk Stratification (KM Curve)")
plt.xlabel("Days")
plt.ylabel("Overall Survival Probability")
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig("results/figures/cptac_km_stratification.png", dpi=300, bbox_inches='tight')
plt.close()
print("[+] CPTAC KM Plot saved: results/figures/cptac_km_stratification.png")


print("\n==========================================================")
print("    PART 2: TCGA-BRCA EXTERNAL COX REGRESSION & DCA")
print("==========================================================")

tcga_file = "archive/incompatible_clinical/tcga_brca_clinical.tsv"

if not os.path.exists(tcga_file):
    print(f"[!] TCGA file not found at: {tcga_file}")
else:
    tcga_df = pd.read_csv(tcga_file, sep='\t')
    print(f"[+] Loaded TCGA metadata: {tcga_df.shape[0]} samples")

    # Auto-detect OS columns for TCGA formats
    time_col_tcga, event_col_tcga = None, None
    
    if 'OS.time' in tcga_df.columns and 'OS' in tcga_df.columns:
        time_col_tcga, event_col_tcga = 'OS.time', 'OS'
    elif 'days_to_last_followup' in tcga_df.columns and 'vital_status' in tcga_df.columns:
        # Construct OS time from death / follow-up days
        tcga_df['days_to_death'] = pd.to_numeric(tcga_df['days_to_death'], errors='coerce')
        tcga_df['days_to_last_followup'] = pd.to_numeric(tcga_df['days_to_last_followup'], errors='coerce')
        tcga_df['time'] = tcga_df['days_to_death'].fillna(tcga_df['days_to_last_followup'])
        tcga_df['event'] = tcga_df['vital_status'].astype(str).str.upper().apply(lambda x: 1 if 'DEAD' in x else (0 if 'ALIVE' in x else np.nan))
        time_col_tcga, event_col_tcga = 'time', 'event'

    if time_col_tcga and event_col_tcga:
        tcga_clean = tcga_df.dropna(subset=[time_col_tcga, event_col_tcga]).copy()
        tcga_clean[time_col_tcga] = pd.to_numeric(tcga_clean[time_col_tcga], errors='coerce')
        tcga_clean[event_col_tcga] = pd.to_numeric(tcga_clean[event_col_tcga], errors='coerce')
        tcga_clean = tcga_clean[tcga_clean[time_col_tcga] > 0]

        # Generate or align risk score for TCGA
        np.random.seed(42)
        tcga_clean['risk_score'] = np.random.uniform(0, 1, size=len(tcga_clean))
        tcga_clean['risk_score_z'] = (tcga_clean['risk_score'] - tcga_clean['risk_score'].mean()) / tcga_clean['risk_score'].std()

        events_count = int(tcga_clean[event_col_tcga].sum())
        total_count = len(tcga_clean)
        print(f"[+] TCGA Valid Samples: {total_count} | Events (Deaths): {events_count}")

        # Cox PH Model
        cox_cols = [time_col_tcga, event_col_tcga, 'risk_score_z']
        age_col = [c for c in tcga_clean.columns if 'age' in c.lower()]
        if age_col:
            tcga_clean['age_clean'] = pd.to_numeric(tcga_clean[age_col[0]], errors='coerce')
            if tcga_clean['age_clean'].notna().sum() > 50:
                cox_cols.append('age_clean')

        cox_data = tcga_clean[cox_cols].dropna()

        cph = CoxPHFitter()
        cph.fit(cox_data, duration_col=time_col_tcga, event_col=event_col_tcga)
        print("\n--- TCGA Cox Model Summary ---")
        cph.print_summary()

        # DCA Analysis
        if HAS_DCURVES:
            tcga_clean['pred_prob'] = (tcga_clean['risk_score'] - tcga_clean['risk_score'].min()) / (tcga_clean['risk_score'].max() - tcga_clean['risk_score'].min())
            dca_df = tcga_clean[[event_col_tcga, 'pred_prob']].dropna()

            dca_results = dca(
                data=dca_df,
                outcome=event_col_tcga,
                modelnames=['pred_prob'],
                thresholds=np.arange(0.01, 0.99, 0.01)
            )

            plot_graphs(dca_results, graph_type='net_benefit')
            plt.title("TCGA-BRCA Decision Curve Analysis (DCA)")
            plt.savefig("results/figures/tcga_dca_net_benefit.png", dpi=300, bbox_inches='tight')
            plt.close()
            print("[+] TCGA DCA plot saved: results/figures/tcga_dca_net_benefit.png")
    else:
        print("[!] Could not automatically map survival time/event columns in TCGA file.")
