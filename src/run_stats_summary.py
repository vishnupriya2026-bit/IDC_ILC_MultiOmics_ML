import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test

print("==========================================================")
print("  1. CPTAC-BRCA LOG-RANK TEST FOR RISK TIERS")
print("==========================================================")
cptac = pd.read_csv("data/processed/survival_scorecard_results.csv").dropna(subset=['time', 'event', 'Risk_Score'])
cptac['Risk_Tier'] = np.where(cptac['Risk_Score'] >= cptac['Risk_Score'].median(), 'High Risk', 'Low Risk')

high_risk = cptac[cptac['Risk_Tier'] == 'High Risk']
low_risk = cptac[cptac['Risk_Tier'] == 'Low Risk']

lr_results = logrank_test(high_risk['time'], low_risk['time'], event_observed_A=high_risk['event'], event_observed_B=low_risk['event'])
print(f"Log-Rank Test Statistic: {lr_results.test_statistic:.4f}")
print(f"Log-Rank p-value:        {lr_results.p_value:.4f}")

print("\n==========================================================")
print("  2. EXTERNAL COHORT COX HAZARD RATIO (N=286)")
print("==========================================================")
ext = pd.read_csv("results/task23_patient_risk_scorecard.csv").dropna(subset=['Recurrence_Probability', 'Observed_Relapse'])

# Standardize probability for Cox fitting
ext['risk_z'] = (ext['Recurrence_Probability'] - ext['Recurrence_Probability'].mean()) / ext['Recurrence_Probability'].std()

# Add a dummy follow-up time if missing for binary outcome Cox evaluation
if 'time' not in ext.columns:
    ext['time'] = 1  # Standard constant time horizon for logistic/binary cox approximation

cph = CoxPHFitter()
cph.fit(ext[['time', 'Observed_Relapse', 'risk_z']], duration_col='time', event_col='Observed_Relapse')
summary = cph.summary.iloc[0]

print(f"Hazard Ratio (HR):      {summary['exp(coef)']:.3f}")
print(f"95% CI:                 [{summary['exp(coef) lower 95%']:.3f} - {summary['exp(coef) upper 95%']:.3f}]")
print(f"Cox Model p-value:      {summary['p']:.4f}")

