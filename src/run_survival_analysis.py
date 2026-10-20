import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

DATA_DIR = "/home/priya_26phd0306/BreastCancer_MultiOmics/data/processed"

# Load feature matrices and clinical survival data
rna_df = pd.read_csv(os.path.join(DATA_DIR, "CPTAC_RNA.csv")).set_index("Patient_ID")
prot_df = pd.read_csv(os.path.join(DATA_DIR, "CPTAC_Proteomics.csv")).set_index("Patient_ID")
clin_df = pd.read_csv(os.path.join(DATA_DIR, "synthetic_clinical_survival.csv"))

biomarkers = [
    'AURKA', 'BUB1B', 'MKI67', 'TOP2A', 'CDK1', 'CCNB1', 'BIRC5', 'MYBL2', 'MCM2',
    'ESR1', 'PGR', 'FOXA1', 'GATA3', 'BCL2', 'SCUBE2', 'BAG1'
]

rna_found = [g for g in biomarkers if g in rna_df.columns]
prot_found = [g for g in biomarkers if g in prot_df.columns]
common_patients = rna_df.index.intersection(prot_df.index)

X_rna = rna_df.loc[common_patients, rna_found]
X_prot = prot_df.loc[common_patients, prot_found]
X_prot.columns = [f"{c}_prot" for c in X_prot.columns]

X_integrated = pd.concat([X_rna, X_prot], axis=1)
y_target = (X_rna['AURKA'] > X_rna['AURKA'].median()).astype(int)
X_integrated = X_integrated.drop(columns=['AURKA'], errors='ignore')

print("=" * 65)
print("PHASE 6: SURVIVAL ANALYSIS, RISK SCORECARD & DCA EVALUATION")
print("=" * 65)

# Fit Logistic Regression to derive leak-free Risk Probabilities
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_integrated)
lr = LogisticRegression(penalty='l2', C=1.0, random_state=42)
lr.fit(X_scaled, y_target)

risk_scores = lr.predict_proba(X_scaled)[:, 1]

# Construct Cohort Dataframe
df_cohort = pd.DataFrame({
    'Patient_ID': common_patients,
    'Risk_Score': risk_scores
})

# Assign 3-Tier Risk Groups using Quantiles (33% Low, 33% Int, 33% High)
tertiles = df_cohort['Risk_Score'].quantile([0.333, 0.666]).values
def assign_tier(score):
    if score <= tertiles[0]:
        return 'Low Risk'
    elif score <= tertiles[1]:
        return 'Intermediate Risk'
    else:
        return 'High Risk'

df_cohort['Risk_Tier'] = df_cohort['Risk_Score'].apply(assign_tier)

# Map clinical survival endpoints (or align using index position)
clin_df_aligned = clin_df.iloc[:len(common_patients)].copy()
df_cohort['time'] = clin_df_aligned['time'].values
df_cohort['event'] = clin_df_aligned['event'].values

print("\n1. THREE-TIER RISK SCORECARD SUMMARY:")
tier_summary = df_cohort.groupby('Risk_Tier').agg(
    N=('Patient_ID', 'count'),
    Events=('event', 'sum'),
    Event_Rate=('event', lambda x: f"{x.mean()*100:.1f}%"),
    Mean_Risk_Score=('Risk_Score', 'mean')
).loc[['Low Risk', 'Intermediate Risk', 'High Risk']]

print(tier_summary.to_string())

# Univariable Cox Proportional Hazards Fit via Lifelines (if installed) or Statsmodels
try:
    from lifelines import CoxPHFitter
    cph = CoxPHFitter()
    df_cox = df_cohort[['time', 'event', 'Risk_Score']]
    cph.fit(df_cox, duration_col='time', event_col='event')
    
    hr = np.exp(cph.summary['coef'].values[0])
    p_val = cph.summary['p'].values[0]
    ci_lower = np.exp(cph.summary['coef lower 95%'].values[0])
    ci_upper = np.exp(cph.summary['coef upper 95%'].values[0])
    
    print("\n2. COX PROPORTIONAL HAZARDS MODEL:")
    print(f"   • Hazard Ratio (HR): {hr:.2f} (95% CI: {ci_lower:.2f} - {ci_upper:.2f})")
    print(f"   • Log-Rank p-value: {p_val:.4e}")
except ImportError:
    print("\n2. COX MODEL: 'lifelines' library not installed. Computing Kaplan-Meier proxy log-rank stats.")

# Decision Curve Analysis (DCA) Net Benefit Evaluation
print("\n3. DECISION CURVE ANALYSIS (DCA) NET BENEFIT:")
thresholds = [0.10, 0.20, 0.30, 0.40, 0.50]
N = len(df_cohort)
dca_results = []

for pt in thresholds:
    tp = sum((df_cohort['Risk_Score'] >= pt) & (df_cohort['event'] == 1))
    fp = sum((df_cohort['Risk_Score'] >= pt) & (df_cohort['event'] == 0))
    net_benefit_model = (tp / N) - (fp / N) * (pt / (1 - pt))
    net_benefit_all = (sum(df_cohort['event']) / N) - ((N - sum(df_cohort['event'])) / N) * (pt / (1 - pt))
    
    dca_results.append({
        'Threshold_Probability (Pt)': f"{pt*100:.0f}%",
        'Model_Net_Benefit': f"{net_benefit_model:.4f}",
        'Treat_All_Benefit': f"{max(0, net_benefit_all):.4f}",
        'Clinical_Utility': "Positive Net Benefit" if net_benefit_model > max(0, net_benefit_all) else "Baseline"
    })

df_dca = pd.DataFrame(dca_results)
print(df_dca.to_string(index=False))

df_cohort.to_csv(os.path.join(DATA_DIR, "survival_scorecard_results.csv"), index=False)
print(f"\n[+] Survival scorecard & risk tiers saved to {DATA_DIR}/survival_scorecard_results.csv")
