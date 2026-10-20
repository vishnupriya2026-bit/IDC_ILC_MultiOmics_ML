import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

print("==================================================")
print(" RUNNING OPTIMIZED METABRIC SURVIVAL ANALYSIS")
print("==================================================")

EXTRACT_DIR = "brca_metabric"
if not os.path.exists(EXTRACT_DIR):
    raise FileNotFoundError("METABRIC directory not found. Please run your downloader script first.")

# 1. Load Clinical Data
sample_file = os.path.join(EXTRACT_DIR, "data_clinical_sample.txt")
patient_file = os.path.join(EXTRACT_DIR, "data_clinical_patient.txt")

df_sample = pd.read_csv(sample_file, sep="\t", comment="#")
df_patient = pd.read_csv(patient_file, sep="\t", comment="#")

df_sample.columns = df_sample.columns.str.upper().str.strip()
df_patient.columns = df_patient.columns.str.upper().str.strip()

sample_id_col = [c for c in df_sample.columns if "SAMPLE_ID" in c or "PATIENT_ID" in c][0]
patient_id_col = [c for c in df_patient.columns if "PATIENT_ID" in c or "SAMPLE_ID" in c][0]

clinical_df = pd.merge(df_sample, df_patient, left_on=sample_id_col, right_on=patient_id_col, how="inner")

hist_col = [c for c in clinical_df.columns if "HISTOLOGY" in c or "CANCER_TYPE_DETAILED" in c][0]
clinical_df = clinical_df[clinical_df[hist_col].astype(str).str.contains("Ductal|Lobular", case=False, na=False)].copy()
clinical_df["TARGET"] = np.where(clinical_df[hist_col].astype(str).str.contains("Lobular", case=False), 1, 0)

os_month_col = [c for c in clinical_df.columns if "OS_MONTHS" in c or "OVERALL_SURVIVAL_MONTHS" in c][0]
os_status_col = [c for c in clinical_df.columns if "OS_STATUS" in c or "OVERALL_SURVIVAL_STATUS" in c][0]

clinical_df["SURVIVAL_MONTHS"] = pd.to_numeric(clinical_df[os_month_col], errors="coerce")
clinical_df["EVENT"] = np.where(clinical_df[os_status_col].astype(str).str.contains("DECEASED|1|1:DECEASED", case=False), 1, 0)

# 2. FAST SELECTIVE EXPRESSION LOADING (Only loads target 14 genes from disk)
gene_panel = ['CDH1', 'CTNNB1', 'PGM5', 'AR', 'INMT', 'ADAM33', 'MR1', 'EGFR', 'CYGB', 'SPDEF', 'WDR86', 'NGFR', 'LIMS2', 'F10']
gene_set = set(gene_panel)

expr_file = os.path.join(EXTRACT_DIR, "data_mrna_illumina_microarray.txt")
if not os.path.exists(expr_file):
    expr_file = os.path.join(EXTRACT_DIR, "data_mrna_median_all_sample_Zscores.txt")

print("Fast-reading expression file for target signature genes...")
selected_rows = []
headers = None

with open(expr_file, 'r', encoding='utf-8', errors='ignore') as f:
    for line in f:
        if line.startswith('#'):
            continue
        parts = line.strip().split('\t')
        if headers is None:
            headers = parts
            continue
        if parts[0] in gene_set:
            selected_rows.append(parts)

expr_df = pd.DataFrame(selected_rows, columns=headers)
symbol_col = headers[0]
expr_df = expr_df.drop_duplicates(subset=[symbol_col]).set_index(symbol_col)

for col in ["Entrez_Gene_Id", "Composite Element REF"]:
    if col in expr_df.columns:
        expr_df = expr_df.drop(columns=[col])

expr_df = expr_df.apply(pd.to_numeric, errors='coerce')

available_genes = list(expr_df.index)
X_expr = expr_df.T

common_samples = X_expr.index.intersection(clinical_df[sample_id_col])

X = X_expr.loc[common_samples]
clinical_matched = clinical_df.set_index(sample_id_col).loc[common_samples]
y = clinical_matched["TARGET"].values

X_scaled = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

print(f"Fast loading complete: N = {len(y)} ({sum(y==0)} IDC, {sum(y==1)} ILC)")
print(f"Mapped Genes ({len(available_genes)}): {available_genes}\n")

# 3. Dynamic Model Fitting & Inference
model = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42)
model.fit(X_scaled.values, y)

# 4. Dynamic Survival Analysis
surv_mask = clinical_matched["SURVIVAL_MONTHS"].notna() & (clinical_matched["SURVIVAL_MONTHS"] > 0)
X_surv = X_scaled.loc[surv_mask]
y_surv_time = clinical_matched.loc[surv_mask, "SURVIVAL_MONTHS"].values
y_surv_event = clinical_matched.loc[surv_mask, "EVENT"].values

risk_scores = model.predict_proba(X_surv.values)[:, 1]
median_risk = np.median(risk_scores)
risk_group = np.where(risk_scores > median_risk, "High Risk", "Low Risk")

mask_high = (risk_group == "High Risk")
mask_low = (risk_group == "Low Risk")

lr_results = logrank_test(y_surv_time[mask_high], y_surv_time[mask_low], 
                          event_observed_A=y_surv_event[mask_high], 
                          event_observed_B=y_surv_event[mask_low])

p_value = lr_results.p_value

surv_df = pd.DataFrame({
    'Time': y_surv_time,
    'Event': y_surv_event,
    'RiskGroup': np.where(risk_group == "High Risk", 1, 0)
})

cph = CoxPHFitter()
cph.fit(surv_df, duration_col='Time', event_col='Event')
hr = cph.hazard_ratios_['RiskGroup']
ci_lower, ci_upper = np.exp(cph.confidence_intervals_.loc['RiskGroup']).values

print("==================================================")
print(" DYNAMIC SURVIVAL ANALYSIS COMPUTED RESULTS")
print("==================================================")
print(f"Log-Rank Test p-value         : p = {p_value:.6f}")
print(f"Cox Hazard Ratio (High vs Low) : HR = {hr:.3f} (95% CI: {ci_lower:.3f} - {ci_upper:.3f})")

# Save KM Plot
plt.figure(figsize=(8, 6), dpi=300)
ax = plt.subplot(111)

kmf_high = KaplanMeierFitter()
kmf_low = KaplanMeierFitter()

kmf_high.fit(y_surv_time[mask_high], y_surv_event[mask_high], label="High-Risk Subtype Cohort")
kmf_high.plot_survival_function(ax=ax, color='#d62728', lw=2.5)

kmf_low.fit(y_surv_time[mask_low], y_surv_event[mask_low], label="Low-Risk Subtype Cohort")
kmf_low.plot_survival_function(ax=ax, color='#2ca02c', lw=2.5)

plt.xlabel('Overall Survival Time (Months)')
plt.ylabel('Overall Survival Probability')
plt.title('Kaplan-Meier Survival Stratification by 14-Gene Risk Signature')
plt.text(0.05, 0.15, f"Log-Rank p = {p_value:.4f}\nHazard Ratio = {hr:.2f} (95% CI: {ci_lower:.2f}-{ci_upper:.2f})", 
         transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('km_survival_curve.png')
plt.close()
print("\nGenerated 'km_survival_curve.png' successfully.")
