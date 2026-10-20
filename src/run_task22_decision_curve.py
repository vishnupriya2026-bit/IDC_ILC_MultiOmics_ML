import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("--- Step 1: Loading Trained Multi-Omic Model & Validation Data ---")
model_dict = joblib.load("models/final_multiomic_model.pkl")
model = model_dict["model"]
scaler = model_dict["scaler"]
features = model_dict["features"]

gse_raw = pd.read_csv("data/processed/GSE2034_harmonized.csv")
gse_clean = gse_raw.groupby("Symbol").mean().T

series_matrix_path = "datasets/GEO/raw/GSE2034_series_matrix.txt"
with open(series_matrix_path, "r") as f:
    lines = f.readlines()

sample_ids = []
for line in lines:
    if line.startswith("!Sample_geo_accession"):
        sample_ids = [x.strip('"\r\n ') for x in line.split("\t")[1:]]
        break

relapse_status = None
for line in lines:
    if line.startswith("!Sample_characteristics_ch1"):
        parts = [x.strip('"\r\n ') for x in line.split("\t")[1:]]
        if parts:
            tag = parts[0].lower()
            if "relapse" in tag and "time" not in tag and not any(s in tag for s in ["bone", "brain", "lung", "loco"]):
                relapse_status = parts

if relapse_status is None:
    for line in lines:
        if line.startswith("!Sample_characteristics_ch1"):
            parts = [x.strip('"\r\n ') for x in line.split("\t")[1:]]
            if parts and "relapse" in parts[0].lower() and "time" not in parts[0].lower():
                relapse_status = parts
                break

y_event = np.array([1 if x.split(":")[-1].strip().lower() in ["1", "yes", "relapse"] else 0 for x in relapse_status[:len(sample_ids)]])

meta_df = pd.DataFrame({"event": y_event}, index=sample_ids)
gse_clean = gse_clean.loc[gse_clean.index.intersection(meta_df.index)]
meta_df = meta_df.loc[gse_clean.index]

X = scaler.transform(gse_clean[features].values)
y = meta_df['event'].values
N = len(y)

# Predict risk probabilities
y_prob = model.predict_proba(X)[:, 1]

print("--- Step 2: Calculating Clinical Net Benefit Across Thresholds ---")
thresholds = np.linspace(0.05, 0.60, 50)
net_benefit_model = []
net_benefit_all = []
net_benefit_none = []

prev = np.mean(y)

for p_t in thresholds:
    # Model Strategy
    pred_pos = (y_prob >= p_t).astype(int)
    tp = np.sum((pred_pos == 1) & (y == 1))
    fp = np.sum((pred_pos == 1) & (y == 0))
    nb_model = (tp / N) - (fp / N) * (p_t / (1 - p_t))
    net_benefit_model.append(nb_model)
    
    # Treat All Strategy
    tp_all = np.sum(y == 1)
    fp_all = np.sum(y == 0)
    nb_all = (tp_all / N) - (fp_all / N) * (p_t / (1 - p_t))
    net_benefit_all.append(nb_all)
    
    # Treat None Strategy
    net_benefit_none.append(0.0)

dca_df = pd.DataFrame({
    "Threshold_Probability": thresholds,
    "Net_Benefit_Model": net_benefit_model,
    "Net_Benefit_Treat_All": net_benefit_all,
    "Net_Benefit_Treat_None": net_benefit_none
})

os.makedirs("results", exist_ok=True)
dca_df.to_csv("results/task22_dca_summary.csv", index=False)

print("--- Step 3: Generating Decision Curve Plot ---")
fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)

ax.plot(thresholds, net_benefit_model, label="Multi-Omic ML Model", color="#1f77b4", lw=2.5)
ax.plot(thresholds, net_benefit_all, label="Treat All Patients", color="black", linestyle="--", lw=1.5)
ax.plot(thresholds, net_benefit_none, label="Treat No Patients", color="gray", linestyle="-.", lw=1.5)

ax.set_ylim(-0.05, max(net_benefit_model) * 1.2)
ax.set_xlim(0.05, 0.60)
ax.set_xlabel("Threshold Probability ($p_t$)", fontweight="bold")
ax.set_ylabel("Clinical Net Benefit", fontweight="bold")
ax.set_title("Decision Curve Analysis (DCA): GSE2034 Clinical Utility", fontweight="bold")
ax.grid(True, linestyle=":", alpha=0.6)
ax.legend(loc="upper right")

plt.tight_layout()
output_path = "results/task22_decision_curve.png"
plt.savefig(output_path, dpi=300)
plt.close()

print(f"Task 22 completed. Saved plot to '{output_path}' and summary metrics to 'results/task22_dca_summary.csv'")
