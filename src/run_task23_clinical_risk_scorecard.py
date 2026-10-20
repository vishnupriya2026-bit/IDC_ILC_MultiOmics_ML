import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("--- Step 1: Loading Trained Multi-Omic ML Pipeline ---")
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

print("--- Step 2: Generating Individualized Patient Risk Scorecards ---")
risk_probs = model.predict_proba(X)[:, 1]

def assign_risk_category(prob):
    if prob < 0.25:
        return "Low Risk"
    elif prob < 0.45:
        return "Intermediate Risk"
    else:
        return "High Risk"

scorecard_df = pd.DataFrame({
    "Sample_ID": gse_clean.index,
    "Recurrence_Probability": np.round(risk_probs, 4),
    "Risk_Group": [assign_risk_category(p) for p in risk_probs],
    "Observed_Relapse": y
})

os.makedirs("results", exist_ok=True)
scorecard_df.to_csv("results/task23_patient_risk_scorecard.csv", index=False)

print("\n=== Risk Category Distribution ===")
summary_stats = scorecard_df.groupby("Risk_Group").agg(
    Sample_Count=("Sample_ID", "count"),
    Observed_Relapses=("Observed_Relapse", "sum"),
    Observed_Relapse_Rate=("Observed_Relapse", "mean")
).reset_index()
print(summary_stats.to_string(index=False))

print("\n--- Step 3: Plotting Risk Tier Distribution & Event Rates ---")
fig, ax1 = plt.subplots(figsize=(7, 5), dpi=300)

groups = ["Low Risk", "Intermediate Risk", "High Risk"]
counts = [scorecard_df[scorecard_df["Risk_Group"] == g].shape[0] for g in groups]
rates = [scorecard_df[scorecard_df["Risk_Group"] == g]["Observed_Relapse"].mean() * 100 for g in groups]

bars = ax1.bar(groups, counts, color=["#2ca02c", "#ff7f0e", "#d62728"], alpha=0.85, edgecolor="black", width=0.5)
ax1.set_ylabel("Patient Count", fontweight="bold", color="black")
ax1.tick_params(axis="y", labelcolor="black")

ax2 = ax1.twinx()
ax2.plot(groups, rates, color="blue", marker="o", linewidth=2.5, markersize=8, label="Relapse Rate (%)")
ax2.set_ylabel("Observed Relapse Rate (%)", fontweight="bold", color="blue")
ax2.tick_params(axis="y", labelcolor="blue")

plt.title("Clinical Risk Stratification Scorecard (GSE2034 Cohort)", fontweight="bold")
plt.tight_layout()

output_path = "results/task23_risk_stratification.png"
plt.savefig(output_path, dpi=300)
plt.close()

print(f"\nTask 23 completed. Saved scorecard to 'results/task23_patient_risk_scorecard.csv' and plot to '{output_path}'")
