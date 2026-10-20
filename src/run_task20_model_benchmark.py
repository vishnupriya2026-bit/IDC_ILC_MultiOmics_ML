import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc, f1_score, precision_score, recall_score, accuracy_score

print("--- Step 1: Loading Processed Multi-Omic Dataset ---")
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

prolif_genes = ['MKI67', 'AURKA', 'CCNB1', 'TOP2A', 'BIRC5', 'CDK1', 'MYBL2', 'BUB1B', 'MCM2']
luminal_genes = ['ESR1', 'PGR', 'FOXA1', 'GATA3', 'BCL2', 'SCUBE2', 'BAG1']
all_genes = [g for g in prolif_genes + luminal_genes if g in gse_clean.columns]

X = gse_clean[all_genes].values
y = meta_df['event'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Dataset shape: X={X_scaled.shape}, y={y.shape} (Relapse events: {y.sum()})")

print("--- Step 2: Benchmarking Machine Learning Classifiers ---")
models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "Support Vector Machine": SVC(probability=True, random_state=42)
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
plt.figure(figsize=(8, 6), dpi=300)

results = []

for name, model in models.items():
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    
    accs, precs, recs, f1s = [], [], [], []
    
    for train, test in cv.split(X_scaled, y):
        model.fit(X_scaled[train], y[train])
        probas = model.predict_proba(X_scaled[test])[:, 1]
        preds = model.predict(X_scaled[test])
        
        fpr, tpr, _ = roc_curve(y[test], probas)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        
        accs.append(accuracy_score(y[test], preds))
        precs.append(precision_score(y[test], preds, zero_division=0))
        recs.append(recall_score(y[test], preds, zero_division=0))
        f1s.append(f1_score(y[test], preds, zero_division=0))
        
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    
    plt.plot(mean_fpr, mean_tpr, label=f"{name} (AUC = {mean_auc:.2f} ± {std_auc:.2f})", lw=2)
    
    results.append({
        "Model": name,
        "AUC_Mean": round(mean_auc, 3),
        "AUC_Std": round(std_auc, 3),
        "Accuracy": round(np.mean(accs), 3),
        "Precision": round(np.mean(precs), 3),
        "Recall": round(np.mean(recs), 3),
        "F1_Score": round(np.mean(f1s), 3)
    })

plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Chance (AUC = 0.50)')
plt.xlabel('False Positive Rate', fontweight='bold')
plt.ylabel('True Positive Rate', fontweight='bold')
plt.title('5-Fold Cross-Validated ROC Curves (GSE2034 Relapse Prediction)', fontweight='bold')
plt.legend(loc='lower right')
plt.grid(True, linestyle=':', alpha=0.6)

os.makedirs("results", exist_ok=True)
os.makedirs("models", exist_ok=True)

plt.tight_layout()
plt.savefig("results/task20_roc_curves.png", dpi=300)
plt.close()

res_df = pd.DataFrame(results).sort_values(by="AUC_Mean", ascending=False)
res_df.to_csv("results/task20_model_benchmark.csv", index=False)

best_model_name = res_df.iloc[0]["Model"]
best_model = models[best_model_name]
best_model.fit(X_scaled, y)
joblib.dump({"model": best_model, "scaler": scaler, "features": all_genes}, "models/final_multiomic_model.pkl")

print("\n=== Classifier Performance Benchmark ===")
print(res_df.to_string(index=False))

print(f"\nTask 20 completed. Saved ROC plot to 'results/task20_roc_curves.png', benchmark to 'results/task20_model_benchmark.csv', and best model ({best_model_name}) to 'models/final_multiomic_model.pkl'")
