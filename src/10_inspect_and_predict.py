import joblib
import pandas as pd
import numpy as np

print("=== Step 10: Inspect Champion Model & Run Inference ===")

# ---------------------------------------------------------
# 1. Load Trained Model & Scaler Artifacts
# ---------------------------------------------------------
model = joblib.load("results/advanced_boost/champion_xgb_model.joblib")
scaler = joblib.load("results/advanced_boost/rna_scaler.joblib")

print("\n Champion Model Successfully Loaded from Disk!")
print("Model Hyperparameters:", model.get_params())

# ---------------------------------------------------------
# 2. Inspect Top Predictive Biomarkers (Feature Importance)
# ---------------------------------------------------------
if hasattr(model, "feature_importances_"):
    importances = model.feature_importances_
    top_indices = np.argsort(importances)[::-1][:10]
    
    print("\n--- Top 10 Most Predictive Biomarkers (XGBoost Feature Importance) ---")
    for rank, idx in enumerate(top_indices, 1):
        print(f"Rank {rank:2d}: Feature Index {idx:3d} | Importance Weight: {importances[idx]:.4f}")

# ---------------------------------------------------------
# 3. Simulate Inference on New Unseen Patient Samples
# ---------------------------------------------------------
print("\n=== Running Clinical Predictions on 3 Unseen Patient Profiles ===")

# Generate synthetic expression profiles matching expected feature dimensions
np.random.seed(42)
n_features = model.n_features_in_
new_patients = np.random.randn(3, n_features)

# Predict outcomes & probabilities
predictions = model.predict(new_patients)
probabilities = model.predict_proba(new_patients)

for i in range(len(new_patients)):
    pred_label = "Class 1 (High Risk / Outcome)" if predictions[i] == 1 else "Class 0 (Low Risk / Outcome)"
    conf_score = probabilities[i][predictions[i]] * 100
    print(f"\nPatient Profile #{i+1}:")
    print(f"  ➜ Predicted Phenotype : {pred_label}")
    print(f"  ➜ Prediction Confidence: {conf_score:.2f}%")
