import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, roc_curve)
from sklearn.utils.class_weight import compute_class_weight

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

# Set seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# 1. Load Data
X_train = pd.read_csv("diablo_exports/X_diablo_train_scaled.csv", index_col=0)
y_train_df = pd.read_csv("diablo_exports/y_diablo_train_encoded.csv", index_col=0)
X_test = pd.read_csv("diablo_exports/X_diablo_test_scaled.csv", index_col=0)
y_test_df = pd.read_csv("diablo_exports/y_test_encoded.csv" if "y_test_encoded.csv" in y_train_df.columns else "diablo_exports/y_diablo_test_encoded.csv", index_col=0)

# Align indices
y_train_df = y_train_df[~y_train_df.index.duplicated(keep='first')]
y_test_df = y_test_df[~y_test_df.index.duplicated(keep='first')]

y_train = y_train_df.reindex(X_train.index).iloc[:, 0].values.astype(int)
y_test = y_test_df.reindex(X_test.index).iloc[:, 0].values.astype(int)

# Calculate class weights for imbalance handling
classes = np.unique(y_train)
weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weight_dict = dict(zip(classes, weights))
scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)

print(f"Train Shape: {X_train.shape}, Test Shape: {X_test.shape}")
print(f"Class Distribution - Train: {np.bincount(y_train)}, Test: {np.bincount(y_test)}")

results = []
roc_curves = {}

# Helper function to record metrics
def evaluate_model(name, y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    auc = roc_auc_score(y_true, y_prob)
    
    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "ROC-AUC": auc
    })
    
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_curves[name] = (fpr, tpr, auc)
    print(f"[{name}] Accuracy: {acc:.4f} | F1: {f1:.4f} | ROC-AUC: {auc:.4f}")

# --- 1. Balanced Random Forest ---
rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
rf.fit(X_train, y_train)
evaluate_model("Random Forest (Balanced)", y_test, rf.predict(X_test), rf.predict_proba(X_test)[:, 1])

# --- 2. XGBoost ---
xgb_model = xgb.XGBClassifier(n_estimators=150, learning_rate=0.05, scale_pos_weight=scale_pos_weight, random_state=42, eval_metric='logloss')
xgb_model.fit(X_train, y_train)
evaluate_model("XGBoost", y_test, xgb_model.predict(X_test), xgb_model.predict_proba(X_test)[:, 1])

# --- 3. LightGBM ---
lgb_model = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05, class_weight='balanced', random_state=42, verbose=-1)
lgb_model.fit(X_train, y_train)
evaluate_model("LightGBM", y_test, lgb_model.predict(X_test), lgb_model.predict_proba(X_test)[:, 1])

# --- 4. CatBoost ---
cb_model = CatBoostClassifier(iterations=200, learning_rate=0.05, auto_class_weights='Balanced', random_seed=42, verbose=0)
cb_model.fit(X_train, y_train)
evaluate_model("CatBoost", y_test, cb_model.predict(X_test), cb_model.predict_proba(X_test)[:, 1])

# --- 5. PyTorch Multi-Layer Perceptron (GPU) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_tr_t = torch.tensor(X_train.values, dtype=torch.float32)
y_tr_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_te_t = torch.tensor(X_test.values, dtype=torch.float32).to(device)

train_ds = TensorDataset(X_tr_t, y_tr_t)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

class MultiOmicsMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

mlp = MultiOmicsMLP(X_train.shape[1]).to(device)
criterion = nn.BCELoss(pos_weight=torch.tensor([scale_pos_weight]).to(device))
optimizer = optim.AdamW(mlp.parameters(), lr=0.001, weight_decay=1e-4)

mlp.train()
for epoch in range(100):
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        out = mlp(batch_x)
        loss = criterion(out, batch_y)
        loss.backward()
        optimizer.step()

mlp.eval()
with torch.no_grad():
    mlp_probs = mlp(X_te_t).cpu().numpy().ravel()
    mlp_preds = (mlp_probs > 0.5).astype(int)

evaluate_model("PyTorch MLP (GPU)", y_test, mlp_preds, mlp_probs)

# --- Summary DataFrame ---
results_df = pd.DataFrame(results).sort_values(by="ROC-AUC", ascending=False)
results_df.to_csv("model_benchmark_results.csv", index=False)
print("\n=== Final Benchmark Results ===")
print(results_df.to_string(index=False))

# --- ROC Curve Plot ---
plt.figure(figsize=(8, 6), dpi=300)
for name, (fpr, tpr, auc_val) in roc_curves.items():
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.3f})", lw=2)

plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Guess')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('ROC Curves - Multi-Omics Cancer Subtype Classification')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("roc_curves.png")
print("\nROC curves saved to 'roc_curves.png' and benchmark saved to 'model_benchmark_results.csv'.")
