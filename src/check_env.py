import torch
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# 1. GPU Verification
print("--- GPU Verification ---")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device Name: {torch.cuda.get_device_name(0)}")

# 2. Load and Align DIABLO Data
print("\n--- Loading & Aligning Dataset ---")
X_train = pd.read_csv("diablo_exports/X_diablo_train_scaled.csv", index_col=0)
y_train_df = pd.read_csv("diablo_exports/y_diablo_train_encoded.csv", index_col=0)

X_test = pd.read_csv("diablo_exports/X_diablo_test_scaled.csv", index_col=0)
y_test_df = pd.read_csv("diablo_exports/y_test_encoded.csv" if "y_test_encoded.csv" in y_train_df.columns else "diablo_exports/y_diablo_test_encoded.csv", index_col=0)

# Remove duplicated block labels and align strictly to X sample IDs
y_train_df = y_train_df[~y_train_df.index.duplicated(keep='first')]
y_test_df = y_test_df[~y_test_df.index.duplicated(keep='first')]

y_train = y_train_df.reindex(X_train.index).iloc[:, 0].values
y_test = y_test_df.reindex(X_test.index).iloc[:, 0].values

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape:  {X_test.shape}")
print(f"y_test shape:   {y_test.shape}")

# 3. Train Baseline Model (Random Forest)
print("\n--- Training Baseline Model ---")
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(f"Baseline Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
