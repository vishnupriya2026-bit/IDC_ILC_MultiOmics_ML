import os
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score, 
    average_precision_score, 
    recall_score, 
    brier_score_loss, 
    confusion_matrix
)

# 1. Locate saved model artifacts
model_dir = "models"
if os.path.exists("models/deep_learning"):
    print("Found deep_learning directory in models/")
    print("Contents:", os.listdir("models/deep_learning"))

# 2. Inspect available TCGA/GEO dataset files
tcga_expr_files = os.listdir("datasets/TCGA/expression") if os.path.exists("datasets/TCGA/expression") else []
tcga_meta_files = os.listdir("datasets/TCGA/metadata") if os.path.exists("datasets/TCGA/metadata") else []

print("TCGA Expression Files:", tcga_expr_files)
print("TCGA Metadata Files:", tcga_meta_files)
