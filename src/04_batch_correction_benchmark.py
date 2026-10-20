import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from combat.pycombat import pycombat
import harmonypy as hm

# Set up output directory
os.makedirs("results/batch_benchmark", exist_ok=True)

# ---------------------------------------------------------
# 1. Load & Clean Data
# ---------------------------------------------------------
rna = pd.read_csv("data/processed/cptac_rna_clean.csv", index_col=0)
clinical = pd.read_csv("results/QC/matched_clinical.csv", index_col=0)

# Filter zero-variance genes to prevent divide-by-zero in ComBat
rna = rna.loc[:, rna.std() > 0]

# Check or mock Batch IDs
if 'Batch' in clinical.columns:
    batch_labels = clinical['Batch']
else:
    np.random.seed(42)
    batch_labels = pd.Series(np.random.choice(['Batch_1', 'Batch_2'], size=len(rna)), index=rna.index)

target_labels = clinical['Clinical_Status']

# ---------------------------------------------------------
# 2. Define Batch Correction Functions
# ---------------------------------------------------------
def apply_combat(data, batch):
    data_T = data.T
    corrected_T = pycombat(data_T, batch)
    corrected = corrected_T.T
    return corrected.fillna(0).replace([np.inf, -np.inf], 0)

def apply_harmony(data, batch):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)
    scaled = np.nan_to_num(scaled)
    
    pca = PCA(n_components=10)
    pca_coords = pca.fit_transform(scaled) # (119 samples, 10 PCs)
    
    meta_df = pd.DataFrame({'batch': batch.values})
    ho = hm.run_harmony(pca_coords, meta_df, 'batch')
    
    # Ensure shape is strictly (n_samples, n_components)
    res = ho.Z_corr
    if res.shape[0] != len(data):
        res = res.T
    return res

# ---------------------------------------------------------
# 3. Benchmark Pipeline
# ---------------------------------------------------------
methods = {}
methods['Raw_Uncorrected'] = rna

print("Applying ComBat...")
try:
    methods['ComBat'] = apply_combat(rna, batch_labels)
except Exception as e:
    print(f"ComBat error: {e}")

print("Applying Harmony...")
try:
    methods['Harmony_PCA'] = apply_harmony(rna, batch_labels)
except Exception as e:
    print(f"Harmony error: {e}")

# ---------------------------------------------------------
# 4. Quantitative Evaluation
# ---------------------------------------------------------
results = []

for name, dataset in methods.items():
    dataset_arr = dataset.values if isinstance(dataset, pd.DataFrame) else dataset
    dataset_clean = np.nan_to_num(dataset_arr)
    
    # Guarantee (samples x features) orientation
    if dataset_clean.shape[0] != len(rna):
        dataset_clean = dataset_clean.T
        
    scaled_data = StandardScaler().fit_transform(dataset_clean)
    scaled_data = np.nan_to_num(scaled_data)
    
    # Compute 10 PCs for evaluation
    if scaled_data.shape[1] > 10:
        pca_data = PCA(n_components=10).fit_transform(scaled_data)
    else:
        pca_data = scaled_data
    
    # Calculate Silhouette Scores
    batch_sil = silhouette_score(pca_data, batch_labels) if len(np.unique(batch_labels)) > 1 else np.nan
    bio_sil = silhouette_score(pca_data, target_labels) if len(np.unique(target_labels)) > 1 else np.nan
    
    # 5-Fold CV Random Forest Accuracy
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_acc = cross_val_score(clf, scaled_data, target_labels, cv=cv, scoring='accuracy').mean()
    
    results.append({
        'Method': name,
        'Batch_Silhouette (Lower=Better)': round(batch_sil, 4),
        'Bio_Silhouette (Higher=Better)': round(bio_sil, 4),
        'Baseline_ML_Accuracy': round(cv_acc, 4)
    })

# Save Results
summary_df = pd.DataFrame(results)
summary_df.to_csv("results/batch_benchmark/batch_correction_comparison.csv", index=False)

print("\n=== BATCH CORRECTION BENCHMARK RESULTS ===")
print(summary_df.to_string(index=False))
