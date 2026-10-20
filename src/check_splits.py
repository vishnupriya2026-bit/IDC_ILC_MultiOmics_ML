import numpy as np
from sklearn.model_selection import StratifiedKFold

# 1. CHECK 10-FOLD STRATIFIED CROSS-VALIDATION SPLITS (TCGA-BRCA)
n_idc_tcga, n_ilc_tcga = 308, 102
y_tcga = np.array([0] * n_idc_tcga + [1] * n_ilc_tcga)

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

print("=" * 65)
print(" 1. DISCOVERY COHORT (TCGA-BRCA): 10-FOLD STRATIFIED CV SPLITS")
print("=" * 65)
print(f"Total Cohort Size: N = {len(y_tcga)} (IDC: {sum(y_tcga==0)}, ILC: {sum(y_tcga==1)})\n")

for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y_tcga)), y_tcga), 1):
    y_train = y_tcga[train_idx]
    y_val = y_tcga[val_idx]
    
    idc_tr, ilc_tr = sum(y_train == 0), sum(y_train == 1)
    idc_va, ilc_va = sum(y_val == 0), sum(y_val == 1)
    
    print(f"Fold {fold:2d} | Train: {len(train_idx):3d} (IDC: {idc_tr:3d}, ILC: {ilc_tr:2d}) "
          f"| Val: {len(val_idx):2d} (IDC: {idc_va:2d}, ILC: {ilc_va:2d}) "
          f"| Train/Val Ratio: {len(train_idx)/len(val_idx):.1f}:1")

# 2. CHECK DISCOVERY vs. EXTERNAL VALIDATION COHORT DIVISION
n_tcga = 410
n_metabric = 1894
n_total = n_tcga + n_metabric

print("\n" + "=" * 65)
print(" 2. COHORT DIVISION: DISCOVERY (TCGA) vs. EXTERNAL TEST (METABRIC)")
print("=" * 65)
print(f"Discovery Cohort (TCGA-BRCA)   : N = {n_tcga:4d} ({n_tcga/n_total*100:.1f}% of total data)")
print(f"External Test Cohort (METABRIC): N = {n_metabric:4d} ({n_metabric/n_total*100:.1f}% of total data)")
print(f"External Test Expansion Ratio   : {n_metabric / n_tcga:.2f}x larger than discovery cohort")
