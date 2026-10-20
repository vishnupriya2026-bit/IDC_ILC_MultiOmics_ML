import re
from pathlib import Path
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

ml_dir = Path("data/processed/ml_ready")
rppa_dir = Path("data/rppa")

# Load Target IDs
y_tr = pd.read_csv(ml_dir / "y_train.csv", index_col=0).bfill().ffill()
y_te = pd.read_csv(ml_dir / "y_test.csv", index_col=0).bfill().ffill()

y_tr.to_csv(ml_dir / "y_train.csv")
y_te.to_csv(ml_dir / "y_test.csv")

train_ids = [str(i)[:12] for i in y_tr.index]
test_ids  = [str(i)[:12] for i in y_te.index]
all_ids   = train_ids + test_ids

# Parse raw TSV files using 'peptide_target' and 'protein_expression'
rppa_files = list(rppa_dir.rglob("*.tsv"))
patient_profiles = {}

for f in rppa_files:
    match = re.search(r'(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})', str(f))
    if not match:
        continue
    
    barcode = match.group(1)
    if barcode not in all_ids:
        continue

    try:
        df = pd.read_csv(f, sep='\t')
        if 'peptide_target' in df.columns and 'protein_expression' in df.columns:
            df['protein_expression'] = pd.to_numeric(df['protein_expression'], errors='coerce')
            series = df.groupby('peptide_target')['protein_expression'].mean()
            
            if barcode not in patient_profiles:
                patient_profiles[barcode] = series.to_dict()
            else:
                s_exist = pd.Series(patient_profiles[barcode])
                patient_profiles[barcode] = pd.concat([s_exist, series], axis=1).mean(axis=1).to_dict()
    except Exception:
        continue

# Construct RPPA Matrix
rppa_df = pd.DataFrame.from_dict(patient_profiles, orient='index').reindex(all_ids)
valid_features = rppa_df.columns[rppa_df.isnull().mean() < 0.2]
rppa_filtered = rppa_df[valid_features]

# Impute & Scale
tr_raw = rppa_filtered.loc[train_ids]
te_raw = rppa_filtered.loc[test_ids]

imp = SimpleImputer(strategy="median")
tr_imp = imp.fit_transform(tr_raw)
te_imp = imp.transform(te_raw)

scaler = StandardScaler()
tr_sc = scaler.fit_transform(tr_imp)
te_sc = scaler.transform(te_imp)

rppa_tr_df = pd.DataFrame(tr_sc, index=train_ids, columns=valid_features)
rppa_te_df = pd.DataFrame(te_sc, index=test_ids, columns=valid_features)

rppa_tr_df.to_csv(ml_dir / "rppa_train.csv")
rppa_te_df.to_csv(ml_dir / "rppa_test.csv")

print("\n==========================================================")
print(" RPPA REPAIR COMPLETE")
print("==========================================================")
print(f"Updated rppa_train.csv Shape: {rppa_tr_df.shape} (Patients x Protein Features)")
print(f"Updated rppa_test.csv  Shape: {rppa_te_df.shape} (Patients x Protein Features)")
print(f"Missing Values: {rppa_tr_df.isnull().sum().sum()}")
