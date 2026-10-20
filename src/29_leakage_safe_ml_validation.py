import os
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, roc_auc_score, average_precision_score, f1_score, accuracy_score

warnings.filterwarnings("ignore")

ROOT = "."
RNA_FILE = "data/processed/CPTAC_RNA.csv"
PROT_FILE = "data/processed/CPTAC_Proteomics.csv"
CLINICAL_CANDIDATES = [
    "datasets/CPTAC/clinical/cptac_brca_clinical.csv",
    "datasets/TCGA/clinical/tcga_brca_clinical.tsv",
]
OUT = "results/ml_validation"
os.makedirs(OUT, exist_ok=True)

print("=" * 70)
print("LEAKAGE-SAFE ML VALIDATION AUDIT")
print("=" * 70)

# ------------------------------------------------------------
# 1. Load genuine processed omics data
# ------------------------------------------------------------
rna = pd.read_csv(RNA_FILE, index_col=0)
prot = pd.read_csv(PROT_FILE, index_col=0)

common = rna.index.intersection(prot.index)

print("\nREAL OMICS DATA")
print("RNA:", rna.shape)
print("Proteomics:", prot.shape)
print("Matched patients:", len(common))

# ------------------------------------------------------------
# 2. Audit clinical files BEFORE choosing an endpoint
# ------------------------------------------------------------
print("\nCLINICAL DATA AUDIT")

clinical_found = []

for f in CLINICAL_CANDIDATES:
    if os.path.exists(f):
        try:
            sep = "\t" if f.endswith(".tsv") else ","
            c = pd.read_csv(f, sep=sep)
            clinical_found.append((f, c))

            print("\nFILE:", f)
            print("Shape:", c.shape)
            print("Columns:", c.columns.tolist())
            print("First rows:")
            print(c.head(3).to_string(index=False))

        except Exception as e:
            print("Could not read:", f, "|", e)

# ------------------------------------------------------------
# 3. Check whether clinical files actually match the 119 patients
# ------------------------------------------------------------
print("\nPATIENT-ID COMPATIBILITY CHECK")

for f, c in clinical_found:
    possible_ids = [
        x for x in c.columns
        if str(x).lower() in {
            "patient_id", "patientid", "sample_id", "sampleid",
            "case_id", "caseid", "bcr_patient_barcode",
            "participant_id", "id"
        }
    ]

    print("\n", f)
    print("Possible ID columns:", possible_ids)

    if possible_ids:
        for col in possible_ids:
            ids = set(c[col].astype(str))
            overlap = len(set(map(str, common)) & ids)
            print("ID:", col, "| overlap with 119:", overlap)

# ------------------------------------------------------------
# 4. IMPORTANT: do not create a synthetic target
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("TARGET VALIDATION")
print("=" * 70)

print("""
The previous ML pipeline created an artificial endpoint such as:

    ESR1 >= median -> class 1
    ESR1 < median  -> class 0

This is NOT an independent clinical phenotype and should NOT be used
as the final prediction endpoint.

Likewise, synthetic survival time/event variables must not be used.

A legitimate supervised ML model requires a real patient-level outcome,
phenotype, subtype, treatment response, or other scientifically
justified label that matches the 119 omics patients.
""")

# ------------------------------------------------------------
# 5. Check existing integrated matrix ONLY for audit purposes
# ------------------------------------------------------------
integrated_file = "results/feature_selection/integrated_feature_matrix.csv"

if os.path.exists(integrated_file):
    integrated = pd.read_csv(integrated_file, index_col=0)
    print("\nEXISTING INTEGRATED MATRIX (AUDIT ONLY)")
    print("Shape:", integrated.shape)
    print("This matrix will NOT be used for the new validation because")
    print("its features were selected before model cross-validation.")

# ------------------------------------------------------------
# 6. Demonstrate the leakage-safe model architecture
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("LEAKAGE-SAFE MODEL ARCHITECTURE")
print("=" * 70)

print("""
Once a REAL endpoint is identified, the correct workflow will be:

Patient-level split
        |
        v
Training fold
        |
        +--> imputation
        |
        +--> scaling
        |
        +--> feature selection
        |
        +--> model fitting
        |
        v
Validation fold
        |
        +--> same fitted preprocessing
        |
        +--> prediction
        |
        v
ROC-AUC / PR-AUC / F1 / Accuracy / Sensitivity / Specificity

Every preprocessing and feature-selection operation must be fitted
inside the training fold.

No synthetic labels.
No synthetic survival data.
No pre-selected feature matrix.
No test-set feature selection.
No CMOT-Net.
""")

# ------------------------------------------------------------
# 7. Save audit
# ------------------------------------------------------------
audit_file = os.path.join(OUT, "ml_validation_audit.txt")

with open(audit_file, "w") as fh:
    fh.write("LEAKAGE-SAFE ML VALIDATION AUDIT\n")
    fh.write(f"RNA shape: {rna.shape}\n")
    fh.write(f"Proteomics shape: {prot.shape}\n")
    fh.write(f"Matched patients: {len(common)}\n")
    fh.write("Existing integrated matrix is NOT used for validation.\n")
    fh.write("Synthetic ESR1/survival endpoints are NOT used.\n")
    fh.write("A real patient-level endpoint is required before supervised ML.\n")

print("\nSaved audit:", audit_file)

print("\n" + "=" * 70)
print("RESULT")
print("=" * 70)
print("""
ML HAS NOT BEEN RERUN WITH A FABRICATED TARGET.

This is intentional.

The next scientifically correct step is to identify a REAL endpoint
that matches the 119 CPTAC patients. Once that endpoint is confirmed,
we can run nested/stratified patient-level cross-validation using
Logistic Regression/LASSO and Random Forest and select the model
based on cross-validated performance.
""")
