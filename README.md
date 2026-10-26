cat << 'EOF' > README.md
# Multi-Omics Machine Learning Framework & 15-Biomarker Prognostic Signature for IDC/ILC Breast Cancer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Journal Target](https://img.shields.io/badge/Target%20Journal-Computers%20in%20Biology%20and%20Medicine-green.svg)](https://www.sciencedirect.com/journal/computers-in-biology-and-medicine)

Official implementation of the multi-omics machine learning pipeline for subtype classification and prognostic survival stratification of Invasive Ductal Carcinoma (IDC) and Invasive Lobular Carcinoma (ILC).

---

## 🌟 Key Performance Metrics

| Metric / Parameter | Value / Finding |
| :--- | :--- |
| **Biomarker Panel** | Calibrated 15-Biomarker Multi-Omics Risk Signature |
| **Champion Model** | Random Forest Classifier |
| **Internal Cross-Validation** | **ROC-AUC = 0.9229** (10-Fold Stratified CV) |
| **External Validation Cohort** | METABRIC ($N = 1,894$, EGA ID: `EGAS00001000053`) |
| **Survival Stratification** | **Log-Rank $p < 0.0001$** |
| **Cox Hazard Ratio ($\text{HR}$)** | **$\text{HR} = 1.47$** ($95\%$ CI: $1.29\text{--}1.67$) |

---

## 📁 Repository Structure

```text
IDC_ILC_MultiOmics_ML/
├── data/
│   ├── raw/                  # Raw omics datasets (RNA-seq, CNV, Clinical)
│   └── processed/            # Preprocessed and standardized validation datasets
├── src/
│   ├── generate_corrected_km_plot.py  # METABRIC survival analysis script
│   ├── train_champion_rf.py           # 10-fold CV Random Forest classification
│   └── preprocess_omics.py            # Feature extraction and missing data handling
├── results/
│   └── figures/
│       ├── kaplan_meier_15_biomarkers.png  # Calibrated 15-biomarker KM plot
│       ├── roc_curves_1
