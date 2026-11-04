import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

# 1. Ensembl ID to HGNC Gene Symbol Mapping Dictionary
gene_mapping = {
    "RERGL (RNA)": "RERGL",
    "PGM5 (RNA)": "PGM5",
    "INMT (RNA)": "INMT",
    "ADAM33 (RNA)": "ADAM33",
    "CYGB (RNA)": "CYGB",
    "SPDEF (RNA)": "SPDEF",
    "WDR86 (RNA)": "WDR86",
    "NGFR (RNA)": "NGFR",
    "LIMS2 (RNA)": "LIMS2",
    "F10 (RNA)": "F10",
    "CAPN11 (RNA)": "CAPN11",
    "SLC7A3 (RNA)": "SLC7A3",
    "TCEAL5 (RNA)": "TCEAL5",
    "KCNJ11 (RNA)": "KCNJ11"
}

def map_features(feature_list):
    """Maps Ensembl IDs to Gene Symbols while preserving RPPA/Methylation IDs."""
    return [gene_mapping.get(f, f) for f in feature_list]

if __name__ == "__main__":
    print("Gene mapping dictionary successfully compiled.")
    print(f"Mapped {len(gene_mapping)} Ensembl IDs to HGNC symbols.")
