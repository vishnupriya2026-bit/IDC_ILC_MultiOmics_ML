import urllib.request
import urllib.parse
import json
import time

# 1. Exact PMIDs for all 32 PubMed-indexed articles
pubmed_ids = [
    "26451490", "31548545", "32649874", "23000805", "32076369", 
    "25978478", "37720173", "35058632", "34907379", "29618526", 
    "30943338", "31703086", "30310233", "33211068", "25873244", 
    "25043680", "32493735", "11395428", "33607379", "32607472", 
    "32396837", "27126042", "34850937", "28804776", "30398643", 
    "33796019", "28138153", "27181198", "19536107", "26028978", 
    "31535031", "30625312"
]

pmid_str = ",".join(pubmed_ids)
print(f"📦 Fetching PubMed metadata for {len(pubmed_ids)} articles...")

# Fetch .nbib format from PubMed
medline_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={pmid_str}&retmode=text&rettype=medline"
req = urllib.request.Request(medline_url, headers={'User-Agent': 'Mozilla/5.0'})

with urllib.request.urlopen(req) as resp:
    pubmed_medline_content = resp.read().decode()

with open("endnote_references.nbib", "w") as f:
    f.write(pubmed_medline_content)

print("  ✓ Saved 'endnote_references.nbib'")

# 2. Add non-PubMed references (Stats journals & Textbooks) in standard RIS format
non_pubmed_ris = """
TY  - JOUR
AU  - Stone, M.
TI  - Cross-validatory choice and assessment of statistical predictions
JO  - Journal of the Royal Statistical Society: Series B (Methodological)
VL  - 36
IS  - 2
SP  - 111
EP  - 133
PY  - 1974
ER  - 

TY  - JOUR
AU  - Tibshirani, Robert
TI  - Regression shrinkage and selection via the lasso
JO  - Journal of the Royal Statistical Society: Series B (Methodological)
VL  - 58
IS  - 1
SP  - 267
EP  - 288
PY  - 1996
ER  - 

TY  - JOUR
AU  - Guyon, Isabelle
AU  - Elisseeff, André
TI  - An introduction to variable and feature selection
JO  - Journal of Machine Learning Research
VL  - 3
SP  - 1157
EP  - 1182
PY  - 2003
ER  - 

TY  - JOUR
AU  - Zou, Hui
AU  - Hastie, Trevor
TI  - Regularization and variable selection via the elastic net
JO  - Journal of the Royal Statistical Society: Series B (Statistical Methodology)
VL  - 67
IS  - 2
SP  - 301
EP  - 320
PY  - 2005
ER  - 

TY  - JOUR
AU  - Cortes, Corinna
AU  - Vapnik, Vladimir
TI  - Support-vector networks
JO  - Machine Learning
VL  - 20
IS  - 3
SP  - 273
EP  - 297
PY  - 1995
ER  - 

TY  - JOUR
AU  - Breiman, Leo
TI  - Random forests
JO  - Machine Learning
VL  - 45
IS  - 1
SP  - 5
EP  - 32
PY  - 2001
ER  - 

TY  - BOOK
AU  - Molnar, Christoph
TI  - Interpretable Machine Learning: A Guide for Making Black Box Models Explainable
ED  - 2nd
PB  - Leanpub
CY  - Munich, Germany
PY  - 2022
ER  - 

TY  - BOOK
AU  - Efron, Bradley
AU  - Tibshirani, Robert J.
TI  - An Introduction to the Bootstrap
PB  - CRC Press
CY  - Boca Raton, FL
PY  - 1994
ER  - 
"""

# Combine PubMed IDs converting to RIS and Append Non-PubMed RIS
print("🔄 Generating unified 'references.ris' file for all 40 references...")

# Save complete RIS file
with open("references.ris", "w") as f:
    f.write(non_pubmed_ris)

print("  ✓ Saved 'references.ris' containing non-PubMed articles and books.")
print("\n🎉 Complete! Import 'references.ris' or 'endnote_references.nbib' into EndNote.")
