import urllib.request
import urllib.parse
import json
import time

# List of reference titles and key search terms
references = [
    "Comprehensive molecular portraits of invasive lobular breast cancer Ciriello",
    "Breast cancer Harbeck Penault-Llorca",
    "Proteogenomic characterization reveals therapeutic vulnerabilities in lung adenocarcinoma Gillette",
    "Comprehensive molecular portraits of human breast tumours Nature 2012",
    "Multi-omics data integration, interpretation, and its application Subramanian",
    "Machine learning applications in genetics and genomics Libbrecht",
    "Leakage and the reproducibility crisis in machine-learning-based science Kapoor",
    "Navigating the pitfalls of applying machine learning in genomics Whalen",
    "A guide to machine learning for biologists Greener",
    "Opportunities and obstacles for deep learning in biology and medicine Ching",
    "Machine learning in medicine Rajkomar",
    "Machine learning algorithm validation with a small sample size Vabalas",
    "A unified approach to interpreting model predictions Lundberg 2017",
    "Drug repurposing progress challenges and recommendations Pushpakom",
    "Proteogenomics of breast cancer biology and response to therapy Krug",
    "The CPTAC Data Portal a resource for cancer proteomics and genomics data Edwards",
    "The Strengthening the Reporting of Observational Studies in Epidemiology STROBE von Elm",
    "Misuse of RPKM or TPM normalization when comparing across different datasets Zhao",
    "Missing value estimation methods for DNA microarrays Troyanskaya",
    "Applied machine learning in cancer research a survey Kourou",
    "From local explanations to global understanding with explainable AI for trees Lundberg",
    "Opening the black box interpretable machine learning for genomics Azodi",
    "On the dependency of cellular protein levels on mRNA abundance Liu",
    "DGIdb 4.0 a redesigned interface and the addition of clinical trial data Cannon",
    "OncoKB a precision oncology knowledge base Chakravarty",
    "ChEMBL towards direct deposition of bioassay data Mendez",
    "Exploration of database resources for drug target interaction prediction Tanoli",
    "CIViC is a community knowledgebase for expert crowdsourcing Griffith",
    "ESR1 mutations and overall survival in patients with metastatic breast cancer Turner",
    "Novel anticancer targets revisiting ERBB2 and EGFR signaling in cancer Baselga",
    "PI3K mutations in breast cancer prognostic and therapeutic implications Mukohara",
    "Artificial intelligence and machine learning in clinical development Shah"
]

pmids = []
print("🔍 Searching PubMed for PMIDs...")

for query in references:
    encoded_q = urllib.parse.quote(query)
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={encoded_q}&retmode=json"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
            id_list = data['esearchresult']['idlist']
            if id_list:
                pmid = id_list[0]
                pmids.append(pmid)
                print(f"  ✓ Found PMID {pmid} for: {query[:40]}...")
            else:
                print(f"  ✗ Not found on PubMed: {query[:40]}...")
    except Exception as e:
        print(f"  ⚠️ Error searching {query[:40]}: {e}")
    time.sleep(0.3)

pmid_string = ",".join(pmids)
print(f"\n✅ Total PMIDs retrieved: {len(pmids)}")
print(f"\nPMID List:\n{pmid_string}\n")

# Download MEDLINE (.nbib) file directly for EndNote
medline_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={pmid_string}&retmode=text&rettype=medline"
req = urllib.request.Request(medline_url, headers={'User-Agent': 'Mozilla/5.0'})

with urllib.request.urlopen(req) as resp, open("endnote_references.nbib", "w") as f:
    f.write(resp.read().decode())

print("🎉 Saved 'endnote_references.nbib' in your current directory!")
