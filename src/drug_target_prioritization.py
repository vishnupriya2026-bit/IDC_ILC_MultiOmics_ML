import os
import requests
import pandas as pd

def query_dgidb_v5(genes: list):
    """
    Query DGIdb v5 GraphQL API for drug-gene interactions.
    """
    print(f"--> Querying DGIdb GraphQL API for top {len(genes)} biomarkers...")
    url = "https://dgidb.org/api/graphql"
    
    query = """
    query getInteractions($genes: [String!]) {
      genes(names: $genes) {
        nodes {
          name
          interactions {
            drug {
              name
            }
            interactionTypes {
              type
            }
          }
        }
      }
    }
    """
    
    variables = {"genes": genes}
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    
    records = []
    try:
        response = requests.post(url, json={'query': query, 'variables': variables}, headers=headers, timeout=20)
        if response.status_code == 200 and 'data' in response.json():
            gene_nodes = response.json()['data']['genes']['nodes']
            for node in gene_nodes:
                gene_name = node.get('name')
                for inter in node.get('interactions', []):
                    drug_name = inter.get('drug', {}).get('name')
                    types = [t.get('type') for t in inter.get('interactionTypes', []) if t.get('type')]
                    interaction_type = ", ".join(types) if types else "Interaction"
                    if drug_name:
                        records.append({
                            "Biomarker_Gene": gene_name,
                            "Candidate_Drug": drug_name,
                            "Interaction_Type": interaction_type
                        })
    except Exception as e:
        print(f"GraphQL Query Notice: {e}")

    # Fallback to MyChem.info REST API if GraphQL records are sparse
    if len(records) < 5:
        print("--> Fetching additional druggability mapping via MyChem/MyGene REST API...")
        for g in genes:
            try:
                mc_url = f"https://mychem.info/v1/query?q={g}&fields=chembl.compound_name,drugbank.name,pharmgkb.trade_names&size=5"
                res = requests.get(mc_url, timeout=5)
                if res.status_code == 200:
                    hits = res.json().get('hits', [])
                    for hit in hits:
                        drug_name = None
                        if 'chembl' in hit:
                            drug_name = hit['chembl'].get('compound_name')
                        elif 'drugbank' in hit:
                            drug_name = hit['drugbank'].get('name')
                        if drug_name:
                            records.append({
                                "Biomarker_Gene": g,
                                "Candidate_Drug": drug_name.upper(),
                                "Interaction_Type": "In-silico Candidate Target"
                            })
            except Exception:
                continue

    return pd.DataFrame(records)


def run_phase_d_pipeline(biomarker_file: str, top_n: int = 20):
    if not os.path.exists(biomarker_file):
        raise FileNotFoundError(f"File not found: {biomarker_file}")
        
    df_bm = pd.read_csv(biomarker_file)
    top_genes = df_bm["Feature"].head(top_n).tolist()
    
    drug_df = query_dgidb_v5(top_genes)
    
    if not drug_df.empty:
        merged_df = drug_df.merge(
            df_bm[["Feature", "IG_Attribution_Score"]], 
            left_on="Biomarker_Gene", 
            right_on="Feature"
        ).drop_duplicates().sort_values(by="IG_Attribution_Score", ascending=False)
        
        output_path = "breast_cancer_repurposed_drugs.csv"
        merged_df.to_csv(output_path, index=False)
        print(f"\n--> Drug target predictions successfully saved to: {output_path}")
        print("\n--- DEEP LEARNING TARGET & REPURPOSED DRUG PREDICTIONS ---")
        print(merged_df.head(15).to_string(index=False))
    else:
        print("No drug target records found for the queried genes.")

if __name__ == "__main__":
    run_phase_d_pipeline("prioritized_dl_biomarkers.csv", top_n=20)
