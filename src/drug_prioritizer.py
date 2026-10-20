import requests
import pandas as pd
from typing import List

class DrugPrioritizationEngine:
    """
    Module for querying DGIdb API (v5 GraphQL) to identify 
    and rank targeted therapeutics for multi-omics biomarkers.
    """
    def __init__(self, genes: List[str]):
        self.genes = [g.upper().strip() for g in genes]
        self.graphql_url = "https://dgidb.org/api/graphql"

    def fetch_dgidb_interactions(self) -> pd.DataFrame:
        """
        Queries DGIdb GraphQL API for candidate genes and returns a structured DataFrame.
        """
        print(f"🔍 Fetching drug interactions for genes: {self.genes}...")
        results = []
        
        query = """
        query GetInteractions($names: [String!]) {
          genes(names: $names) {
            nodes {
              name
              interactions {
                drug {
                  name
                }
                interactionTypes {
                  type
                }
                sources {
                  sourceDbName
                }
              }
            }
          }
        }
        """
        
        try:
            response = requests.post(
                self.graphql_url,
                json={"query": query, "variables": {"names": self.genes}},
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                gene_nodes = data.get("data", {}).get("genes", {}).get("nodes", [])
                
                for gene_node in gene_nodes:
                    gene_name = gene_node.get("name")
                    for inter in gene_node.get("interactions", []):
                        drug_info = inter.get("drug", {})
                        drug_name = drug_info.get("name") if drug_info else "Unknown"
                        
                        types = [t.get("type") for t in inter.get("interactionTypes", []) if t.get("type")]
                        interaction_types = ", ".join(types) if types else "N/A"
                        
                        sources = [s.get("sourceDbName") for s in inter.get("sources", []) if s.get("sourceDbName")]
                        source_str = ", ".join(sources) if sources else "N/A"
                        
                        results.append({
                            "Gene": gene_name,
                            "Drug": drug_name,
                            "Interaction_Type": interaction_types,
                            "Sources": source_str,
                            "Evidence_Count": len(sources)
                        })
            else:
                print(f"⚠️ API returned status code {response.status_code}")
                
        except Exception as e:
            print(f"❌ Connection error: {e}")

        df = pd.DataFrame(results)
        return df

    def prioritize_drugs(self, interactions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Ranks drugs based on target evidence count and specificity.
        """
        if interactions_df.empty:
            print("⚠️ No drug interactions found for provided genes.")
            return pd.DataFrame()

        # Sort by evidence density
        ranked_df = interactions_df.sort_values(by="Evidence_Count", ascending=False).reset_index(drop=True)
        return ranked_df

if __name__ == "__main__":
    sample_biomarkers = ["ERBB2", "ESR1", "PIK3CA", "BRCA1"]
    
    engine = DrugPrioritizationEngine(genes=sample_biomarkers)
    raw_interactions = engine.fetch_dgidb_interactions()
    prioritized_report = engine.prioritize_drugs(raw_interactions)
    
    print("\n--- TOP PRIORITIZED DRUGS FOR BIOMARKERS ---")
    print(prioritized_report.head(10).to_string())
    
    output_path = "data/processed/prioritized_drugs.csv"
    prioritized_report.to_csv(output_path, index=False)
    print(f"\n✅ Results saved successfully to {output_path}")
