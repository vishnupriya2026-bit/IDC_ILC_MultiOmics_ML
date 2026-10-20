import os
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def run_ppi_analysis():
    deg_file = "results/feature_selection/differential_expression_results.csv"
    out_csv = "results/feature_selection/ppi_hub_genes.csv"
    out_fig = "results/publication_figures/ppi_network.png"
    
    os.makedirs("results/feature_selection", exist_ok=True)
    os.makedirs("results/publication_figures", exist_ok=True)
    
    # Select top 25 DEGs sorted by FDR
    df = pd.read_csv(deg_file)
    top_genes = df[df['status'] != 'Not Significant'].sort_values('fdr').head(25)['feature'].tolist()
    
    print(f"Querying STRING API for PPI network of top {len(top_genes)} genes...")
    
    string_api_url = "https://string-db.org/api/json/network"
    params = {
        "identifiers": "\r".join(top_genes),
        "species": 9606,  # Human
        "caller_identity": "multiomics_pipeline"
    }
    
    try:
        response = requests.post(string_api_url, data=params, timeout=10)
        edges = response.json()
    except Exception as e:
        print(f"STRING API request failed or timed out: {e}. Building local network graph...")
        edges = []

    G = nx.Graph()
    for gene in top_genes:
        G.add_node(gene)
        
    if isinstance(edges, list) and len(edges) > 0:
        for edge in edges:
            p1, p2 = edge.get('preferredName_A'), edge.get('preferredName_B')
            score = edge.get('score', 0)
            if p1 and p2 and score > 0.4:
                G.add_edge(p1, p2, weight=score)
                
    # Calculate degree centrality (Hub Genes)
    centrality = nx.degree_centrality(G)
    hub_df = pd.DataFrame(list(centrality.items()), columns=['gene', 'degree_centrality'])
    hub_df.sort_values(by='degree_centrality', ascending=False, inplace=True)
    hub_df.to_csv(out_csv, index=False)
    print(f"Saved PPI Hub Genes to: {out_csv}")
    
    # Plot PPI Network Graph
    plt.figure(figsize=(8, 8), dpi=300)
    pos = nx.spring_layout(G, k=0.5, seed=42)
    
    node_sizes = [v * 3000 + 300 for v in centrality.values()]
    
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='#377eb8', alpha=0.9)
    nx.draw_networkx_edges(G, pos, alpha=0.4, edge_color='gray')
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold')
    
    plt.title("Protein-Protein Interaction (PPI) Network", fontsize=12)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_fig)
    plt.close()
    print(f"Saved PPI Network Plot to: {out_fig}")

if __name__ == "__main__":
    run_ppi_analysis()
