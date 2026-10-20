import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

print("--- Step 1: Loading & Aligning Multi-Omic Data ---")
rna_path = "data/processed/CPTAC_RNA.csv"
prot_path = "data/processed/CPTAC_Proteomics.csv" if os.path.exists("data/processed/CPTAC_Proteomics.csv") else "data/processed/CPTAC_Prot.csv"

rna_df = pd.read_csv(rna_path)
prot_df = pd.read_csv(prot_path)

rna_id = next(c for c in rna_df.columns if c.lower() in ['patient_id', 'sample_id', 'id', 'case_id'])
prot_id = next(c for c in prot_df.columns if c.lower() in ['patient_id', 'sample_id', 'id', 'case_id'])

rna_df['merge_id'] = rna_df[rna_id].astype(str).str.strip()
prot_df['merge_id'] = prot_df[prot_id].astype(str).str.strip()

merged_df = pd.merge(rna_df, prot_df, on='merge_id', suffixes=('_rna', '_prot'))

prolif_genes = ['MKI67', 'AURKA', 'CCNB1', 'TOP2A', 'BIRC5', 'CDK1', 'MYBL2', 'BUB1B', 'MCM2']
luminal_genes = ['ESR1', 'PGR', 'FOXA1', 'GATA3', 'BCL2', 'SCUBE2', 'BAG1']
target_genes = prolif_genes + luminal_genes

rna_cols = [c for c in merged_df.columns if any(c == g or c == f"{g}_rna" for g in target_genes)]
prot_cols = [c for c in merged_df.columns if any(c == f"{g}_prot" for g in target_genes)]

data_mat = merged_df[rna_cols + prot_cols].select_dtypes(include=[np.number])

print("--- Step 2: Calculating Pearson Cross-Correlation Matrix ---")
corr_matrix = data_mat.corr(method='pearson')

# Filter edges with correlation threshold |r| >= 0.40
threshold = 0.40
G = nx.Graph()

for col in data_mat.columns:
    node_type = 'RNA' if col in rna_cols else 'Proteomics'
    gene_symbol = col.replace('_rna', '').replace('_prot', '')
    module = 'Proliferation' if gene_symbol in prolif_genes else 'Luminal'
    G.add_node(col, type=node_type, gene=gene_symbol, module=module)

for i in range(len(corr_matrix.columns)):
    for j in range(i + 1, len(corr_matrix.columns)):
        node1 = corr_matrix.columns[i]
        node2 = corr_matrix.columns[j]
        r_val = corr_matrix.iloc[i, j]
        if abs(r_val) >= threshold:
            G.add_edge(node1, node2, weight=r_val)

print(f"Graph topology: {G.number_of_nodes()} Nodes | {G.number_of_edges()} Edges (|r| >= {threshold})")

print("--- Step 3: Rendering Multi-Omic Network Visualization ---")
plt.figure(figsize=(10, 8), dpi=300)
pos = nx.spring_layout(G, seed=42, k=0.35)

node_colors = []
for node, attr in G.nodes(data=True):
    if attr['module'] == 'Luminal':
        node_colors.append('#1f77b4' if attr['type'] == 'RNA' else '#aec7e8')
    else:
        node_colors.append('#d62728' if attr['type'] == 'RNA' else '#ff9896')

edges = G.edges(data=True)
edge_colors = ['#2ca02c' if e[2]['weight'] > 0 else '#d62728' for e in edges]
edge_widths = [abs(e[2]['weight']) * 2.5 for e in edges]

nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=600, alpha=0.9, edgecolors='black')
nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths, alpha=0.6)

labels = {node: node for node in G.nodes()}
nx.draw_networkx_labels(G, pos, labels=labels, font_size=7, font_weight='bold')

plt.title(f"Cross-Modality Co-Expression Network (|r| >= {threshold})", fontweight='bold', fontsize=12)
plt.axis('off')
plt.tight_layout()

os.makedirs("results", exist_ok=True)
output_path = "results/task19_multiomics_network.png"
plt.savefig(output_path, dpi=300)
plt.close()

# Save Edge Metrics CSV
edge_list = []
for u, v, d in G.edges(data=True):
    edge_list.append({"Source": u, "Target": v, "Pearson_R": d['weight']})
edge_df = pd.DataFrame(edge_list).sort_values(by="Pearson_R", ascending=False)
edge_df.to_csv("results/task19_network_edges.csv", index=False)

print(f"Task 19 completed. Saved network figure to '{output_path}' and edge metrics to 'results/task19_network_edges.csv'")
