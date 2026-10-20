import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from captum.attr import IntegratedGradients

# --- 1. MODEL ARCHITECTURE ---
class CrossModalAttentionFusion(nn.Module):
    def __init__(self, rna_dim: int, prot_dim: int, embed_dim: int = 64, num_classes: int = 2):
        super(CrossModalAttentionFusion, self).__init__()
        
        self.rna_proj = nn.Sequential(
            nn.Linear(rna_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.prot_proj = nn.Sequential(
            nn.Linear(prot_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.attention_net = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )

    def forward(self, rna_x: torch.Tensor, prot_x: torch.Tensor):
        h_rna = self.rna_proj(rna_x)
        h_prot = self.prot_proj(prot_x)
        stacked = torch.stack([h_rna, h_prot], dim=1)
        attn_logits = self.attention_net(stacked).squeeze(-1)
        attn_weights = torch.softmax(attn_logits, dim=-1)
        fused = torch.sum(stacked * attn_weights.unsqueeze(-1), dim=1)
        logits = self.classifier(fused)
        return logits


# --- 2. INTEGRATED GRADIENTS EXPLAINABILITY ---
def run_integrated_gradients(rna_path: str, prot_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--> Operating on device: {device}")
    
    rna_df = pd.read_csv(rna_path, index_col=0)
    prot_df = pd.read_csv(prot_path, index_col=0)
    
    common_samples = rna_df.index.intersection(prot_df.index)
    rna_df = rna_df.loc[common_samples]
    prot_df = prot_df.loc[common_samples]
    
    X_rna = torch.tensor(rna_df.values, dtype=torch.float32).to(device)
    X_prot = torch.tensor(prot_df.values, dtype=torch.float32).to(device)
    
    model = CrossModalAttentionFusion(
        rna_dim=rna_df.shape[1], 
        prot_dim=prot_df.shape[1]
    ).to(device)
    model.eval()
    
    # Initialize Captum Integrated Gradients
    ig = IntegratedGradients(model)
    
    # Baselines (zero tensors matching input shape)
    baseline_rna = torch.zeros_like(X_rna)
    baseline_prot = torch.zeros_like(X_prot)
    
    print("--> Computing Integrated Gradients across all multi-omics features...")
    
    # Target class = 1 (High Risk / Outcome)
    attributions_rna, attributions_prot = ig.attribute(
        inputs=(X_rna, X_prot),
        baselines=(baseline_rna, baseline_prot),
        target=1,
        n_steps=50
    )
    
    # Average absolute attribution scores across cohort
    mean_rna_attr = torch.abs(attributions_rna).mean(dim=0).cpu().detach().numpy()
    mean_prot_attr = torch.abs(attributions_prot).mean(dim=0).cpu().detach().numpy()
    
    # Construct Feature Importance DataFrames
    rna_importance = pd.DataFrame({
        "Feature": rna_df.columns,
        "Modality": "RNA-Seq",
        "IG_Attribution_Score": mean_rna_attr
    })
    
    prot_importance = pd.DataFrame({
        "Feature": prot_df.columns,
        "Modality": "Proteomics",
        "IG_Attribution_Score": mean_prot_attr
    })
    
    full_attributions = pd.concat([rna_importance, prot_importance]).sort_values(
        by="IG_Attribution_Score", ascending=False
    ).reset_index(drop=True)
    
    output_path = "prioritized_dl_biomarkers.csv"
    full_attributions.to_csv(output_path, index=False)
    
    print(f"--> Ranked feature attributions saved to: {output_path}")
    print("\n--- TOP 10 DEEP LEARNING PRIORITIZED BIOMARKERS ---")
    print(full_attributions.head(10).to_string(index=False))


if __name__ == "__main__":
    RNA_PATH = "/home/priya_26phd0306/BreastCancer_MultiOmics/data/processed/CPTAC_RNA.csv"
    PROT_PATH = "/home/priya_26phd0306/BreastCancer_MultiOmics/data/processed/CPTAC_Proteomics.csv"
    
    if os.path.exists(RNA_PATH) and os.path.exists(PROT_PATH):
        run_integrated_gradients(RNA_PATH, PROT_PATH)
    else:
        print("Data paths not found.")
