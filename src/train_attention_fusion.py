import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

# --- 1. CROSS-MODAL ATTENTION MODULE ---
class CrossModalAttentionFusion(nn.Module):
    def __init__(self, rna_dim: int, prot_dim: int, embed_dim: int = 64, num_classes: int = 2):
        """
        Cross-Modal Attention Network for dynamic weighting across Omics layers.
        """
        super(CrossModalAttentionFusion, self).__init__()
        
        # Linear projection to unified feature dimension
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
        
        # Attention scoring network
        self.attention_net = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        
        # Downstream Classifier
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )

    def forward(self, rna_x: torch.Tensor, prot_x: torch.Tensor):
        # 1. Project modalities to shared space: (Batch, embed_dim)
        h_rna = self.rna_proj(rna_x)
        h_prot = self.prot_proj(prot_x)
        
        # 2. Stack modalities: (Batch, 2, embed_dim)
        stacked = torch.stack([h_rna, h_prot], dim=1)
        
        # 3. Calculate attention logits & Softmax weights
        attn_logits = self.attention_net(stacked).squeeze(-1)  # (Batch, 2)
        attn_weights = torch.softmax(attn_logits, dim=-1)      # α_rna, α_prot summing to 1
        
        # 4. Weighted multi-modal fusion
        fused = torch.sum(stacked * attn_weights.unsqueeze(-1), dim=1)  # (Batch, embed_dim)
        
        # 5. Prediction
        logits = self.classifier(fused)
        return logits, attn_weights


# --- 2. TRAINING PIPELINE ---
def train_attention_model(rna_path: str, prot_path: str, epochs: int = 100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--> Using device: {device}")
    
    rna_df = pd.read_csv(rna_path, index_col=0)
    prot_df = pd.read_csv(prot_path, index_col=0)
    
    # Align common samples
    common_samples = rna_df.index.intersection(prot_df.index)
    rna_df = rna_df.loc[common_samples]
    prot_df = prot_df.loc[common_samples]
    
    print(f"--> Aligned samples across Omics layers: {len(common_samples)}")
    
    X_rna = torch.tensor(rna_df.values, dtype=torch.float32).to(device)
    X_prot = torch.tensor(prot_df.values, dtype=torch.float32).to(device)
    
    # Generate target labels (synthetic or clinical phenotype)
    # Binary classification placeholder: High vs Low risk split
    y_labels = torch.tensor((rna_df.mean(axis=1) > rna_df.mean(axis=1).median()).astype(int).values, dtype=torch.long).to(device)
    
    model = CrossModalAttentionFusion(
        rna_dim=rna_df.shape[1], 
        prot_dim=prot_df.shape[1], 
        embed_dim=64
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    model.train()
    print("--> Training Cross-Modal Attention Fusion Network...")
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        logits, attn_weights = model(X_rna, X_prot)
        loss = criterion(logits, y_labels)
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0 or epoch == 1:
            mean_rna_attn = attn_weights[:, 0].mean().item()
            mean_prot_attn = attn_weights[:, 1].mean().item()
            print(f"    Epoch [{epoch}/{epochs}] - Loss: {loss.item():.4f} | Attn Weights -> RNA: {mean_rna_attn:.3f}, Prot: {mean_prot_attn:.3f}")
            
    # Save modality attention weights per patient
    model.eval()
    with torch.no_grad():
        _, final_attn = model(X_rna, X_prot)
        
    attn_df = pd.DataFrame(
        final_attn.cpu().numpy(),
        index=common_samples,
        columns=["Attention_Weight_RNA", "Attention_Weight_Proteomics"]
    )
    attn_df.to_csv("patient_omics_attention_weights.csv")
    print("--> Saved patient omics attention weights to: patient_omics_attention_weights.csv")


if __name__ == "__main__":
    RNA_PATH = "/home/priya_26phd0306/BreastCancer_MultiOmics/data/processed/CPTAC_RNA.csv"
    PROT_PATH = "/home/priya_26phd0306/BreastCancer_MultiOmics/data/processed/CPTAC_Proteomics.csv"
    
    if os.path.exists(RNA_PATH) and os.path.exists(PROT_PATH):
        train_attention_model(RNA_PATH, PROT_PATH)
    else:
        print("Dataset paths not found. Verify file paths.")
