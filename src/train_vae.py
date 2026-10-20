import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np

# --- 1. VAE ARCHITECTURE ---
class MultiOmicsVAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 32, hidden_dims: list = [256, 128]):
        super(MultiOmicsVAE, self).__init__()
        
        # Encoder
        encoder_layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, h_dim))
            encoder_layers.append(nn.BatchNorm1d(h_dim))
            encoder_layers.append(nn.LeakyReLU(0.2))
            encoder_layers.append(nn.Dropout(0.2))
            in_dim = h_dim
            
        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoder
        decoder_layers = []
        in_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, h_dim))
            decoder_layers.append(nn.BatchNorm1d(h_dim))
            decoder_layers.append(nn.LeakyReLU(0.2))
            decoder_layers.append(nn.Dropout(0.2))
            in_dim = h_dim
            
        decoder_layers.append(nn.Linear(hidden_dims[0], input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor):
        hidden = self.encoder(x)
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar, z


class VAELoss(nn.Module):
    def __init__(self, beta: float = 0.5):
        super(VAELoss, self).__init__()
        self.beta = beta
        self.mse = nn.MSELoss(reduction='sum')

    def forward(self, x_recon: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        recon_loss = self.mse(x_recon, x)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return (recon_loss + self.beta * kl_loss) / x.size(0)


# --- 2. TRAINING & EMBEDDING EXTRACTION ---
def run_vae_training(data_path: str, latent_dim: int = 32, epochs: int = 150, batch_size: int = 16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--> Operating on computing device: {device}")
    
    # Load dataset
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Update DATA_PATH variable.")
        
    df = pd.read_csv(data_path, index_col=0)
    print(f"--> Loaded multi-omics matrix with shape: {df.shape} (Samples x Features)")
    
    X_data = df.values.astype(np.float32)
    input_dim = X_data.shape[1]
    
    # Tensor dataset
    tensor_x = torch.tensor(X_data, dtype=torch.float32)
    dataset = TensorDataset(tensor_x)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Model, Loss, Optimizer
    model = MultiOmicsVAE(input_dim=input_dim, latent_dim=latent_dim).to(device)
    criterion = VAELoss(beta=0.5)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # Training Loop
    model.train()
    print("--> Starting Multi-Omics VAE Training...")
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for batch in dataloader:
            x_batch = batch[0].to(device)
            optimizer.zero_grad()
            
            x_recon, mu, logvar, _ = model(x_batch)
            loss = criterion(x_recon, x_batch, mu, logvar)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x_batch.size(0)
            
        if epoch % 30 == 0 or epoch == 1:
            print(f"    Epoch [{epoch}/{epochs}] - Loss: {total_loss / len(dataset):.4f}")
            
    # Extract deterministic latent space representation (mu)
    model.eval()
    with torch.no_grad():
        _, mu, _, _ = model(tensor_x.to(device))
        
    latent_df = pd.DataFrame(
        mu.cpu().numpy(),
        index=df.index,
        columns=[f"Latent_Dim_{i+1}" for i in range(latent_dim)]
    )
    
    output_path = "multiomics_latent_embeddings.csv"
    latent_df.to_csv(output_path)
    print(f"--> Extracted latent representation saved to: {output_path}")
    return latent_df


if __name__ == "__main__":
    # Update path to your preprocessed multi-omics dataset file
    DATA_PATH = "/home/priya_26phd0306/BreastCancer_MultiOmics/results/feature_selection/integrated_feature_matrix.csv" 
    
    # Run training if dataset file exists
    if os.path.exists(DATA_PATH):
        run_vae_training(DATA_PATH)
    else:
        print(f"Please specify the path to your preprocessed multi-omics CSV file in DATA_PATH.")
