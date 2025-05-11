import torch
import torch.nn as nn

class PhaseEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
        nn.Linear(input_dim, 128),
        nn.ReLU(),
        nn.LayerNorm(128),
        nn.Linear(128, latent_dim),
        nn.ReLU(),
        )


    def forward(self, obs):
        # obs: [B, T, D]
        B, T, D = obs.shape
        obs = obs.view(B * T, D)  # Flatten batch and time
        out = self.net(obs)       # [B*T, latent_dim]
        out = out.view(B, T, -1)  # [B, T, latent_dim]
        out = out.mean(dim=1)     # [B, latent_dim]
        return out
