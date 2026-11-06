import torch
import torch.nn as nn


class ImaginaryMatrixGenerator(nn.Module):

    def __init__(
        self,
        in_channels: int,
        img_size: int = 28
    ):

        super().__init__()

        flat_dim = in_channels * img_size * img_size

        self.mlp = nn.Sequential(

            nn.Linear(flat_dim, 256),

            nn.GELU(),

            nn.Linear(256, 128),

            nn.GELU()
        )

        self.mu_head = nn.Linear(128, flat_dim)

        self.sigma_head = nn.Linear(128, flat_dim)

    def forward(self, x):

        batch, channels, height, width = x.shape

        flat = x.reshape(batch, -1)

        features = self.mlp(flat)

        mu = torch.tanh(
            self.mu_head(features)
        )

        sigma = torch.tanh(
            self.sigma_head(features)
        ).abs() + 1e-6

        epsilon = torch.randn_like(mu)

        imaginary = mu + sigma * epsilon

        return imaginary.reshape(
            batch,
            channels,
            height,
            width
        )