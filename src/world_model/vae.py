"""Convolutional VAE used by the Stage 13 world model."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ConvVAE(nn.Module):
    def __init__(self, image_size: int = 64, channels: int = 3, latent_size: int = 64) -> None:
        super().__init__()
        if image_size != 64:
            raise ValueError("ConvVAE currently supports image_size=64")

        self.image_size = image_size
        self.channels = channels
        self.latent_size = latent_size

        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_size)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_size)
        self.fc_decode = nn.Linear(latent_size, 256 * 4 * 4)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        h = self.encoder(x)
        h = h.flatten(start_dim=1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: Tensor) -> Tensor:
        h = self.fc_decode(z).view(z.shape[0], 256, 4, 4)
        return self.decoder(h)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar


def vae_loss(reconstruction: Tensor, target: Tensor, mu: Tensor, logvar: Tensor, kl_weight: float = 1.0) -> tuple[Tensor, dict[str, float]]:
    recon_loss = torch.mean((reconstruction - target) ** 2)
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total = recon_loss + float(kl_weight) * kl_loss
    return total, {
        "total": float(total.detach().item()),
        "reconstruction_mse": float(recon_loss.detach().item()),
        "kl": float(kl_loss.detach().item()),
    }
