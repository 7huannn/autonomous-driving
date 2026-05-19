from __future__ import annotations

import torch

from world_model.vae import ConvVAE, vae_loss


def test_vae_forward_shapes() -> None:
    model = ConvVAE(image_size=64, channels=3, latent_size=64)
    x = torch.rand(2, 3, 64, 64)
    recon, mu, logvar = model(x)
    assert recon.shape == (2, 3, 64, 64)
    assert mu.shape == (2, 64)
    assert logvar.shape == (2, 64)


def test_vae_loss_outputs_metrics() -> None:
    model = ConvVAE(image_size=64, channels=3, latent_size=64)
    x = torch.rand(4, 3, 64, 64)
    recon, mu, logvar = model(x)
    loss, metrics = vae_loss(recon, x, mu, logvar, kl_weight=1.0)

    assert loss.ndim == 0
    assert loss.item() >= 0.0
    assert metrics["total"] >= 0.0
    assert metrics["reconstruction_mse"] >= 0.0
    assert metrics["kl"] >= 0.0
