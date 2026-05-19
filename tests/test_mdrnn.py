from __future__ import annotations

import torch

from world_model.mdrnn import MDRNN
from world_model.mdrnn_loss import mdrnn_loss


def test_mdrnn_forward_shapes() -> None:
    model = MDRNN(latent_size=64, action_size=3, hidden_size=256, num_gaussians=5)
    z_t = torch.randn(2, 7, 64)
    a_t = torch.randn(2, 7, 3)
    preds, hidden = model(z_t, a_t)

    assert preds["pi_logits"].shape == (2, 7, 5)
    assert preds["mu"].shape == (2, 7, 5, 64)
    assert preds["logsigma"].shape == (2, 7, 5, 64)
    assert preds["reward"].shape == (2, 7)
    assert preds["done_logits"].shape == (2, 7)
    h, c = hidden
    assert h.shape == (1, 2, 256)
    assert c.shape == (1, 2, 256)


def test_mdrnn_loss_terms_are_finite() -> None:
    model = MDRNN(latent_size=8, action_size=3, hidden_size=32, num_gaussians=3)
    z_t = torch.randn(4, 5, 8)
    a_t = torch.randn(4, 5, 3)
    z_next = torch.randn(4, 5, 8)
    reward = torch.randn(4, 5)
    done = torch.randint(0, 2, (4, 5), dtype=torch.float32)

    preds, _ = model(z_t, a_t)
    loss, metrics = mdrnn_loss(preds, z_next, reward, done)

    assert torch.isfinite(loss)
    assert metrics["total"] == metrics["total"]
    assert metrics["gmm_nll"] >= 0.0
    assert metrics["reward_mse"] >= 0.0
    assert metrics["done_bce"] >= 0.0
