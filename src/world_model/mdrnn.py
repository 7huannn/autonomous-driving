"""Mixture-density recurrent dynamics model for Stage 13."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class MDRNN(nn.Module):
    def __init__(self, latent_size: int = 64, action_size: int = 3, hidden_size: int = 256, num_gaussians: int = 5) -> None:
        super().__init__()
        self.latent_size = latent_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.num_gaussians = num_gaussians

        self.rnn = nn.LSTM(input_size=latent_size + action_size, hidden_size=hidden_size, batch_first=True)
        self.pi_head = nn.Linear(hidden_size, num_gaussians)
        self.mu_head = nn.Linear(hidden_size, num_gaussians * latent_size)
        self.logsigma_head = nn.Linear(hidden_size, num_gaussians * latent_size)
        self.reward_head = nn.Linear(hidden_size, 1)
        self.done_head = nn.Linear(hidden_size, 1)

    def forward(
        self,
        z_t: Tensor,
        actions: Tensor,
        hidden: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[dict[str, Tensor], tuple[Tensor, Tensor]]:
        x = torch.cat([z_t, actions], dim=-1)
        out, hidden_next = self.rnn(x, hidden)

        bsz, steps, _ = out.shape
        pi_logits = self.pi_head(out)
        mu = self.mu_head(out).view(bsz, steps, self.num_gaussians, self.latent_size)
        logsigma = self.logsigma_head(out).view(bsz, steps, self.num_gaussians, self.latent_size)
        reward = self.reward_head(out).squeeze(-1)
        done_logits = self.done_head(out).squeeze(-1)

        preds = {
            "pi_logits": pi_logits,
            "mu": mu,
            "logsigma": logsigma,
            "reward": reward,
            "done_logits": done_logits,
        }
        return preds, hidden_next
