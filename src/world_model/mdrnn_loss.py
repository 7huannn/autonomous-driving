"""Losses for MDRNN: GMM-NLL + reward MSE + done BCE."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def gmm_nll(pi_logits: Tensor, mu: Tensor, logsigma: Tensor, target: Tensor) -> Tensor:
    target_expanded = target.unsqueeze(2)
    sigma = torch.exp(logsigma)
    normalized = (target_expanded - mu) / (sigma + 1e-8)
    log_prob = -0.5 * (normalized.pow(2) + 2.0 * logsigma + math.log(2.0 * math.pi)).sum(dim=-1)
    log_mix = torch.log_softmax(pi_logits, dim=-1)
    mixture_log_prob = torch.logsumexp(log_mix + log_prob, dim=-1)
    return -mixture_log_prob.mean()


def mdrnn_loss(
    preds: dict[str, Tensor],
    z_next: Tensor,
    reward_target: Tensor,
    done_target: Tensor,
    reward_weight: float = 1.0,
    done_weight: float = 1.0,
) -> tuple[Tensor, dict[str, float]]:
    nll = gmm_nll(preds["pi_logits"], preds["mu"], preds["logsigma"], z_next)
    reward_mse = torch.mean((preds["reward"] - reward_target) ** 2)
    done_bce = torch.nn.functional.binary_cross_entropy_with_logits(preds["done_logits"], done_target)
    total = nll + float(reward_weight) * reward_mse + float(done_weight) * done_bce
    return total, {
        "total": float(total.detach().item()),
        "gmm_nll": float(nll.detach().item()),
        "reward_mse": float(reward_mse.detach().item()),
        "done_bce": float(done_bce.detach().item()),
    }
