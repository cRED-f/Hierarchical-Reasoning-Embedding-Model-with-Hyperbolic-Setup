import torch
from typing import List


def expmap0_poincare(u: torch.Tensor, c: float) -> torch.Tensor:
    """
    Exp_0^c(u) = tanh(sqrt(c)*||u||) * u / (sqrt(c)*||u||)
    """
    if c <= 0:
        raise ValueError("c must be > 0")
    sqrt_c = u.new_tensor(c).sqrt()
    u_norm = u.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)
    scaled = sqrt_c * u_norm
    coef = torch.tanh(scaled) / (sqrt_c * u_norm)
    return coef * u


def poincare_distance_batch(q: torch.Tensor, cand: torch.Tensor, c: float, eps: float = 1e-6) -> torch.Tensor:
    """
    q:    [B,d]
    cand: [B,C,d]
    returns distances [B,C]
    """
    if c <= 0:
        raise ValueError("c must be > 0")

    q = q.float()
    cand = cand.float()

    sqrt_c = q.new_tensor(c).sqrt()

    q2 = (q * q).sum(dim=-1, keepdim=True)
    b2 = (cand * cand).sum(dim=-1)
    diff2 = ((q.unsqueeze(1) - cand) ** 2).sum(dim=-1)

    denom = (1.0 - c * q2).clamp_min(eps) * (1.0 - c * b2).clamp_min(eps)
    arg = 1.0 + 2.0 * c * diff2 / denom
    arg = arg.clamp_min(1.0 + eps)

    return torch.acosh(arg) / sqrt_c


def poincare_distance_matrix(a: torch.Tensor, b: torch.Tensor, c: float, eps: float = 1e-6) -> torch.Tensor:
    """
    Compute pairwise Poincaré distance between rows of a and b.
    a: [N, d]
    b: [M, d]
    returns: [N, M]
    """
    if c <= 0:
        raise ValueError("c must be > 0")

    a = a.float()
    b = b.float()

    sqrt_c = a.new_tensor(c).sqrt()

    a2 = (a * a).sum(dim=-1, keepdim=True)
    b2 = (b * b).sum(dim=-1)
    diff2 = ((a.unsqueeze(1) - b.unsqueeze(0)) ** 2).sum(dim=-1)

    denom = (1.0 - c * a2).clamp_min(eps) * (1.0 - c * b2).clamp_min(eps)
    arg = 1.0 + 2.0 * c * diff2 / denom
    arg = arg.clamp_min(1.0 + eps)

    return torch.acosh(arg) / sqrt_c