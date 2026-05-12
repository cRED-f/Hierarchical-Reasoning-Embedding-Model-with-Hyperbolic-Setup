import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Dict, Any


class TokenAttentionPooler(nn.Module):
    """
    scores = score_net(H) -> [B,L,heads]
    alpha  = softmax(scores over tokens with mask)
    pooled = sum_i alpha_i * H_i -> [B,heads,d]
    out    = mean heads -> [B,d]
    """

    def __init__(self, d_model: int, heads: int = 1, hidden_mult: int = 2, dropout: float = 0.0):
        super().__init__()
        self.heads = heads
        hdim = max(32, hidden_mult * d_model)

        self.score_net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hdim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hdim, heads),
        )
        for m in self.score_net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        scores = self.score_net(last_hidden_state)

        mask = attention_mask.to(dtype=torch.bool)
        mask_value = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(~mask.unsqueeze(-1), mask_value)

        alpha = torch.softmax(scores, dim=1)
        pooled = (alpha.unsqueeze(-1) * last_hidden_state.unsqueeze(2)).sum(dim=1)

        if self.heads > 1:
            return pooled.mean(dim=1)
        return pooled.squeeze(1)


class HRMRefinementEncoder(nn.Module):
    """
    Frozen backbone -> token states H (no grad)
    Trainable pooler -> x
    Trainable projection -> u0
    HRM dynamics -> tangents h^(m)
    """

    def __init__(
        self,
        backbone: nn.Module,
        d_base: int,
        d_hrm: int = 512,
        n_cycles: int = 2,
        t_low: int = 2,
        hidden_mult: int = 4,
        pool_heads: int = 1,
        pool_hidden_mult: int = 2,
        pool_dropout: float = 0.0,
        proj_scale: float = 1.0,
    ):
        super().__init__()
        self.backbone = backbone
        self.d_base = d_base
        self.d_hrm = d_hrm
        self.n_cycles = n_cycles
        self.t_low = t_low

        self.pooler = TokenAttentionPooler(
            d_model=d_base,
            heads=pool_heads,
            hidden_mult=pool_hidden_mult,
            dropout=pool_dropout,
        )
        self.in_proj = nn.Linear(d_base, d_hrm)

        hdim = hidden_mult * d_hrm
        self.low_mlp = nn.Sequential(
            nn.LayerNorm(3 * d_hrm),
            nn.Linear(3 * d_hrm, hdim),
            nn.GELU(),
            nn.Linear(hdim, d_hrm),
        )
        self.high_mlp = nn.Sequential(
            nn.LayerNorm(2 * d_hrm),
            nn.Linear(2 * d_hrm, hdim),
            nn.GELU(),
            nn.Linear(hdim, d_hrm),
        )
        self.out_proj = nn.Linear(d_hrm, d_hrm)

        self.z0_H = nn.Parameter(torch.zeros(1, d_hrm))
        self.z0_L = nn.Parameter(torch.zeros(1, d_hrm))

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.zeros_(self.in_proj.bias)
        for seq in (self.low_mlp, self.high_mlp):
            for m in seq:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    @torch.no_grad()
    def encode_backbone_tokens(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        return out.last_hidden_state

    def pool_tokens(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.pooler(last_hidden_state, attention_mask)

    def project_to_hrm(self, x: torch.Tensor) -> torch.Tensor:
        return self.in_proj(x)

    def init_state(self, B: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        zH = self.z0_H.expand(B, -1).to(device)
        zL = self.z0_L.expand(B, -1).to(device)
        return zH, zL

    def low_update(self, zL: torch.Tensor, zH: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        h = torch.cat([zL, zH, x], dim=-1)
        return zL + self.low_mlp(h)

    def high_update(self, zH: torch.Tensor, zL: torch.Tensor) -> torch.Tensor:
        h = torch.cat([zH, zL], dim=-1)
        return zH + self.high_mlp(h)

    def run_segment(
        self,
        x: torch.Tensor,
        zH: torch.Tensor,
        zL: torch.Tensor,
        grad_window: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        total_steps = max(1, self.n_cycles * self.t_low)

        if grad_window is None or grad_window <= 0:
            grad_window = total_steps
        grad_window = min(int(grad_window), total_steps)
        burn_in = total_steps - grad_window

        if burn_in > 0:
            with torch.no_grad():
                for step in range(burn_in):
                    zL = self.low_update(zL, zH, x)
                    if (step + 1) % self.t_low == 0:
                        zH = self.high_update(zH, zL)

        for step in range(grad_window):
            gidx = burn_in + step
            zL = self.low_update(zL, zH, x)
            if (gidx + 1) % self.t_low == 0:
                zH = self.high_update(zH, zL)

        h = self.out_proj(zH) + x
        return zH, zL, h


def collect_hrm_tangents(model: HRMRefinementEncoder, u0: torch.Tensor, num_segments: int, grad_window: Optional[int]) -> List[torch.Tensor]:
    B = u0.size(0)
    device = u0.device
    zH, zL = model.init_state(B, device)
    outs: List[torch.Tensor] = []
    for _ in range(num_segments):
        zH, zL, h = model.run_segment(u0, zH, zL, grad_window=grad_window)
        outs.append(h)
        zH = zH.detach()
        zL = zL.detach()
    return outs


def tangents_to_poincare_points(
    tangents: List[torch.Tensor],
    s_scales: List[float],
    c: float,
    eps: float = 1e-8,
) -> List[torch.Tensor]:
    """
    Hierarchy by construction:
      normalize direction + apply increasing scale s_m
      then expmap to Poincaré ball
    """
    from .poincare_utils import expmap0_poincare as _expmap0

    if len(tangents) != len(s_scales):
        raise ValueError(f"len(tangents)={len(tangents)} must equal len(s_scales)={len(s_scales)}")

    points: List[torch.Tensor] = []
    for h, s in zip(tangents, s_scales):
        n = h.norm(p=2, dim=-1, keepdim=True).clamp_min(eps)
        h_hat = h / n
        u = float(s) * h_hat
        y = _expmap0(u, c=c)
        points.append(y)
    return points


def encode_texts_to_segments(
    model: HRMRefinementEncoder,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    num_segments: int,
    hrm_grad_window: Optional[int],
    s_scales: List[float],
    hyp_c: float,
) -> List[torch.Tensor]:
    """
    Returns list of hyperbolic points per segment: [y^(1),...,y^(M)], each [N,d_hrm]
    """
    from .poincare_utils import expmap0_poincare as _expmap0

    H = model.encode_backbone_tokens(input_ids, attention_mask)
    x = model.pool_tokens(H, attention_mask)
    u0 = model.in_proj(x)
    tangents = collect_hrm_tangents(model, u0, num_segments=num_segments, grad_window=hrm_grad_window)
    return tangents_to_poincare_points(tangents, s_scales=s_scales, c=hyp_c)