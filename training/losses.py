import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from models import poincare_distance_batch, encode_texts_to_segments, HRMRefinementEncoder


def nce_loss_hyperbolic(
    y_q: torch.Tensor,
    y_pos: torch.Tensor,
    y_negs: torch.Tensor,
    temperature: float,
    hyp_c: float,
) -> torch.Tensor:
    """
    Explicit-negative NCE:
      candidates = [pos, neg1..negK]
      logits_j = -d(q, cand_j)/tau
      label = 0
    """
    B = y_q.size(0)
    cand = torch.cat([y_pos.unsqueeze(1), y_negs], dim=1)
    dist = poincare_distance_batch(y_q, cand, c=hyp_c)
    logits = -dist / float(temperature)
    labels = torch.zeros(B, dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, labels)


def batch_total_loss(
    model: HRMRefinementEncoder,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    num_segments: int,
    hrm_grad_window: Optional[int],
    s_scales: List[float],
    w_segments: List[float],
    hyp_c: float,
    temperature: float,
    coarse_fine_alphas: List[float],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    q_ids = batch["q_input_ids"].to(device)
    q_mask = batch["q_attention_mask"].to(device)

    pc_ids = batch["pc_input_ids"].to(device)
    pc_mask = batch["pc_attention_mask"].to(device)

    pf_ids = batch["pf_input_ids"].to(device)
    pf_mask = batch["pf_attention_mask"].to(device)

    n_ids = batch["n_input_ids"].to(device)
    n_mask = batch["n_attention_mask"].to(device)

    B, K, L = n_ids.shape
    n_ids_flat = n_ids.view(B * K, L)
    n_mask_flat = n_mask.view(B * K, L)

    yq_segs = encode_texts_to_segments(model, q_ids, q_mask, num_segments, hrm_grad_window, s_scales, hyp_c)
    ypc_segs = encode_texts_to_segments(model, pc_ids, pc_mask, num_segments, hrm_grad_window, s_scales, hyp_c)
    ypf_segs = encode_texts_to_segments(model, pf_ids, pf_mask, num_segments, hrm_grad_window, s_scales, hyp_c)
    yn_segs_flat = encode_texts_to_segments(model, n_ids_flat, n_mask_flat, num_segments, hrm_grad_window, s_scales, hyp_c)

    total = torch.zeros([], device=device)
    coarse_sum = 0.0
    fine_sum = 0.0

    for m in range(num_segments):
        yq = yq_segs[m]
        ypc = ypc_segs[m]
        ypf = ypf_segs[m]
        yn = yn_segs_flat[m].view(B, K, -1)

        loss_coarse = nce_loss_hyperbolic(yq, ypc, yn, temperature, hyp_c)
        loss_fine = nce_loss_hyperbolic(yq, ypf, yn, temperature, hyp_c)

        alpha = float(coarse_fine_alphas[m])
        loss_m = (1.0 - alpha) * loss_coarse + alpha * loss_fine
        total = total + float(w_segments[m]) * loss_m

        coarse_sum += float(loss_coarse.item())
        fine_sum += float(loss_fine.item())

    metrics = {
        "loss": float(total.item()),
        "coarse": coarse_sum / float(num_segments),
        "fine": fine_sum / float(num_segments),
    }
    return total, metrics