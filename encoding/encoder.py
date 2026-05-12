import torch
import torch.nn.functional as F
from typing import List, Optional

from models import HRMRefinementEncoder, poincare_distance_matrix


@torch.no_grad()
def hrm_encode_last_segment_hyperbolic(
    model: HRMRefinementEncoder,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    num_segments: int,
    hyp_c: float,
) -> torch.Tensor:
    from models.poincare_utils import expmap0_poincare

    H = model.encode_backbone_tokens(input_ids, attention_mask)
    x_base = model.pool_tokens(H, attention_mask)
    u0 = model.project_to_hrm(x_base)

    B = u0.size(0)
    device = u0.device
    zH, zL = model.init_state(B, device)

    h_last = None
    for _ in range(max(1, int(num_segments))):
        zH, zL, h_last = model.run_segment(u0, zH, zL, grad_window=None)

    return expmap0_poincare(h_last, c=hyp_c)


@torch.no_grad()
def backbone_encode_cosine(
    backbone: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    outputs = backbone(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
    last_hidden = outputs.last_hidden_state
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden * mask).sum(dim=1)
    lengths = mask.sum(dim=1).clamp(min=1e-9)
    emb = summed / lengths
    return F.normalize(emb, p=2, dim=-1)


def embed_texts(
    model,
    tokenizer,
    texts: List[str],
    batch_size: int,
    max_length: int,
    device: torch.device,
    num_segments: int,
    use_hrm: bool,
    hyp_c: float,
) -> torch.Tensor:
    """
    HRM mode: returns hyperbolic points (Poincaré ball), no L2 norm.
    Backbone mode: returns L2-normalized Euclidean embeddings.
    Returns on CPU float32 (keeps RAM stable + lets scoring pick device later).
    """
    all_embs: List[torch.Tensor] = []
    model.eval()

    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start:start + batch_size]
            enc = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            if use_hrm:
                emb = hrm_encode_last_segment_hyperbolic(
                    model=model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    num_segments=num_segments,
                    hyp_c=hyp_c,
                )
            else:
                emb = backbone_encode_cosine(
                    backbone=model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

            all_embs.append(emb.float().cpu())

    return torch.cat(all_embs, dim=0) if all_embs else torch.empty((0, 1), dtype=torch.float32)