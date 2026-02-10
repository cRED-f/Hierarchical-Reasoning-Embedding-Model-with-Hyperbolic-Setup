import argparse
import os
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer
from tqdm.auto import tqdm




# ----------------------------
# Utils
# ----------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def warmup_linear_lr(step: int, total_steps: int, base_lr: float, warmup_ratio: float) -> float:
    if total_steps <= 0:
        return base_lr
    warmup_steps = max(int(total_steps * warmup_ratio), 1)
    if step <= warmup_steps:
        return base_lr * float(step) / float(warmup_steps)
    remain = total_steps - warmup_steps
    after = step - warmup_steps
    factor = max(0.0, float(remain - after) / float(max(1, remain)))
    return base_lr * factor


def parse_float_list(s: Optional[str], expected_len: int, name: str) -> Optional[List[float]]:
    if s is None or s.strip() == "":
        return None
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    vals = [float(p) for p in parts]
    if len(vals) != expected_len:
        raise ValueError(f"{name} must have length {expected_len}, got {len(vals)} from: {s}")
    return vals


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

    # fp32 for stability
    q = q.float()
    cand = cand.float()

    sqrt_c = q.new_tensor(c).sqrt()

    q2 = (q * q).sum(dim=-1, keepdim=True)               # [B,1]
    b2 = (cand * cand).sum(dim=-1)                       # [B,C]
    diff2 = ((q.unsqueeze(1) - cand) ** 2).sum(dim=-1)   # [B,C]

    denom = (1.0 - c * q2).clamp_min(eps) * (1.0 - c * b2).clamp_min(eps)
    arg = 1.0 + 2.0 * c * diff2 / denom
    arg = arg.clamp_min(1.0 + eps)

    return torch.acosh(arg) / sqrt_c




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
        scores = self.score_net(last_hidden_state)  # [B,L,heads]

        mask = attention_mask.to(dtype=torch.bool)  # [B,L]
        mask_value = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(~mask.unsqueeze(-1), mask_value)

        alpha = torch.softmax(scores, dim=1)  # [B,L,heads]
        pooled = (alpha.unsqueeze(-1) * last_hidden_state.unsqueeze(2)).sum(dim=1)  # [B,heads,d]

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
        backbone: AutoModel,
        d_base: int,
        d_hrm: int = 512,
        n_cycles: int = 2,
        t_low: int = 2,
        hidden_mult: int = 4,
        pool_heads: int = 1,
        pool_hidden_mult: int = 2,
        pool_dropout: float = 0.0,
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
    if len(tangents) != len(s_scales):
        raise ValueError(f"len(tangents)={len(tangents)} must equal len(s_scales)={len(s_scales)}")

    points: List[torch.Tensor] = []
    for h, s in zip(tangents, s_scales):
        n = h.norm(p=2, dim=-1, keepdim=True).clamp_min(eps)
        h_hat = h / n
        u = float(s) * h_hat
        y = expmap0_poincare(u, c=c)
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
    H = model.encode_backbone_tokens(input_ids, attention_mask)     # frozen
    x = model.pool_tokens(H, attention_mask)                        # trainable
    u0 = model.in_proj(x)                                           # trainable
    tangents = collect_hrm_tangents(model, u0, num_segments=num_segments, grad_window=hrm_grad_window)
    return tangents_to_poincare_points(tangents, s_scales=s_scales, c=hyp_c)




_QUERY_RE = re.compile(r"(?i)\bquery\s*:\s*")

def extract_query_text(text: str) -> str:
    m = _QUERY_RE.search(text)
    if not m:
        return text.strip()
    return text[m.end():].strip()


def coarsen_text(text: str, max_chars: int = 60) -> str:
    q = extract_query_text(text)
    cut = None
    for ch in ["。", ".", "?", "!", "？", "！"]:
        idx = q.find(ch)
        if idx != -1:
            cut = idx + 1
            break
    if cut is not None and cut > 0:
        q2 = q[:cut].strip()
    else:
        q2 = q.strip()

    q2 = q2[:max_chars].strip()
    if not q2:
        q2 = q[:max_chars].strip() if q else "dummy query"

    return f"Instruct: Retrieve semantically similar text.\nQuery: {q2}"




def _as_str_list(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, str):
        s = x.strip()
        return [s] if s else []
    if isinstance(x, (list, tuple)):
        out = []
        for v in x:
            if isinstance(v, str):
                s = v.strip()
                if s:
                    out.append(s)
        return out
    return []


@dataclass
class KaLMHierNCECollator:
    tokenizer: Any
    max_length: int
    num_negs: int = 4
    coarse_max_chars: int = 60
    rng: random.Random = None

    def __post_init__(self):
        if self.rng is None:
            self.rng = random.Random(42)

    def _pick_coarse_fine(self, pos_list: List[str]) -> Tuple[str, str]:
        if len(pos_list) >= 2:
            a, b = self.rng.sample(pos_list, 2)
            if len(a) <= len(b):
                return a, b
            return b, a
        fine = pos_list[0] if pos_list else "Instruct: Retrieve semantically similar text.\nQuery: dummy pos"
        coarse = coarsen_text(fine, max_chars=self.coarse_max_chars)
        return coarse, fine

    def _sample_negs(self, neg_list: List[str]) -> List[str]:
        if len(neg_list) == 0:
            neg_list = ["Instruct: Retrieve semantically similar text.\nQuery: dummy neg"]
        if len(neg_list) >= self.num_negs:
            return self.rng.sample(neg_list, self.num_negs)
        out = list(neg_list)
        while len(out) < self.num_negs:
            out.append(self.rng.choice(neg_list))
        return out[: self.num_negs]

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        qs: List[str] = []
        pcs: List[str] = []
        pfs: List[str] = []
        negs_flat: List[str] = []

        for ex in examples:
            q = (ex.get("query") or "").strip()
            pos_list = _as_str_list(ex.get("pos"))
            neg_list = _as_str_list(ex.get("neg"))

            if not q or not pos_list:
                continue

            coarse, fine = self._pick_coarse_fine(pos_list)
            negs = self._sample_negs(neg_list)

            qs.append(q)
            pcs.append(coarse)
            pfs.append(fine)
            negs_flat.extend(negs)

        if len(qs) == 0:
            qs = ["Instruct: Retrieve semantically similar text.\nQuery: dummy query"]
            pcs = ["Instruct: Retrieve semantically similar text.\nQuery: dummy coarse"]
            pfs = ["Instruct: Retrieve semantically similar text.\nQuery: dummy fine"]
            negs_flat = ["Instruct: Retrieve semantically similar text.\nQuery: dummy neg"] * self.num_negs

        q_batch = self.tokenizer(qs, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
        pc_batch = self.tokenizer(pcs, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
        pf_batch = self.tokenizer(pfs, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
        n_batch = self.tokenizer(negs_flat, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")

        B = q_batch["input_ids"].size(0)
        K = self.num_negs
        n_input_ids = n_batch["input_ids"].view(B, K, -1)
        n_attention_mask = n_batch["attention_mask"].view(B, K, -1)

        return {
            "q_input_ids": q_batch["input_ids"],
            "q_attention_mask": q_batch["attention_mask"],
            "pc_input_ids": pc_batch["input_ids"],
            "pc_attention_mask": pc_batch["attention_mask"],
            "pf_input_ids": pf_batch["input_ids"],
            "pf_attention_mask": pf_batch["attention_mask"],
            "n_input_ids": n_input_ids,
            "n_attention_mask": n_attention_mask,
        }




def nce_loss_hyperbolic(
    y_q: torch.Tensor,        # [B,d]
    y_pos: torch.Tensor,      # [B,d]
    y_negs: torch.Tensor,     # [B,K,d]
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
    cand = torch.cat([y_pos.unsqueeze(1), y_negs], dim=1)  # [B,1+K,d]
    dist = poincare_distance_batch(y_q, cand, c=hyp_c)     # [B,1+K]
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

    n_ids = batch["n_input_ids"].to(device)           # [B,K,L]
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
        yq = yq_segs[m]                     # [B,d]
        ypc = ypc_segs[m]                   # [B,d]
        ypf = ypf_segs[m]                   # [B,d]
        yn = yn_segs_flat[m].view(B, K, -1) # [B,K,d]

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



def load_kalm_finetune_dataset(args):
    """
    Loads KaLM finetune dataset, optionally filtering parquet shards by filename.
    """
    if args.no_exclude_filter:
        ds = load_dataset(args.repo_id, split="train")
        return ds

    parquet_files = None
    try:
        from huggingface_hub import list_repo_files
        all_files = list_repo_files(args.repo_id, repo_type="dataset")
        kept: List[str] = []
        for f in all_files:
            if not f.endswith(".parquet"):
                continue
            low = f.lower()
            if any(ex in low for ex in EXCLUDE_DATASETS):
                continue
            kept.append(f)
        if len(kept) == 0:
            raise ValueError("No parquet files remained after applying EXCLUDE_DATASETS.")
        parquet_files = kept
        print(f"[Data] Kept {len(parquet_files)} parquet files after exclusions.")
    except Exception as e:
        print(f"[Data] Warning: could not list/filter repo files ({e}). Falling back to unfiltered load.")
        parquet_files = None

    if parquet_files is not None:
        ds = load_dataset(args.repo_id, split="train", data_files=parquet_files)
    else:
        ds = load_dataset(args.repo_id, split="train")
    return ds


def make_loaders(args, tokenizer) -> Tuple[DataLoader, Optional[DataLoader]]:
    ds = load_kalm_finetune_dataset(args)

    if args.max_train_examples > 0 and args.max_train_examples < len(ds):
        ds = ds.select(range(args.max_train_examples))

    ds = ds.shuffle(seed=args.seed)

    val_loader = None
    if args.val_size > 0 and args.val_size < len(ds):
        splits = ds.train_test_split(test_size=args.val_size, seed=args.seed)
        train_ds, val_ds = splits["train"], splits["test"]
    else:
        train_ds, val_ds = ds, None

    collator = KaLMHierNCECollator(
        tokenizer=tokenizer,
        max_length=args.max_length,
        num_negs=args.num_negs,
        coarse_max_chars=args.coarse_max_chars,
        rng=random.Random(args.seed),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=args.num_workers,
            pin_memory=True,
        )

    return train_loader, val_loader




def save_checkpoint(model, optimizer, epoch, global_step, best_val, args, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best_val": best_val,
            "args": vars(args),
        },
        path,
    )
    print(f"[Checkpoint] Saved: {path}")


def load_checkpoint(path: str, model, optimizer, device: torch.device) -> Tuple[int, int, float]:
    print(f"[Checkpoint] Loading: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optim_state"])
    epoch = int(ckpt.get("epoch", 0)) + 1
    global_step = int(ckpt.get("global_step", 0))
    best_val = float(ckpt.get("best_val", float("inf")))
    print(f"[Checkpoint] Resume epoch={epoch} global_step={global_step} best_val={best_val:.4f}")
    return epoch, global_step, best_val




def train(args) -> None:
    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")

    print(f"[Model] Loading backbone: {args.backbone_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.backbone_name)
    backbone = AutoModel.from_pretrained(args.backbone_name)

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.cls_token is not None:
            tokenizer.pad_token = tokenizer.cls_token

    backbone.to(device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    d_base = backbone.config.hidden_size
    model = HRMRefinementEncoder(
        backbone=backbone,
        d_base=d_base,
        d_hrm=args.d_hrm,
        n_cycles=args.n_cycles,
        t_low=args.t_low,
        hidden_mult=args.hrm_hidden_mult,
        pool_heads=args.pool_heads,
        pool_hidden_mult=args.pool_hidden_mult,
        pool_dropout=args.pool_dropout,
    ).to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"[Model] Trainable params (pooler+HRM): {sum(p.numel() for p in trainable):,}")

    M = args.num_segments
    s_scales = parse_float_list(args.s_scales, M, "s_scales") or [float(i + 1) for i in range(M)]

    w_segments = parse_float_list(args.w_segments, M, "w_segments")
    if w_segments is None:
        raw = [float(i + 1) for i in range(M)]
        ssum = sum(raw)
        w_segments = [r / ssum for r in raw]
    else:
        ssum = sum(w_segments)
        if abs(ssum - 1.0) > 1e-6:
            w_segments = [w / ssum for w in w_segments]

    alpha_list = parse_float_list(args.alpha_segments, M, "alpha_segments")
    if alpha_list is None:
        if M == 1:
            alpha_list = [1.0]
        else:
            alpha_list = [float(m) / float(M - 1) for m in range(M)]

    print(f"[Config] segments={M} s_scales={s_scales} w_segments={w_segments} alpha={alpha_list}")
    print(f"[Config] hyp_c={args.hyp_c} temp={args.temperature} num_negs={args.num_negs}")
    print(f"[Config] repo_id={args.repo_id} exclude_filter={'off' if args.no_exclude_filter else 'on'}")
    print(f"[Config] batch={args.batch_size} max_len={args.max_length} amp={args.amp} hrm_grad_window={args.hrm_grad_window}")

    train_loader, val_loader = make_loaders(args, tokenizer)

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    start_epoch = 1
    global_step = 0
    best_val = float("inf")
    if args.resume_from:
        start_epoch, global_step, best_val = load_checkpoint(args.resume_from, model, optimizer, device)

    total_steps = args.epochs * len(train_loader)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp and device.type == "cuda"))
    hrm_grad_window = args.hrm_grad_window if args.hrm_grad_window > 0 else None

    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running = 0.0

        pbar = tqdm(train_loader, desc=f"[Train] epoch {epoch}/{args.epochs}", total=len(train_loader))
        for step, batch in enumerate(pbar, start=1):
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=bool(args.amp and device.type == "cuda")):
                loss, mets = batch_total_loss(
                    model=model,
                    batch=batch,
                    device=device,
                    num_segments=M,
                    hrm_grad_window=hrm_grad_window,
                    s_scales=s_scales,
                    w_segments=w_segments,
                    hyp_c=args.hyp_c,
                    temperature=args.temperature,
                    coarse_fine_alphas=alpha_list,
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            lr = warmup_linear_lr(global_step, total_steps, args.lr, args.warmup_ratio)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            running += float(loss.item())
            if step % args.log_every == 0:
                pbar.set_postfix(
                    {
                        "loss": f"{(running / args.log_every):.4f}",
                        "lr": f"{lr:.2e}",
                        "coarse": f"{mets['coarse']:.3f}",
                        "fine": f"{mets['fine']:.3f}",
                    }
                )
                running = 0.0

        val_loss = None
        if val_loader is not None:
            model.eval()
            vsum = 0.0
            vsteps = 0
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"[Val] epoch {epoch}", total=len(val_loader), leave=False):
                    loss, _ = batch_total_loss(
                        model=model,
                        batch=batch,
                        device=device,
                        num_segments=M,
                        hrm_grad_window=None,
                        s_scales=s_scales,
                        w_segments=w_segments,
                        hyp_c=args.hyp_c,
                        temperature=args.temperature,
                        coarse_fine_alphas=alpha_list,
                    )
                    vsum += float(loss.item())
                    vsteps += 1
            val_loss = vsum / max(1, vsteps)
            print(f"[Val] epoch {epoch} loss={val_loss:.4f}")
            model.train()

        last_path = os.path.join(args.output_dir, "checkpoint_last.pt")
        save_checkpoint(model, optimizer, epoch, global_step, best_val, args, last_path)

        if val_loss is not None and val_loss < best_val:
            best_val = val_loss
            best_path = os.path.join(args.output_dir, "checkpoint_best.pt")
            save_checkpoint(model, optimizer, epoch, global_step, best_val, args, best_path)

    final_path = os.path.join(args.output_dir, "checkpoint_final.pt")
    save_checkpoint(model, optimizer, args.epochs, global_step, best_val, args, final_path)
    print(f"[Done] Saved final checkpoint: {final_path}")



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Frozen backbone + HRM + hyperbolic hierarchical NCE.")

    p.add_argument("--backbone_name", type=str, default="BAAI/bge-small-en-v1.5")
    p.add_argument("--d_hrm", type=int, default=512)
    p.add_argument("--n_cycles", type=int, default=2)
    p.add_argument("--t_low", type=int, default=2)
    p.add_argument("--hrm_hidden_mult", type=int, default=4)

    p.add_argument("--pool_heads", type=int, default=1)
    p.add_argument("--pool_hidden_mult", type=int, default=2)
    p.add_argument("--pool_dropout", type=float, default=0.0)

    p.add_argument("--hyp_c", type=float, default=1.0)
    p.add_argument("--temperature", type=float, default=0.05)

    p.add_argument("--num_segments", type=int, default=4)
    p.add_argument("--s_scales", type=str, default="", help="Comma list length M, e.g. '1,2,3,4'. Default: 1..M")
    p.add_argument("--w_segments", type=str, default="", help="Comma list length M. Default: increasing normalized.")
    p.add_argument("--alpha_segments", type=str, default="", help="Comma list length M. Default: linear 0..1")

    p.add_argument("--repo_id", type=str, default="KaLM-Embedding/KaLM-embedding-finetuning-data")
    p.add_argument("--no_exclude_filter", action="store_true", help="Disable filename-based exclusion filtering.")
    p.add_argument("--max_train_examples", type=int, default=0)
    p.add_argument("--val_size", type=int, default=5000)
    p.add_argument("--num_negs", type=int, default=4)
    p.add_argument("--coarse_max_chars", type=int, default=60)

    p.add_argument("--max_length", type=int, default=252)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--hrm_grad_window", type=int, default=0)

    p.add_argument("--amp", action="store_true")
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", type=str, default="hrm_hyp_hier_runs")

    p.add_argument("--resume_from", type=str, default=None)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
