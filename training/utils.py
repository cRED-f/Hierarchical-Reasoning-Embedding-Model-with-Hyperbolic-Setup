import random
from typing import Optional, List

import torch


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