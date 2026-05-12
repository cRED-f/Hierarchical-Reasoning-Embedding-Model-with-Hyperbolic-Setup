import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from datasets import load_dataset


EXCLUDE_DATASETS = ["test", "eval", "benchmark", "sample"]


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