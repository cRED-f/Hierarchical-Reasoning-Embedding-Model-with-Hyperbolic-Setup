import argparse
import os
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModel

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DEFAULT_INSTRUCTIONS, TASK_TO_DATASET, TASK_ALIASES
from models import HRMRefinementEncoder
from training import set_seed, get_device
from data import load_mteb_retrieval_dataset
from encoding import embed_texts
from evaluation import evaluate_retrieval


def run_task_evaluation(
    task: str,
    model,
    tokenizer,
    device: torch.device,
    args: argparse.Namespace,
    use_hrm: bool,
    max_length: int,
    hyp_c: float,
) -> None:
    if task not in TASK_TO_DATASET:
        raise ValueError(f"Unknown canonical task '{task}'.")

    ds_name = TASK_TO_DATASET[task]
    print(f"[Data] Loading {ds_name} (split={args.split}, max_corpus={args.max_corpus})...")

    kept_doc_ids, kept_doc_texts, eval_query_ids, eval_query_texts_raw, qrels_by_qid = load_mteb_retrieval_dataset(
        ds_name=ds_name,
        split=args.split,
        max_corpus=args.max_corpus,
        seed=args.seed,
    )

    print(f"[Data] Kept corpus docs: {len(kept_doc_ids)}")
    print(f"[Data] Eval queries (>=1 kept positive): {len(eval_query_ids)}")

    instruction = args.instruction.strip() if args.instruction.strip() else DEFAULT_INSTRUCTIONS.get(task, "")
    print(f'[Data] Instruction for "{task}": "{instruction}"')

    if instruction:
        eval_query_texts = [f"{instruction} {q}".strip() for q in eval_query_texts_raw]
    else:
        eval_query_texts = eval_query_texts_raw

    print("[Encode] Encoding corpus...")
    doc_embs = embed_texts(
        model=model,
        tokenizer=tokenizer,
        texts=kept_doc_texts,
        batch_size=args.batch_size,
        max_length=max_length,
        device=device,
        num_segments=args.num_segments_eval,
        use_hrm=use_hrm,
        hyp_c=hyp_c,
    )
    print(f"[Encode] Corpus embs: {tuple(doc_embs.shape)}")

    print("[Encode] Encoding queries...")
    query_embs = embed_texts(
        model=model,
        tokenizer=tokenizer,
        texts=eval_query_texts,
        batch_size=args.batch_size,
        max_length=max_length,
        device=device,
        num_segments=args.num_segments_eval,
        use_hrm=use_hrm,
        hyp_c=hyp_c,
    )
    print(f"[Encode] Query embs: {tuple(query_embs.shape)}")

    print("[Eval] Computing metrics...")
    metrics = evaluate_retrieval(
        query_embs=query_embs,
        doc_embs=doc_embs,
        eval_query_ids=eval_query_ids,
        qrels_by_qid=qrels_by_qid,
        use_hyperbolic=use_hrm,
        hyp_c=hyp_c,
        k_values=(1, 10, 100),
        max_k_for_mrr=10,
        max_k_for_map=(10, 100),
        query_chunk=args.query_chunk,
        dist_fp32=not args.no_dist_fp32,
        device=device if device.type in ("cuda", "cpu", "mps") else None,
    )

    num_q = int(metrics.pop("num_queries", 0))
    mode_label = "HRM + Backbone (hyperbolic distance)" if use_hrm else "Backbone Only (cosine)"

    print(f"\n=== {task.upper()} Results ({mode_label}) ===")
    print(f"Dataset: {ds_name}")
    print(f"Queries evaluated: {num_q}")
    for k in [1, 10, 100]:
        print(
            f"hits@{k: <3}: {metrics[f'hits@{k}']*100:6.2f}% | "
            f"recall@{k: <3}: {metrics[f'recall@{k}']*100:6.2f}% | "
            f"ndcg@{k: <3}: {metrics[f'ndcg@{k}']*100:6.2f}%"
        )
    print(f"mrr@10  : {metrics['mrr@10']*100:6.2f}%")
    for K in [10, 100]:
        print(f"map@{K:<3}: {metrics[f'map@{K}']*100:6.2f}%")


def parse_args() -> argparse.Namespace:
    all_task_keys = sorted(TASK_ALIASES.keys())
    p = argparse.ArgumentParser(description="Evaluate HRM+Backbone (hyperbolic) or Backbone-only (cosine) on MTEB retrieval tasks.")

    p.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint (.pt).")
    p.add_argument("--task", type=str.lower, default="fever", choices=all_task_keys)
    p.add_argument("--split", type=str, default="test", choices=["train", "dev", "test"])
    p.add_argument("--max_corpus", type=int, default=200_000)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_segments_eval", type=int, default=1, help="HRM segments at inference (HRM mode only).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--instruction", type=str, default="")
    p.add_argument("--no_hrm", action="store_true")
    p.add_argument("--backbone_name", type=str, default="", help="Override backbone HF name (mainly for --no_hrm).")
    p.add_argument("--max_length", type=int, default=0, help="Override token max_length (0 uses ckpt max_length or 512).")
    p.add_argument("--hyp_c", type=float, default=0.0, help="Override hyperbolic curvature c (0 means read from checkpoint).")
    p.add_argument("--query_chunk", type=int, default=64, help="Queries per chunk for scoring against corpus.")
    p.add_argument("--no_dist_fp32", action="store_true", help="Do NOT cast embeddings to fp32 for distance computations.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")

    canonical = TASK_ALIASES.get(args.task, args.task)
    tasks_to_run = list(TASK_TO_DATASET.keys()) if canonical == "all" else [canonical]

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ckpt_args = ckpt.get("args", {}) or {}

    ckpt_backbone_name = ckpt.get("backbone_name", ckpt_args.get("backbone_name", None))
    backbone_name = args.backbone_name if args.backbone_name else ckpt_backbone_name
    if backbone_name is None:
        raise ValueError("No backbone_name found. Provide --backbone_name or use a checkpoint that stores backbone_name.")

    tokenizer_name = backbone_name

    max_length = int(args.max_length) if (args.max_length and args.max_length > 0) else int(ckpt_args.get("max_length", 512))
    hyp_c = float(args.hyp_c) if (args.hyp_c and args.hyp_c > 0) else float(ckpt_args.get("hyp_c", 1.0))

    print(f"[Model] Backbone:  {backbone_name}")
    print(f"[Model] Tokenizer: {tokenizer_name}")
    print(f"[Config] max_length={max_length} | hyp_c={hyp_c}")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.cls_token

    backbone = AutoModel.from_pretrained(backbone_name).to(device).eval()
    for p in backbone.parameters():
        p.requires_grad = False

    d_base = backbone.config.hidden_size

    if args.no_hrm:
        print("[Mode] BACKBONE ONLY (cosine)")
        model = backbone
        use_hrm = False
    else:
        print("[Mode] HRM + BACKBONE (hyperbolic distance)")

        d_hrm = int(ckpt_args.get("d_hrm", 512))
        n_cycles = int(ckpt_args.get("n_cycles", 2))
        t_low = int(ckpt_args.get("t_low", 2))
        hrm_hidden_mult = int(ckpt_args.get("hrm_hidden_mult", 4))

        pool_heads = int(ckpt_args.get("pool_heads", 1))
        pool_hidden_mult = int(ckpt_args.get("pool_hidden_mult", 2))
        pool_dropout = float(ckpt_args.get("pool_dropout", 0.0))
        proj_scale = float(ckpt_args.get("proj_scale", 1.0))

        model = HRMRefinementEncoder(
            backbone=backbone,
            d_base=d_base,
            d_hrm=d_hrm,
            n_cycles=n_cycles,
            t_low=t_low,
            hidden_mult=hrm_hidden_mult,
            pool_heads=pool_heads,
            pool_hidden_mult=pool_hidden_mult,
            pool_dropout=pool_dropout,
            proj_scale=proj_scale,
        )

        if "hrm_state_dict" not in ckpt:
            raise ValueError("Checkpoint missing 'hrm_state_dict'. Use a HRM checkpoint or pass --no_hrm.")

        model.load_state_dict(ckpt["hrm_state_dict"], strict=True)
        model.to(device).eval()
        use_hrm = True

    for t in tasks_to_run:
        print("\n" + "=" * 80)
        print(f"Running task: {t}")
        print("=" * 80)
        run_task_evaluation(
            task=t,
            model=model,
            tokenizer=tokenizer,
            device=device,
            args=args,
            use_hrm=use_hrm,
            max_length=max_length,
            hyp_c=hyp_c,
        )


if __name__ == "__main__":
    main()