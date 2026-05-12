import argparse
import os
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from tqdm.auto import tqdm

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DEFAULT_INSTRUCTIONS, TASK_TO_DATASET, TASK_ALIASES
from models import HRMRefinementEncoder
from training import set_seed, get_device, warmup_linear_lr, parse_float_list
from training.losses import batch_total_loss
from data import load_mteb_retrieval_dataset, make_loaders
from encoding import embed_texts
from evaluation import evaluate_retrieval


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
            "hrm_state_dict": model.state_dict(),
            "backbone_name": args.backbone_name,
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
    p.add_argument("--proj_scale", type=float, default=1.0)

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