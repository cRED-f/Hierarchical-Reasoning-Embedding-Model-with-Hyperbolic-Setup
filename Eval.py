import argparse
import os
import random
import math
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Optional

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel

from train_hrm_backbone import (
    HRMRefinementEncoder,
    set_seed,
    get_device,
    expmap0_poincare,
    poincare_distance_matrix,
)


DEFAULT_INSTRUCTIONS: Dict[str, str] = {
    "fever": "Given a claim, retrieve passages that support or refute the claim.",
    "scifact": "Given a scientific claim, retrieve abstracts that support or refute the claim.",
    "nfcorpus": "Given a question, retrieve relevant documents that answer the question.",
    "dbpedia": "Given an entity or topic, retrieve documents that describe it.",
    "hotpotqa": "Given a multi-hop question, retrieve documents containing the evidence needed to answer the question.",
    "nq": "Given a question, retrieve Wikipedia passages that answer the question.",
    "scidocs": "Given a scientific paper, retrieve other papers that are relevant to it.",
    "fiqa": "Given a financial question, retrieve relevant financial passages that answer the question.",
    "cqadupstack": "Given a community question, retrieve existing questions that have the same meaning.",
    "climatefever": "Given a climate change claim, retrieve passages that support or refute the claim.",
    "arguana": "Given a query, retrieve relevant documents that answer the query.",
}





def pick_first(candidates: List[str], cols: List[str]) -> Optional[str]:
    for c in candidates:
        if c in cols:
            return c
    return None


def load_mteb_retrieval_dataset(
    ds_name: str,
    split: str = "test",
    max_corpus: int = 200_000,
    seed: int = 42,
) -> Tuple[
    List[str],            # kept_doc_ids
    List[str],            # kept_doc_texts
    List[str],            # eval_query_ids
    List[str],            # eval_query_texts_raw
    Dict[str, Set[int]],  # qid -> set(doc_idx)
]:
    rng = random.Random(seed)

    corpus = load_dataset(ds_name, "corpus", split="corpus")
    queries = load_dataset(ds_name, "queries", split="queries")
    qrels = load_dataset(ds_name, "default", split=split)

    c_cols = corpus.column_names
    q_cols = queries.column_names
    qr_cols = qrels.column_names

    qrels_qid_col = pick_first(["query-id", "query_id", "qid", "id", "question_id"], qr_cols)
    if qrels_qid_col is None:
        raise ValueError(f"[{ds_name}] No query-id column in qrels; cols={qr_cols}")

    qrels_did_col = pick_first(
        ["corpus-id", "corpus_id", "doc_id", "document_id", "pid", "id"],
        [c for c in qr_cols if c != qrels_qid_col],
    )
    if qrels_did_col is None:
        raise ValueError(f"[{ds_name}] No corpus-id column in qrels; cols={qr_cols}")

    qid_col = pick_first(["id", "query-id", "query_id", "qid", "question_id", "_id"], q_cols)
    if qid_col is None:
        common = [c for c in q_cols if c in qr_cols]
        if common:
            qid_col = common[0]
        else:
            raise ValueError(f"[{ds_name}] No query ID column; queries cols={q_cols}, qrels cols={qr_cols}")

    q_text_col = pick_first(["text", "query", "question", "claim", "title"], [c for c in q_cols if c != qid_col])
    if q_text_col is None:
        for c in q_cols:
            if c != qid_col:
                q_text_col = c
                break
    if q_text_col is None:
        raise ValueError(f"[{ds_name}] No query text column; queries cols={q_cols}")

    id_to_query_text: Dict[str, str] = {}
    for ex in queries:
        qid = ex[qid_col]
        text = (ex.get(q_text_col, "") or "").strip()
        id_to_query_text[qid] = text

    c_id_col = pick_first(["id", "corpus-id", "corpus_id", "doc_id", "document_id", "pid", "_id"], c_cols)
    if c_id_col is None:
        common = [c for c in c_cols if c in qr_cols]
        if common:
            c_id_col = common[0]
        else:
            raise ValueError(f"[{ds_name}] No corpus ID column; corpus cols={c_cols}, qrels cols={qr_cols}")

    c_text_col = pick_first(["text", "passage", "content", "body", "abstract"], [c for c in c_cols if c != c_id_col])
    if c_text_col is None:
        if "title" in c_cols and c_id_col != "title":
            c_text_col = "title"
        else:
            for c in c_cols:
                if c != c_id_col:
                    c_text_col = c
                    break
    if c_text_col is None:
        raise ValueError(f"[{ds_name}] No corpus text column; corpus cols={c_cols}")

    title_col = "title" if "title" in c_cols and c_text_col != "title" else None

    id_to_doc_text: Dict[str, str] = {}
    for ex in corpus:
        did = ex[c_id_col]
        title = (ex.get(title_col, "") or "").strip() if title_col else ""
        body = (ex.get(c_text_col, "") or "").strip()
        id_to_doc_text[did] = (title + " " + body).strip()

    all_doc_ids = list(id_to_doc_text.keys())

    pos_doc_ids: Set[str] = set()
    for ex in qrels:
        did = ex[qrels_did_col]
        if did in id_to_doc_text:
            pos_doc_ids.add(did)

    if max_corpus is None or max_corpus <= 0:
        kept_doc_ids = all_doc_ids
    else:
        pos_list = sorted(pos_doc_ids)
        remaining = max(0, max_corpus - len(pos_list))
        if remaining <= 0:
            kept_doc_ids = pos_list
        else:
            neg_candidates = [d for d in all_doc_ids if d not in pos_doc_ids]
            remaining = min(remaining, len(neg_candidates))
            kept_doc_ids = pos_list + rng.sample(neg_candidates, remaining)

    kept_doc_texts = [id_to_doc_text[did] for did in kept_doc_ids]
    doc_id_to_idx = {did: i for i, did in enumerate(kept_doc_ids)}

    qrels_by_qid: Dict[str, Set[int]] = defaultdict(set)
    for ex in qrels:
        qid = ex[qrels_qid_col]
        did = ex[qrels_did_col]
        if did in doc_id_to_idx:
            qrels_by_qid[qid].add(doc_id_to_idx[did])

    eval_query_ids, eval_query_texts_raw = [], []
    for qid, rels in qrels_by_qid.items():
        if not rels:
            continue
        text = id_to_query_text.get(qid, "").strip()
        if not text:
            continue
        eval_query_ids.append(qid)
        eval_query_texts_raw.append(text)

    return kept_doc_ids, kept_doc_texts, eval_query_ids, eval_query_texts_raw, qrels_by_qid




@torch.no_grad()
def hrm_encode_last_segment_hyperbolic(
    model: HRMRefinementEncoder,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    num_segments: int,
    hyp_c: float,
) -> torch.Tensor:
    # EXACTLY matches updated training script
    H = model.encode_backbone_tokens(input_ids, attention_mask)   # [B,L,d_base]
    x_base = model.pool_tokens(H, attention_mask)                 # [B,d_base]
    u0 = model.project_to_hrm(x_base)                             # [B,d_hrm]

    B = u0.size(0)
    device = u0.device
    zH, zL = model.init_state(B, device)

    h_last = None
    for _ in range(max(1, int(num_segments))):
        zH, zL, h_last = model.run_segment(u0, zH, zL, grad_window=None)

    return expmap0_poincare(h_last, c=hyp_c)  # [B,d_hrm] in ball


@torch.no_grad()
def backbone_encode_cosine(
    backbone: AutoModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    outputs = backbone(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
    last_hidden = outputs.last_hidden_state  # [B,L,H]
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
                    backbone=model,  # here, model == backbone
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

            all_embs.append(emb.float().cpu())

    return torch.cat(all_embs, dim=0) if all_embs else torch.empty((0, 1), dtype=torch.float32)




def evaluate_retrieval(
    query_embs: torch.Tensor,
    doc_embs: torch.Tensor,
    eval_query_ids: List[str],
    qrels_by_qid: Dict[str, Set[int]],
    *,
    use_hyperbolic: bool,
    hyp_c: float,
    k_values: Tuple[int, ...] = (1, 10, 100),
    max_k_for_mrr: int = 10,
    max_k_for_map: Tuple[int, ...] = (10, 100),
    query_chunk: int = 64,
    dist_fp32: bool = True,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    If use_hyperbolic:
        score = -PoincaréDistance (larger is better)
    Else:
        score = cosine similarity
    """
    assert query_embs.size(0) == len(eval_query_ids)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    doc_embs = doc_embs.to(device, non_blocking=True)
    query_embs = query_embs.to(device, non_blocking=True)

    if not use_hyperbolic:
        doc_embs = F.normalize(doc_embs, p=2, dim=-1)
        query_embs = F.normalize(query_embs, p=2, dim=-1)

    num_queries = 0
    hits_at_k = {k: 0 for k in k_values}
    recall_at_k = {k: 0.0 for k in k_values}
    ndcg_at_k = {k: 0.0 for k in k_values}
    mrr = 0.0
    map_at_k = {K: 0.0 for K in max_k_for_map}

    max_k = max(max(k_values), *max_k_for_map, max_k_for_mrr)
    Nd = doc_embs.size(0)

    for start in range(0, query_embs.size(0), query_chunk):
        end = min(start + query_chunk, query_embs.size(0))
        q_chunk = query_embs[start:end]
        qids_chunk = eval_query_ids[start:end]

        if use_hyperbolic:
            a = q_chunk.float() if dist_fp32 else q_chunk
            b = doc_embs.float() if dist_fp32 else doc_embs
            dist = poincare_distance_matrix(a, b, c=hyp_c)  # [B,Nd]
            sims = -dist
        else:
            sims = q_chunk @ doc_embs.t()

        for i in range(sims.size(0)):
            qid = qids_chunk[i]
            rel_indices = qrels_by_qid.get(qid, set())
            if not rel_indices:
                continue

            num_queries += 1
            num_rel = len(rel_indices)

            scores = sims[i]
            _, top_idx = torch.topk(scores, k=min(max_k, Nd))
            top_idx = top_idx.tolist()

            for k in k_values:
                kk = min(k, len(top_idx))
                if kk == 0:
                    continue

                if any(idx in rel_indices for idx in top_idx[:kk]):
                    hits_at_k[k] += 1

                rel_in_top = sum(1 for idx in top_idx[:kk] if idx in rel_indices)
                recall_at_k[k] += rel_in_top / float(num_rel) if num_rel > 0 else 0.0

                dcg = 0.0
                for r, idx in enumerate(top_idx[:kk], start=1):
                    if idx in rel_indices:
                        dcg += 1.0 / math.log2(r + 1.0)

                ideal_hits = min(num_rel, kk)
                if ideal_hits > 0:
                    idcg = sum(1.0 / math.log2(r + 1.0) for r in range(1, ideal_hits + 1))
                    ndcg_at_k[k] += (dcg / idcg) if idcg > 0 else 0.0

            # MRR@max_k_for_mrr
            rank = None
            kk = min(max_k_for_mrr, len(top_idx))
            for r, idx in enumerate(top_idx[:kk], start=1):
                if idx in rel_indices:
                    rank = r
                    break
            if rank is not None:
                mrr += 1.0 / rank

            # MAP@K
            for K in max_k_for_map:
                kk = min(K, len(top_idx))
                if kk == 0:
                    continue
                hits = 0
                ap = 0.0
                for r, idx in enumerate(top_idx[:kk], start=1):
                    if idx in rel_indices:
                        hits += 1
                        ap += hits / float(r)
                ap = ap / min(len(rel_indices), kk) if hits > 0 else 0.0
                map_at_k[K] += ap

    if num_queries == 0:
        raise RuntimeError("No queries with relevant documents after corpus sampling. Try increasing --max_corpus.")

    results: Dict[str, float] = {}
    for k in k_values:
        results[f"hits@{k}"] = hits_at_k[k] / float(num_queries)
        results[f"recall@{k}"] = recall_at_k[k] / float(num_queries)
        results[f"ndcg@{k}"] = ndcg_at_k[k] / float(num_queries)
    results[f"mrr@{max_k_for_mrr}"] = mrr / float(num_queries)
    for K in max_k_for_map:
        results[f"map@{K}"] = map_at_k[K] / float(num_queries)
    results["num_queries"] = float(num_queries)
    return results




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
