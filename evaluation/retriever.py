import math
from typing import Dict, List, Set, Optional, Tuple

import torch

from models import poincare_distance_matrix


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
        doc_embs = torch.nn.functional.normalize(doc_embs, p=2, dim=-1)
        query_embs = torch.nn.functional.normalize(query_embs, p=2, dim=-1)

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
            dist = poincare_distance_matrix(a, b, c=hyp_c)
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

            rank = None
            kk = min(max_k_for_mrr, len(top_idx))
            for r, idx in enumerate(top_idx[:kk], start=1):
                if idx in rel_indices:
                    rank = r
                    break
            if rank is not None:
                mrr += 1.0 / rank

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