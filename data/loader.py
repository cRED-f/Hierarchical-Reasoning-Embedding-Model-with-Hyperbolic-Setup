import random
from typing import Dict, List, Tuple, Set, Optional

from datasets import load_dataset


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
    List[str],
    List[str],
    List[str],
    List[str],
    Dict[str, Set[int]],
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

    from collections import defaultdict
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