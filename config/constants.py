from typing import Dict

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

TASK_TO_DATASET: Dict[str, str] = {
    "fever": "mteb/FEVER",
    "scifact": "mteb/SciFact",
    "nfcorpus": "mteb/NFCorpus",
    "dbpedia": "mteb/DBPedia",
    "hotpotqa": "mteb/HotpotQA",
    "nq": "mteb/NQ",
    "scidocs": "mteb/SCIDOCS",
    "fiqa": "mteb/FiQA2018",
    "cqadupstack": "mteb/CQADupstack",
    "climatefever": "mteb/ClimateFEVER",
    "arguana": "mteb/Arguana",
}

TASK_ALIASES: Dict[str, str] = {
    "all": "all",
    "fever": "fever",
    "scifact": "scifact",
    "nfcorpus": "nfcorpus",
    "dbpedia": "dbpedia",
    "hotpotqa": "hotpotqa",
    "nq": "nq",
    "scidocs": "scidocs",
    "fiqa": "fiqa",
    "cqadupstack": "cqadupstack",
    "climatefever": "climatefever",
    "arguana": "arguana",
}