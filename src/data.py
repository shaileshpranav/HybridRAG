"""Loaders for the normalized dataset written by scripts/download_scifact.py."""

import csv
import json
from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"


def _jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_corpus(dataset: str = "scifact") -> dict[str, dict]:
    """doc_id -> {"id", "title", "text"}"""
    return {d["id"]: d for d in _jsonl(DATA_ROOT / dataset / "corpus.jsonl")}


def load_queries(dataset: str = "scifact") -> dict[str, str]:
    """query_id -> query text (test split only)"""
    return {q["id"]: q["text"] for q in _jsonl(DATA_ROOT / dataset / "queries.jsonl")}


def load_qrels(dataset: str = "scifact") -> dict[str, dict[str, int]]:
    """query_id -> {doc_id: relevance}, in the shape pytrec_eval expects."""
    qrels: dict[str, dict[str, int]] = {}
    with (DATA_ROOT / dataset / "qrels.tsv").open(encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # header
        for qid, did, score in reader:
            qrels.setdefault(qid, {})[did] = int(score)
    return qrels
