"""Download BEIR SciFact from HuggingFace and write normalized files to data/scifact/.

Outputs:
    corpus.jsonl   {"id", "title", "text"}
    queries.jsonl  {"id", "text"}
    qrels.tsv      query_id, doc_id, score  (test split only)

queries.jsonl is filtered to the queries that appear in the test qrels.

Idempotent: if all three files already exist, the download is skipped.
"""

import csv
import json
import os
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "scifact"
CORPUS = OUT_DIR / "corpus.jsonl"
QUERIES = OUT_DIR / "queries.jsonl"
QRELS = OUT_DIR / "qrels.tsv"

DATA_REPO = "BeIR/scifact"
QRELS_REPO = "BeIR/scifact-qrels"

EXPECTED_QUERIES = 300  # SciFact test split


def _atomic_write(path: Path, write):
    """Write via a temp file so an interrupted run never leaves a partial file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        write(f)
    os.replace(tmp, path)


def _read_parquet(filename: str) -> list[dict]:
    path = hf_hub_download(DATA_REPO, filename, repo_type="dataset")
    return pq.read_table(path).to_pylist()


def _write_jsonl(path: Path, rows):
    def write(f):
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    _atomic_write(path, write)


def download():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    corpus = _read_parquet("corpus/corpus-00000-of-00001.parquet")
    _write_jsonl(
        CORPUS,
        (
            {"id": str(r["_id"]), "title": r["title"] or "", "text": r["text"] or ""}
            for r in corpus
        ),
    )

    test_tsv = hf_hub_download(QRELS_REPO, "test.tsv", repo_type="dataset")
    with open(test_tsv, encoding="utf-8", newline="") as src:
        reader = csv.reader(src, delimiter="\t")
        next(reader)  # source header: query-id, corpus-id, score
        qrels = [(qid, did, int(score)) for qid, did, score in reader]

    def write_qrels(f):
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["query_id", "doc_id", "score"])
        w.writerows(qrels)

    _atomic_write(QRELS, write_qrels)

    # Keep only queries that have test qrels (the HF queries file also holds train).
    qrel_query_ids = {qid for qid, _, _ in qrels}
    queries = _read_parquet("queries/queries-00000-of-00001.parquet")
    _write_jsonl(
        QUERIES,
        (
            {"id": str(r["_id"]), "text": r["text"]}
            for r in queries
            if str(r["_id"]) in qrel_query_ids
        ),
    )


def validate_and_stats():
    """Check the files are mutually consistent, then print summary stats.

    Raises AssertionError on a wrong dataset/split or a stale queries file.
    """
    doc_ids = set()
    n_chars = n_tokens = 0
    with CORPUS.open(encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            doc_ids.add(doc["id"])
            full = f"{doc['title']} {doc['text']}".strip()
            n_chars += len(full)
            n_tokens += len(full.split())
    n_docs = len(doc_ids)

    with QUERIES.open(encoding="utf-8") as f:
        query_ids = [json.loads(line)["id"] for line in f]

    with QRELS.open(encoding="utf-8") as f:
        rows = list(csv.reader(f, delimiter="\t"))[1:]
    qrel_query_ids = {r[0] for r in rows}
    qrel_doc_ids = {r[1] for r in rows}

    assert qrel_doc_ids <= doc_ids, (
        f"{len(qrel_doc_ids - doc_ids)} qrel doc ids missing from corpus"
    )
    assert qrel_query_ids <= set(query_ids), (
        f"{len(qrel_query_ids - set(query_ids))} qrel query ids missing from queries"
    )
    assert len(query_ids) == len(qrel_query_ids) == EXPECTED_QUERIES, (
        f"expected {EXPECTED_QUERIES} queries, got {len(query_ids)} queries "
        f"and {len(qrel_query_ids)} qrel queries"
    )

    print(f"Docs:     {n_docs}")
    print(f"Queries:  {len(query_ids)}")
    print(f"Qrels:    {len(rows)} rows over {len(qrel_query_ids)} queries")
    print(
        f"Avg doc length (title + text): {n_chars / n_docs:.0f} chars, "
        f"{n_tokens / n_docs:.0f} whitespace tokens"
    )


def main():
    if all(p.exists() and p.stat().st_size > 0 for p in (CORPUS, QUERIES, QRELS)):
        print(f"Files already exist in {OUT_DIR}, skipping download.")
    else:
        print(f"Downloading {DATA_REPO} + {QRELS_REPO} -> {OUT_DIR}")
        download()
    validate_and_stats()


if __name__ == "__main__":
    main()
