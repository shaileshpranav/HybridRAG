"""Download BEIR SciFact from HuggingFace and write normalized files to data/scifact/.

Outputs:
    corpus.jsonl   {"id", "title", "text"}
    queries.jsonl  {"id", "text"}
    qrels.tsv      query_id, doc_id, score  (test split only)

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

    queries = _read_parquet("queries/queries-00000-of-00001.parquet")
    _write_jsonl(QUERIES, ({"id": str(r["_id"]), "text": r["text"]} for r in queries))

    test_tsv = hf_hub_download(QRELS_REPO, "test.tsv", repo_type="dataset")
    with open(test_tsv, encoding="utf-8", newline="") as src:
        reader = csv.reader(src, delimiter="\t")
        next(reader)  # source header: query-id, corpus-id, score

        def write(f):
            w = csv.writer(f, delimiter="\t", lineterminator="\n")
            w.writerow(["query_id", "doc_id", "score"])
            for qid, did, score in reader:
                w.writerow([qid, did, int(score)])

        _atomic_write(QRELS, write)


def print_stats():
    n_docs = n_chars = n_tokens = 0
    with CORPUS.open(encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            full = f"{doc['title']} {doc['text']}".strip()
            n_docs += 1
            n_chars += len(full)
            n_tokens += len(full.split())

    with QUERIES.open(encoding="utf-8") as f:
        n_queries = sum(1 for _ in f)

    with QRELS.open(encoding="utf-8") as f:
        rows = list(csv.reader(f, delimiter="\t"))[1:]
    n_qrels = len(rows)
    n_qrels_queries = len({r[0] for r in rows})

    print(f"Docs:     {n_docs}")
    print(f"Queries:  {n_queries} (in queries.jsonl)")
    print(f"Qrels:    {n_qrels} rows over {n_qrels_queries} test queries")
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
    print_stats()


if __name__ == "__main__":
    main()
