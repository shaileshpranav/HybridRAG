"""Embed the corpus and load it into Qdrant.

    uv run python -m src.ingest.ingest --dataset scifact [--force] [--recreate]

Idempotent: point ids are deterministic, so a rerun overwrites. If the collection
already holds every chunk, the (slow) embedding step is skipped unless --force.
The collection also declares a sparse vector slot, populated on the BM25 day.
"""

import argparse
import time
from itertools import islice

from fastembed import TextEmbedding
from qdrant_client import QdrantClient, models
from tqdm import tqdm

from src.config import (
    DENSE_VECTOR,
    EMBED_DIM,
    EMBED_MODEL,
    QDRANT_GRPC_PORT,
    QDRANT_URL,
    SPARSE_VECTOR,
    collection_name,
)
from src.data import load_corpus
from src.ingest.chunking import Chunk, chunk_corpus

EMBED_BATCH = 64
UPSERT_BATCH = 256


def get_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL, grpc_port=QDRANT_GRPC_PORT, prefer_grpc=True)


def ensure_collection(client: QdrantClient, name: str, recreate: bool) -> None:
    if recreate and client.collection_exists(name):
        client.delete_collection(name)
    if client.collection_exists(name):
        return
    client.create_collection(
        collection_name=name,
        vectors_config={
            DENSE_VECTOR: models.VectorParams(size=EMBED_DIM, distance=models.Distance.COSINE)
        },
        sparse_vectors_config={
            SPARSE_VECTOR: models.SparseVectorParams(modifier=models.Modifier.IDF)
        },
    )


def batched(items, n):
    it = iter(items)
    while batch := list(islice(it, n)):
        yield batch


def ingest(dataset: str, force: bool = False, recreate: bool = False) -> None:
    corpus = load_corpus(dataset)
    chunks = chunk_corpus(corpus)
    name = collection_name(dataset)

    client = get_client()
    ensure_collection(client, name, recreate)

    existing = client.count(name, exact=True).count
    if existing == len(chunks) and not force:
        print(f"'{name}' already holds {existing} points, skipping (use --force to re-embed).")
        return

    model = TextEmbedding(EMBED_MODEL)
    start = time.perf_counter()
    with tqdm(total=len(chunks), desc="embed+upsert", unit="chunk") as bar:
        for batch in batched(chunks, UPSERT_BATCH):
            vectors = list(model.embed([c.text for c in batch], batch_size=EMBED_BATCH))
            client.upsert(
                collection_name=name,
                points=[_point(c, v) for c, v in zip(batch, vectors)],
                wait=True,
            )
            bar.update(len(batch))
    elapsed = time.perf_counter() - start

    total = client.count(name, exact=True).count
    assert total == len(chunks), f"expected {len(chunks)} points, collection has {total}"
    print(f"Ingested {len(chunks)} chunks into '{name}' in {elapsed:.0f}s ({total} points total).")


def _point(chunk: Chunk, vector) -> models.PointStruct:
    return models.PointStruct(
        id=chunk.point_id,
        vector={DENSE_VECTOR: vector.tolist()},
        payload={"doc_id": chunk.doc_id, "chunk_index": chunk.chunk_index, "title": chunk.title},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="scifact")
    parser.add_argument("--force", action="store_true", help="re-embed even if the collection is full")
    parser.add_argument("--recreate", action="store_true", help="drop and recreate the collection")
    args = parser.parse_args()
    ingest(args.dataset, args.force, args.recreate)
