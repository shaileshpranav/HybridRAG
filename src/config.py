"""Frozen settings shared by ingest, retrieval and eval. Changing the embedder invalidates every result."""

import uuid

QDRANT_URL = "http://localhost:6333"
QDRANT_GRPC_PORT = 6334

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384

DENSE_VECTOR = "dense"
SPARSE_VECTOR = "sparse"

# Fixed namespace so point ids are reproducible across runs and machines.
POINT_ID_NAMESPACE = uuid.UUID("6f1d5a8e-2c3b-4d7a-9e10-5b8c4a7d3f21")


def collection_name(dataset: str) -> str:
    return dataset
