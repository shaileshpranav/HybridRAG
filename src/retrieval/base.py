"""The contract every pipeline configuration implements.

The eval harness only ever sees this interface, so each ablation row
(dense, BM25, hybrid, +rerank, ...) is just a different Retriever.
"""

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class SearchResult:
    hits: list[tuple[str, float]]  # (doc_id, score), best first
    timings: dict[str, float] = field(default_factory=dict)  # stage -> seconds
    meta: dict = field(default_factory=dict)  # cache_hit, tokens, rewritten query, ...


class Retriever(Protocol):
    name: str

    def search(self, query: str, k: int) -> SearchResult: ...
