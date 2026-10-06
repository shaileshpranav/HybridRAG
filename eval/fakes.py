"""Fake retrievers with known expected scores, used to prove the harness is correct."""

import random
import time

from src.retrieval.base import SearchResult


def _hits(doc_ids: list[str]) -> SearchResult:
    n = len(doc_ids)
    return SearchResult(hits=[(d, float(n - i)) for i, d in enumerate(doc_ids)])


class OracleRetriever:
    """Returns the relevant docs first, then junk. Optionally pushes them down `offset` ranks."""

    def __init__(self, queries, qrels, corpus_ids, offset: int = 0, name: str = "oracle"):
        self.name = name
        self.text_to_qid = {text: qid for qid, text in queries.items()}
        self.qrels = qrels
        self.junk = sorted(corpus_ids)
        self.offset = offset

    def search(self, query: str, k: int) -> SearchResult:
        relevant = sorted(self.qrels[self.text_to_qid[query]])
        rel_set = set(relevant)
        non_relevant = [d for d in self.junk if d not in rel_set]
        head, tail = non_relevant[: self.offset], non_relevant[self.offset :]
        return _hits((head + relevant + tail)[:k])


class RandomRetriever:
    def __init__(self, corpus_ids, seed: int = 0, name: str = "random"):
        self.name = name
        self.ids = sorted(corpus_ids)
        self.rng = random.Random(seed)

    def search(self, query: str, k: int) -> SearchResult:
        return _hits(self.rng.sample(self.ids, k))


class SleepRetriever:
    """Wraps another retriever and adds a fixed delay, to test the latency math."""

    def __init__(self, inner, delay_s: float, name: str = "sleep"):
        self.name = name
        self.inner = inner
        self.delay_s = delay_s

    def search(self, query: str, k: int) -> SearchResult:
        time.sleep(self.delay_s)
        return self.inner.search(query, k)
