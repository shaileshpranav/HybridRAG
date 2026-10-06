"""Turn corpus documents into embeddable chunks.

SciFact abstracts are short, so the default strategy keeps one chunk per document.
The window strategy exists for corpora with long documents (full PDFs, FiQA threads).
"""

import uuid
from dataclasses import dataclass

from src.config import POINT_ID_NAMESPACE


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    chunk_index: int
    title: str
    text: str  # what gets embedded

    @property
    def point_id(self) -> str:
        """Deterministic UUID, so re-ingesting overwrites instead of duplicating."""
        return str(uuid.uuid5(POINT_ID_NAMESPACE, f"{self.doc_id}:{self.chunk_index}"))


def _full_text(doc: dict) -> str:
    # Some BEIR corpora have empty titles; avoid a dangling ". " prefix.
    title, text = doc["title"].strip(), doc["text"].strip()
    return f"{title}. {text}" if title and text else title or text


def chunk_document(doc: dict, max_words: int | None = None, overlap: int = 0) -> list[Chunk]:
    """One chunk per doc by default. With max_words, split into overlapping word windows."""
    full = _full_text(doc)
    if max_words is None:
        return [Chunk(doc["id"], 0, doc["title"], full)]

    assert 0 <= overlap < max_words, "overlap must be smaller than the window"
    words = full.split()
    step = max_words - overlap
    windows = [words[i : i + max_words] for i in range(0, max(len(words), 1), step)]
    # Drop a trailing window that is entirely covered by the previous one.
    if len(windows) > 1 and len(windows[-1]) <= overlap:
        windows.pop()
    return [Chunk(doc["id"], i, doc["title"], " ".join(w)) for i, w in enumerate(windows)]


def chunk_corpus(corpus: dict[str, dict], **kwargs) -> list[Chunk]:
    return [c for doc in corpus.values() for c in chunk_document(doc, **kwargs)]
