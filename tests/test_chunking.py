import pytest

from src.ingest.chunking import chunk_corpus, chunk_document
from src.data import load_corpus


def doc(i="1", title="T", text="word " * 100):
    return {"id": i, "title": title, "text": text}


def test_default_is_one_chunk_per_doc():
    chunks = chunk_document(doc())
    assert len(chunks) == 1
    assert chunks[0].text.startswith("T. word")


def test_empty_title_has_no_dangling_prefix():
    assert chunk_document(doc(title=""))[0].text.startswith("word")
    assert chunk_document(doc(text="", title="Only title"))[0].text == "Only title"


def test_point_ids_are_deterministic_and_unique():
    a, b = chunk_document(doc("42"))[0], chunk_document(doc("42"))[0]
    assert a.point_id == b.point_id
    other = chunk_document(doc("43"))[0]
    assert a.point_id != other.point_id
    windows = chunk_document(doc("42"), max_words=40, overlap=10)
    assert len({c.point_id for c in windows}) == len(windows)


def test_window_chunking_overlaps_and_covers_everything():
    words = [f"w{i}" for i in range(95)]
    chunks = chunk_document(doc(title="", text=" ".join(words)), max_words=40, overlap=10)
    assert all(len(c.text.split()) <= 40 for c in chunks)
    covered = {w for c in chunks for w in c.text.split()}
    assert covered == set(words)
    # consecutive windows share exactly `overlap` words
    assert chunks[0].text.split()[-10:] == chunks[1].text.split()[:10]


def test_overlap_must_be_smaller_than_window():
    with pytest.raises(AssertionError):
        chunk_document(doc(), max_words=10, overlap=10)


def test_scifact_has_unique_point_ids():
    chunks = chunk_corpus(load_corpus())
    assert len(chunks) == 5183
    assert len({c.point_id for c in chunks}) == 5183
