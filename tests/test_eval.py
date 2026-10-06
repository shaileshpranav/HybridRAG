"""Prove the eval harness is correct before any real retriever exists."""

import math

import pytest

from src.data import load_corpus, load_qrels, load_queries
from eval.fakes import OracleRetriever, RandomRetriever, SleepRetriever
from eval.metrics import evaluate, to_run
from eval.runner import run_retriever
from src.retrieval.base import SearchResult


@pytest.fixture(scope="module")
def data():
    return load_corpus(), load_queries(), load_qrels()


def _run(retriever, queries, qrels, warmup=0):
    return run_retriever(retriever, queries, qrels, k=100, warmup=warmup)


def test_data_shape(data):
    corpus, queries, qrels = data
    assert len(corpus) == 5183
    assert len(queries) == len(qrels) == 300
    assert len(set(queries.values())) == len(queries), "oracle looks queries up by text"


def test_oracle_is_perfect(data):
    corpus, queries, qrels = data
    out = _run(OracleRetriever(queries, qrels, corpus), queries, qrels)
    assert out.metrics["recall_10"] == pytest.approx(1.0)
    assert out.metrics["ndcg_cut_10"] == pytest.approx(1.0)


def test_random_is_near_zero(data):
    corpus, queries, qrels = data
    out = _run(RandomRetriever(corpus, seed=0), queries, qrels)
    assert out.metrics["recall_10"] < 0.01
    assert out.metrics["ndcg_cut_10"] < 0.01


def test_cutoffs_are_respected(data):
    corpus, queries, qrels = data
    out = _run(OracleRetriever(queries, qrels, corpus, offset=10), queries, qrels)
    assert out.metrics["recall_10"] == pytest.approx(0.0)
    assert out.metrics["recall_100"] == pytest.approx(1.0)


def test_rank_position_matters(data):
    corpus, queries, qrels = data
    single = {q: r for q, r in qrels.items() if len(r) == 1}
    assert single, "need single-relevant queries"
    single_queries = {q: queries[q] for q in single}
    out = _run(OracleRetriever(single_queries, single, corpus, offset=1), single_queries, single)
    assert out.metrics["ndcg_cut_10"] == pytest.approx(1 / math.log2(3))


def test_latency_math(data):
    corpus, queries, qrels = data
    few = dict(list(queries.items())[:20])
    few_qrels = {q: qrels[q] for q in few}
    inner = OracleRetriever(few, few_qrels, corpus)
    out = _run(SleepRetriever(inner, delay_s=0.05), few, few_qrels, warmup=2)
    assert 50 <= out.latency_ms["p50"] < 70
    assert out.latency_ms["p95"] < 100


def test_hand_computed_ndcg():
    """3 queries, binary relevance, computed by hand.

    q1: relevant {a}; ranking [a, x, y]        -> nDCG = 1
    q2: relevant {b, c}; ranking [x, b, c]     -> DCG = 1/log2(3) + 1/log2(4); IDCG = 1 + 1/log2(3)
    q3: relevant {d}; ranking [x, y, z]        -> nDCG = 0
    """
    qrels = {"q1": {"a": 1}, "q2": {"b": 1, "c": 1}, "q3": {"d": 1}}
    results = {
        "q1": SearchResult(hits=[("a", 0.9), ("x", 0.5), ("y", 0.1)]),
        "q2": SearchResult(hits=[("x", 0.9), ("b", 0.5), ("c", 0.1)]),
        "q3": SearchResult(hits=[("x", 0.9), ("y", 0.5), ("z", 0.1)]),
    }
    q2 = (1 / math.log2(3) + 1 / math.log2(4)) / (1 + 1 / math.log2(3))
    means, _ = evaluate(to_run(results), qrels)
    assert means["ndcg_cut_10"] == pytest.approx((1 + q2 + 0) / 3)
    assert means["recall_10"] == pytest.approx((1 + 1 + 0) / 3)


def test_ties_do_not_scramble_ranking():
    """Equal pipeline scores must not let pytrec_eval reorder by doc id."""
    qrels = {"q": {"a": 1}}
    tied = {"q": SearchResult(hits=[("z", 1.0), ("y", 1.0), ("a", 1.0)])}  # relevant is last
    means, _ = evaluate(to_run(tied), qrels)
    assert means["ndcg_cut_10"] == pytest.approx(1 / math.log2(4))


def test_missing_queries_are_rejected():
    qrels = {"q1": {"a": 1}, "q2": {"b": 1}}
    run = {"q1": {"a": 1.0}}
    with pytest.raises(AssertionError):
        evaluate(run, qrels)
