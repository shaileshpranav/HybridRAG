"""Quality metrics via pytrec_eval, with guards against its silent failure modes."""

import numpy as np
import pytrec_eval

from src.retrieval.base import SearchResult

DEFAULT_MEASURES = {"recall_10", "recall_100", "ndcg_cut_10"}


def to_run(results: dict[str, SearchResult]) -> dict[str, dict[str, float]]:
    """Convert search results to pytrec_eval's run format.

    pytrec_eval re-sorts by score and breaks ties by doc id, which would scramble
    the pipeline's ranking (RRF produces ties). So the score handed to it is
    rank-derived and strictly decreasing, regardless of the pipeline's own scores.
    """
    run = {}
    for qid, result in results.items():
        n = len(result.hits)
        run[qid] = {str(doc_id): float(n - rank) for rank, (doc_id, _) in enumerate(result.hits)}
    return run


def evaluate(
    run: dict[str, dict[str, float]],
    qrels: dict[str, dict[str, int]],
    measures: set[str] = DEFAULT_MEASURES,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Return (mean of each measure over all qrels queries, per-query scores)."""
    # pytrec_eval silently drops queries missing from the run, inflating the mean.
    missing = set(qrels) - set(run)
    extra = set(run) - set(qrels)
    assert not missing, f"{len(missing)} queries have qrels but no run results"
    assert not extra, f"{len(extra)} queries in the run have no qrels"

    per_query = pytrec_eval.RelevanceEvaluator(qrels, measures).evaluate(run)
    means = {
        m: float(np.mean([scores[m] for scores in per_query.values()]))
        for m in sorted(measures)
    }
    return means, per_query
