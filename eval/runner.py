"""Run a Retriever over the queries, time it, score it, and write a results file."""

import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from eval.metrics import evaluate, to_run
from src.retrieval.base import Retriever, SearchResult

RESULTS_DIR = Path(__file__).resolve().parent / "results"
WARMUP_QUERIES = 10


@dataclass
class RunOutput:
    name: str
    k: int
    metrics: dict[str, float]
    latency_ms: dict[str, float]
    results: dict[str, SearchResult]
    per_query: dict[str, dict[str, float]]


def latency_summary(latencies_s: list[float]) -> dict[str, float]:
    ms = np.asarray(latencies_s) * 1000
    return {
        "p50": float(np.percentile(ms, 50)),
        "p95": float(np.percentile(ms, 95)),
        "mean": float(ms.mean()),
        "max": float(ms.max()),
    }


def run_retriever(
    retriever: Retriever,
    queries: dict[str, str],
    qrels: dict[str, dict[str, int]],
    k: int = 100,
    warmup: int = WARMUP_QUERIES,
) -> RunOutput:
    """Sequentially query the retriever. Concurrency is the load test's job, not this one's."""
    # Warm-up (model load, connection setup) is excluded from timing and scoring.
    for text in list(queries.values())[:warmup]:
        retriever.search(text, k)

    results: dict[str, SearchResult] = {}
    latencies: list[float] = []
    for qid, text in queries.items():
        start = time.perf_counter()
        results[qid] = retriever.search(text, k)
        latencies.append(time.perf_counter() - start)

    metrics, per_query = evaluate(to_run(results), qrels)
    return RunOutput(
        name=retriever.name,
        k=k,
        metrics=metrics,
        latency_ms=latency_summary(latencies),
        results=results,
        per_query=per_query,
    )


def _git_sha() -> str:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def write_results(out: RunOutput, dataset: str, config: dict | None = None, include_run: bool = False) -> Path:
    payload = {
        "name": out.name,
        "dataset": dataset,
        "git_sha": _git_sha(),
        "k": out.k,
        "n_queries": len(out.results),
        "config": config or {},
        "metrics": out.metrics,
        "latency_ms": out.latency_ms,
    }
    if include_run:
        payload["run"] = {
            qid: [[doc_id, score] for doc_id, score in r.hits] for qid, r in out.results.items()
        }
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{dataset}__{out.name}.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path
