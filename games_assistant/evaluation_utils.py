from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd


SearchFunction = Callable[..., Iterable[Mapping[str, Any]]]


def hit_rate(relevant_id: int | str, results: Iterable[Mapping[str, Any]]) -> float:
    """Return 1.0 when the relevant document appears in the results, else 0.0.

    Args:
        relevant_id: ID of the FAQ document relevant to the query.
        results: Ranked search results. Elasticsearch hits may contain the
            document ID in ``_source['id']`` or in ``_id``.
    """
    return float(_relevant_rank(relevant_id, results) is not None)


def mrr(relevant_id: int | str, results: Iterable[Mapping[str, Any]]) -> float:
    """Return the reciprocal rank of the relevant document, or 0.0 if absent.

    Args:
        relevant_id: ID of the FAQ document relevant to the query.
        results: Ranked search results in descending relevance order.
    """
    rank = _relevant_rank(relevant_id, results)
    return 1 / rank if rank is not None else 0.0


def evaluate_search_function(
    search_function: SearchFunction,
    ground_truth: pd.DataFrame | str | Path,
    index_name: str,
    **kwargs: Any,
) -> dict[str, float]:
    """Calculate aggregate hit rate and MRR for a search function.

    The ground-truth data must contain ``id`` and ``question`` columns. Each
    row represents one query whose relevant document is identified by ``id``.

    Args:
        search_function: Callable accepting ``query``, ``index_name``, and
            optional search arguments, returning ranked result mappings.
        ground_truth: Ground-truth DataFrame or path to a CSV file.
        index_name: Index name passed to the search function for every query.
        **kwargs: Additional search arguments, such as ``size``, ``client``,
            or vector-search settings.

    Returns:
        A dictionary containing aggregate ``hit_rate`` and ``mrr`` values.
    """
    dataset = _load_ground_truth(ground_truth)
    if dataset.empty:
        raise ValueError("ground_truth must contain at least one row")
    missing_columns = {"id", "question"} - set(dataset.columns)
    if missing_columns:
        raise ValueError(f"ground_truth is missing columns: {sorted(missing_columns)}")

    hit_scores = []
    reciprocal_ranks = []
    for row in dataset.itertuples(index=False):
        results = search_function(
            query=row.question,
            index_name=index_name,
            **kwargs,
        )
        results = list(results)
        hit_scores.append(hit_rate(row.id, results))
        reciprocal_ranks.append(mrr(row.id, results))

    return {
        "hit_rate": sum(hit_scores) / len(hit_scores),
        "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks),
    }


def _load_ground_truth(ground_truth: pd.DataFrame | str | Path) -> pd.DataFrame:
    if isinstance(ground_truth, pd.DataFrame):
        return ground_truth
    return pd.read_csv(ground_truth)


def _relevant_rank(
    relevant_id: int | str,
    results: Iterable[Mapping[str, Any]],
) -> int | None:
    expected_id = str(relevant_id)
    for rank, result in enumerate(results, start=1):
        source = result.get("_source", result)
        result_id = source.get("id", result.get("_id"))
        if result_id is not None and str(result_id) == expected_id:
            return rank
    return None