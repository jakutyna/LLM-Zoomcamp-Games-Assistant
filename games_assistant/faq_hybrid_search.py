from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping
from typing import Any

from elasticsearch import Elasticsearch

import faq_text_search as text_search
import faq_vector_search as vector_search
from constants import (
    BEST_BOOST_PARAMS,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_ES_URL,
    DEFAULT_INDEX,
    DEFAULT_VECTOR_INDEX,
)


def reciprocal_rank_fusion(
    ranked_results: Iterable[Iterable[Mapping[str, Any]]],
    size: int = 5,
    rank_constant: int = 60,
) -> list[dict[str, Any]]:
    """Combine ranked result lists using reciprocal rank fusion (RRF).

    The fused score adds $1 / (k + rank)$ for each ranked result, where ``k`` is
    ``rank_constant``. Documents found in both result lists therefore receive
    a higher score than similarly ranked documents found in only one list. A
    document contributes at most once per result list.
    """
    if size < 1:
        raise ValueError("size must be at least 1")
    if rank_constant < 0:
        raise ValueError("rank_constant must be non-negative")

    fused_hits: dict[str, dict[str, Any]] = {}
    for results in ranked_results:
        seen_document_ids: set[str] = set()
        for rank, hit in enumerate(results, start=1):
            document_id = hit.get("_id")
            if document_id is None or document_id in seen_document_ids:
                continue
            seen_document_ids.add(document_id)
            if document_id not in fused_hits:
                fused_hits[document_id] = {**hit, "_score": 0.0}
            fused_hits[document_id]["_score"] += 1 / (rank_constant + rank)

    return sorted(
        fused_hits.values(),
        key=lambda hit: hit["_score"],
        reverse=True,
    )[:size]


def search_faq(
    client: Elasticsearch,
    query: str,
    index_name: str = DEFAULT_INDEX,
    vector_index_name: str = DEFAULT_VECTOR_INDEX,
    size: int = 5,
    category: str | None = None,
    tag: str | None = None,
    boost_dict: dict[str, int] | None = None,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    vector_field: str = "question_answer_vector",
    num_candidates: int = 100,
    cache_folder: str | None = None,
    local_files_only: bool = False,
    rank_constant: int = 60,
) -> list[dict[str, Any]]:
    """Search lexical and vector indices, combining results with RRF.

    Args:
        client: Connected Elasticsearch client.
        index_name: Text-search index name.
        query: Natural-language search query.
        size: Maximum number of fused results to return.
        vector_index_name: Vector-search index name.
        category: Exact FAQ category by which to filter results.
        tag: Exact FAQ tag by which to filter results.
        boost_dict: Field weights used by text search.
        model_name: Sentence transformer model used for vector search.
        vector_field: Dense vector field to search.
        num_candidates: Vector kNN candidates considered by each shard.
        cache_folder: Optional directory for the Hugging Face model cache.
        local_files_only: Load only an embedding model already cached locally.
        rank_constant: RRF rank constant, typically 60.
    """
    text_hits = text_search.search_faq(
        client=client,
        index_name=index_name,
        query=query,
        size=size,
        category=category,
        tag=tag,
        boost_dict=boost_dict,
    )
    vector_hits = vector_search.search_faq(
        client=client,
        index_name=vector_index_name,
        query=query,
        size=size,
        model_name=model_name,
        category=category,
        tag=tag,
        vector_field=vector_field,
        num_candidates=num_candidates,
        cache_folder=cache_folder,
        local_files_only=local_files_only,
    )
    return reciprocal_rank_fusion(
        (text_hits, vector_hits),
        size=size,
        rank_constant=rank_constant,
    )


def main() -> None:
    """Parse CLI arguments, run a hybrid FAQ search, and print matches."""
    parser = argparse.ArgumentParser(description="Hybrid RRF search on the GeForce NOW FAQ")
    parser.add_argument("query", nargs="?", help="Search query")
    parser.add_argument("--index", default=DEFAULT_INDEX, help="Text-search index name")
    parser.add_argument("--vector-index", default=DEFAULT_VECTOR_INDEX, help="Vector-search index name")
    parser.add_argument("--es-url", default=DEFAULT_ES_URL, help="Elasticsearch URL")
    parser.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL, help="Sentence transformer model name")
    parser.add_argument("--size", type=int, default=5, help="Number of top results")
    parser.add_argument("--category", help="Restrict results to an exact FAQ category")
    parser.add_argument("--tag", help="Restrict results to an exact FAQ tag")
    parser.add_argument("--rank-constant", type=int, default=60, help="RRF rank constant")
    parser.add_argument("--local-files-only", action="store_true", help="Use only cached model files")
    args = parser.parse_args()

    query = args.query or input("Search the GeForce NOW FAQ: ").strip()
    if not query:
        raise ValueError("A search query is required")

    client = Elasticsearch(args.es_url)
    if not client.ping():
        raise RuntimeError(f"Cannot connect to Elasticsearch at {args.es_url}")

    hits = search_faq(
        client=client,
        index_name=args.index,
        vector_index_name=args.vector_index,
        query=query,
        size=args.size,
        category=args.category,
        tag=args.tag,
        boost_dict=BEST_BOOST_PARAMS,
        model_name=args.model,
        local_files_only=args.local_files_only,
        rank_constant=args.rank_constant,
    )
    if not hits:
        print("No results found")
        return

    for position, hit in enumerate(hits, start=1):
        faq = hit["_source"]
        print(
            f"{position}. [{faq['category']}] {faq['question']} "
            f"(rrf_score={hit['_score']:.4f})\n"
            f"   {faq['answer']}\n"
        )


if __name__ == "__main__":
    main()