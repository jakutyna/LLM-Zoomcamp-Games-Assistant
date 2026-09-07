from __future__ import annotations

import argparse
from threading import Lock

from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

from constants import DEFAULT_EMBEDDING_MODEL, DEFAULT_ES_URL, DEFAULT_VECTOR_INDEX

VECTOR_FIELDS = (
    "question_vector",
    "answer_vector",
    "question_answer_vector",
    "tag_vector",
)

_model_cache: dict[tuple[str, str | None, bool], SentenceTransformer] = {}
_model_cache_lock = Lock()


def get_embedding_model(
    model_name: str,
    cache_folder: str | None = None,
    local_files_only: bool = False,
) -> SentenceTransformer:
    """Load an embedding model once per process and reuse it for searches.

    Model files are stored in Hugging Face's cache by default. Set
    ``local_files_only`` after the first download to prevent network requests.
    """
    cache_key = (model_name, cache_folder, local_files_only)
    with _model_cache_lock:
        if cache_key not in _model_cache:
            _model_cache[cache_key] = SentenceTransformer(
                model_name,
                cache_folder=cache_folder,
                local_files_only=local_files_only,
            )
        return _model_cache[cache_key]


def search_faq(
    client: Elasticsearch,
    index_name: str,
    query: str,
    size: int = 5,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    category: str | None = None,
    tag: str | None = None,
    vector_field: str = "question_answer_vector",
    num_candidates: int = 100,
    cache_folder: str | None = None,
    local_files_only: bool = False,
) -> list[dict]:
    """Search FAQ entries by embedding similarity with optional metadata filters.

    Args:
        client: Connected Elasticsearch client.
        index_name: Name of the FAQ vector index to search.
        query: Natural-language search query.
        model_name: Sentence transformer model name used to encode the query.
        size: Maximum number of matching documents to return.
        category: Exact FAQ category by which to filter results.
        tag: Exact FAQ tag by which to filter results.
        vector_field: Dense vector field against which to run the kNN search.
        num_candidates: Number of candidates each shard considers before ranking.
        cache_folder: Optional directory for the Hugging Face model cache.
        local_files_only: Load only model files already available locally.

    Returns:
        Elasticsearch hit objects ordered by similarity score.
    """
    if vector_field not in VECTOR_FIELDS:
        raise ValueError(f"vector_field must be one of {VECTOR_FIELDS}")

    filters = []
    if category:
        filters.append({"term": {"category": category}})
    if tag:
        filters.append({"term": {"tag.keyword": tag}})

    model = get_embedding_model(model_name, cache_folder, local_files_only)
    knn = {
        "field": vector_field,
        "query_vector": model.encode(query).tolist(),
        "k": size,
        "num_candidates": num_candidates,
    }
    if filters:
        knn["filter"] = filters

    response = client.search(
        index=index_name,
        knn=knn,
        size=size,
        # source_excludes=list(VECTOR_FIELDS),
    )
    return response["hits"]["hits"]


def main() -> None:
    """Parse CLI arguments, run an FAQ vector search, and print matching results."""
    parser = argparse.ArgumentParser(description="Vector search on the GeForce NOW FAQ index")
    parser.add_argument("query", nargs="?", help="Search query")
    parser.add_argument("--index", default=DEFAULT_VECTOR_INDEX, help="Elasticsearch index name")
    parser.add_argument("--es-url", default=DEFAULT_ES_URL, help="Elasticsearch URL")
    parser.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL, help="Sentence transformer model name")
    parser.add_argument("--size", type=int, default=5, help="Number of top results")
    parser.add_argument("--category", help="Restrict results to an exact FAQ category")
    parser.add_argument("--tag", help="Restrict results to an exact FAQ tag")
    parser.add_argument(
        "--vector-field",
        default="question_answer_vector",
        choices=VECTOR_FIELDS,
        help="Dense vector field to search against",
    )
    parser.add_argument("--num-candidates", type=int, default=100, help="Candidates considered per shard")
    parser.add_argument("--cache-folder", help="Directory for the Hugging Face model cache")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load only a model already available in the local cache",
    )
    args = parser.parse_args()

    query = args.query or input("Search the GeForce NOW FAQ: ").strip()
    if not query:
        raise ValueError("A search query is required")

    client = Elasticsearch(args.es_url)
    if not client.ping():
        raise RuntimeError(f"Cannot connect to Elasticsearch at {args.es_url}")

    hits = search_faq(
        client,
        args.index,
        query,
        args.model,
        args.size,
        args.category,
        args.tag,
        args.vector_field,
        args.num_candidates,
        args.cache_folder,
        args.local_files_only,
    )
    if not hits:
        print("No results found")
        return

    for position, hit in enumerate(hits, start=1):
        faq = hit["_source"]
        print(
            f"{position}. [{faq['category']}] {faq['question']} "
            f"(tag={faq['tag']}, score={hit['_score']:.3f})\n"
            f"   {faq['answer']}\n"
        )


if __name__ == "__main__":
    main()
