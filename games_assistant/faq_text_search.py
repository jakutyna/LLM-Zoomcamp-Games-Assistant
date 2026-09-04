from __future__ import annotations

import argparse

from elasticsearch import Elasticsearch

from constants import DEFAULT_ES_URL, DEFAULT_INDEX




def search_faq(
    client: Elasticsearch,
    index_name: str,
    query: str,
    size: int = 5,
    category: str | None = None,
    tag: str | None = None,
    boost_dict: dict[str, int] | None = None,
) -> list[dict]:
    """Search FAQ questions and answers with optional exact metadata filters.

    Args:
        client: Connected Elasticsearch client.
        index_name: Name of the FAQ index to search.
        query: Natural-language search query.
        size: Maximum number of matching documents to return.
        category: Exact FAQ category by which to filter results.
        tag: Exact FAQ tag by which to filter results.
        boost_dict: Field weights for boosting relevance scores.
            If None, all fields have equal weight.

    Returns:
        Elasticsearch hit objects ordered by relevance score.
    """
    filters = []
    if category:
        filters.append({"term": {"category": category}})
    if tag:
        filters.append({"term": {"tag": tag}})

    if boost_dict:
        fields = [f"{field}^{boost}" for field, boost in boost_dict.items()]
    else:
        fields = ["question", "tag", "answer"]
    
    body = {
        "size": size,
        "query": {
            "bool": {
                "must": [
                    {
                        "multi_match": {
                            "query": query,
                            "fields": fields,
                            "type": "best_fields",
                        }
                    }
                ],
                "filter": filters,
            }
        },
    }
    response = client.search(index=index_name, body=body)
    return response["hits"]["hits"]




def main() -> None:
    """Parse CLI arguments, run an FAQ search, and print matching results."""
    parser = argparse.ArgumentParser(description="Lexical search on the GeForce NOW FAQ index")
    parser.add_argument("query", nargs="?", help="Search query")
    parser.add_argument("--index", default=DEFAULT_INDEX, help="Elasticsearch index name")
    parser.add_argument("--es-url", default=DEFAULT_ES_URL, help="Elasticsearch URL")
    parser.add_argument("--size", type=int, default=5, help="Number of top results")
    parser.add_argument("--category", help="Restrict results to an exact FAQ category")
    parser.add_argument("--tag", help="Restrict results to an exact FAQ tag")
    args = parser.parse_args()

    query = args.query or input("Search the GeForce NOW FAQ: ").strip()
    if not query:
        raise ValueError("A search query is required")

    client = Elasticsearch(args.es_url)
    if not client.ping():
        raise RuntimeError(f"Cannot connect to Elasticsearch at {args.es_url}")

    hits = search_faq(client, args.index, query, args.size, args.category, args.tag)
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
