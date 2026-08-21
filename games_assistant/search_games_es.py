from __future__ import annotations

import argparse

from elasticsearch import Elasticsearch

DEFAULT_INDEX = "games"
DEFAULT_ES_URL = "http://localhost:9200"


def search_games(client: Elasticsearch, index_name: str, query: str, size: int) -> list[dict]:
    body = {
        "size": size,
        "query": {
            "multi_match": {
                "query": query,
                "fields": ["rank", "title^2", "description", "genre^2", "developer", "publisher", "platforms"],
                "type": "best_fields",
            }
        },
    }

    response = client.search(index=index_name, body=body)
    return response["hits"]["hits"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Lexical search on games Elasticsearch index")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--index", default=DEFAULT_INDEX, help="Elasticsearch index name")
    parser.add_argument("--es-url", default=DEFAULT_ES_URL, help="Elasticsearch URL")
    parser.add_argument("--size", type=int, default=5, help="Number of top results")
    args = parser.parse_args()

    client = Elasticsearch(args.es_url)
    if not client.ping():
        raise RuntimeError(f"Cannot connect to Elasticsearch at {args.es_url}")

    hits = search_games(client, args.index, args.query, args.size)

    if not hits:
        print("No results found")
        return

    for i, hit in enumerate(hits, start=1):
        src = hit["_source"]
        print(
            f"{i}. [{src['rank']}] {src['title']} | score={hit['_score']:.3f}\\n"
            f"   genre={src['genre']} | release_date={src['release_date']}\\n"
            f"   developer={src['developer']} | publisher={src['publisher']}"
        )


if __name__ == "__main__":
    main()
