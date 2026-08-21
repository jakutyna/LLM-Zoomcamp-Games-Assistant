from __future__ import annotations

import argparse
import ast
from pathlib import Path

import pandas as pd
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

DEFAULT_INDEX = "games"
DEFAULT_ES_URL = "http://localhost:9200"


def parse_platforms(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value]

    if not isinstance(value, str):
        return []

    text = value.strip()
    if not text:
        return []

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(v) for v in parsed]
    except (ValueError, SyntaxError):
        pass

    return [text]


def create_index(client: Elasticsearch, index_name: str, recreate: bool) -> None:
    if recreate and client.indices.exists(index=index_name):
        client.indices.delete(index=index_name)

    if client.indices.exists(index=index_name):
        return

    # mappings = {
    #     "properties": {
    #         "rank": {"type": "integer"},
    #         "title": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
    #         "genre": {"type": "keyword"},
    #         "platforms": {"type": "keyword"},
    #         "release_date": {"type": "date", "format": "yyyy-MM-dd"},
    #         "developer": {"type": "keyword"},
    #         "publisher": {"type": "keyword"},
    #         "description": {"type": "text"},
    #     }
    # }
    mappings = {
        "properties": {
            "rank": {"type": "keyword"},
            "title": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
            "genre": {"type": "keyword"},
            "platforms": {"type": "keyword"},
            "release_date": {"type": "text"},
            "developer": {"type": "text"},
            "publisher": {"type": "text"},
            "description": {"type": "text"},
        }
    }

    client.indices.create(index=index_name, mappings=mappings)


def index_csv_data(client: Elasticsearch, index_name: str, csv_path: Path) -> int:
    df = pd.read_csv(csv_path)

    actions = []
    for row in df.to_dict(orient="records"):
        doc = {
            "rank": int(row["rank"]),
            "title": str(row["title"]),
            "genre": str(row["genre"]),
            "platforms": parse_platforms(row.get("platforms")),
            "release_date": str(row["release_date"]),
            "developer": str(row["developer"]),
            "publisher": str(row["publisher"]),
            "description": str(row["description"]),
        }

        actions.append(
            {
                "_index": index_name,
                "_id": doc["rank"],
                "_source": doc,
            }
        )

    if not actions:
        return 0

    success, _ = bulk(client=client, actions=actions, refresh="wait_for")
    return success


def main() -> None:
    parser = argparse.ArgumentParser(description="Create and populate Elasticsearch index from games_data.csv")
    default_csv = Path(__file__).resolve().parent.parent / "data" / "games_data.csv"
    parser.add_argument("--csv", default=str(default_csv), help="Path to CSV file")
    parser.add_argument("--index", default=DEFAULT_INDEX, help="Elasticsearch index name")
    parser.add_argument("--es-url", default=DEFAULT_ES_URL, help="Elasticsearch URL")
    parser.add_argument("--recreate", action="store_true", help="Delete index if it already exists")
    args = parser.parse_args()

    client = Elasticsearch(args.es_url)
    if not client.ping():
        raise RuntimeError(f"Cannot connect to Elasticsearch at {args.es_url}")

    create_index(client, args.index, recreate=args.recreate)
    inserted = index_csv_data(client, args.index, Path(args.csv))
    print(f"Indexed {inserted} documents into '{args.index}'")


if __name__ == "__main__":
    main()
