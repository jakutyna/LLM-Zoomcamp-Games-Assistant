from __future__ import annotations

import argparse
import csv
from pathlib import Path

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

from constants import DEFAULT_ES_URL, DEFAULT_INDEX


def create_index(
        client: Elasticsearch,
        index_name: str,
        recreate: bool = False
    ) -> None:
    """Create the FAQ index when it does not already exist.

    Args:
        client: Connected Elasticsearch client.
        index_name: Name of the index to create.
        recreate: Whether to delete an existing index before creation.
    """
    if recreate and client.indices.exists(index=index_name):
        client.indices.delete(index=index_name)

    if client.indices.exists(index=index_name):
        return

    mappings = {
        "properties": {
            "id": {"type": "integer"},
            "category": {"type": "keyword"},
            "tag": {
                "type": "text",
                "fields": {"keyword": {"type": "keyword"}},
            },
            "question": {"type": "text"},
            "answer": {"type": "text"},
        }
    }
    client.indices.create(index=index_name, mappings=mappings)


def index_csv_data(client: Elasticsearch, index_name: str, csv_path: Path) -> int:
    """Read FAQ CSV rows and bulk-index them into Elasticsearch.

    Args:
        client: Connected Elasticsearch client.
        index_name: Name of the target index.
        csv_path: Path to the GeForce NOW FAQ CSV file.

    Returns:
        Number of successfully indexed FAQ documents.
    """
    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        documents = list(csv.DictReader(csv_file))

    actions = [
        {
            "_index": index_name,
            "_id": document["id"],
            "_source": document,
        }
        for document in documents
    ]

    if not actions:
        return 0

    success, _ = bulk(client=client, actions=actions, refresh="wait_for")
    return success


def main() -> None:
    """Parse CLI arguments and create and populate the FAQ index."""
    parser = argparse.ArgumentParser(description="Create and populate an Elasticsearch FAQ index")
    default_csv = Path(__file__).resolve().parent.parent / "data" / "csv" / "geforce_now_faq.csv"
    parser.add_argument("--csv", default=str(default_csv), help="Path to FAQ CSV file")
    parser.add_argument("--index", default=DEFAULT_INDEX, help="Elasticsearch index name")
    parser.add_argument("--es-url", default=DEFAULT_ES_URL, help="Elasticsearch URL")
    parser.add_argument("--recreate", action="store_true", help="Delete the index if it already exists")
    args = parser.parse_args()

    client = Elasticsearch(args.es_url)
    if not client.ping():
        raise RuntimeError(f"Cannot connect to Elasticsearch at {args.es_url}")

    create_index(client, args.index, recreate=args.recreate)
    inserted = index_csv_data(client, args.index, Path(args.csv))
    print(f"Indexed {inserted} FAQ documents into '{args.index}'")


if __name__ == "__main__":
    main()