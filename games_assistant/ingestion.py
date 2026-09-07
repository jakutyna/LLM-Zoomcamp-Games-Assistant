from __future__ import annotations

from pathlib import Path

from elasticsearch import Elasticsearch

from constants import DEFAULT_ES_URL, DEFAULT_INDEX, DEFAULT_VECTOR_INDEX
from faq_text_index import create_index as create_text_index
from faq_text_index import index_csv_data as index_text_data
from faq_vector_index import create_default_index as create_vector_index
from faq_vector_index import index_csv_data as index_vector_data


DEFAULT_CSV_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "csv" / "geforce_now_faq.csv"
)


def ingest_default_indexes(
    client: Elasticsearch,
    csv_path: Path = DEFAULT_CSV_PATH,
) -> None:
    """Create and populate missing default FAQ indexes."""
    if client.indices.exists(index=DEFAULT_INDEX):
        print(f"Text index '{DEFAULT_INDEX}' already exists; skipping.")
    else:
        create_text_index(client, DEFAULT_INDEX)
        inserted = index_text_data(client, DEFAULT_INDEX, csv_path)
        print(f"Indexed {inserted} FAQ documents into '{DEFAULT_INDEX}'.")

    if client.indices.exists(index=DEFAULT_VECTOR_INDEX):
        print(f"Vector index '{DEFAULT_VECTOR_INDEX}' already exists; skipping.")
    else:
        create_vector_index(client, DEFAULT_VECTOR_INDEX)
        inserted = index_vector_data(client, DEFAULT_VECTOR_INDEX, csv_path)
        print(f"Indexed {inserted} FAQ documents into '{DEFAULT_VECTOR_INDEX}'.")


def main() -> None:
    """Connect to Elasticsearch and ingest any missing default indexes."""
    client = Elasticsearch(DEFAULT_ES_URL)
    if not client.ping():
        raise RuntimeError(f"Cannot connect to Elasticsearch at {DEFAULT_ES_URL}")

    ingest_default_indexes(client)


if __name__ == "__main__":
    main()