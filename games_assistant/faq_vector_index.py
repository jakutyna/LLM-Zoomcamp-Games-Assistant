from __future__ import annotations

import argparse
import csv
from pathlib import Path

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from sentence_transformers import SentenceTransformer

from constants import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_ES_URL,
    DEFAULT_VECTOR_INDEX,
)


def create_index(
        client: Elasticsearch,
        index_name: str,
        dims: int,
        recreate: bool = False
    ) -> None:
    """Create the FAQ vector index when it does not already exist.

    Args:
        client: Connected Elasticsearch client.
        index_name: Name of the index to create.
        dims: Dimensionality of the embedding vectors.
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
            "tag_vector": {
                "type": "dense_vector",
                "dims": dims,
                "index": True,
                "similarity": "cosine",
            },
            "question_vector": {
                "type": "dense_vector",
                "dims": dims,
                "index": True,
                "similarity": "cosine",
            },
            "answer_vector": {
                "type": "dense_vector",
                "dims": dims,
                "index": True,
                "similarity": "cosine",
            },
            "question_answer_vector": {
                "type": "dense_vector",
                "dims": dims,
                "index": True,
                "similarity": "cosine",
            },
        }
    }
    client.indices.create(
        index=index_name,
        mappings=mappings,
        settings={"number_of_replicas": 0},
    )


def create_default_index(
        client: Elasticsearch,
        index_name: str = DEFAULT_VECTOR_INDEX,
        recreate: bool = False
    ) -> None:
    """Create the FAQ vector index using the default embedding model's dimensions.

    Args:
        client: Connected Elasticsearch client.
        index_name: Name of the index to create.
        recreate: Whether to delete an existing index before creation.
    """
    dims = SentenceTransformer(DEFAULT_EMBEDDING_MODEL).get_embedding_dimension()
    create_index(client, index_name, dims, recreate=recreate)


def index_csv_data(
    client: Elasticsearch,
    index_name: str,
    csv_path: Path,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> int:
    """Embed FAQ CSV rows and bulk-index them into Elasticsearch.

    Args:
        client: Connected Elasticsearch client.
        index_name: Name of the target index.
        csv_path: Path to the GeForce NOW FAQ CSV file.
        model_name: Sentence transformer model name used to encode the FAQ text.

    Returns:
        Number of successfully indexed FAQ documents.
    """
    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        documents = list(csv.DictReader(csv_file))

    if not documents:
        return 0

    model = SentenceTransformer(model_name)
    questions = [document["question"] for document in documents]
    answers = [document["answer"] for document in documents]
    pairs = [f"{question} {answer}" for question, answer in zip(questions, answers)]
    # Tags are slugs, so hyphens become spaces to give the tokenizer natural words.
    tags = [document["tag"].replace("-", " ") for document in documents]

    question_vectors = model.encode(questions, show_progress_bar=True)
    answer_vectors = model.encode(answers, show_progress_bar=True)
    pair_vectors = model.encode(pairs, show_progress_bar=True)
    tag_vectors = model.encode(tags, show_progress_bar=True)

    actions = [
        {
            "_index": index_name,
            "_id": document["id"],
            "_source": {
                **document,
                "question_vector": question_vector.tolist(),
                "answer_vector": answer_vector.tolist(),
                "question_answer_vector": pair_vector.tolist(),
                "tag_vector": tag_vector.tolist(),
            },
        }
        for document, question_vector, answer_vector, pair_vector, tag_vector in zip(
            documents, question_vectors, answer_vectors, pair_vectors, tag_vectors
        )
    ]

    success, _ = bulk(client=client, actions=actions, refresh="wait_for")
    return success


def main() -> None:
    """Parse CLI arguments and create and populate the FAQ vector index."""
    parser = argparse.ArgumentParser(description="Create and populate an Elasticsearch FAQ vector index")
    default_csv = Path(__file__).resolve().parent.parent / "data" / "csv" / "geforce_now_faq.csv"
    parser.add_argument("--csv", default=str(default_csv), help="Path to FAQ CSV file")
    parser.add_argument("--index", default=DEFAULT_VECTOR_INDEX, help="Elasticsearch index name")
    parser.add_argument("--es-url", default=DEFAULT_ES_URL, help="Elasticsearch URL")
    parser.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL, help="Sentence transformer model name")
    parser.add_argument("--recreate", action="store_true", help="Delete the index if it already exists")
    args = parser.parse_args()

    client = Elasticsearch(args.es_url)
    if not client.ping():
        raise RuntimeError(f"Cannot connect to Elasticsearch at {args.es_url}")

    dims = SentenceTransformer(args.model).get_embedding_dimension()
    create_index(client, args.index, dims, recreate=args.recreate)

    inserted = index_csv_data(client, args.index, Path(args.csv), args.model)
    print(f"Indexed {inserted} FAQ documents into '{args.index}'")


if __name__ == "__main__":
    main()
