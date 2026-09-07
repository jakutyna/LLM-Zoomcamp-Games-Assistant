from __future__ import annotations

import os

from elasticsearch import Elasticsearch

from constants import DEFAULT_ES_URL
from database import create_tables
from ingestion import ingest_default_indexes


def main() -> None:
    """Initialize application storage, then replace this process with Streamlit."""
    client = Elasticsearch(DEFAULT_ES_URL)
    if not client.ping():
        raise RuntimeError(f"Cannot connect to Elasticsearch at {DEFAULT_ES_URL}")

    ingest_default_indexes(client)
    create_tables()
    os.execvp(
        "streamlit",
        [
            "streamlit",
            "run",
            "games_assistant/app.py",
            "--server.address=0.0.0.0",
            "--server.port=8501",
            "--server.fileWatcherType=none",
        ],
    )


if __name__ == "__main__":
    main()