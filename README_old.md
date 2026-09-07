# LLM-Zoomcamp-Games-Assistant

- project for LLM Zoomcamp 2026
- assistant that anwsers questions about NVIDIA Geforce Now service.
- the anwsers are grounded with the data pulled from NVIDIA FAQ page.

FAQ Source: [https://www.nvidia.com/en-us/geforce-now/faq/](https://www.nvidia.com/en-us/geforce-now/faq/).

## Prerequisites

- Docker with the Compose plugin
- OpenAI account

Create the local environment file and add an OpenAI API key:

```sh
cp .env.example .env
```

The `.env` file is ignored by Git. Change the default PostgreSQL and Grafana
passwords in that file when exposing the services beyond local development.


## Data

The FAQ data questions were scraped from the source page with [data/scrape_nvidia_faq.py](data/scrape_nvidia_faq.py) python script into [data/geforce_now_faq_raw.csv](data/geforce_now_faq_raw.csv). The data can be recreated from the original page by running the script with command:

```sh
uv run data/scrape_nvidia_faq.py
```

The FAQ dataset was additionally enhanced with a tag column. A tag is an expression that briefly indicates what each question is about. All of the tags were generated with a script [data/generate_tags.py](data/generate_tags.py) that calls OpenAI API with a templated prompt for each FAQ question.

The calls are done asynchronously with a [`concurrent.futures`](https://docs.python.org/3/library/concurrent.futures.html) module. The responses are returned in a proper format - a `FAQ` pydantic model defined in [data/faq_model.py](data/faq_model.py).

Tags can be recreated by running:
```sh
uv run data/generate_tags.py
```


## Quick Start

Build and launch the complete stack:

```sh
docker compose up --build
```

Open the services after startup:

- Streamlit assistant: [http://localhost:8501](http://localhost:8501)
- Grafana: [http://localhost:3000](http://localhost:3000)

PostgreSQL is available to host tools on `localhost:5434`. Elasticsearch is
available only within the Compose network.

Grafana uses the credentials from `.env` (`admin` / `admin` by default). The
provisioned **Games Assistant / RAG Metrics** dashboard reports calls, cost,
latency, token usage, feedback, and recent interactions.

The first app startup downloads the embedding model and creates both
Elasticsearch indexes, so it can take several minutes. Subsequent starts reuse
the persisted index and model-cache volumes.

Stop the services with:

```sh
docker compose down
```

Add `-v` only when you also want to delete Elasticsearch, PostgreSQL, Grafana,
and model-cache volumes.


## Components

- **Streamlit app** runs hybrid FAQ retrieval (lexical and vector search with
	reciprocal-rank fusion), sends grounded context to OpenAI, and records each
	interaction and rating.
- **Elasticsearch** stores separate text and vector FAQ indexes. Missing indexes
	are populated automatically from `data/csv/geforce_now_faq.csv` at app startup.
- **PostgreSQL** stores questions, answers, prompts, token counts, cost, response
	time, category, timestamps, and thumbs-up/down feedback.
- **Grafana** reads PostgreSQL through an automatically provisioned datasource
	and dashboard.

For local Python development outside Docker, install dependencies with:

```sh
uv sync
```

Generate 50 mock metric rows distributed across the last 12 hours for the
Grafana dashboard:

```sh
uv run data/generate_mock_metrics.py
```

This command runs on the host and connects to the Compose PostgreSQL service on
port `5434`. Use `--count`, `--hours`, `--seed`, or `--database-url` to override
its defaults.