## Retrieval Setup (Elasticsearch)

Focus: lexical retrieval only (for RAG retrieval stage).

1) Start local Elasticsearch (free)

```sh
docker run --name games-es \
	-p 9200:9200 \
	-e discovery.type=single-node \
	-e xpack.security.enabled=false \
	-e ES_JAVA_OPTS='-Xms1g -Xmx1g' \
	docker.elastic.co/elasticsearch/elasticsearch:9.5.1
```

2) Install dependencies

```sh
uv sync
```

3) Create index and load CSV

```sh
uv run games_assistant/index_games_es.py --recreate
```

4) Run lexical search

```sh
uv run games_assistant/search_games_es.py "psychological survival horror"
```

## GeForce NOW FAQ Retrieval

Create the FAQ index from the enriched FAQ dataset:

```sh
uv run games_assistant/index_faq_es.py --recreate
```

Search with a command-line query or run the command without one to enter it interactively:

```sh
uv run games_assistant/search_faq_es.py "How much does Ultimate cost?"
```

Optional exact filters narrow the retrieval set for a RAG pipeline:

```sh
uv run games_assistant/search_faq_es.py "playtime" --category Memberships
uv run games_assistant/search_faq_es.py "membership" --tag ultimate-membership
```

Optional container commands:

```sh
docker stop games-es
docker start games-es
```