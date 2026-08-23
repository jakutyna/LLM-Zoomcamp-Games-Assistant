# LLM-Zoomcamp-Games-Assistant

- project for LLM Zoomcamp 2026
- assistant that anwsers questions about NVIDIA Geforce Now service.
- the anwsers are grounded with the data pulled from NVIDIA FAQ page.

FAQ Source: [https://www.nvidia.com/en-us/geforce-now/faq/](https://www.nvidia.com/en-us/geforce-now/faq/).

## Prerequisites

- uv
- docker
- OpenAI account

Install `uv` (Mac/Linux):
```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

OpenAI API key (`.env`):

```sh
OPENAI_API_KEY=sk-<YOUR_KEY_HERE>
```

Note: Make sure `.env` is included in `.gitignore` file so you never accidentally commit your key.


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

<docker composer instructions>


## Components

<each point explained seperately>