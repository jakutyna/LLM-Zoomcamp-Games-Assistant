FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY data ./data
COPY games_assistant ./games_assistant
COPY README.md ./
RUN uv sync --frozen --no-dev

EXPOSE 8501

CMD ["python", "games_assistant/startup.py"]