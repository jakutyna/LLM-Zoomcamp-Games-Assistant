from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from uuid import UUID

import psycopg
from psycopg.types.json import Jsonb


DEFAULT_DATABASE_URL = "postgresql://games_assistant:games_assistant@localhost:5432/games_assistant"


@dataclass(frozen=True)
class Interaction:
    id: UUID
    created_at: datetime


def database_url() -> str:
    return os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)


def create_tables() -> None:
    """Create the application metrics table when it does not exist."""
    with psycopg.connect(database_url()) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS interactions (
                id UUID PRIMARY KEY,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                prompt JSONB NOT NULL,
                input_tokens INTEGER NOT NULL CHECK (input_tokens >= 0),
                output_tokens INTEGER NOT NULL CHECK (output_tokens >= 0),
                cost_usd NUMERIC(14, 10) NOT NULL CHECK (cost_usd >= 0),
                response_time_seconds DOUBLE PRECISION NOT NULL
                    CHECK (response_time_seconds >= 0),
                category TEXT,
                feedback SMALLINT CHECK (feedback IN (-1, 1)),
                created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                feedback_at TIMESTAMPTZ
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS interactions_created_at_idx
            ON interactions (created_at DESC)
            """
        )


def save_interaction(
    *,
    interaction_id: UUID,
    question: str,
    answer: str,
    prompt: list[dict[str, str]],
    input_tokens: int,
    output_tokens: int,
    cost_usd: float,
    response_time_seconds: float,
    category: str | None,
) -> Interaction:
    """Persist one completed RAG call and return its identity and timestamp."""
    with psycopg.connect(database_url()) as connection:
        row = connection.execute(
            """
            INSERT INTO interactions (
                id, question, answer, prompt, input_tokens, output_tokens,
                cost_usd, response_time_seconds, category
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id, created_at
            """,
            (
                interaction_id,
                question,
                answer,
                Jsonb(prompt),
                input_tokens,
                output_tokens,
                Decimal(str(cost_usd)),
                response_time_seconds,
                category,
            ),
        ).fetchone()
    if row is None:
        raise RuntimeError("The interaction was not saved")
    return Interaction(id=row[0], created_at=row[1])


def save_feedback(interaction_id: UUID, rating: int) -> None:
    """Set or replace a thumbs-up (1) or thumbs-down (-1) rating."""
    if rating not in {-1, 1}:
        raise ValueError("rating must be 1 or -1")

    with psycopg.connect(database_url()) as connection:
        result = connection.execute(
            """
            UPDATE interactions
            SET feedback = %s, feedback_at = CURRENT_TIMESTAMP
            WHERE id = %s
            """,
            (rating, interaction_id),
        )
        if result.rowcount != 1:
            raise LookupError(f"Unknown interaction: {interaction_id}")