from __future__ import annotations

import argparse
import os
import random
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any
from uuid import uuid4

import psycopg
from dotenv import load_dotenv
from psycopg.types.json import Jsonb

DEFAULT_COUNT = 50
DEFAULT_HOURS = 12
INPUT_PRICE_PER_MILLION_TOKENS = Decimal("0.75")
OUTPUT_PRICE_PER_MILLION_TOKENS = Decimal("4.50")
INSTRUCTIONS = "Answer the GeForce NOW question using only the provided FAQ context."
FAQ_SAMPLES = (
    {
        "category": "General Questions",
        "question": "What is GeForce NOW?",
        "answer": "GeForce NOW is NVIDIA's cloud gaming service.",
    },
    {
        "category": "Memberships",
        "question": "What membership options are available?",
        "answer": "GeForce NOW offers Free, Performance, and Ultimate plans.",
    },
    {
        "category": "Memberships",
        "question": "How much monthly playtime do I get?",
        "answer": "Performance and Ultimate members receive 100 hours per month.",
    },
    {
        "category": "Install-to-Play",
        "question": "What is Install-to-Play?",
        "answer": "Premium members can install supported Steam games on cloud storage.",
    },
    {
        "category": "Android",
        "question": "What Android phones are compatible?",
        "answer": "Android devices need at least 1 GB of memory and Android 7.0.",
    },
    {
        "category": "iPhone and iPad",
        "question": "How do I use GeForce NOW on iPhone or iPad?",
        "answer": "Open play.geforcenow.com in Safari and add it to the home screen.",
    },
)


def default_database_url() -> str:
    """Build the default URL for PostgreSQL exposed on localhost."""
    password = os.getenv("POSTGRES_PASSWORD", "games_assistant")
    return (
        "postgresql://games_assistant:"
        f"{password}@localhost:5434/games_assistant"
    )


def build_rows(count: int, hours: float, seed: int) -> list[tuple[Any, ...]]:
    """Generate mock interaction rows distributed across a time window."""
    if count < 1:
        raise ValueError("count must be at least 1")
    if hours <= 0:
        raise ValueError("hours must be greater than 0")

    randomizer = random.Random(seed)
    now = datetime.now(UTC)
    start = now - timedelta(hours=hours)
    interval = timedelta(hours=hours) / max(count - 1, 1)
    rows = []

    for position in range(count):
        faq = randomizer.choice(FAQ_SAMPLES)
        if count == 1 or position == count - 1:
            created_at = now
        else:
            created_at = start + interval * position
        input_tokens = randomizer.randint(450, 1_800)
        output_tokens = randomizer.randint(35, 240)
        response_time = round(randomizer.uniform(0.7, 8.5), 3)
        feedback = randomizer.choices([1, -1, None], weights=[6, 2, 2], k=1)[0]
        feedback_at = (
            min(created_at + timedelta(minutes=randomizer.randint(1, 20)), now)
            if feedback is not None
            else None
        )
        prompt = [
            {"role": "developer", "content": INSTRUCTIONS},
            {
                "role": "user",
                "content": (
                    f"Context:\nCategory: {faq['category']}\n"
                    f"Question: {faq['question']}\nAnswer: {faq['answer']}\n\n"
                    f"User query:\n{faq['question']}"
                ),
            },
        ]
        cost = (
            input_tokens * INPUT_PRICE_PER_MILLION_TOKENS
            + output_tokens * OUTPUT_PRICE_PER_MILLION_TOKENS
        ) / 1_000_000
        rows.append(
            (
                uuid4(),
                faq["question"],
                faq["answer"],
                Jsonb(prompt),
                input_tokens,
                output_tokens,
                Decimal(str(cost)),
                response_time,
                faq["category"],
                feedback,
                created_at,
                feedback_at,
            )
        )

    return rows


def insert_rows(connection_url: str, rows: list[tuple[Any, ...]]) -> int:
    """Insert mock rows into an existing interactions table."""
    with psycopg.connect(connection_url) as connection:
        with connection.cursor() as cursor:
            cursor.executemany(
                """
                INSERT INTO interactions (
                    id, question, answer, prompt, input_tokens, output_tokens,
                    cost_usd, response_time_seconds, category, feedback,
                    created_at, feedback_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                rows,
            )
    return len(rows)


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description="Insert mock RAG metrics into the Compose PostgreSQL database"
    )
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--hours", type=float, default=DEFAULT_HOURS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--database-url",
        help="PostgreSQL URL (defaults to localhost:5434 and .env credentials)",
    )
    return parser.parse_args()


def main() -> None:
    """Generate and insert mock dashboard metrics."""
    load_dotenv()
    args = parse_args()
    connection_url = args.database_url or os.getenv("DATABASE_URL") or default_database_url()
    rows = build_rows(args.count, args.hours, args.seed)
    inserted = insert_rows(connection_url, rows)
    print(
        f"Inserted {inserted} mock interactions spanning the last "
        f"{args.hours:g} hours."
    )


if __name__ == "__main__":
    main()