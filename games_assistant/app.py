from __future__ import annotations

import csv
import time
from functools import partial
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import streamlit as st
from elasticsearch import Elasticsearch
from openai import OpenAI

from constants import BEST_BOOST_PARAMS, DEFAULT_ES_URL, DEFAULT_INDEX
from database import create_tables, save_feedback, save_interaction
from evaluation_utils import calculate_prompt_cost
from faq_hybrid_search import search_faq
from ingestion import DEFAULT_CSV_PATH, ingest_default_indexes
from rag import RAG


ALL_CATEGORIES = "All FAQ categories"
MODEL_NAME = "gpt-5.4-mini"


@st.cache_resource(show_spinner="Preparing the FAQ search indexes...")
def initialize_services() -> tuple[Elasticsearch, RAG]:
    """Initialize external services once per Streamlit process."""
    client = Elasticsearch(DEFAULT_ES_URL)
    if not client.ping():
        raise RuntimeError(f"Cannot connect to Elasticsearch at {DEFAULT_ES_URL}")

    ingest_default_indexes(client)
    create_tables()
    hybrid_search = partial(search_faq, client=client)
    return client, RAG(DEFAULT_INDEX, hybrid_search, OpenAI(), model_name=MODEL_NAME)


@st.cache_data
def faq_categories(csv_path: Path = DEFAULT_CSV_PATH) -> list[str]:
    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        return sorted({row["category"] for row in csv.DictReader(csv_file)})


def render_feedback(interaction_id: UUID, current_rating: int | None) -> None:
    positive, negative, status = st.columns([0.12, 0.12, 0.76])
    if positive.button(
        "👍",
        key=f"up-{interaction_id}",
        type="primary" if current_rating == 1 else "secondary",
        help="Helpful",
    ):
        save_feedback(interaction_id, 1)
        st.session_state.ratings[str(interaction_id)] = 1
        st.rerun()
    if negative.button(
        "👎",
        key=f"down-{interaction_id}",
        type="primary" if current_rating == -1 else "secondary",
        help="Not helpful",
    ):
        save_feedback(interaction_id, -1)
        st.session_state.ratings[str(interaction_id)] = -1
        st.rerun()
    if current_rating is not None:
        status.caption("Thanks for your feedback.")


def render_answer(message: dict[str, Any]) -> None:
    with st.chat_message("assistant"):
        st.markdown(message["content"])
        st.caption(
            f"Call cost: ${message['cost_usd']:.6f} · "
            f"Response time: {message['response_time_seconds']:.2f}s"
        )
        interaction_id = UUID(message["interaction_id"])
        render_feedback(
            interaction_id,
            st.session_state.ratings.get(str(interaction_id)),
        )


def answer_question(rag: RAG, question: str, category: str | None) -> dict[str, Any]:
    started_at = time.perf_counter()
    records = rag.retrieve(
        question,
        size=7,
        rank_constant=1,
        boost_dict=BEST_BOOST_PARAMS,
        category=category,
    )
    prompt = rag.augment(question, records)
    answer = rag.generate(prompt)
    response_time = time.perf_counter() - started_at
    usage = rag.last_token_usage
    if usage is None:
        raise RuntimeError("The LLM response did not include token usage")

    cost = calculate_prompt_cost(MODEL_NAME, usage.input_tokens, usage.output_tokens)
    interaction_id = uuid4()
    save_interaction(
        interaction_id=interaction_id,
        question=question,
        answer=answer,
        prompt=prompt,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cost_usd=cost,
        response_time_seconds=response_time,
        category=category,
    )
    return {
        "role": "assistant",
        "content": answer,
        "cost_usd": cost,
        "response_time_seconds": response_time,
        "interaction_id": str(interaction_id),
    }


def main() -> None:
    st.set_page_config(page_title="GeForce NOW Assistant", page_icon="🎮")
    st.title("GeForce NOW Assistant")

    try:
        _, rag = initialize_services()
    except Exception as error:
        st.error(f"Application startup failed: {error}")
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "ratings" not in st.session_state:
        st.session_state.ratings = {}

    selected_category = st.selectbox(
        "Search scope",
        [ALL_CATEGORIES, *faq_categories()],
    )
    category = None if selected_category == ALL_CATEGORIES else selected_category

    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.markdown(message["content"])
        else:
            render_answer(message)

    question = st.chat_input("Ask about GeForce NOW")
    if not question:
        return

    user_message = {"role": "user", "content": question}
    st.session_state.messages.append(user_message)
    with st.chat_message("user"):
        st.markdown(question)

    try:
        with st.spinner("Searching the FAQ and preparing an answer..."):
            assistant_message = answer_question(rag, question, category)
    except Exception as error:
        st.error(f"Could not answer the question: {error}")
        return

    st.session_state.messages.append(assistant_message)
    render_answer(assistant_message)


if __name__ == "__main__":
    main()