from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

from openai import OpenAI


DEFAULT_INSTRUCTIONS = """You answer questions about the GeForce NOW service.
Use only the information in the provided context to answer the user's query.
If the answer cannot be found in the context, answer exactly: I don't know.
Keep the answer concise and factual."""

SearchFunction = Callable[..., Iterable[Mapping[str, Any]]]
Prompt = list[dict[str, str]]


class RAG:
    """Run retrieval-augmented generation for GeForce NOW FAQ questions."""

    def __init__(
        self,
        index_name: str,
        search_function: SearchFunction,
        llm_client: OpenAI,
        instructions: str = DEFAULT_INSTRUCTIONS,
        model_name: str = "gpt-5.4-mini",
    ) -> None:
        self.search_function = search_function
        self.index_name = index_name
        self.instructions = instructions
        self.llm_client = llm_client
        self.model_name = model_name

    def retrieve(self, query: str, **kwargs: Any) -> list[Mapping[str, Any]]:
        """Retrieve FAQ records using the configured index and search options."""
        return list(self.search_function(query=query, index_name=self.index_name, **kwargs))

    def augment(
        self,
        query: str,
        retrieved_records: Iterable[Mapping[str, Any]],
    ) -> Prompt:
        """Build developer and user messages for the LLM."""
        context = self._format_context(retrieved_records)
        return [
            {"role": "developer", "content": self.instructions},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nUser query:\n{query}",
            },
        ]

    def generate(self, prompt: Prompt) -> str:
        """Generate an answer by passing the augmented prompt to the LLM."""
        return self.llm(prompt)

    def llm(self, prompt: Prompt) -> str:
        """Call the configured OpenAI model with an augmented prompt."""
        client = self.llm_client or OpenAI()
        response = client.responses.create(
            model=self.model_name,
            input=prompt,
        )
        return response.output_text

    def run(self, query: str, **kwargs: Any) -> str:
        """Run retrieval, augmentation, and generation for a user query."""
        records = self.retrieve(query, **kwargs)
        prompt = self.augment(query, records)
        return self.generate(prompt)

    @staticmethod
    def _format_context(records: Iterable[Mapping[str, Any]]) -> str:
        entries = []
        for position, record in enumerate(records, start=1):
            faq = record.get("_source", record)
            entries.append(
                f"{position}. Category: {faq.get('category', '')}\n"
                f"   Question: {faq.get('question', '')}\n"
                f"   Answer: {faq.get('answer', '')}"
            )
        return "\n\n".join(entries) or "No FAQ context was retrieved."