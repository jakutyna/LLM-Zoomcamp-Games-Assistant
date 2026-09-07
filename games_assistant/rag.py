from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from openai import OpenAI


DEFAULT_INSTRUCTIONS = """You answer questions about the GeForce NOW service.
Use only the information in the provided context to answer the user's query.
If the answer cannot be found in the context, answer exactly:
The answer for your question was not found in FAQ database.
Keep the answer concise and factual."""

SearchFunction = Callable[..., Iterable[Mapping[str, Any]]]
Prompt = list[dict[str, str]]


@dataclass(frozen=True)
class TokenUsage:
    """Token counts reported by the OpenAI Responses API."""

    input_tokens: int
    output_tokens: int

    @property
    def total_tokens(self) -> int:
        """Return the total number of input and output tokens."""
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True)
class RAGResponse:
    """Generated answer and its token usage."""

    answer: str
    usage: TokenUsage


class RAG:
    """Run retrieval-augmented generation for GeForce NOW FAQ questions.

    The class coordinates three steps: retrieving FAQ records with a caller-
    supplied search function, building a role-separated prompt, and generating
    an answer with an OpenAI client.
    """

    def __init__(
        self,
        index_name: str,
        search_function: SearchFunction,
        llm_client: OpenAI,
        instructions: str = DEFAULT_INSTRUCTIONS,
        model_name: str = "gpt-5.4-mini",
    ) -> None:
        """Initialize a RAG workflow for one FAQ index.

        Args:
            index_name: FAQ index passed to the search function on every query.
            search_function: Callable that accepts ``query``, ``index_name``,
                and any additional keyword search options, then returns FAQ
                records or Elasticsearch hit objects.
            llm_client: OpenAI-compatible client exposing
                ``responses.create``.
            instructions: Developer-level instructions that guide the model's
                answers.
            model_name: OpenAI model used for answer generation.
        """
        self.search_function = search_function
        self.index_name = index_name
        self.instructions = instructions
        self.llm_client = llm_client
        self.model_name = model_name
        self.last_token_usage: TokenUsage | None = None

    def retrieve(self, query: str, **kwargs: Any) -> list[Mapping[str, Any]]:
        """Retrieve FAQ records for a query.

        Args:
            query: User question to pass to the search function.
            **kwargs: Additional search options, such as ``size``, ``category``,
                or ``tag``.

        Returns:
            A list of records returned by the configured search function.
        """
        return list(self.search_function(query=query, index_name=self.index_name, **kwargs))

    def augment(
        self,
        query: str,
        retrieved_records: Iterable[Mapping[str, Any]],
    ) -> Prompt:
        """Build the developer and user messages sent to the LLM.

        Args:
            query: User question to include in the user message.
            retrieved_records: FAQ records to format as context. Records may
                be raw FAQ mappings or Elasticsearch hits containing an
                ``_source`` mapping.

        Returns:
            Two prompt messages: developer instructions followed by the user
            message containing the retrieved context and query.
        """
        context = self._format_context(retrieved_records)
        return [
            {"role": "developer", "content": self.instructions},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nUser query:\n{query}",
            },
        ]

    def generate(self, prompt: Prompt) -> str:
        """Generate an answer from an augmented prompt.

        Args:
            prompt: Developer and user messages produced by :meth:`augment`.

        Returns:
            The text returned by the configured language model.
        """
        return self.llm(prompt)

    def llm(self, prompt: Prompt) -> str:
        """Call the configured OpenAI model with an augmented prompt.

        Args:
            prompt: Role-separated messages passed to the OpenAI Responses API.

        Returns:
            The model's generated response text.
        """
        return self._llm_response(prompt).answer

    def run(self, query: str, **kwargs: Any) -> str:
        """Run the complete retrieval, augmentation, and generation workflow.

        Args:
            query: User question to retrieve context for and answer.
            **kwargs: Additional search options forwarded to :meth:`retrieve`.

        Returns:
            A generated answer grounded in the retrieved FAQ context.
        """
        records = self.retrieve(query, **kwargs)
        prompt = self.augment(query, records)
        return self.generate(prompt)

    def run_with_usage(self, query: str, **kwargs: Any) -> RAGResponse:
        """Run the RAG workflow and return the answer with token usage."""
        records = self.retrieve(query, **kwargs)
        prompt = self.augment(query, records)
        return self._llm_response(prompt)

    def _llm_response(self, prompt: Prompt) -> RAGResponse:
        client = self.llm_client or OpenAI()
        response = client.responses.create(
            model=self.model_name,
            input=prompt,
        )
        if response.usage is None:
            raise RuntimeError("The LLM response did not include token usage")

        usage = TokenUsage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
        self.last_token_usage = usage
        return RAGResponse(answer=response.output_text, usage=usage)

    @staticmethod
    def _format_context(records: Iterable[Mapping[str, Any]]) -> str:
        """Format FAQ records as numbered category, question, and answer text.

        Args:
            records: Raw FAQ mappings or Elasticsearch hits with FAQ data in
                ``_source``.

        Returns:
            Formatted FAQ context, or a placeholder when no records exist.
        """
        entries = []
        for position, record in enumerate(records, start=1):
            faq = record.get("_source", record)
            entries.append(
                f"{position}. Category: {faq.get('category', '')}\n"
                f"   Question: {faq.get('question', '')}\n"
                f"   Answer: {faq.get('answer', '')}"
            )
        return "\n\n".join(entries) or "No FAQ context was retrieved."