"""Base types and protocols for LLM summarizer and interactive chat providers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(slots=True)
class SummaryResult:
    """Result of summarizing a transcript or answering a chat query."""

    summary: str
    language: str
    provider: str


class SummarizerProvider(Protocol):
    """Protocol implemented by all LLM providers."""

    name: str

    async def summarize(
        self, transcript: str, speaker_name: str, language: str
    ) -> SummaryResult:
        """Summarize transcript with speaker attribution in the target language."""

    async def chat(
        self,
        chat_history: str,
        session_turns: list[tuple[str, str]],
        question: str,
        asker_name: str,
        mode: str = "chat",
        language: str = "fa",
        web_context: str | None = None,
    ) -> SummaryResult:
        """Generate a conversational reply grounded in chat history and optional live web context."""
