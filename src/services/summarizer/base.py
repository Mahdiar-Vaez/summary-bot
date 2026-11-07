"""Base types and protocols for LLM summarizer providers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(slots=True)
class SummaryResult:
    """Result of summarizing a voice transcript."""

    summary: str
    language: str
    provider: str


class SummarizerProvider(Protocol):
    """Protocol implemented by all LLM summarizer providers."""

    name: str

    async def summarize(
        self, transcript: str, speaker_name: str, language: str
    ) -> SummaryResult:
        """Summarize transcript with speaker attribution in the target language."""
