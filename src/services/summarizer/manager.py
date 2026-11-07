"""LLM Summarizer Fallback Chain Manager."""

from __future__ import annotations

import logging
from typing import Sequence

from src.exceptions import ProviderChainExhaustedError, ProviderError
from src.services.summarizer.base import SummarizerProvider, SummaryResult

logger = logging.getLogger(__name__)


class SummarizerManager:
    """Orchestrates LLM summarizer providers with automatic failover."""

    def __init__(self, providers: Sequence[SummarizerProvider]) -> None:
        self.providers = list(providers)

    async def summarize(
        self, transcript: str, speaker_name: str, language: str
    ) -> SummaryResult:
        """Try each SummarizerProvider in order until one returns a non-empty summary."""
        errors: list[str] = []
        for provider in self.providers:
            try:
                result = await provider.summarize(
                    transcript=transcript,
                    speaker_name=speaker_name,
                    language=language,
                )
                cleaned = result.summary.strip()
                if not cleaned:
                    raise ProviderError("Provider returned whitespace-only summary")
                return SummaryResult(
                    summary=cleaned,
                    language=result.language,
                    provider=result.provider,
                )
            except Exception as exc:  # noqa: BLE001
                err_msg = f"{provider.name}: {exc}"
                logger.warning("Summarizer provider %s failed: %s", provider.name, exc)
                errors.append(err_msg)

        raise ProviderChainExhaustedError(errors)
