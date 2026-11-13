"""LLM Summarizer and Chat Q&A Fallback Chain Manager."""

from __future__ import annotations

import inspect
import logging
from typing import Callable, Awaitable, Sequence

from src.exceptions import ProviderChainExhaustedError, ProviderError
from src.services.http_utils import redact_secrets
from src.services.summarizer.base import SummarizerProvider, SummaryResult

logger = logging.getLogger(__name__)


class SummarizerManager:
    """Orchestrates LLM summarizer & chat providers with automatic failover."""

    def __init__(self, providers: Sequence[SummarizerProvider]) -> None:
        self.providers = list(providers)

    async def _run_chain(
        self, caller: Callable[[SummarizerProvider], Awaitable[SummaryResult]]
    ) -> SummaryResult:
        errors: list[str] = []
        for provider in self.providers:
            try:
                result = await caller(provider)
                cleaned = result.summary.strip()
                if not cleaned:
                    raise ProviderError("Provider returned whitespace-only response")
                return SummaryResult(
                    summary=cleaned,
                    language=result.language,
                    provider=result.provider,
                )
            except Exception as exc:  # noqa: BLE001
                safe_exc = redact_secrets(str(exc))
                err_msg = f"{provider.name}: {safe_exc}"
                logger.warning("LLM provider %s failed: %s", provider.name, safe_exc)
                errors.append(err_msg)

        raise ProviderChainExhaustedError(errors)

    async def summarize(
        self, transcript: str, speaker_name: str, language: str
    ) -> SummaryResult:
        """Try each provider in order until one returns a non-empty summary."""
        return await self._run_chain(
            lambda p: p.summarize(
                transcript=transcript,
                speaker_name=speaker_name,
                language=language,
            )
        )

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
        """Try each provider in order to generate an isolated chat Q&A or personality response."""

        def _invoke(p: SummarizerProvider) -> Awaitable[SummaryResult]:
            sig = inspect.signature(p.chat)
            accepts_web = "web_context" in sig.parameters or any(
                param.kind == inspect.Parameter.VAR_KEYWORD
                for param in sig.parameters.values()
            )
            if accepts_web:
                return p.chat(
                    chat_history=chat_history,
                    session_turns=session_turns,
                    question=question,
                    asker_name=asker_name,
                    mode=mode,
                    language=language,
                    web_context=web_context,
                )
            effective_q = (
                f"{question}\n\n[نتایج جست‌وجوی وب]:\n{web_context}"
                if web_context
                else question
            )
            return p.chat(
                chat_history=chat_history,
                session_turns=session_turns,
                question=effective_q,
                asker_name=asker_name,
                mode=mode,
                language=language,
            )

        return await self._run_chain(_invoke)
