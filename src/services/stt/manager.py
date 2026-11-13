"""Speech-to-Text Fallback Chain Manager."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

from src.exceptions import ProviderChainExhaustedError, ProviderError
from src.services.http_utils import redact_secrets
from src.services.stt.base import STTProvider, TranscriptionResult

logger = logging.getLogger(__name__)


class STTManager:
    """Orchestrates STT providers with automatic failover."""

    def __init__(self, providers: Sequence[STTProvider]) -> None:
        self.providers = list(providers)

    async def transcribe(self, audio_path: Path) -> TranscriptionResult:
        """Try each STT provider in order until one succeeds with non-empty text."""
        errors: list[str] = []
        for provider in self.providers:
            try:
                result = await provider.transcribe(audio_path)
                cleaned = result.text.strip()
                if not cleaned:
                    raise ProviderError("Provider returned whitespace-only transcript")
                return TranscriptionResult(
                    text=cleaned,
                    language=result.language,
                    provider=result.provider,
                )
            except Exception as exc:  # noqa: BLE001
                safe_exc = redact_secrets(str(exc))
                err_msg = f"{provider.name}: {safe_exc}"
                logger.warning("STT provider %s failed: %s", provider.name, safe_exc)
                errors.append(err_msg)

        raise ProviderChainExhaustedError(errors)
