"""Text-to-Speech Fallback Chain Manager."""

from __future__ import annotations

import logging
from typing import Sequence

from src.exceptions import ProviderChainExhaustedError, ProviderError
from src.services.http_utils import redact_secrets
from src.services.tts.base import TTSProvider, TTSResult

logger = logging.getLogger(__name__)


class TTSManager:
    """Orchestrates TTS providers with automatic failover."""

    def __init__(self, providers: Sequence[TTSProvider]) -> None:
        self.providers = list(providers)

    async def synthesize(self, text: str, language: str) -> TTSResult:
        """Try each TTSProvider in order until one returns non-empty audio bytes."""
        errors: list[str] = []
        for provider in self.providers:
            try:
                result = await provider.synthesize(text=text, language=language)
                if not result.audio_bytes:
                    raise ProviderError("Provider returned empty audio bytes")
                return result
            except Exception as exc:  # noqa: BLE001
                safe_exc = redact_secrets(str(exc))
                err_msg = f"{provider.name}: {safe_exc}"
                logger.warning("TTS provider %s failed: %s", provider.name, safe_exc)
                errors.append(err_msg)

        raise ProviderChainExhaustedError(errors)
