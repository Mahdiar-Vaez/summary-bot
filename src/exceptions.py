"""Custom exception hierarchy for Telegram Voice Summary Bot."""

from __future__ import annotations


class VoiceBotError(Exception):
    """Base exception for all bot errors."""


class ProviderError(VoiceBotError):
    """Raised when a single STT or LLM provider fails."""


class ProviderChainExhaustedError(VoiceBotError):
    """Raised when all configured providers in a fallback chain fail."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        joined = "; ".join(errors) if errors else "no providers configured"
        super().__init__(f"All providers failed: {joined}")


class FileTooLargeError(VoiceBotError):
    """Raised when an audio file exceeds the allowed download limit."""
