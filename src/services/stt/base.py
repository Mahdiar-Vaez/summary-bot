"""Base interfaces and helpers for Speech-to-Text providers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(slots=True)
class TranscriptionResult:
    """Result of an audio transcription."""

    text: str
    language: str
    provider: str


class STTProvider(Protocol):
    """Protocol implemented by all Speech-to-Text providers."""

    name: str

    async def transcribe(self, audio_path: Path) -> TranscriptionResult:
        """Transcribe an audio file into text and detected language."""


def normalize_language(raw_lang: str | None, text: str) -> str:
    """Normalize detected language code to 'fa' (Persian) or 'en' (English)."""
    cleaned = (raw_lang or "").strip().lower()
    if cleaned in {"fa", "pes", "persian", "farsi", "fa-ir", "fa_ir"}:
        return "fa"

    letters = [ch for ch in text if ch.isalpha()]
    if letters:
        persian_count = sum(1 for ch in letters if "\u0600" <= ch <= "\u06ff")
        if persian_count / len(letters) > 0.2:
            return "fa"

    return "en"
