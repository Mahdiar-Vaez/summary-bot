"""Base types, protocols, and text cleaners for Text-to-Speech providers."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Protocol


@dataclass(slots=True)
class TTSResult:
    """Result of synthesizing summary text into audio."""

    audio_bytes: bytes
    language: str
    provider: str


class TTSProvider(Protocol):
    """Protocol implemented by all Text-to-Speech providers."""

    name: str

    async def synthesize(self, text: str, language: str) -> TTSResult:
        """Synthesize text into audio bytes in the specified language ('fa' or 'en')."""


def clean_text_for_tts(text: str) -> str:
    """Strip headers, HTML tags, markdown formatting, and emojis so TTS reads smoothly."""
    cleaned = re.sub(r"<[^>]+>", "", text or "")
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if lines and (
        "📌" in lines[0]
        or "خلاصه" in lines[0]
        or lines[0].lower().startswith("summary")
    ):
        lines = lines[1:]

    cleaned = "\n".join(lines)
    cleaned = cleaned.replace("📌", "").replace("•", "، ")
    cleaned = re.sub(r"[*_`~#>]+", "", cleaned)
    cleaned = re.sub(r"\n+", ". ", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip(" .،")
