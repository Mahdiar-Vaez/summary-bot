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


_HTML_TAG_RE = re.compile(
    r"</?(?:b|strong|i|em|u|ins|s|strike|del|code|pre|a|blockquote|tg-spoiler|tg-emoji)(?:\s[^>]*)?>",
    flags=re.IGNORECASE,
)
_URL_RE = re.compile(r"https?://\S+")


def clean_text_for_tts(text: str) -> str:
    """Strip headers, known HTML tags, URLs, and markdown formatting so TTS reads smoothly."""
    cleaned = _HTML_TAG_RE.sub("", text or "")
    cleaned = _URL_RE.sub("", cleaned)
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if len(lines) > 1:
        first = lines[0]
        is_header = (
            first.startswith("📌")
            or ("خلاصه" in first and first.endswith(":"))
            or first.lower().startswith("summary")
        )
        if is_header:
            lines = lines[1:]

    cleaned = "\n".join(lines)
    cleaned = cleaned.replace("📌", "").replace("•", "، ")
    cleaned = re.sub(r"[*_`~#]+", "", cleaned)
    cleaned = re.sub(r"\n+", ". ", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip(" .،")
