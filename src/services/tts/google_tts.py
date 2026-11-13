"""Google Translate HTTP TTS fallback provider."""

from __future__ import annotations

from urllib.parse import quote
import httpx

from src.exceptions import ProviderError
from src.services.stt.base import normalize_language
from src.services.tts.base import TTSResult, clean_text_for_tts


def _split_tts_chunks(text: str, max_chars: int = 180) -> list[str]:
    words = text.split()
    chunks: list[str] = []
    current = ""
    for word in words:
        if len(word) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            for i in range(0, len(word), max_chars):
                part = word[i : i + max_chars]
                if len(part) == max_chars:
                    chunks.append(part)
                else:
                    current = part
            continue
        candidate = f"{current} {word}".strip() if current else word
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = word
    if current:
        chunks.append(current)
    return chunks


class GoogleTranslateTTSProvider:
    """Fallback TTS provider using Google Translate's TTS endpoint."""

    name = "google-tts"

    def __init__(
        self,
        client: httpx.AsyncClient | None = None,
        timeout: float = 30.0,
    ) -> None:
        self._client = client
        self._timeout = timeout

    @staticmethod
    def _validate_audio_response(resp: httpx.Response) -> None:
        if resp.status_code != 200 or not resp.content:
            raise ProviderError(f"Google TTS HTTP {resp.status_code}")
        ctype = (resp.headers.get("content-type") or "").lower()
        if ctype and not ctype.startswith(("audio/", "application/octet-stream")):
            raise ProviderError(f"Google TTS returned non-audio content-type: {ctype}")

    async def synthesize(self, text: str, language: str) -> TTSResult:
        norm_lang = normalize_language(language, text)
        spoken_text = clean_text_for_tts(text)
        if not spoken_text:
            raise ProviderError("Cannot synthesize empty text")

        chunks = _split_tts_chunks(spoken_text, max_chars=180)
        audio_parts: list[bytes] = []

        try:
            if self._client is not None:
                for chunk in chunks:
                    url = (
                        f"https://translate.google.com/translate_tts"
                        f"?ie=UTF-8&client=tw-ob&tl={norm_lang}&q={quote(chunk)}"
                    )
                    resp = await self._client.get(url)
                    self._validate_audio_response(resp)
                    audio_parts.append(resp.content)
            else:
                async with httpx.AsyncClient(
                    timeout=self._timeout, follow_redirects=True
                ) as client:
                    for chunk in chunks:
                        url = (
                            f"https://translate.google.com/translate_tts"
                            f"?ie=UTF-8&client=tw-ob&tl={norm_lang}&q={quote(chunk)}"
                        )
                        resp = await client.get(url)
                        self._validate_audio_response(resp)
                        audio_parts.append(resp.content)
        except httpx.HTTPError as exc:
            raise ProviderError(f"Google TTS network error: {exc}") from exc

        audio_bytes = b"".join(audio_parts)
        if not audio_bytes:
            raise ProviderError("Google TTS returned empty audio")

        return TTSResult(audio_bytes=audio_bytes, language=norm_lang, provider=self.name)
