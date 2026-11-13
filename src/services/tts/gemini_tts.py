"""Google Gemini 3.8 / 3.1 Flash Native TTS Provider (natural Persian & English speech)."""

from __future__ import annotations

import io
import re
import wave
import httpx

from src.exceptions import ProviderError
from src.services.http_utils import (
    extract_gemini_inline_audio,
    parse_json_response,
    redact_secrets,
)
from src.services.stt.base import normalize_language
from src.services.tts.base import TTSResult, clean_text_for_tts

_RATE_RE = re.compile(r"rate=(\d+)", re.IGNORECASE)
_CHANNELS_RE = re.compile(r"channels=(\d+)", re.IGNORECASE)
_COMPRESSED_AUDIO_TYPES = ("wav", "mp3", "mpeg", "ogg", "opus", "aac", "flac", "webm")


def _ensure_wav_bytes(raw_bytes: bytes, mime_type: str) -> bytes:
    """Wrap raw PCM L16 audio in a WAV header without corrupting compressed audio formats."""
    lower_mime = (mime_type or "").lower()
    if raw_bytes.startswith((b"RIFF", b"ID3", b"OggS", b"fLaC")) or any(
        fmt in lower_mime for fmt in _COMPRESSED_AUDIO_TYPES
    ):
        return raw_bytes

    rate_match = _RATE_RE.search(lower_mime)
    chan_match = _CHANNELS_RE.search(lower_mime)
    sample_rate = int(rate_match.group(1)) if rate_match else 24000
    channels = int(chan_match.group(1)) if chan_match else 1

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(raw_bytes)
    return buf.getvalue()


class GeminiTTSProvider:
    """Synthesizes natural conversational Persian and English audio using Gemini TTS models."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-3.8-flash-tts",
        voice_name: str = "Kore",
        client: httpx.AsyncClient | None = None,
        timeout: float = 35.0,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.voice_name = voice_name
        self.name = f"gemini-tts:{model}"
        self._client = client
        self._timeout = timeout

    async def synthesize(self, text: str, language: str) -> TTSResult:
        norm_lang = normalize_language(language, text)
        spoken_text = clean_text_for_tts(text)
        if not spoken_text:
            raise ProviderError("Cannot synthesize empty text")

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent"
        )
        headers = {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "contents": [{"parts": [{"text": spoken_text}]}],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {
                    "voiceConfig": {
                        "prebuiltVoiceConfig": {"voiceName": self.voice_name}
                    }
                },
            },
        }

        try:
            if self._client is not None:
                response = await self._client.post(url, headers=headers, json=payload)
            else:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    response = await client.post(url, headers=headers, json=payload)
        except httpx.HTTPError as exc:
            raise ProviderError(f"Gemini TTS network error: {redact_secrets(str(exc))}") from exc

        if response.status_code != 200:
            raise ProviderError(
                redact_secrets(f"Gemini TTS HTTP {response.status_code}: {response.text[:200]}")
            )

        data = parse_json_response(response, "Gemini TTS")
        raw_audio, mime_type = extract_gemini_inline_audio(data, "Gemini TTS")

        if not raw_audio:
            raise ProviderError("Gemini TTS returned empty audio")

        wav_bytes = _ensure_wav_bytes(raw_audio, mime_type)
        return TTSResult(
            audio_bytes=wav_bytes,
            language=norm_lang,
            provider=self.name,
        )
