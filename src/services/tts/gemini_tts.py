"""Google Gemini 3.8 / 3.1 Flash Native TTS Provider (natural Persian & English speech)."""

from __future__ import annotations

import base64
import io
import wave
import httpx

from src.exceptions import ProviderError
from src.services.stt.base import normalize_language
from src.services.tts.base import TTSResult, clean_text_for_tts


def _ensure_wav_bytes(raw_bytes: bytes, mime_type: str) -> bytes:
    """Wrap raw PCM L16 audio in a standard 24kHz 16-bit mono WAV header if needed."""
    if raw_bytes.startswith(b"RIFF") or "wav" in (mime_type or "").lower():
        return raw_bytes
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)
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
            f"{self.model}:generateContent?key={self.api_key}"
        )
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
                response = await self._client.post(url, json=payload)
            else:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    response = await client.post(url, json=payload)
        except httpx.HTTPError as exc:
            raise ProviderError(f"Gemini TTS network error: {exc}") from exc

        if response.status_code != 200:
            raise ProviderError(
                f"Gemini TTS HTTP {response.status_code}: {response.text[:200]}"
            )

        data = response.json()
        try:
            inline_data = data["candidates"][0]["content"]["parts"][0]["inlineData"]
            b64_audio = inline_data["data"]
            mime_type = inline_data.get("mimeType", "audio/wav")
            raw_audio = base64.b64decode(b64_audio)
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ProviderError(f"Gemini TTS unexpected response: {data}") from exc

        if not raw_audio:
            raise ProviderError("Gemini TTS returned empty audio")

        wav_bytes = _ensure_wav_bytes(raw_audio, mime_type)
        return TTSResult(
            audio_bytes=wav_bytes,
            language=norm_lang,
            provider=self.name,
        )
