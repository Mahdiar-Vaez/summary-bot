"""Google Gemini Multimodal Audio Speech-to-Text provider."""

from __future__ import annotations

import base64
from pathlib import Path
import httpx

from src.exceptions import ProviderError
from src.services.stt.base import TranscriptionResult, normalize_language

_MIME_BY_SUFFIX = {
    ".ogg": "audio/ogg",
    ".oga": "audio/ogg",
    ".mp3": "audio/mp3",
    ".wav": "audio/wav",
    ".m4a": "audio/mp4",
}


class GeminiAudioSTTProvider:
    """Transcribes audio files using Google Gemini multimodal inline_data."""

    name = "gemini-stt"

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.0-flash",
        client: httpx.AsyncClient | None = None,
        timeout: float = 90.0,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self._client = client
        self._timeout = timeout

    async def transcribe(self, audio_path: Path) -> TranscriptionResult:
        if not audio_path.exists():
            raise ProviderError(f"Audio file not found: {audio_path}")

        mime_type = _MIME_BY_SUFFIX.get(audio_path.suffix.lower(), "audio/ogg")
        encoded_audio = base64.b64encode(audio_path.read_bytes()).decode("ascii")

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={self.api_key}"
        )
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": encoded_audio,
                            }
                        },
                        {
                            "text": (
                                "Transcribe this audio accurately word-for-word in its original "
                                "spoken language (Persian or English). Output ONLY the transcription text, "
                                "with no extra commentary."
                            )
                        },
                    ],
                }
            ],
            "generationConfig": {"temperature": 0.0},
        }

        try:
            if self._client is not None:
                response = await self._client.post(url, json=payload)
            else:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    response = await client.post(url, json=payload)
        except httpx.HTTPError as exc:
            raise ProviderError(f"Gemini STT network error: {exc}") from exc

        if response.status_code != 200:
            raise ProviderError(
                f"Gemini STT HTTP {response.status_code}: {response.text[:250]}"
            )

        data = response.json()
        try:
            parts = data["candidates"][0]["content"]["parts"]
            text = "".join(p.get("text", "") for p in parts).strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"Gemini STT unexpected response: {data}") from exc

        if not text:
            raise ProviderError("Gemini STT returned empty text")

        language = normalize_language(None, text)
        return TranscriptionResult(text=text, language=language, provider=self.name)
