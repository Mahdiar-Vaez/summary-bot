"""Groq Whisper-large-v3 Speech-to-Text provider."""

from __future__ import annotations

from pathlib import Path
import httpx

from src.exceptions import ProviderError
from src.services.stt.base import TranscriptionResult, normalize_language

GROQ_AUDIO_URL = "https://api.groq.com/openai/v1/audio/transcriptions"


class GroqWhisperProvider:
    """Transcribes audio files using Groq's ultra-fast Whisper API."""

    name = "groq"

    def __init__(
        self,
        api_key: str,
        model: str = "whisper-large-v3",
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

        headers = {"Authorization": f"Bearer {self.api_key}"}
        data = {
            "model": self.model,
            "response_format": "verbose_json",
            "temperature": "0",
        }

        audio_bytes = audio_path.read_bytes()
        files = {"file": (audio_path.name, audio_bytes, "audio/ogg")}

        try:
            if self._client is not None:
                response = await self._client.post(
                    GROQ_AUDIO_URL, headers=headers, data=data, files=files
                )
            else:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    response = await client.post(
                        GROQ_AUDIO_URL, headers=headers, data=data, files=files
                    )
        except httpx.HTTPError as exc:
            raise ProviderError(f"Groq network error: {exc}") from exc

        if response.status_code != 200:
            raise ProviderError(
                f"Groq STT HTTP {response.status_code}: {response.text[:250]}"
            )

        payload = response.json()
        text = (payload.get("text") or "").strip()
        raw_lang = payload.get("language") or ""
        if not text:
            raise ProviderError("Groq STT returned empty text")

        language = normalize_language(raw_lang, text)
        return TranscriptionResult(text=text, language=language, provider=self.name)
