"""Cloudflare Workers AI Whisper Speech-to-Text fallback provider."""

from __future__ import annotations

from pathlib import Path
import httpx

from src.exceptions import ProviderError
from src.services.stt.base import TranscriptionResult, normalize_language


class CloudflareWhisperProvider:
    """Transcribes audio files using Cloudflare Workers AI (@cf/openai/whisper)."""

    name = "cloudflare"

    def __init__(
        self,
        account_id: str,
        api_token: str,
        model: str = "@cf/openai/whisper",
        client: httpx.AsyncClient | None = None,
        timeout: float = 90.0,
    ) -> None:
        self.account_id = account_id
        self.api_token = api_token
        self.model = model
        self._client = client
        self._timeout = timeout

    async def transcribe(self, audio_path: Path) -> TranscriptionResult:
        if not audio_path.exists():
            raise ProviderError(f"Audio file not found: {audio_path}")

        url = f"https://api.cloudflare.com/client/v4/accounts/{self.account_id}/ai/run/{self.model}"
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/octet-stream",
        }
        audio_bytes = audio_path.read_bytes()

        try:
            if self._client is not None:
                response = await self._client.post(url, headers=headers, content=audio_bytes)
            else:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    response = await client.post(url, headers=headers, content=audio_bytes)
        except httpx.HTTPError as exc:
            raise ProviderError(f"Cloudflare STT network error: {exc}") from exc

        if response.status_code != 200:
            raise ProviderError(
                f"Cloudflare STT HTTP {response.status_code}: {response.text[:250]}"
            )

        payload = response.json()
        result_obj = payload.get("result") or {}
        text = (result_obj.get("text") or "").strip()
        if not text:
            raise ProviderError("Cloudflare STT returned empty text")

        language = normalize_language(result_obj.get("language"), text)
        return TranscriptionResult(text=text, language=language, provider=self.name)
