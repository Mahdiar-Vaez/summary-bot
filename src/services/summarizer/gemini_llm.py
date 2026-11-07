"""Google Gemini Flash LLM Summarizer Provider."""

from __future__ import annotations

import httpx

from src.exceptions import ProviderError
from src.services.stt.base import normalize_language
from src.services.summarizer.base import SummaryResult
from src.services.summarizer.prompts import build_summary_prompt


class GeminiSummarizerProvider:
    """Summarizes transcripts using Google Gemini REST API."""

    name = "gemini"

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-3.5-flash",
        client: httpx.AsyncClient | None = None,
        timeout: float = 45.0,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self._client = client
        self._timeout = timeout

    async def summarize(
        self, transcript: str, speaker_name: str, language: str
    ) -> SummaryResult:
        norm_lang = normalize_language(language, transcript)
        system_prompt, user_prompt = build_summary_prompt(
            transcript=transcript, speaker_name=speaker_name, language=norm_lang
        )
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={self.api_key}"
        )
        payload = {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {"temperature": 0.3},
        }

        try:
            if self._client is not None:
                response = await self._client.post(url, json=payload)
            else:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    response = await client.post(url, json=payload)
        except httpx.HTTPError as exc:
            raise ProviderError(f"Gemini network error: {exc}") from exc

        if response.status_code != 200:
            raise ProviderError(
                f"Gemini HTTP {response.status_code}: {response.text[:250]}"
            )

        data = response.json()
        try:
            parts = data["candidates"][0]["content"]["parts"]
            summary_text = "".join(p.get("text", "") for p in parts).strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"Gemini unexpected response structure: {data}") from exc

        if not summary_text:
            raise ProviderError("Gemini returned empty summary")

        return SummaryResult(
            summary=summary_text, language=norm_lang, provider=self.name
        )
