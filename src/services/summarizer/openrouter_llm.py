"""OpenRouter Free LLM Summarizer Fallback Provider."""

from __future__ import annotations

import httpx

from src.exceptions import ProviderError
from src.services.stt.base import normalize_language
from src.services.summarizer.base import SummaryResult
from src.services.summarizer.prompts import build_summary_prompt

OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"


class OpenRouterSummarizerProvider:
    """Summarizes transcripts using OpenRouter free-tier models."""

    name = "openrouter-llm"

    def __init__(
        self,
        api_key: str,
        model: str = "meta-llama/llama-3.3-70b-instruct:free",
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
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.3,
        }

        try:
            if self._client is not None:
                response = await self._client.post(
                    OPENROUTER_CHAT_URL, headers=headers, json=payload
                )
            else:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    response = await client.post(
                        OPENROUTER_CHAT_URL, headers=headers, json=payload
                    )
        except httpx.HTTPError as exc:
            raise ProviderError(f"OpenRouter network error: {exc}") from exc

        if response.status_code != 200:
            raise ProviderError(
                f"OpenRouter HTTP {response.status_code}: {response.text[:250]}"
            )

        data = response.json()
        try:
            summary_text = (
                data["choices"][0]["message"]["content"] or ""
            ).strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"OpenRouter unexpected response: {data}") from exc

        if not summary_text:
            raise ProviderError("OpenRouter returned empty summary")

        return SummaryResult(
            summary=summary_text, language=norm_lang, provider=self.name
        )
