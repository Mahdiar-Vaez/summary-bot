"""OpenRouter Free LLM Summarizer & Chat Q&A Fallback Provider."""

from __future__ import annotations

import httpx

from src.exceptions import ProviderError
from src.services.stt.base import normalize_language
from src.services.summarizer.base import SummaryResult
from src.services.summarizer.prompts import build_chat_qa_prompt, build_summary_prompt

OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"


class OpenRouterSummarizerProvider:
    """Summarizes transcripts and handles chat Q&A using OpenRouter free-tier models."""

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

    async def _generate(
        self,
        system_prompt: str,
        user_prompt: str,
        norm_lang: str,
        temperature: float = 0.3,
        max_tokens: int | None = None,
    ) -> SummaryResult:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, object] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

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
            output_text = (
                data["choices"][0]["message"]["content"] or ""
            ).strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"OpenRouter unexpected response: {data}") from exc

        if not output_text:
            raise ProviderError("OpenRouter returned empty response")

        return SummaryResult(
            summary=output_text, language=norm_lang, provider=self.name
        )

    async def summarize(
        self, transcript: str, speaker_name: str, language: str
    ) -> SummaryResult:
        norm_lang = normalize_language(language, transcript)
        system_prompt, user_prompt = build_summary_prompt(
            transcript=transcript, speaker_name=speaker_name, language=norm_lang
        )
        return await self._generate(system_prompt, user_prompt, norm_lang, temperature=0.3)

    async def chat(
        self,
        chat_history: str,
        session_turns: list[tuple[str, str]],
        question: str,
        asker_name: str,
        mode: str = "chat",
        language: str = "fa",
    ) -> SummaryResult:
        norm_lang = normalize_language(language, f"{question}\n{chat_history}")
        system_prompt, user_prompt = build_chat_qa_prompt(
            chat_history=chat_history,
            session_turns=session_turns,
            question=question,
            asker_name=asker_name,
            mode=mode,
            language=norm_lang,
        )
        return await self._generate(
            system_prompt,
            user_prompt,
            norm_lang,
            temperature=0.7,
            max_tokens=450,
        )
