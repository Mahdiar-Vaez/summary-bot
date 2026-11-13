"""Google Gemini Flash LLM Summarizer and Chat Q&A Provider."""

from __future__ import annotations

import httpx

from src.exceptions import ProviderError
from src.services.http_utils import (
    extract_gemini_parts_text,
    parse_json_response,
    redact_secrets,
)
from src.services.stt.base import normalize_language
from src.services.summarizer.base import SummaryResult
from src.services.summarizer.prompts import build_chat_qa_prompt, build_summary_prompt


class GeminiSummarizerProvider:
    """Summarizes transcripts and handles chat Q&A using Google Gemini REST API."""

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

    async def _generate(
        self,
        system_prompt: str,
        user_prompt: str,
        norm_lang: str,
        temperature: float = 0.3,
        max_output_tokens: int | None = None,
    ) -> SummaryResult:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent"
        )
        headers = {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json",
        }
        gen_cfg: dict[str, object] = {"temperature": temperature}
        if max_output_tokens is not None:
            gen_cfg["maxOutputTokens"] = max_output_tokens
        payload = {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": gen_cfg,
        }

        try:
            if self._client is not None:
                response = await self._client.post(url, headers=headers, json=payload)
            else:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    response = await client.post(url, headers=headers, json=payload)
        except httpx.HTTPError as exc:
            raise ProviderError(f"Gemini network error: {redact_secrets(str(exc))}") from exc

        if response.status_code != 200:
            raise ProviderError(
                redact_secrets(f"Gemini HTTP {response.status_code}: {response.text[:250]}")
            )

        data = parse_json_response(response, "Gemini")
        output_text = extract_gemini_parts_text(data, "Gemini")

        if not output_text:
            raise ProviderError("Gemini returned empty response")

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
        web_context: str | None = None,
    ) -> SummaryResult:
        norm_lang = normalize_language(language, f"{question}\n{chat_history}")
        system_prompt, user_prompt = build_chat_qa_prompt(
            chat_history=chat_history,
            session_turns=session_turns,
            question=question,
            asker_name=asker_name,
            mode=mode,
            language=norm_lang,
            web_context=web_context,
        )
        return await self._generate(
            system_prompt,
            user_prompt,
            norm_lang,
            temperature=0.7,
            max_output_tokens=1200,
        )
