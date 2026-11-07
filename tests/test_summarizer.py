import httpx
import pytest
from src.exceptions import ProviderChainExhaustedError
from src.services.summarizer.prompts import build_summary_prompt
from src.services.summarizer.gemini_llm import GeminiSummarizerProvider
from src.services.summarizer.groq_llm import GroqSummarizerProvider
from src.services.summarizer.openrouter_llm import OpenRouterSummarizerProvider
from src.services.summarizer.manager import SummarizerManager


def test_build_summary_prompt_persian_includes_speaker_attribution():
    system_prompt, user_prompt = build_summary_prompt(
        transcript="سلام، فردا ساعت ده جلسه داریم و باید گزارش فروش آماده بشه.",
        speaker_name="مهدیار",
        language="fa",
    )
    assert "مهدیار میگه که" in system_prompt
    assert "📌 خلاصه صحبت‌های مهدیار" in system_prompt
    assert "مهدیار" in user_prompt


def test_build_summary_prompt_english_includes_speaker_attribution():
    system_prompt, user_prompt = build_summary_prompt(
        transcript="Hey team, let's deploy the release tomorrow morning at 9am.",
        speaker_name="Alex",
        language="en",
    )
    assert "Alex says that" in system_prompt
    assert "📌 Summary of Alex's voice" in system_prompt


@pytest.mark.asyncio
async def test_summarizer_manager_falls_back_from_gemini_to_groq_to_openrouter():
    def gemini_fail_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": {"message": "Quota exceeded"}})

    def groq_fail_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": {"message": "Service unavailable"}})

    def openrouter_ok_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": "📌 خلاصه صحبت‌های مهدیار:\nمهدیار میگه که فردا جلسه داریم."
                        }
                    }
                ]
            },
        )

    gemini = GeminiSummarizerProvider(
        api_key="AIza_test",
        client=httpx.AsyncClient(transport=httpx.MockTransport(gemini_fail_handler)),
    )
    groq = GroqSummarizerProvider(
        api_key="gsk_test",
        client=httpx.AsyncClient(transport=httpx.MockTransport(groq_fail_handler)),
    )
    openrouter = OpenRouterSummarizerProvider(
        api_key="or_test",
        client=httpx.AsyncClient(transport=httpx.MockTransport(openrouter_ok_handler)),
    )

    manager = SummarizerManager([gemini, groq, openrouter])
    result = await manager.summarize(
        transcript="سلام فردا جلسه داریم",
        speaker_name="مهدیار",
        language="fa",
    )
    assert result.provider == "openrouter-llm"
    assert "مهدیار میگه که" in result.summary


@pytest.mark.asyncio
async def test_gemini_summarizer_provider_parses_candidates():
    def gemini_ok_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"text": "📌 خلاصه صحبت‌های مهدیار:\nمهدیار میگه که کارها ردیفه."}
                            ]
                        }
                    }
                ]
            },
        )

    gemini = GeminiSummarizerProvider(
        api_key="AIza_test",
        client=httpx.AsyncClient(transport=httpx.MockTransport(gemini_ok_handler)),
    )
    result = await gemini.summarize("کارها ردیفه", "مهدیار", "fa")
    assert result.provider == "gemini"
    assert "مهدیار میگه که" in result.summary


@pytest.mark.asyncio
async def test_summarizer_manager_raises_when_all_fail():
    def fail_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="Internal Server Error")

    groq = GroqSummarizerProvider(
        api_key="gsk_test",
        client=httpx.AsyncClient(transport=httpx.MockTransport(fail_handler)),
    )
    manager = SummarizerManager([groq])
    with pytest.raises(ProviderChainExhaustedError):
        await manager.summarize("متن تست", "مهدیار", "fa")
