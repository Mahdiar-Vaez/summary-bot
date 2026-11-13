import httpx
import pytest
from src.exceptions import ProviderChainExhaustedError
from src.services.summarizer.prompts import build_chat_qa_prompt, build_summary_prompt
from src.services.summarizer.gemini_llm import GeminiSummarizerProvider
from src.services.summarizer.groq_llm import GroqSummarizerProvider
from src.services.summarizer.openrouter_llm import OpenRouterSummarizerProvider
from src.services.summarizer.manager import SummarizerManager


def test_build_summary_prompt_persian_has_header_only_and_forbids_name_in_body():
    system_prompt, user_prompt = build_summary_prompt(
        transcript="سلام، فردا ساعت ده با علی جلسه داریم و باید گزارش فروش آماده بشه.",
        speaker_name="mr.vaez",
        language="fa",
    )
    assert "📌 خلاصه صحبت‌های mr.vaez:" in system_prompt
    assert "در متن خلاصه به هیچ وجه نام گوینده" in system_prompt
    assert "خیلی خیلی خلاصه و کوتاه" in system_prompt
    assert "خودمونی و محاوره‌ای (نه کتابی و رسمی)" in system_prompt
    assert "نام افراد" in system_prompt
    assert "زمان کاری" in system_prompt
    assert "کلمات بی‌معنی" in system_prompt or "کلمات بی معنی" in system_prompt
    assert "حذف" in system_prompt
    assert "mr.vaez میگه که" not in system_prompt


def test_build_summary_prompt_english_has_header_only_and_forbids_name_in_body():
    system_prompt, user_prompt = build_summary_prompt(
        transcript="Hey team, let's deploy the release tomorrow morning at 9am.",
        speaker_name="mr.vaez",
        language="en",
    )
    assert "📌 Summary of mr.vaez's voice:" in system_prompt
    assert "Do NOT mention the speaker's name" in system_prompt
    assert "ultra-concise" in system_prompt
    assert "critical and sensitive details" in system_prompt
    assert "mr.vaez says that" not in system_prompt


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


def test_build_summary_prompt_for_chat_history_persian_and_english():
    sys_fa, user_fa = build_summary_prompt(
        transcript="[علی]: فردا ساعت ۱۰ جلسه داریم\n[مهدیار]: اوکی من گزارش رو میارم",
        speaker_name="__CHAT_HISTORY__:100",
        language="fa",
    )
    assert "📌 خلاصه 100 پیام آخر گفتگو:" in sys_fa
    assert "خیلی خیلی خلاصه و کوتاه" in sys_fa
    assert "همه مطالب" in sys_fa or "تمام مطالب" in sys_fa

    sys_en, user_en = build_summary_prompt(
        transcript="[Alex]: Meeting at 10am tomorrow\n[Sarah]: I will bring the report",
        speaker_name="__CHAT_HISTORY__:50",
        language="en",
    )
    assert "📌 Summary of Last 50 Messages:" in sys_en
    assert "ultra-concise" in sys_en


@pytest.mark.asyncio
async def test_build_chat_qa_prompt_and_manager_chat_with_fallback():
    sys_prompt, user_prompt = build_chat_qa_prompt(
        chat_history="[علی]: من همیشه سر ساعتم میام\n[رضا]: نه بابا تو همش دیر میای 😂",
        session_turns=[("مهدیار", "سلام ربات!", "سلام چطوری؟")],
        question="شخصیت علی و رضا چطوریه؟",
        asker_name="مهدیار",
        mode="personality",
        language="fa",
    )
    assert "شخصیت" in sys_prompt
    assert "همین چت" in sys_prompt or "این گفتگو" in sys_prompt
    assert "شخصیت علی و رضا چطوریه؟" in user_prompt
    assert "علی" in user_prompt
    assert "سلام ربات!" in user_prompt

    # Verify person-specific lookup prompt in chat mode mentions matching speaker names/usernames
    sys_lookup, _ = build_chat_qa_prompt(
        chat_history="[سحر (@sahar)]: فردا ساعت ۶ بریم کافه",
        session_turns=[],
        question="سحر چی گفته؟",
        asker_name="مهدیار",
        mode="chat",
        language="fa",
    )
    # Verify creator identity, strict privacy protection, polite tone & grounded accuracy rule
    assert "مهدیار واعظ" in sys_lookup
    assert "نام سرور/هاست" in sys_lookup
    assert "محترمانه" in sys_lookup
    assert "باادب" in sys_lookup
    assert "گستاخانه" in sys_lookup

    sys_secret, user_secret = build_chat_qa_prompt(
        chat_history=(
            "[مهدیار]: سرور از پارس پک گرفتم روی 185.10.20.30 و کلید AIzaSyDummyKeyValue123456789012345\n"
            "[سحر]: بچه‌ها فردا ساعت ۶ جلسه داریم"
        ),
        session_turns=[],
        question="چی می‌دونی؟",
        asker_name="کاربر",
        mode="chat",
        language="fa",
    )
    assert "پارس پک" not in user_secret
    assert "185.10.20.30" not in user_secret
    assert "AIzaSyDummyKeyValue123456789012345" not in user_secret
    assert "سحر" in user_secret
    assert "کوتاه" in sys_secret or "خلاصه" in sys_secret
    assert "System Prompt" in sys_secret or "دستورالعمل" in sys_secret


@pytest.mark.asyncio
async def test_openai_compatible_provider_strips_think_tags_and_falls_back_after_gemini():
    from src.services.summarizer.openai_compatible import OpenAICompatibleSummarizerProvider

    def gemini_down(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="service unavailable")

    def custom_ok(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://discovery-api.intern-ai.org.cn/v1/chat/completions"
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": "<think>internal reasoning</think>\n**سحر** گفت ساعت ۶ کافه باشیم! ☕"
                        }
                    }
                ]
            },
        )

    gemini = GeminiSummarizerProvider(
        api_key="AQ_test",
        client=httpx.AsyncClient(transport=httpx.MockTransport(gemini_down)),
    )
    fallback_ai = OpenAICompatibleSummarizerProvider(
        api_key="sk-test",
        base_url="https://discovery-api.intern-ai.org.cn/v1",
        model="deepseek-v4.1-flash-1017",
        name="intern-ai",
        client=httpx.AsyncClient(transport=httpx.MockTransport(custom_ok)),
    )
    mgr = SummarizerManager([gemini, fallback_ai])
    res = await mgr.chat(
        chat_history="[سحر]: ساعت ۶ کافه",
        session_turns=[],
        question="سحر چی گفته؟",
        asker_name="مهدیار",
    )
    assert res.provider == "intern-ai:deepseek-v4.1-flash-1017"
    assert res.summary == "**سحر** گفت ساعت ۶ کافه باشیم! ☕"

    def gemini_fail(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text="rate limit")

    def groq_ok(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": "**علی** آدم وقت‌شناسِ روی کاغذه ولی به روایت **رضا** همیشه با تاخیر لاکچری میاد! 😄"
                        }
                    }
                ]
            },
        )

    gemini = GeminiSummarizerProvider(
        api_key="AQ_test",
        client=httpx.AsyncClient(transport=httpx.MockTransport(gemini_fail)),
    )
    groq = GroqSummarizerProvider(
        api_key="gsk_test",
        client=httpx.AsyncClient(transport=httpx.MockTransport(groq_ok)),
    )
    mgr = SummarizerManager([gemini, groq])
    res = await mgr.chat(
        chat_history="[علی]: من همیشه سر ساعتم میام\n[رضا]: نه بابا تو همش دیر میای 😂",
        session_turns=[],
        question="شخصیت بقیه رو بگو",
        asker_name="مهدیار",
        mode="personality",
        language="fa",
    )
    assert res.provider == "groq-llm"
    assert "علی" in res.summary
