"""TDD tests for WebSearchService, pre-search, 1-hop auto-search, and open-ended chatbot routing."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import httpx
import pytest

from src.services.search.web_search import (
    WebSearchService,
    extract_auto_search_query,
    should_presearch_query,
    strip_search_directives,
)
from src.services.summarizer.base import SummaryResult


def test_search_heuristics_and_directive_parsing():
    assert should_presearch_query("قیمت طلای ۱۸ عیار امروز چنده؟") is True
    assert should_presearch_query("دلار الان چند شده؟") is True
    assert should_presearch_query("آب و هوای تهران چطوره؟") is True
    assert should_presearch_query("سحر چی گفته؟") is False
    assert should_presearch_query("شخصیت علی چطوره؟") is False

    assert (
        extract_auto_search_query("[[SEARCH: قیمت بیت کوین امروز]]")
        == "قیمت بیت کوین امروز"
    )
    assert extract_auto_search_query("پاسخ معمولی بدون تگ") is None
    assert (
        strip_search_directives("متن تست\n[[SEARCH: something]]") == "متن تست"
    )


@pytest.mark.asyncio
async def test_web_search_service_parses_bing_gnews_wiki_and_tgju():
    bing_xml = """<?xml version="1.0" encoding="utf-8" ?>
    <rss version="2.0"><channel>
      <item>
        <title>قیمت طلای ۱۸ عیار امروز</title>
        <link>https://example.com/gold</link>
        <description>آخرین نرخ &lt;b&gt;طلای ۱۸ عیار&lt;/b&gt; به‌روزرسانی شد.</description>
        <pubDate>Fri, 14 Nov 2026 08:00:00 GMT</pubDate>
      </item>
    </channel></rss>"""

    gnews_xml = """<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0"><channel>
      <item>
        <title>خبر تازه بازار طلا</title>
        <link>https://news.example.com/gold</link>
        <description>نوسان محدود در بازار امروز</description>
        <pubDate>Fri, 14 Nov 2026 08:15:00 GMT</pubDate>
      </item>
    </channel></rss>"""

    tgju_json = {
        "data": [
            [
                "251,784,000",
                "249,954,000",
                "251,784,000",
                "250,388,000",
                '<span class="low">1399000</span>',
                '<span class="low">0.56%</span>',
                "2026/11/12",
                "1405/08/21",
            ]
        ]
    }

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "bing.com/search" in url:
            return httpx.Response(200, text=bing_xml)
        if "news.google.com" in url:
            return httpx.Response(200, text=gnews_xml)
        if "api.tgju.org" in url:
            return httpx.Response(200, json=tgju_json)
        if "wikipedia.org" in url:
            return httpx.Response(200, json={"query": {"search": []}})
        return httpx.Response(404, text="not found")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    service = WebSearchService(client=client)
    context = await service.search("قیمت طلای ۱۸ عیار امروز", language="fa")

    assert "قیمت طلای ۱۸ عیار امروز" in context
    assert "خبر تازه بازار طلا" in context
    assert "25,178,400 تومان" in context
    assert "1405/08/21" in context


@pytest.mark.asyncio
async def test_group_chatbot_presearch_and_auto_search_one_hop(tmp_path):
    from aiogram import Bot, Dispatcher
    from aiogram.client.session.base import BaseSession
    from aiogram.methods import SendMessage
    from aiogram.types import Chat, Message, Update, User
    from src.bot.cache import TranscriptCache
    from src.bot.handlers import create_bot_router

    sent_texts: list[str] = []

    class DummySession(BaseSession):
        async def make_request(self, bot, method, timeout=None):
            if isinstance(method, SendMessage):
                sent_texts.append(method.text)
                return Message(
                    message_id=500 + len(sent_texts),
                    date=datetime.now(timezone.utc),
                    chat=Chat(id=int(method.chat_id), type="supergroup"),
                    text=method.text,
                )
            return True

        async def stream_content(self, url, timeout=None, chunk_size=65536, raise_for_status=True):
            if False:
                yield b""

        async def close(self):
            return None

    fake_bot = Bot(token="123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11", session=DummySession())
    fake_bot._me = User(id=9048492533, is_bot=True, first_name="voice-summary", username="vaez_summary_bot")

    recorded_llm_calls: list[dict] = []

    class AutoSearchSummarizer:
        async def chat(
            self,
            chat_history: str,
            session_turns: list,
            question: str,
            asker_name: str,
            mode: str = "chat",
            language: str = "fa",
            web_context: str | None = None,
        ) -> SummaryResult:
            recorded_llm_calls.append({"question": question, "web_context": web_context})
            if not web_context and "پایتون" in question:
                return SummaryResult(
                    summary="[[SEARCH: latest python version features]]",
                    language="fa",
                    provider="dummy",
                )
            if web_context:
                return SummaryResult(
                    summary=f"طبق سرچ: {web_context.splitlines()[0]}",
                    language="fa",
                    provider="dummy",
                )
            return SummaryResult(
                summary="پاسخ عادی چت‌بات بدون سرچ!",
                language="fa",
                provider="dummy",
            )

    class FakeSearchService:
        def __init__(self):
            self.queries: list[str] = []

        async def search(self, query: str, language: str = "fa") -> str:
            self.queries.append(query)
            return f"نتیجه زنده برای: {query} — عدد تست 12345"

    fake_search = FakeSearchService()
    cache = TranscriptCache(
        persistence_path=tmp_path / "chat.json",
        rate_limit_max_requests=5,
        rate_limit_window_seconds=300,
    )
    router = create_bot_router(
        stt_manager=SimpleNamespace(),
        summarizer_manager=AutoSearchSummarizer(),
        cache=cache,
        web_search=fake_search,
    )
    dp = Dispatcher()
    dp.include_router(router)

    group_chat = Chat(id=-100888, type="supergroup", title="گروه تست")
    human = User(id=77, is_bot=False, first_name="مهدیار", username="vaezCore")

    # 1. Pre-search triggered by group @mention ("قیمت طلا چنده؟") -> 1 rate-limit slot used
    await dp.feed_update(
        fake_bot,
        Update(
            update_id=1,
            message=Message(
                message_id=1,
                date=datetime.now(timezone.utc),
                chat=group_chat,
                from_user=human,
                text="@vaez_summary_bot قیمت طلا چنده؟",
            ),
        ),
    )
    assert fake_search.queries == ["قیمت طلا چنده؟"]
    assert len(recorded_llm_calls) == 1
    assert "12345" in (recorded_llm_calls[0]["web_context"] or "")
    assert "12345" in sent_texts[-1]

    # 2. LLM auto-search (`[[SEARCH: ...]]`) on /ask for off-topic question -> 1-hop retry cleanly!
    await dp.feed_update(
        fake_bot,
        Update(
            update_id=2,
            message=Message(
                message_id=2,
                date=datetime.now(timezone.utc),
                chat=group_chat,
                from_user=human,
                text="/ask آخرین نسخه پایتون چیه؟",
            ),
        ),
    )
    assert fake_search.queries[-1] == "latest python version features"
    assert len(recorded_llm_calls) == 3  # 1 (presearch) + 2 (auto-search 1-hop)
    assert "[[SEARCH:" not in sent_texts[-1]
    assert "12345" in sent_texts[-1]

    # 3. Verify total rate limit consumed is ONLY 2 (NOT 3!)
    assert len(cache._chat_rate_windows[str(group_chat.id)]) == 2

    # 4. General off-topic chat by replying to bot's previous message works naturally without search
    bot_prev_msg = Message(
        message_id=502,
        date=datetime.now(timezone.utc),
        chat=group_chat,
        from_user=fake_bot._me,
        text="جواب قبلی",
    )
    await dp.feed_update(
        fake_bot,
        Update(
            update_id=3,
            message=Message(
                message_id=3,
                date=datetime.now(timezone.utc),
                chat=group_chat,
                from_user=human,
                text="یه فیلم خوب معرفی کن",
                reply_to_message=bot_prev_msg,
            ),
        ),
    )
    assert "پاسخ عادی چت‌بات" in sent_texts[-1]
