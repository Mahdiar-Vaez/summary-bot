from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
import pytest
from src.bot.cache import TranscriptCache, chunk_text
from src.bot.extractors import extract_voice_target
from src.bot.formatting import format_telegram_html
from src.bot.handlers import (
    MAX_CHAT_SUMMARY_MESSAGES,
    build_help_text,
    create_bot_router,
    handle_chat_interaction,
    handle_chat_summary_callback,
    handle_chat_summary_command,
    handle_feedback_callback,
    handle_stats_command,
    handle_transcript_callback,
    handle_tts_callback,
    process_voice_request,
    record_incoming_chat_message,
    should_trigger_chat_interaction,
    should_trigger_on_reply,
    should_trigger_on_voice_message,
    track_update_activity,
)
from src.config import Settings
from src.exceptions import ProviderChainExhaustedError
from src.main import (
    build_stt_manager,
    build_summarizer_manager,
    build_tts_manager,
    configure_bot_profile,
)
from src.services.stt.base import TranscriptionResult
from src.services.summarizer.base import SummaryResult
from src.services.tts.base import TTSResult


def test_extract_voice_target_from_reply_and_hidden_forward():
    replied_msg = SimpleNamespace(
        voice=SimpleNamespace(file_id="v1", file_size=1024, duration=120),
        audio=None,
        forward_origin=SimpleNamespace(
            type="hidden_user",
            sender_user_name="مهدیار",
            sender_user=None,
            sender_chat=None,
            chat=None,
        ),
        from_user=SimpleNamespace(full_name="Ali", first_name="Ali"),
    )
    msg = SimpleNamespace(
        voice=None,
        audio=None,
        reply_to_message=replied_msg,
        from_user=SimpleNamespace(full_name="Reza", first_name="Reza"),
    )
    target = extract_voice_target(msg)
    assert target is not None
    media, speaker = target
    assert media.file_id == "v1"
    assert speaker == "مهدیار"


def test_extract_voice_target_from_channel_forward():
    msg = SimpleNamespace(
        voice=None,
        audio=SimpleNamespace(file_id="a1", file_size=2048, duration=60),
        reply_to_message=None,
        forward_origin=SimpleNamespace(
            type="channel",
            sender_user_name=None,
            sender_user=None,
            sender_chat=None,
            chat=SimpleNamespace(title="کانال تکنولوژی"),
        ),
        from_user=SimpleNamespace(full_name="Reza", first_name="Reza"),
    )
    target = extract_voice_target(msg)
    assert target is not None
    media, speaker = target
    assert media.file_id == "a1"
    assert speaker == "کانال تکنولوژی"


def test_transcript_cache_and_chunking():
    cache = TranscriptCache(ttl_seconds=60, max_items=10)
    cid = cache.put(speaker_name="مهدیار", text="A" * 8500)
    entry = cache.get(cid)
    assert entry is not None
    assert entry.speaker_name == "مهدیار"
    chunks = chunk_text(entry.text, max_len=4000)
    assert len(chunks) == 3
    assert all(len(c) <= 4000 for c in chunks)


@pytest.mark.asyncio
async def test_process_voice_request_cleans_up_temp_file_and_edits_message():
    initial_replies = []
    status_edits = []
    recorded_paths: list[Path] = []

    class DummyStatusMsg:
        async def edit_text(self, text: str, reply_markup=None):
            status_edits.append((text, reply_markup))

    class DummyMessage:
        voice = SimpleNamespace(file_id="voice_123", file_size=500_000, duration=300)
        audio = None
        reply_to_message = None
        forward_origin = None
        from_user = SimpleNamespace(full_name="مهدیار", first_name="مهدیار")

        async def reply(self, text: str):
            initial_replies.append(text)
            return DummyStatusMsg()

    class DummyBot:
        async def get_file(self, file_id: str):
            return SimpleNamespace(file_path="voice/file_1.oga")

        async def download_file(self, file_path: str, destination: Path):
            Path(destination).write_bytes(b"OggS-voice-content")
            recorded_paths.append(Path(destination))

    class DummySTT:
        async def transcribe(self, audio_path: Path) -> TranscriptionResult:
            assert audio_path.exists()
            return TranscriptionResult(
                text="سلام من مهدیارم و می‌خوام پروژه رو استارت بزنیم",
                language="fa",
                provider="groq",
            )

    class DummySummarizer:
        async def summarize(
            self, transcript: str, speaker_name: str, language: str
        ) -> SummaryResult:
            return SummaryResult(
                summary=f"📌 خلاصه صحبت‌های {speaker_name}:\n{speaker_name} میگه که پروژه رو استارت بزنیم.",
                language="fa",
                provider="gemini",
            )

    cache = TranscriptCache()
    await process_voice_request(
        message=DummyMessage(),
        bot=DummyBot(),
        stt_manager=DummySTT(),
        summarizer_manager=DummySummarizer(),
        cache=cache,
        max_file_size_mb=20,
    )

    assert len(initial_replies) == 1
    assert "در حال خلاصه‌سازی" in initial_replies[0]
    assert "تبدیل ویس" not in initial_replies[0]
    assert len(status_edits) == 1
    final_text, final_markup = status_edits[-1]
    assert "مهدیار میگه که" in final_text
    assert final_markup is not None
    assert len(recorded_paths) == 1
    assert not recorded_paths[0].exists()


@pytest.mark.asyncio
async def test_process_voice_request_cleans_up_temp_file_on_provider_failure():
    status_edits = []
    recorded_paths: list[Path] = []

    class DummyStatusMsg:
        async def edit_text(self, text: str, reply_markup=None):
            status_edits.append((text, reply_markup))

    class DummyMessage:
        voice = SimpleNamespace(file_id="voice_err", file_size=500_000, duration=60)
        audio = None
        reply_to_message = None
        forward_origin = None
        from_user = SimpleNamespace(full_name="مهدیار", first_name="مهدیار")

        async def reply(self, text: str):
            return DummyStatusMsg()

    class DummyBot:
        async def get_file(self, file_id: str):
            return SimpleNamespace(file_path="voice/file_err.oga")

        async def download_file(self, file_path: str, destination: Path):
            Path(destination).write_bytes(b"OggS-voice-content")
            recorded_paths.append(Path(destination))

    class FailingSTT:
        async def transcribe(self, audio_path: Path) -> TranscriptionResult:
            raise ProviderChainExhaustedError(["groq: 429"])

    class DummySummarizer:
        async def summarize(self, transcript: str, speaker_name: str, language: str) -> SummaryResult:
            raise AssertionError("Should not be called")

    cache = TranscriptCache()
    await process_voice_request(
        message=DummyMessage(),
        bot=DummyBot(),
        stt_manager=FailingSTT(),
        summarizer_manager=DummySummarizer(),
        cache=cache,
        max_file_size_mb=20,
    )

    assert len(recorded_paths) == 1
    assert not recorded_paths[0].exists()
    assert "توکنام تموم شده" in status_edits[-1][0]


@pytest.mark.asyncio
async def test_handle_transcript_callback_expired_and_valid():
    answers = []
    replies = []

    class DummyCallback:
        data = "transcript:missing_id"
        message = SimpleNamespace(reply=lambda text: replies.append(text))

        async def answer(self, text: str | None = None, show_alert: bool = False):
            answers.append((text, show_alert))

    cache = TranscriptCache()
    await handle_transcript_callback(DummyCallback(), cache)
    assert answers[-1][1] is True
    assert "منقضی" in (answers[-1][0] or "")


def test_build_managers_with_gemini_creates_multi_model_fallback_chain():
    settings = Settings(
        telegram_bot_token="123:ABC",
        gemini_api_key="AQ.test_key",
        _env_file=None,
    )
    stt_mgr = build_stt_manager(settings)
    sum_mgr = build_summarizer_manager(settings)
    tts_mgr = build_tts_manager(settings)

    stt_models = [getattr(p, "model", "") for p in stt_mgr.providers]
    sum_models = [getattr(p, "model", "") for p in sum_mgr.providers]
    tts_names = [getattr(p, "name", "") for p in tts_mgr.providers]

    assert stt_models == [
        "gemini-3.5-flash-lite",
        "gemini-3.8-flash",
        "gemini-3.1-flash-lite",
        "gemini-3.5-flash",
    ]
    assert sum_models == [
        "gemini-3.5-flash-lite",
        "gemini-3.8-flash",
        "gemini-3.1-flash-lite",
        "gemini-3.5-flash",
    ]
    assert tts_names[:3] == [
        "gemini-tts:gemini-3.8-flash-tts",
        "gemini-tts:gemini-3.8-flash-lite-tts",
        "gemini-tts:gemini-3.1-flash-tts-preview",
    ]


def test_should_trigger_on_reply_detects_voice_replies_in_private_and_group():
    replied_voice = SimpleNamespace(voice=SimpleNamespace(file_id="v1"), audio=None)
    private_reply = SimpleNamespace(
        chat=SimpleNamespace(type="private"),
        text="چی میگه؟",
        reply_to_message=replied_voice,
    )
    group_mention_reply = SimpleNamespace(
        chat=SimpleNamespace(type="supergroup"),
        text="@vaez_summary_bot خلاصه کن",
        reply_to_message=replied_voice,
    )
    group_username_without_at_reply = SimpleNamespace(
        chat=SimpleNamespace(type="supergroup"),
        text="my_custom_bot",
        reply_to_message=replied_voice,
    )
    group_bot_name_reply = SimpleNamespace(
        chat=SimpleNamespace(type="supergroup"),
        text="voice-summary",
        reply_to_message=replied_voice,
    )
    group_keyword_reply = SimpleNamespace(
        chat=SimpleNamespace(type="group"),
        text="خلاصه",
        reply_to_message=replied_voice,
    )
    group_unrelated_reply = SimpleNamespace(
        chat=SimpleNamespace(type="supergroup"),
        text="باشه مرسی",
        reply_to_message=replied_voice,
    )
    group_conversational_sentence_with_kholase = SimpleNamespace(
        chat=SimpleNamespace(type="supergroup"),
        text="دمت گرم بابت این خلاصه خیلی خوب بود و به کارم اومد",
        reply_to_message=replied_voice,
    )
    private_unrelated_reply = SimpleNamespace(
        chat=SimpleNamespace(type="private"),
        text="مرسی دستت درد نکنه",
        reply_to_message=replied_voice,
    )
    bot_own_reply = SimpleNamespace(
        chat=SimpleNamespace(type="supergroup"),
        text="خلاصه",
        from_user=SimpleNamespace(is_bot=True),
        reply_to_message=replied_voice,
    )
    slash_command_with_bot_username = SimpleNamespace(
        chat=SimpleNamespace(type="supergroup"),
        text="/summary@vaez_summary_bot",
        reply_to_message=replied_voice,
    )

    assert should_trigger_on_reply(private_reply, bot_username="my_custom_bot") is True
    assert should_trigger_on_reply(group_mention_reply, bot_username="vaez_summary_bot") is True
    assert should_trigger_on_reply(group_username_without_at_reply, bot_username="my_custom_bot") is True
    assert should_trigger_on_reply(group_bot_name_reply, bot_username="my_custom_bot") is True
    assert should_trigger_on_reply(group_keyword_reply, bot_username="my_custom_bot") is True
    assert should_trigger_on_reply(slash_command_with_bot_username, bot_username="vaez_summary_bot") is True
    assert should_trigger_on_reply(group_unrelated_reply, bot_username="my_custom_bot") is False
    assert should_trigger_on_reply(group_conversational_sentence_with_kholase, bot_username="my_custom_bot") is False
    assert should_trigger_on_reply(private_unrelated_reply, bot_username="my_custom_bot") is False
    assert should_trigger_on_reply(bot_own_reply, bot_username="my_custom_bot") is False


def test_should_trigger_on_voice_message_private_vs_group():
    private_voice = SimpleNamespace(
        chat=SimpleNamespace(type="private"),
        voice=SimpleNamespace(file_id="v1"),
        caption=None,
    )
    group_plain_voice = SimpleNamespace(
        chat=SimpleNamespace(type="supergroup"),
        voice=SimpleNamespace(file_id="v2"),
        caption=None,
    )
    group_mentioned_voice = SimpleNamespace(
        chat=SimpleNamespace(type="supergroup"),
        voice=SimpleNamespace(file_id="v3"),
        caption="@vaez_summary_bot خلاصه",
    )

    assert should_trigger_on_voice_message(private_voice, bot_username="vaez_summary_bot") is True
    assert should_trigger_on_voice_message(group_plain_voice, bot_username="vaez_summary_bot") is False
    assert should_trigger_on_voice_message(group_mentioned_voice, bot_username="vaez_summary_bot") is True


@pytest.mark.asyncio
async def test_process_voice_request_deduplicates_same_voice_in_group():
    initial_replies = []
    status_edits = []

    class DummyStatusMsg:
        async def edit_text(self, text: str, reply_markup=None):
            status_edits.append((text, reply_markup))

    replied_voice_msg = SimpleNamespace(
        message_id=777,
        voice=SimpleNamespace(
            file_id="voice_group_1",
            file_unique_id="uniq_group_1",
            file_size=50_000,
            duration=30,
        ),
        audio=None,
        forward_origin=None,
        from_user=SimpleNamespace(full_name="مهدیار", first_name="مهدیار"),
    )

    class DummyGroupReplyMessage:
        chat = SimpleNamespace(id=-100999888, type="supergroup")
        message_id = 778
        voice = None
        audio = None
        reply_to_message = replied_voice_msg
        forward_origin = None
        from_user = SimpleNamespace(full_name="کاربر ۱", first_name="کاربر")

        async def reply(self, text: str):
            initial_replies.append(text)
            return DummyStatusMsg()

    class DummyBot:
        async def get_file(self, file_id: str):
            return SimpleNamespace(file_path="voice/file_g1.oga")

        async def download_file(self, file_path: str, destination: Path):
            Path(destination).write_bytes(b"OggS-voice-content")

    class DummySTT:
        async def transcribe(self, audio_path: Path) -> TranscriptionResult:
            return TranscriptionResult(text="متن تست گروه", language="fa", provider="gemini")

    class DummySummarizer:
        async def summarize(self, transcript: str, speaker_name: str, language: str) -> SummaryResult:
            return SummaryResult(summary="📌 خلاصه صحبت‌های مهدیار:\nتست", language="fa", provider="gemini")

    cache = TranscriptCache()
    msg1 = DummyGroupReplyMessage()
    msg2 = DummyGroupReplyMessage()

    await process_voice_request(
        message=msg1,
        bot=DummyBot(),
        stt_manager=DummySTT(),
        summarizer_manager=DummySummarizer(),
        cache=cache,
    )
    # Second request for the exact same voice in the same group should be ignored
    await process_voice_request(
        message=msg2,
        bot=DummyBot(),
        stt_manager=DummySTT(),
        summarizer_manager=DummySummarizer(),
        cache=cache,
    )

    assert len(initial_replies) == 1
    assert len(status_edits) == 1


@pytest.mark.asyncio
async def test_handle_tts_and_transcript_callbacks_disable_buttons_after_click():
    sent_voices = []
    sent_transcripts = []
    edited_markups = []
    callback_answers = []

    class DummyMsg:
        async def reply_voice(self, voice, caption=None, reply_markup=None):
            sent_voices.append((voice, caption, reply_markup))

        async def reply(self, text: str, reply_markup=None):
            sent_transcripts.append((text, reply_markup))

        async def edit_reply_markup(self, reply_markup=None):
            edited_markups.append(reply_markup)

    class DummyCallback:
        def __init__(self, prefix: str, cid: str, msg: DummyMsg):
            self.data = f"{prefix}:{cid}"
            self.message = msg

        async def answer(self, text: str | None = None, show_alert: bool = False):
            callback_answers.append((text, show_alert))

    class DummyTTS:
        async def synthesize(self, text: str, language: str) -> TTSResult:
            return TTSResult(audio_bytes=b"MP3AUDIO", language=language, provider="edge-tts")

    cache = TranscriptCache()
    cid = cache.put(
        speaker_name="mr.vaez",
        text="متن کامل ویس",
        summary="📌 خلاصه صحبت‌های mr.vaez:\nجلسه فردا برگزار می‌شود.",
        language="fa",
    )
    shared_msg = DummyMsg()

    # 1. Click TTS button -> sends voice, removes TTS button (leaves transcript button)
    await handle_tts_callback(DummyCallback("tts", cid, shared_msg), cache, DummyTTS())
    assert len(sent_voices) == 1
    assert "mr.vaez" in (sent_voices[0][1] or "")
    assert len(edited_markups) == 1
    remaining_buttons = [
        btn.callback_data
        for row in edited_markups[-1].inline_keyboard
        for btn in row
    ]
    assert remaining_buttons == [f"transcript:{cid}"]

    # 2. Clicking TTS button again should NOT synthesize or send voice again
    await handle_tts_callback(DummyCallback("tts", cid, shared_msg), cache, DummyTTS())
    assert len(sent_voices) == 1

    # 3. Click Transcript button -> sends transcript, removes last button (reply_markup becomes None)
    await handle_transcript_callback(DummyCallback("transcript", cid, shared_msg), cache)
    assert len(sent_transcripts) == 1
    assert edited_markups[-1] is None

    # 4. Clicking Transcript button again should NOT send transcript again
    await handle_transcript_callback(DummyCallback("transcript", cid, shared_msg), cache)
    assert len(sent_transcripts) == 1


def test_chat_history_store_caps_at_250_messages_and_ignores_commands():
    assert MAX_CHAT_SUMMARY_MESSAGES == 250
    cache = TranscriptCache()
    chat_id = -100123456

    for i in range(1, 271):
        msg = SimpleNamespace(
            chat=SimpleNamespace(id=chat_id, type="supergroup"),
            from_user=SimpleNamespace(
                full_name=f"User{i}",
                first_name=f"User{i}",
                username=f"user_{i}",
                is_bot=False,
            ),
            text=f"پیام شماره {i} درباره پروژه",
            caption=None,
        )
        record_incoming_chat_message(msg, cache)

    # Commands should not be recorded
    cmd_msg = SimpleNamespace(
        chat=SimpleNamespace(id=chat_id, type="supergroup"),
        from_user=SimpleNamespace(full_name="Admin", first_name="Admin", is_bot=False),
        text="/chat_summary 200",
        caption=None,
    )
    record_incoming_chat_message(cmd_msg, cache)

    recent_all = cache.get_recent_chat_messages(chat_id, limit=250)
    assert len(recent_all) == 250
    assert len(cache.get_recent_chat_messages(chat_id, limit=200)) == 200
    assert recent_all[0].text == "پیام شماره 21 درباره پروژه"
    assert recent_all[-1].text == "پیام شماره 270 درباره پروژه"
    assert "@user_270" in recent_all[-1].sender


@pytest.mark.asyncio
async def test_chat_summary_uses_bottom_reply_keyboard_and_deletes_status_message(tmp_path: Path):
    replies = []
    deleted_count = 0

    class DummyStatusOrPromptMsg:
        message_id = 901

        async def delete(self):
            nonlocal deleted_count
            deleted_count += 1

        async def edit_text(self, text: str, reply_markup=None):
            replies.append(("edit", text, reply_markup))

    class DummyIncomingMsg:
        chat = SimpleNamespace(id=-100555, type="supergroup")
        message_id = 900
        text = "/chat_summary"
        caption = None
        reply_to_message = None
        from_user = SimpleNamespace(id=111, full_name="مهدیار", first_name="مهدیار", is_bot=False)
        bot = None

        async def reply(self, text: str, reply_markup=None):
            replies.append(("reply", text, reply_markup))
            return DummyStatusOrPromptMsg()

    class DummySummarizer:
        async def summarize(self, transcript: str, speaker_name: str, language: str) -> SummaryResult:
            assert speaker_name.startswith("__CHAT_HISTORY__:")
            return SummaryResult(
                summary="📌 خلاصه 3 پیام آخر گفتگو:\n• هماهنگی جلسه فردا ساعت ۱۰ صبح با علی",
                language="fa",
                provider="gemini",
            )

    history_file = tmp_path / "chat_history.json"
    cache = TranscriptCache(persistence_path=history_file)
    cache.add_chat_message(-100555, "علی", "سلام بچه‌ها فردا ساعت ۱۰ صبح جلسه داریم")
    cache.add_chat_message(-100555, "رضا", "گزارش مالی رو هم بیارید")
    cache.add_chat_message(-100555, "مهدیار", "حله من آماده می‌کنم")

    # Verify persistence across restarts
    reloaded_cache = TranscriptCache(persistence_path=history_file)
    assert len(reloaded_cache.get_recent_chat_messages(-100555, limit=150)) == 3

    # 1. Calling /chat_summary without number shows ReplyKeyboardMarkup at bottom input area
    await handle_chat_summary_command(DummyIncomingMsg(), DummySummarizer(), reloaded_cache)
    assert len(replies) == 1
    kind, menu_text, menu_markup = replies[0]
    assert hasattr(menu_markup, "keyboard")  # ReplyKeyboardMarkup (bottom input bar), NOT inline_keyboard
    assert menu_markup.resize_keyboard is True
    assert menu_markup.one_time_keyboard is True

    # 2. Selecting an option from the bottom keyboard summarizes, deletes temporary status message, and sends final output
    replies.clear()
    select_msg = DummyIncomingMsg()
    select_msg.text = "📊 خلاصه ۱۰۰ پیام آخر"
    await handle_chat_summary_command(select_msg, DummySummarizer(), reloaded_cache)

    assert deleted_count >= 1  # Status message was deleted!
    final_kind, final_text, final_markup = replies[-1]
    assert "📌" in final_text
    assert final_markup is not None  # Must have feedback buttons!
    fb_datas = [btn.callback_data for row in final_markup.inline_keyboard for btn in row]
    assert all(cb.startswith("fb:") for cb in fb_datas)
    assert not any(cb.startswith("tts:") for cb in fb_datas)  # No audio button on group chat summary


def test_format_telegram_html_converts_markdown_and_escapes_html():
    raw = "📌 خلاصه صحبت‌های mr.vaez:\n• جلسه فردا **ساعت ۱۰ صبح** با `علی` (هزینه < ۱۰۰)"
    formatted = format_telegram_html(raw)
    assert "<b>📌 خلاصه صحبت‌های mr.vaez:</b>" in formatted
    assert "<b>ساعت ۱۰ صبح</b>" in formatted
    assert "<code>علی</code>" in formatted
    assert "&lt; ۱۰۰" in formatted
    assert "**" not in formatted


def test_should_trigger_chat_interaction_detection():
    bot_id = 9048492533
    private_text = SimpleNamespace(
        chat=SimpleNamespace(type="private"),
        from_user=SimpleNamespace(id=1, is_bot=False),
        text="سلام ربات! شخصیت علی چطوریه؟",
        caption=None,
        reply_to_message=None,
    )
    group_mention = SimpleNamespace(
        chat=SimpleNamespace(type="supergroup"),
        from_user=SimpleNamespace(id=1, is_bot=False),
        text="@vaez_summary_bot نظرت راجع به شخصیت بچه‌های گروه چیه؟",
        caption=None,
        reply_to_message=None,
    )
    group_reply_to_bot = SimpleNamespace(
        chat=SimpleNamespace(type="supergroup"),
        from_user=SimpleNamespace(id=1, is_bot=False),
        text="جدی میگی؟ بیشتر توضیح بده!",
        caption=None,
        reply_to_message=SimpleNamespace(
            from_user=SimpleNamespace(id=bot_id, username="vaez_summary_bot", is_bot=True),
            voice=None,
            audio=None,
        ),
    )
    group_normal_chat = SimpleNamespace(
        chat=SimpleNamespace(type="supergroup"),
        from_user=SimpleNamespace(id=1, is_bot=False),
        text="بچه‌ها ساعت ۵ جلسه داریم",
        caption=None,
        reply_to_message=None,
    )

    assert should_trigger_chat_interaction(private_text, bot_id=bot_id, bot_username="vaez_summary_bot") is True
    assert should_trigger_chat_interaction(group_mention, bot_id=bot_id, bot_username="vaez_summary_bot") is True
    assert should_trigger_chat_interaction(group_reply_to_bot, bot_id=bot_id, bot_username="vaez_summary_bot") is True
    assert should_trigger_chat_interaction(group_normal_chat, bot_id=bot_id, bot_username="vaez_summary_bot") is False


@pytest.mark.asyncio
async def test_chat_interaction_isolates_sessions_by_chat_id(tmp_path: Path):
    seen_contexts = []
    sent_replies = []

    class DummySummarizer:
        async def chat(
            self,
            chat_history: str,
            session_turns: list[tuple[str, str]],
            question: str,
            asker_name: str,
            mode: str = "chat",
            language: str = "fa",
        ) -> SummaryResult:
            seen_contexts.append((chat_history, list(session_turns), question, mode))
            return SummaryResult(
                summary=f"جواب فان برای {asker_name}: عالیه!",
                language="fa",
                provider="gemini",
            )

    class DummyMsg:
        def __init__(self, chat_id: int, text: str, sender: str):
            self.chat = SimpleNamespace(id=chat_id, type="supergroup", title=f"Chat {chat_id}")
            self.text = text
            self.caption = None
            self.from_user = SimpleNamespace(id=42, full_name=sender, first_name=sender, is_bot=False)

        async def reply(self, text: str, reply_markup=None):
            sent_replies.append((text, reply_markup))
            return SimpleNamespace(edit_text=lambda *a, **kw: None)

    cache = TranscriptCache(
        persistence_path=tmp_path / "hist.json",
        feedback_path=tmp_path / "fb.json",
        admin_config_path=tmp_path / "admin.json",
    )
    cache.add_chat_message(-1001, "سارا", "راز گروه اول: فردا جشن داریم")
    cache.add_chat_message(-1002, "نیما", "راز گروه دوم: سرور آپدیت میشه")

    # Ask in Chat -1001
    await handle_chat_interaction(
        DummyMsg(-1001, "@vaez_summary_bot شخصیت بچه‌ها چطوریه؟", "مهدیار"),
        DummySummarizer(),
        cache,
        bot_username="vaez_summary_bot",
        mode="personality",
    )
    # Ask in Chat -1002
    await handle_chat_interaction(
        DummyMsg(-1002, "@vaez_summary_bot چه خبره اینجا؟", "علی"),
        DummySummarizer(),
        cache,
        bot_username="vaez_summary_bot",
    )

    assert "سارا" in seen_contexts[0][0]
    assert "نیما" not in seen_contexts[0][0]
    assert "نیما" in seen_contexts[1][0]
    assert "سارا" not in seen_contexts[1][0]
    assert len(seen_contexts[1][1]) == 0  # Session turns MUST be isolated per chat!

    # Reply must include feedback buttons
    reply_text, reply_markup = sent_replies[-1]
    assert reply_markup is not None
    assert any(
        btn.callback_data.startswith("fb:")
        for row in reply_markup.inline_keyboard
        for btn in row
    )


@pytest.mark.asyncio
async def test_feedback_buttons_disappear_on_click_and_notify_admin(tmp_path: Path):
    edited_markups = []
    admin_notifications = []
    callback_answers = []

    class DummyBot:
        async def send_message(self, chat_id: int, text: str):
            admin_notifications.append((chat_id, text))

    class DummyMsg:
        def __init__(self, chat_id: int):
            self.chat = SimpleNamespace(id=chat_id, type="supergroup", title="گروه تست")
            self.bot = DummyBot()

        async def edit_reply_markup(self, reply_markup=None):
            edited_markups.append(reply_markup)

    class DummyCallback:
        def __init__(self, data: str, msg: DummyMsg):
            self.data = data
            self.message = msg
            self.from_user = SimpleNamespace(
                id=555, full_name="حسین", first_name="حسین", username="hosein"
            )

        async def answer(self, text: str | None = None, show_alert: bool = False):
            callback_answers.append((text, show_alert))

    fb_file = tmp_path / "feedbacks.json"
    admin_file = tmp_path / "admin.json"
    cache = TranscriptCache(
        persistence_path=tmp_path / "hist.json",
        feedback_path=fb_file,
        admin_config_path=admin_file,
        admin_chat_id=99999,
    )

    # Case 1: Standalone output (e.g. TTS audio or chat summary) -> buttons vanish completely (None)
    fb_id_voice = cache.create_feedback_target(
        output_type="tts_voice",
        chat_id=-1001,
        chat_title="گروه تست",
        preview="خلاصه صوتی صحبت‌ها...",
    )
    msg1 = DummyMsg(-1001)
    await handle_feedback_callback(DummyCallback(f"fb:{fb_id_voice}:up", msg1), cache)
    assert edited_markups[-1] is None
    assert len(admin_notifications) == 1
    assert admin_notifications[0][0] == 99999
    assert "👍" in admin_notifications[0][1]
    assert "حسین" in admin_notifications[0][1]

    # Clicking again does not duplicate admin notification
    await handle_feedback_callback(DummyCallback(f"fb:{fb_id_voice}:down", msg1), cache)
    assert len(admin_notifications) == 1

    # Case 2: Voice summary with TTS + Transcript buttons -> ONLY feedback row vanishes, other buttons remain!
    cid = cache.put(speaker_name="مهدیار", text="متن", summary="خلاصه", language="fa")
    fb_id_sum = cache.create_feedback_target(
        output_type="voice_summary",
        chat_id=-1001,
        chat_title="گروه تست",
        preview="خلاصه متنی...",
        linked_transcript_cache_id=cid,
    )
    entry = cache.get(cid)
    entry.fb_id = fb_id_sum
    msg2 = DummyMsg(-1001)
    await handle_feedback_callback(DummyCallback(f"fb:{fb_id_sum}:down", msg2), cache)
    remaining_cbs = [
        btn.callback_data
        for row in edited_markups[-1].inline_keyboard
        for btn in row
    ]
    assert f"tts:{cid}" in remaining_cbs
    assert f"transcript:{cid}" in remaining_cbs
    assert not any(cb.startswith("fb:") for cb in remaining_cbs)
    assert len(admin_notifications) == 2
    assert "👎" in admin_notifications[1][1]

    # Verify persistence across reload
    reloaded = TranscriptCache(
        persistence_path=tmp_path / "hist.json",
        feedback_path=fb_file,
        admin_config_path=admin_file,
    )
    assert len(reloaded.get_recent_chat_messages(-1001, limit=200)) >= 0
    stats = reloaded.get_feedback_stats()
    assert stats["up"] == 1
    assert stats["down"] == 1
    assert stats["total"] == 2


@pytest.mark.asyncio
async def test_chat_interaction_receives_at_least_200_messages_and_matches_named_members(tmp_path: Path):
    captured = []

    class DummySummarizer:
        async def chat(
            self,
            chat_history: str,
            session_turns: list[tuple[str, str]],
            question: str,
            asker_name: str,
            mode: str = "chat",
            language: str = "fa",
        ) -> SummaryResult:
            captured.append((chat_history, question, mode))
            return SummaryResult(summary="**سحر** گفت ساعت ۶ آماده باشیم.", language="fa", provider="gemini")

    class DummyMsg:
        def __init__(self, text: str):
            self.chat = SimpleNamespace(id=-100777, type="supergroup", title="گروه دوستان")
            self.text = text
            self.caption = None
            self.from_user = SimpleNamespace(id=7, full_name="مهدیار", first_name="مهدیار", is_bot=False)

        async def reply(self, text: str, reply_markup=None):
            return SimpleNamespace()

    cache = TranscriptCache(persistence_path=tmp_path / "hist.json")
    cache.add_chat_message(-100777, "سحر (@sahar)", "بچه‌ها ساعت ۶ کافه قرار بذاریم")
    for i in range(210):
        cache.add_chat_message(-100777, f"مهدی_{i % 3}", f"پیام تستی شماره {i}")

    await handle_chat_interaction(
        DummyMsg("@vaez_summary_bot سحر چی گفته؟"),
        DummySummarizer(),
        cache,
        bot_username="vaez_summary_bot",
    )
    history_passed, q_passed, mode_passed = captured[-1]
    assert "سحر (@sahar)" in history_passed
    assert "پیام تستی شماره 209" in history_passed
    assert history_passed.count("\n") + 1 == 211  # > 200 messages passed to Q&A!
    assert q_passed == "سحر چی گفته؟"
    assert mode_passed == "chat"

    await handle_chat_interaction(
        DummyMsg("@vaez_summary_bot تیپ شخصیت مهدی طبق این چتا چیه؟"),
        DummySummarizer(),
        cache,
        bot_username="vaez_summary_bot",
    )
    assert captured[-1][2] == "personality"


@pytest.mark.asyncio
async def test_active_users_stats_tracking_and_admin_only_command(tmp_path: Path):
    stats_file = tmp_path / "users_stats.json"
    admin_file = tmp_path / "admin.json"
    cache = TranscriptCache(
        users_stats_path=stats_file,
        admin_config_path=admin_file,
    )

    now = datetime(2026, 11, 13, 12, 0, tzinfo=timezone.utc)
    # 1. Direct active user within 24h
    cache.record_activity(
        user_id=5190248080,
        user_name="مهدیار",
        username="vaezm",
        chat_id=5190248080,
        chat_title="مهدیار",
        chat_type="private",
        is_direct=True,
        now=now - timedelta(hours=2),
    )
    # 2. Passive user in a supergroup 3 days ago
    cache.record_activity(
        user_id=2002,
        user_name="سحر",
        username="sahar",
        chat_id=-100999,
        chat_title="گروه بچه‌ها",
        chat_type="supergroup",
        is_direct=False,
        now=now - timedelta(days=3),
    )
    # 3. Older user 15 days ago
    cache.record_activity(
        user_id=3003,
        user_name="رضا",
        username=None,
        chat_id=-100999,
        chat_title="گروه بچه‌ها",
        chat_type="supergroup",
        is_direct=True,
        now=now - timedelta(days=15),
    )

    # Track update helper (ignores bots, classifies mention as direct)
    track_update_activity(
        SimpleNamespace(
            chat=SimpleNamespace(id=-100999, type="supergroup", title="گروه بچه‌ها"),
            from_user=SimpleNamespace(id=4004, full_name="سارا", username="sara", is_bot=False),
            text="@vaez_summary_bot سحر چی گفته؟",
            caption=None,
            reply_to_message=None,
        ),
        cache,
        {"id": 999, "username": "vaez_summary_bot", "first_name": "SummaryBot"},
    )
    track_update_activity(
        SimpleNamespace(
            chat=SimpleNamespace(id=-100999, type="supergroup", title="گروه بچه‌ها"),
            from_user=SimpleNamespace(id=999, full_name="Bot", is_bot=True),
            text="bot msg",
            caption=None,
        ),
        cache,
    )

    reloaded = TranscriptCache(users_stats_path=stats_file, admin_config_path=admin_file)
    ustats = reloaded.get_active_users_stats(now=now)
    assert ustats["total_users"] == 4
    assert ustats["direct_users"] == 3
    assert ustats["active_24h"] == 2  # user 5190248080 + user 4004
    assert ustats["active_7d"] == 3
    assert ustats["active_30d"] == 4
    assert ustats["group_chats"] == 1
    assert ustats["private_chats"] == 1

    replies = []

    class DummyCmdMsg:
        def __init__(self, uid: int):
            self.chat = SimpleNamespace(id=uid, type="private")
            self.from_user = SimpleNamespace(id=uid, full_name="User", is_bot=False)

        async def reply(self, text: str):
            replies.append(text)

    # Before admin is set -> prompts to run /set_admin
    await handle_stats_command(DummyCmdMsg(5190248080), reloaded)
    assert "/set_admin" in replies[-1]

    # Register admin -> non-admin rejected, admin gets full active user report
    reloaded.set_admin_chat_id(5190248080)
    await handle_stats_command(DummyCmdMsg(2002), reloaded)
    assert "فقط برای ادمین" in replies[-1]

    await handle_stats_command(DummyCmdMsg(5190248080), reloaded)
    assert "کاربران فعال" in replies[-1]
    assert "5190248080" in replies[-1]
    assert "سارا" in replies[-1] or "مهدیار" in replies[-1]


@pytest.mark.asyncio
async def test_per_chat_rate_limiter_5_per_5_minutes_and_token_exhaustion_message(tmp_path: Path):
    cache = TranscriptCache(
        persistence_path=tmp_path / "hist.json",
        rate_limit_max_requests=5,
        rate_limit_window_seconds=300,
    )

    # 1. Exact sliding window unit test: 5 allowed in chat A, 6th blocked, chat B untouched
    t0 = 1000.0
    for i in range(5):
        allowed, wait = cache.check_and_consume_chat_rate_limit(-1001, now=t0 + i)
        assert allowed is True
        assert wait == 0

    blocked, wait_sec = cache.check_and_consume_chat_rate_limit(-1001, now=t0 + 10)
    assert blocked is False
    assert wait_sec >= 289

    # Different chat is completely independent
    allowed_b, _ = cache.check_and_consume_chat_rate_limit(-1002, now=t0 + 10)
    assert allowed_b is True

    # After 5 minutes (300s) passes, chat A unlocks again
    allowed_after, _ = cache.check_and_consume_chat_rate_limit(-1001, now=t0 + 301)
    assert allowed_after is True

    # 2. Handler integration: 6th message blocked cleanly without calling LLM
    sent = []
    llm_calls = 0

    class DummySummarizer:
        async def chat(self, **kwargs):
            nonlocal llm_calls
            llm_calls += 1
            return SummaryResult(summary="پاسخ کوتاه و فان!", language="fa", provider="gemini")

    class DummyMsg:
        def __init__(self, chat_id: int, text: str):
            self.chat = SimpleNamespace(id=chat_id, type="supergroup", title="گروه تست")
            self.text = text
            self.caption = None
            self.from_user = SimpleNamespace(id=123, full_name="علی", username="ali", is_bot=False)

        async def reply(self, text: str, reply_markup=None):
            sent.append(text)
            return SimpleNamespace()

    chat_c = -100333
    for i in range(5):
        await handle_chat_interaction(
            DummyMsg(chat_c, f"@vaez_summary_bot سوال {i}"),
            DummySummarizer(),
            cache,
            bot_username="vaez_summary_bot",
        )
    assert llm_calls == 5

    # 6th call must be rejected by rate limiter (LLM not called)
    await handle_chat_interaction(
        DummyMsg(chat_c, "@vaez_summary_bot سوال ششم"),
        DummySummarizer(),
        cache,
        bot_username="vaez_summary_bot",
    )
    assert llm_calls == 5
    assert "محدودیت پیام برخوردید" in sent[-1]
    assert "زمان باقی‌مانده" in sent[-1]
    assert "5 پیام در هر 5 دقیقه" in sent[-1]

    # 3. Token/quota exhausted error sends the requested Persian admin-refill message
    class QuotaFailSummarizer:
        async def chat(self, **kwargs):
            raise ProviderChainExhaustedError(["gemini: HTTP 429 RESOURCE_EXHAUSTED quota exceeded"])

    await handle_chat_interaction(
        DummyMsg(-100444, "@vaez_summary_bot سحر چی گفته؟"),
        QuotaFailSummarizer(),
        cache,
        bot_username="vaez_summary_bot",
    )
    assert "فعلاً توکنام تموم شده" in sent[-1]
    assert "ادمینم" in sent[-1]
    assert "توکن رایگان" in sent[-1]


def test_help_text_and_fallback_chain_order():
    help_msg = build_help_text()
    assert "@vaez_summary_bot" in help_msg
    assert "/ask" in help_msg
    assert "/vibe" in help_msg
    assert "/chat_summary" in help_msg
    assert "/summary" in help_msg
    assert "۵ درخواست هوشمند در هر ۵ دقیقه" in help_msg

    settings = Settings(
        telegram_bot_token="123:ABC",
        gemini_api_key="AIza_test",
        fallback_llm_api_key="sk-fallback-test",
        groq_api_key="gsk_test",
        openrouter_api_key="sk-or-test",
        _env_file=None,
    )
    mgr = build_summarizer_manager(settings)
    names = [p.name for p in mgr.providers]
    assert names[:4] == ["gemini", "gemini", "gemini", "gemini"]
    assert names[4:6] == ["intern-ai:deepseek-v4.1-flash-1017", "intern-ai:qwen3.8-27b"]
    assert names[6:] == ["groq-llm", "openrouter-llm"]


@pytest.mark.asyncio
async def test_configure_bot_profile_respects_telegram_limits():
    captured = {}

    class DummyBot:
        async def set_my_short_description(self, short_description: str):
            captured["short"] = short_description

        async def set_my_description(self, description: str):
            captured["desc"] = description

        async def set_my_commands(self, commands):
            captured["commands"] = commands

    await configure_bot_profile(DummyBot())
    assert len(captured["short"]) <= 120
    assert len(captured["desc"]) <= 512
    assert "@vaez_summary_bot" in captured["desc"]
    assert "/help" in captured["desc"]
    cmd_names = [c.command for c in captured["commands"]]
    assert "help" in cmd_names
    assert "ask" in cmd_names
    assert "vibe" in cmd_names
