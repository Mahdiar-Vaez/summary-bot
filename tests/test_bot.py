from pathlib import Path
from types import SimpleNamespace
import pytest
from src.bot.cache import TranscriptCache, chunk_text
from src.bot.extractors import extract_voice_target
from src.bot.handlers import (
    create_bot_router,
    handle_transcript_callback,
    handle_tts_callback,
    process_voice_request,
    should_trigger_on_reply,
    should_trigger_on_voice_message,
)
from src.config import Settings
from src.exceptions import ProviderChainExhaustedError
from src.main import build_stt_manager, build_summarizer_manager
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
    assert "خطا" in status_edits[-1][0]


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

    stt_models = [getattr(p, "model", "") for p in stt_mgr.providers]
    sum_models = [getattr(p, "model", "") for p in sum_mgr.providers]

    assert stt_models == [
        "gemini-3.5-flash",
        "gemini-3.6-flash",
        "gemini-3.5-flash-lite",
        "gemini-3.1-flash-lite",
    ]
    assert sum_models == [
        "gemini-3.5-flash",
        "gemini-3.6-flash",
        "gemini-3.5-flash-lite",
        "gemini-3.1-flash-lite",
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
        async def reply_voice(self, voice, caption=None):
            sent_voices.append((voice, caption))

        async def reply(self, text: str):
            sent_transcripts.append(text)

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
