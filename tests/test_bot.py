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

    assert should_trigger_on_reply(private_reply, bot_username="my_custom_bot") is True
    assert should_trigger_on_reply(group_mention_reply, bot_username="vaez_summary_bot") is True
    assert should_trigger_on_reply(group_username_without_at_reply, bot_username="my_custom_bot") is True
    assert should_trigger_on_reply(group_bot_name_reply, bot_username="my_custom_bot") is True
    assert should_trigger_on_reply(group_keyword_reply, bot_username="my_custom_bot") is True
    assert should_trigger_on_reply(group_unrelated_reply, bot_username="my_custom_bot") is False


@pytest.mark.asyncio
async def test_handle_tts_callback_sends_voice_reply():
    sent_voices = []

    class DummyMsg:
        async def reply_voice(self, voice, caption=None):
            sent_voices.append((voice, caption))

    class DummyCallback:
        def __init__(self, cid: str):
            self.data = f"tts:{cid}"
            self.message = DummyMsg()

        async def answer(self, text: str | None = None, show_alert: bool = False):
            pass

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
    await handle_tts_callback(DummyCallback(cid), cache, DummyTTS())
    assert len(sent_voices) == 1
    assert "mr.vaez" in (sent_voices[0][1] or "")
