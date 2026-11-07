"""Telegram bot handlers for voice/audio messages, replies, and inline callbacks."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

from aiogram import F, Router
from aiogram.filters import Command
from aiogram.types import BufferedInputFile, CallbackQuery, Message

from src.bot.cache import TranscriptCache, chunk_text
from src.bot.extractors import extract_voice_target
from src.bot.keyboards import build_transcript_keyboard
from src.exceptions import FileTooLargeError, ProviderChainExhaustedError
from src.services.stt.manager import STTManager
from src.services.summarizer.manager import SummarizerManager
from src.services.tts.manager import TTSManager

logger = logging.getLogger(__name__)

_REPLY_TRIGGER_KEYWORDS = (
    "خلاصه",
    "چی میگه",
    "چی گفت",
    "summary",
    "summarize",
    "tldr",
)


def should_trigger_on_reply(
    message: Any,
    bot_username: str | None = None,
    bot_first_name: str | None = None,
) -> bool:
    """Return True if a reply message should trigger voice summarization."""
    replied = getattr(message, "reply_to_message", None)
    if replied is None:
        return False
    has_media = bool(
        getattr(replied, "voice", None) or getattr(replied, "audio", None)
    )
    if not has_media:
        return False

    chat = getattr(message, "chat", None)
    chat_type = getattr(chat, "type", "private")
    if chat_type == "private":
        return True

    text = (getattr(message, "text", None) or getattr(message, "caption", None) or "").lower()
    clean_username = (bot_username or "vaez_summary_bot").lstrip("@").strip().lower()
    if clean_username and clean_username in text:
        return True

    clean_first_name = (bot_first_name or "voice-summary").strip().lower()
    if clean_first_name and clean_first_name in text:
        return True

    return any(kw in text for kw in _REPLY_TRIGGER_KEYWORDS)


async def process_voice_request(
    message: Any,
    bot: Any,
    stt_manager: STTManager,
    summarizer_manager: SummarizerManager,
    cache: TranscriptCache,
    max_file_size_mb: int = 20,
) -> None:
    """Orchestrate downloading, transcribing, and summarizing a voice/audio message."""
    target = extract_voice_target(message)
    if target is None:
        await message.reply(
            "🎙️ لطفاً یک ویس یا فایل صوتی بفرستید، یا روی ویس شخص دیگری ریپلای بزنید."
        )
        return

    media, speaker_name = target
    file_size = getattr(media, "file_size", 0) or 0
    max_bytes = max_file_size_mb * 1024 * 1024
    if file_size > max_bytes:
        await message.reply(
            f"⚠️ حجم فایل صوتی بیش از حد مجاز ({max_file_size_mb} مگابایت) است."
        )
        return

    status_msg = await message.reply(f"⏳ در حال خلاصه‌سازی ویس «{speaker_name}»...")
    tmp_path: Path | None = None

    try:
        file_info = await bot.get_file(media.file_id)
        suffix = Path(getattr(file_info, "file_path", "") or "voice.ogg").suffix or ".ogg"

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)

        await bot.download_file(file_info.file_path, destination=tmp_path)

        stt_result = await stt_manager.transcribe(tmp_path)

        summary_result = await summarizer_manager.summarize(
            transcript=stt_result.text,
            speaker_name=speaker_name,
            language=stt_result.language,
        )

        cache_id = cache.put(
            speaker_name=speaker_name,
            text=stt_result.text,
            summary=summary_result.summary,
            language=summary_result.language,
        )
        keyboard = build_transcript_keyboard(
            cache_id=cache_id, language=summary_result.language
        )

        await status_msg.edit_text(summary_result.summary, reply_markup=keyboard)

    except FileTooLargeError:
        await status_msg.edit_text(
            f"⚠️ حجم فایل صوتی بیش از سقف مجاز ({max_file_size_mb} مگابایت) است."
        )
    except ProviderChainExhaustedError as exc:
        logger.error("All providers failed during voice processing: %s", exc)
        await status_msg.edit_text(
            "❌ خطا در پردازش ویس: تمامی سرویس‌های هوش مصنوعی در حال حاضر با محدودیت مواجه شدند. لطفاً چند لحظه دیگر دوباره تلاش کنید."
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error processing voice message: %s", exc)
        await status_msg.edit_text(
            "❌ متأسفانه خطایی هنگام پردازش ویس رخ داد. لطفاً دوباره امتحان کنید."
        )
    finally:
        if tmp_path is not None and tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError as unlink_err:
                logger.warning("Failed to delete temp file %s: %s", tmp_path, unlink_err)


async def handle_transcript_callback(
    callback: Any, cache: TranscriptCache
) -> None:
    """Send the cached full transcript when the user clicks the inline button."""
    data = getattr(callback, "data", "") or ""
    cache_id = data.split(":", 1)[1] if ":" in data else ""
    entry = cache.get(cache_id)

    if entry is None:
        await callback.answer(
            "⏳ متن کامل این ویس از حافظه موقت منقضی شده است. لطفاً دوباره ویس را ارسال کنید.",
            show_alert=True,
        )
        return

    await callback.answer()
    header = f"📜 متن کامل صحبت‌های {entry.speaker_name}:\n\n"
    chunks = chunk_text(header + entry.text, max_len=4000)
    msg = getattr(callback, "message", None)
    if msg is not None:
        for chunk in chunks:
            await msg.reply(chunk)


async def handle_tts_callback(
    callback: Any, cache: TranscriptCache, tts_manager: TTSManager
) -> None:
    """Synthesize and send the voice summary when the user clicks the audio summary button."""
    data = getattr(callback, "data", "") or ""
    cache_id = data.split(":", 1)[1] if ":" in data else ""
    entry = cache.get(cache_id)

    if entry is None:
        await callback.answer(
            "⏳ خلاصه این ویس از حافظه موقت منقضی شده است. لطفاً دوباره ویس را ارسال کنید.",
            show_alert=True,
        )
        return

    await callback.answer("🔊 در حال تولید فایل صوتی خلاصه...")
    msg = getattr(callback, "message", None)
    if msg is None:
        return

    try:
        tts_result = await tts_manager.synthesize(
            text=entry.summary, language=entry.language
        )
        caption = (
            f"🔊 نسخه صوتی خلاصه صحبت‌های {entry.speaker_name}"
            if entry.language == "fa"
            else f"🔊 Audio summary of {entry.speaker_name}'s voice"
        )
        audio_file = BufferedInputFile(
            tts_result.audio_bytes, filename=f"summary_{cache_id}.mp3"
        )
        await msg.reply_voice(voice=audio_file, caption=caption)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to generate TTS audio summary: %s", exc)
        await msg.reply("❌ متأسفانه در تولید فایل صوتی خلاصه خطایی رخ داد.")


def create_bot_router(
    stt_manager: STTManager,
    summarizer_manager: SummarizerManager,
    cache: TranscriptCache,
    tts_manager: TTSManager | None = None,
    max_file_size_mb: int = 20,
) -> Router:
    """Create and wire the aiogram Router with all commands, voice handlers, and callbacks."""
    router = Router(name="voice_summary_router")

    @router.message(Command("start", "help"))
    async def cmd_start(message: Message) -> None:
        await message.reply(
            "👋 سلام! من ربات هوشمند خلاصه‌ساز ویس هستم.\n\n"
            "✨ چطور از من استفاده کنی؟\n"
            "۱. در چت خصوصی: هر ویس یا فایل صوتی را مستقیم برایم بفرست یا فوروارد کن.\n"
            "۲. در گروه‌ها: روی هر ویسی ریپلای بزن و بنویس خلاصه (یا دستور /summary).\n\n"
            "من زبان ویس (فارسی یا انگلیسی) و نام گوینده را تشخیص می‌دهم و خلاصه‌ی دقیق صحبت‌هایش را به صورت متنی و صوتی تحویلت می‌دهم!"
        )

    @router.message(Command("summary", "summarize", "خلاصه"))
    async def cmd_summary(message: Message) -> None:
        await process_voice_request(
            message=message,
            bot=message.bot,
            stt_manager=stt_manager,
            summarizer_manager=summarizer_manager,
            cache=cache,
            max_file_size_mb=max_file_size_mb,
        )

    @router.message(F.voice | F.audio)
    async def on_voice_or_audio(message: Message) -> None:
        await process_voice_request(
            message=message,
            bot=message.bot,
            stt_manager=stt_manager,
            summarizer_manager=summarizer_manager,
            cache=cache,
            max_file_size_mb=max_file_size_mb,
        )

    bot_identity: dict[str, str | None] = {"username": None, "first_name": None}

    @router.message(F.reply_to_message)
    async def on_reply_to_voice(message: Message) -> None:
        if bot_identity["username"] is None and message.bot is not None:
            me = await message.bot.get_me()
            bot_identity["username"] = getattr(me, "username", None)
            bot_identity["first_name"] = getattr(me, "first_name", None)

        if should_trigger_on_reply(
            message,
            bot_username=bot_identity["username"],
            bot_first_name=bot_identity["first_name"],
        ):
            await process_voice_request(
                message=message,
                bot=message.bot,
                stt_manager=stt_manager,
                summarizer_manager=summarizer_manager,
                cache=cache,
                max_file_size_mb=max_file_size_mb,
            )

    @router.callback_query(F.data.startswith("transcript:"))
    async def on_transcript_button(callback: CallbackQuery) -> None:
        await handle_transcript_callback(callback, cache)

    @router.callback_query(F.data.startswith("tts:"))
    async def on_tts_button(callback: CallbackQuery) -> None:
        if tts_manager is not None:
            await handle_tts_callback(callback, cache, tts_manager)

    return router
