"""Telegram bot handlers for voice/audio messages, chat history summaries, replies, and inline callbacks."""

from __future__ import annotations

import logging
import re
import tempfile
from pathlib import Path
from typing import Any

from aiogram import F, Router
from aiogram.filters import Command
from aiogram.types import (
    BufferedInputFile,
    CallbackQuery,
    Message,
    ReplyKeyboardRemove,
)

from src.bot.cache import MAX_CHAT_HISTORY_PER_CHAT, TranscriptCache, chunk_text
from src.bot.extractors import extract_speaker_name, extract_voice_target
from src.bot.keyboards import (
    build_chat_summary_reply_keyboard,
    build_transcript_keyboard,
)
from src.exceptions import FileTooLargeError, ProviderChainExhaustedError
from src.services.stt.manager import STTManager
from src.services.summarizer.manager import SummarizerManager
from src.services.summarizer.prompts import CHAT_HISTORY_SPEAKER_PREFIX
from src.services.tts.manager import TTSManager

logger = logging.getLogger(__name__)

MAX_CHAT_SUMMARY_MESSAGES = MAX_CHAT_HISTORY_PER_CHAT

_EXACT_TRIGGER_PHRASES = frozenset(
    {
        "خلاصه",
        "خلاصه کن",
        "خلاصش کن",
        "خلاصه لطفا",
        "خلاصه لطفاً",
        "چی میگه",
        "چی گفت",
        "چی میگه تو ویس",
        "چی گفت تو ویس",
        "summary",
        "summarize",
        "tldr",
        "tl;dr",
    }
)

_COMMAND_PREFIXES = (
    "/summary",
    "/summarize",
    "/خلاصه",
    "/chat_summary",
    "/messages",
    "/history",
    "/summary_10",
    "/summary_30",
    "/summary_50",
    "/summary_100",
    "/summary_150",
)

_PERSIAN_DIGIT_MAP = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")


def _extract_requested_count(raw_text: str) -> int | None:
    """Extract an optional message count (clamped to 1..150) from command or text."""
    normalized = (raw_text or "").translate(_PERSIAN_DIGIT_MAP)
    cmd_match = re.search(r"/summary_(\d{1,4})\b", normalized)
    if cmd_match:
        return min(max(int(cmd_match.group(1)), 1), MAX_CHAT_SUMMARY_MESSAGES)

    match = re.search(r"\b(\d{1,4})\b", normalized)
    if not match:
        return None
    val = int(match.group(1))
    if val <= 0:
        return None
    return min(val, MAX_CHAT_SUMMARY_MESSAGES)


def _is_bottom_keyboard_selection(raw_text: str) -> bool:
    """Return True if the text came from tapping one of the bottom ReplyKeyboardMarkup buttons."""
    text = (raw_text or "").strip()
    return (
        text.startswith("📊 خلاصه")
        or text.startswith("🔥 خلاصه")
        or bool(
            re.match(
                r"^خلاصه\s+[۰-۹0-9]+\s+پیام",
                text,
            )
        )
    )


def _is_explicit_summary_text(
    raw_text: str,
    bot_username: str | None = None,
    bot_first_name: str | None = None,
) -> bool:
    """Return True only when text explicitly mentions the bot or is a direct summary command/phrase."""
    text = (raw_text or "").strip().lower()
    if not text:
        return False

    first_token = text.split()[0]
    if any(
        first_token == cmd or first_token.startswith(f"{cmd}@")
        for cmd in _COMMAND_PREFIXES
    ):
        return True

    clean_username = (bot_username or "vaez_summary_bot").lstrip("@").strip().lower()
    if clean_username and clean_username in text:
        return True

    clean_first_name = (bot_first_name or "voice-summary").strip().lower()
    if clean_first_name and clean_first_name in text:
        return True

    normalized = re.sub(r"[?؟!.،,\-_]+", " ", text)
    normalized = " ".join(normalized.split())
    return normalized in _EXACT_TRIGGER_PHRASES


def should_trigger_on_voice_message(
    message: Any,
    bot_username: str | None = None,
    bot_first_name: str | None = None,
) -> bool:
    """Return True if a direct voice/audio message should auto-summarize."""
    sender = getattr(message, "from_user", None)
    if getattr(sender, "is_bot", False):
        return False

    chat = getattr(message, "chat", None)
    chat_type = getattr(chat, "type", "private")
    if chat_type == "private":
        return True

    caption = getattr(message, "caption", None) or getattr(message, "text", None) or ""
    return _is_explicit_summary_text(
        caption, bot_username=bot_username, bot_first_name=bot_first_name
    )


def should_trigger_on_reply(
    message: Any,
    bot_username: str | None = None,
    bot_first_name: str | None = None,
) -> bool:
    """Return True if a reply message should trigger voice summarization."""
    sender = getattr(message, "from_user", None)
    if getattr(sender, "is_bot", False):
        return False

    replied = getattr(message, "reply_to_message", None)
    if replied is None:
        return False
    has_media = bool(
        getattr(replied, "voice", None) or getattr(replied, "audio", None)
    )
    if not has_media:
        return False

    text = getattr(message, "text", None) or getattr(message, "caption", None) or ""
    return _is_explicit_summary_text(
        text, bot_username=bot_username, bot_first_name=bot_first_name
    )


def _should_record_text(text: str) -> bool:
    if not text or text.startswith("/") or _is_bottom_keyboard_selection(text):
        return False
    normalized = re.sub(r"[?؟!.،,\-_]+", " ", text.lower())
    normalized = " ".join(normalized.split())
    if normalized in _EXACT_TRIGGER_PHRASES:
        return False
    return True


def record_incoming_chat_message(message: Any, cache: TranscriptCache) -> None:
    """Record non-command text/caption messages from a chat or channel into the rolling 150-message store."""
    sender_user = getattr(message, "from_user", None)
    if getattr(sender_user, "is_bot", False):
        return

    chat = getattr(message, "chat", None)
    chat_id = getattr(chat, "id", None)
    if chat_id is None:
        return

    # Also capture replied-to message text if someone replies to an earlier message in the group
    replied = getattr(message, "reply_to_message", None)
    if replied is not None:
        replied_sender = getattr(replied, "from_user", None)
        if not getattr(replied_sender, "is_bot", False):
            replied_text = (
                getattr(replied, "text", None) or getattr(replied, "caption", None) or ""
            ).strip()
            if _should_record_text(replied_text):
                cache.add_chat_message(
                    chat_id=chat_id,
                    sender=extract_speaker_name(replied),
                    text=replied_text,
                )

    text = (getattr(message, "text", None) or getattr(message, "caption", None) or "").strip()
    if not _should_record_text(text):
        return

    speaker = extract_speaker_name(message)
    cache.add_chat_message(chat_id=chat_id, sender=speaker, text=text)


def _build_voice_dedup_key(message: Any, media: Any) -> str:
    """Build a per-chat deduplication key for a target voice/audio message."""
    chat = getattr(message, "chat", None)
    chat_id = getattr(chat, "id", None)
    file_uid = (
        getattr(media, "file_unique_id", None)
        or getattr(media, "file_id", None)
        or ""
    )
    if not file_uid:
        return ""
    return f"{chat_id if chat_id is not None else 'private'}:{file_uid}"


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
            "🎙️ لطفاً یک ویس یا فایل صوتی بفرستید، یا روی ویس ریپلای بزنید (یا برای خلاصه پیام‌های گروه/کانال از دستور /chat_summary استفاده کنید)."
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

    voice_key = _build_voice_dedup_key(message, media)
    if not cache.try_acquire_voice(voice_key):
        logger.info("Skipping duplicate/in-flight summary request for voice_key=%s", voice_key)
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

        chat = getattr(message, "chat", None)
        chat_id = getattr(chat, "id", None)
        if chat_id is not None and stt_result.text:
            cache.add_chat_message(
                chat_id=chat_id,
                sender=f"{speaker_name} (ویس)",
                text=stt_result.text,
            )

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
        cache.mark_voice_completed(voice_key)

    except FileTooLargeError:
        cache.release_voice(voice_key)
        await status_msg.edit_text(
            f"⚠️ حجم فایل صوتی بیش از سقف مجاز ({max_file_size_mb} مگابایت) است."
        )
    except ProviderChainExhaustedError as exc:
        cache.release_voice(voice_key)
        logger.error("All providers failed during voice processing: %s", exc)
        await status_msg.edit_text(
            "❌ خطا در پردازش ویس: تمامی سرویس‌های هوش مصنوعی در حال حاضر با محدودیت مواجه شدند. لطفاً چند لحظه دیگر دوباره تلاش کنید."
        )
    except Exception as exc:  # noqa: BLE001
        cache.release_voice(voice_key)
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


async def _delete_and_send_output(
    origin_message: Any,
    status_msg: Any,
    text: str,
    reply_markup: Any = None,
) -> None:
    """Delete the temporary status message and send the final summary output cleanly."""
    if status_msg is not None and hasattr(status_msg, "delete"):
        try:
            await status_msg.delete()
            if origin_message is not None and hasattr(origin_message, "reply"):
                await origin_message.reply(text, reply_markup=reply_markup)
                return
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not delete status message, falling back to edit_text: %s", exc)

    if status_msg is not None and hasattr(status_msg, "edit_text"):
        await status_msg.edit_text(text, reply_markup=reply_markup)
    elif origin_message is not None and hasattr(origin_message, "reply"):
        await origin_message.reply(text, reply_markup=reply_markup)


async def _run_chat_history_summary(
    chat_id: int | str,
    requested_count: int,
    status_updater: Any,
    summarizer_manager: SummarizerManager,
    cache: TranscriptCache,
    origin_message: Any = None,
) -> None:
    """Summarize the last `requested_count` (up to 150) messages, delete status message, and output summary."""
    clamped_count = max(1, min(int(requested_count), MAX_CHAT_SUMMARY_MESSAGES))
    entries = cache.get_recent_chat_messages(chat_id, limit=clamped_count)
    if not entries:
        empty_msg = (
            "📭 هنوز پیامی از این گفتگو در حافظه ربات ثبت نشده است.\n\n"
            "💡 **چرا پیام‌های قبلی دیده نشد؟**\n"
            "۱. تلگرام به ربات‌ها اجازه دانلود پیام‌های قبل از عضویت را نمی‌دهد و ربات از این لحظه به بعد پیام‌ها را (تا ۱۵۰ پیام) ذخیره می‌کند.\n"
            "۲. برای اینکه ربات در گروه پیام‌های عادی بچه‌ها را ببیند، حتماً باید در گروه **Admin (ادمین)** باشد (یا در `@BotFather` گزینه `Group Privacy` روی `Turn off` گذاشته شود)."
        )
        await _delete_and_send_output(
            origin_message=origin_message,
            status_msg=status_updater,
            text=empty_msg,
            reply_markup=ReplyKeyboardRemove() if origin_message is not None else None,
        )
        return

    actual_count = len(entries)
    formatted_lines = [f"[{item.sender}]: {item.text}" for item in entries]
    combined_transcript = "\n".join(formatted_lines)

    try:
        summary_result = await summarizer_manager.summarize(
            transcript=combined_transcript,
            speaker_name=f"{CHAT_HISTORY_SPEAKER_PREFIX}{actual_count}",
            language="fa",
        )
        cache_id = cache.put(
            speaker_name=f"{actual_count} پیام آخر",
            text=combined_transcript,
            summary=summary_result.summary,
            language=summary_result.language,
        )
        entry = cache.get(cache_id)
        if entry is not None:
            entry.transcript_used = True
        keyboard = build_transcript_keyboard(
            cache_id=cache_id,
            language=summary_result.language,
            show_tts=True,
            show_transcript=False,
        )
        await _delete_and_send_output(
            origin_message=origin_message,
            status_msg=status_updater,
            text=summary_result.summary,
            reply_markup=keyboard,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to summarize chat history: %s", exc)
        await _delete_and_send_output(
            origin_message=origin_message,
            status_msg=status_updater,
            text="❌ متأسفانه در خلاصه‌سازی پیام‌های گفتگو خطایی رخ داد. لطفاً چند لحظه دیگر دوباره تلاش کنید.",
            reply_markup=None,
        )


async def handle_chat_summary_command(
    message: Any,
    summarizer_manager: SummarizerManager,
    cache: TranscriptCache,
    requested_count: int | None = None,
) -> None:
    """Show the bottom input ReplyKeyboardMarkup (10..150) or summarize & delete temporary messages."""
    chat = getattr(message, "chat", None)
    chat_id = getattr(chat, "id", "private")
    text = getattr(message, "text", None) or getattr(message, "caption", None) or ""
    count = requested_count if requested_count is not None else _extract_requested_count(text)

    if count is None:
        keyboard = build_chat_summary_reply_keyboard()
        prompt_msg = await message.reply(
            "👇 از دکمه‌های پایین صفحه (قسمت تایپ) تعداد پیام‌ها را برای خلاصه‌سازی انتخاب کنید:",
            reply_markup=keyboard,
        )
        cache.set_prompt_message_id(chat_id, getattr(prompt_msg, "message_id", None))
        return

    # Delete previous prompt message if one was sent
    prev_prompt_id = cache.pop_prompt_message_id(chat_id)
    bot = getattr(message, "bot", None)
    if prev_prompt_id is not None and bot is not None and hasattr(bot, "delete_message"):
        try:
            await bot.delete_message(chat_id=chat_id, message_id=prev_prompt_id)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not delete prompt message %s: %s", prev_prompt_id, exc)

    status_msg = await message.reply(
        f"⏳ در حال خلاصه‌سازی {min(count, MAX_CHAT_SUMMARY_MESSAGES)} پیام آخر...",
        reply_markup=ReplyKeyboardRemove(),
    )
    await _run_chat_history_summary(
        chat_id=chat_id,
        requested_count=count,
        status_updater=status_msg,
        summarizer_manager=summarizer_manager,
        cache=cache,
        origin_message=message,
    )


async def handle_chat_summary_callback(
    callback: Any,
    summarizer_manager: SummarizerManager,
    cache: TranscriptCache,
) -> None:
    """Handle click on `chatsum:<count>` inline buttons if used."""
    data = getattr(callback, "data", "") or ""
    raw_count = data.split(":", 1)[1] if ":" in data else "50"
    try:
        count = max(1, min(int(raw_count), MAX_CHAT_SUMMARY_MESSAGES))
    except ValueError:
        count = 50

    await callback.answer(f"⏳ خلاصه‌سازی {count} پیام آخر...")
    msg = getattr(callback, "message", None)
    if msg is None:
        return

    chat = getattr(msg, "chat", None)
    chat_id = getattr(chat, "id", "private")
    await _run_chat_history_summary(
        chat_id=chat_id,
        requested_count=count,
        status_updater=msg,
        summarizer_manager=summarizer_manager,
        cache=cache,
        origin_message=msg,
    )


async def _update_message_buttons(
    msg: Any, cache_id: str, entry: Any
) -> None:
    """Update the inline keyboard on the summary message to remove already-clicked buttons."""
    if msg is None or not hasattr(msg, "edit_reply_markup"):
        return
    updated_markup = build_transcript_keyboard(
        cache_id=cache_id,
        language=entry.language,
        show_tts=not entry.tts_used,
        show_transcript=not entry.transcript_used,
    )
    try:
        await msg.edit_reply_markup(reply_markup=updated_markup)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not update inline reply markup for %s: %s", cache_id, exc)


async def handle_transcript_callback(
    callback: Any, cache: TranscriptCache
) -> None:
    """Send the cached full transcript once and disable the transcript button."""
    data = getattr(callback, "data", "") or ""
    cache_id = data.split(":", 1)[1] if ":" in data else ""
    entry = cache.get(cache_id)

    if entry is None:
        await callback.answer(
            "⏳ متن کامل این ویس از حافظه موقت منقضی شده است. لطفاً دوباره ویس را ارسال کنید.",
            show_alert=True,
        )
        return

    if entry.transcript_used:
        await callback.answer("📜 متن کامل این ویس قبلاً ارسال شده است.", show_alert=False)
        return

    entry.transcript_used = True
    msg = getattr(callback, "message", None)
    await _update_message_buttons(msg, cache_id, entry)
    await callback.answer()

    header = f"📜 متن کامل صحبت‌های {entry.speaker_name}:\n\n"
    chunks = chunk_text(header + entry.text, max_len=4000)
    if msg is not None:
        for chunk in chunks:
            await msg.reply(chunk)


async def handle_tts_callback(
    callback: Any, cache: TranscriptCache, tts_manager: TTSManager
) -> None:
    """Synthesize and send the voice summary once and disable the audio summary button."""
    data = getattr(callback, "data", "") or ""
    cache_id = data.split(":", 1)[1] if ":" in data else ""
    entry = cache.get(cache_id)

    if entry is None:
        await callback.answer(
            "⏳ خلاصه این ویس از حافظه موقت منقضی شده است. لطفاً دوباره ویس را ارسال کنید.",
            show_alert=True,
        )
        return

    if entry.tts_used:
        await callback.answer("🔊 نسخه صوتی خلاصه قبلاً ارسال شده است.", show_alert=False)
        return

    entry.tts_used = True
    msg = getattr(callback, "message", None)
    await _update_message_buttons(msg, cache_id, entry)
    await callback.answer("🔊 در حال تولید فایل صوتی خلاصه...")

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
        entry.tts_used = False
        await _update_message_buttons(msg, cache_id, entry)
        logger.error("Failed to generate TTS audio summary: %s", exc)
        await msg.reply("❌ متأسفانه در تولید فایل صوتی خلاصه خطایی رخ داد.")


def create_bot_router(
    stt_manager: STTManager,
    summarizer_manager: SummarizerManager,
    cache: TranscriptCache,
    tts_manager: TTSManager | None = None,
    max_file_size_mb: int = 20,
) -> Router:
    """Create and wire the aiogram Router with all commands, voice handlers, chat history, and callbacks."""
    router = Router(name="voice_summary_router")
    bot_identity: dict[str, str | None] = {"username": None, "first_name": None}

    async def _ensure_bot_identity(message: Message) -> None:
        if bot_identity["username"] is None and message.bot is not None:
            me = await message.bot.get_me()
            bot_identity["username"] = getattr(me, "username", None)
            bot_identity["first_name"] = getattr(me, "first_name", None)

    @router.message(Command("start", "help"))
    async def cmd_start(message: Message) -> None:
        await message.reply(
            "👋 سلام! من ربات هوشمند خلاصه‌ساز ویس و پیام‌های گروه و کانال هستم.\n\n"
            "✨ قابلیت‌های من:\n"
            "۱. 🎙️ **خلاصه‌سازی ویس در چت خصوصی:** هر ویس یا فایل صوتی را مستقیم برایم بفرست یا فوروارد کن.\n"
            "۲. 👥 **خلاصه‌سازی ویس در گروه‌ها:** روی هر ویسی ریپلای بزن و بنویس `خلاصه` (یا دستور `/summary` یا منشن `@vaez_summary_bot`).\n"
            "۳. 💬 **خلاصه‌سازی پیام‌های گروه و چنل (۱۰ تا ۱۵۰ پیام آخر):** دستور `/chat_summary` را بزن تا دکمه‌های انتخاب تعداد در پایین صفحه (قسمت تایپ) باز شود، یا مستقیم از منوی `/` دستورات `/summary_10` تا `/summary_150` را انتخاب کن!\n"
            "۴. 🔊 **خلاصه صوتی:** زیر تمام خلاصه‌ها می‌توانی نسخه صوتی خلاصه را هم بشنوی.\n\n"
            "⚠️ **نکته مهم در گروه‌ها:** حتماً ربات را در گروه **Admin (ادمین)** کنید تا بتواند پیام‌های گروه را ببیند و خلاصه کند."
        )

    @router.message(
        Command(
            "chat_summary",
            "messages",
            "history",
            "summary_10",
            "summary_30",
            "summary_50",
            "summary_100",
            "summary_150",
        )
    )
    async def cmd_chat_summary(message: Message) -> None:
        await handle_chat_summary_command(
            message=message,
            summarizer_manager=summarizer_manager,
            cache=cache,
        )

    @router.message(Command("summary", "summarize", "خلاصه"))
    async def cmd_summary(message: Message) -> None:
        if extract_voice_target(message) is not None:
            await process_voice_request(
                message=message,
                bot=message.bot,
                stt_manager=stt_manager,
                summarizer_manager=summarizer_manager,
                cache=cache,
                max_file_size_mb=max_file_size_mb,
            )
        else:
            await handle_chat_summary_command(
                message=message,
                summarizer_manager=summarizer_manager,
                cache=cache,
            )

    @router.message(F.voice | F.audio)
    async def on_voice_or_audio(message: Message) -> None:
        await _ensure_bot_identity(message)
        if should_trigger_on_voice_message(
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

    @router.message(F.reply_to_message)
    async def on_reply_to_voice(message: Message) -> None:
        text = (getattr(message, "text", None) or getattr(message, "caption", None) or "").strip()
        if _is_bottom_keyboard_selection(text):
            await handle_chat_summary_command(
                message=message,
                summarizer_manager=summarizer_manager,
                cache=cache,
            )
            return

        record_incoming_chat_message(message, cache)
        await _ensure_bot_identity(message)
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

    @router.channel_post()
    async def on_channel_post(message: Message) -> None:
        text = (getattr(message, "text", None) or getattr(message, "caption", None) or "").strip()
        if text.startswith(("/chat_summary", "/summary", "/خلاصه")) or _is_bottom_keyboard_selection(text):
            await handle_chat_summary_command(
                message=message,
                summarizer_manager=summarizer_manager,
                cache=cache,
            )
            return
        record_incoming_chat_message(message, cache)

    @router.message()
    async def on_any_chat_message(message: Message) -> None:
        text = (getattr(message, "text", None) or getattr(message, "caption", None) or "").strip()
        lowered = text.lower()
        if _is_bottom_keyboard_selection(text) or any(
            phrase in lowered
            for phrase in ("خلاصه پیام", "خلاصه چت", "خلاصه گروه", "خلاصه چنل", "خلاصه کانال")
        ):
            await handle_chat_summary_command(
                message=message,
                summarizer_manager=summarizer_manager,
                cache=cache,
            )
            return
        record_incoming_chat_message(message, cache)

    @router.callback_query(F.data.startswith("chatsum:"))
    async def on_chat_summary_button(callback: CallbackQuery) -> None:
        await handle_chat_summary_callback(callback, summarizer_manager, cache)

    @router.callback_query(F.data.startswith("transcript:"))
    async def on_transcript_button(callback: CallbackQuery) -> None:
        await handle_transcript_callback(callback, cache)

    @router.callback_query(F.data.startswith("tts:"))
    async def on_tts_button(callback: CallbackQuery) -> None:
        if tts_manager is not None:
            await handle_tts_callback(callback, cache, tts_manager)

    return router
