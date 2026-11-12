"""Telegram bot handlers for voice/audio summaries, chat history, interactive Q&A, and feedback."""

from __future__ import annotations

import html
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
    InlineKeyboardMarkup,
    Message,
    ReplyKeyboardRemove,
)

from src.bot.cache import (
    MAX_CHAT_HISTORY_PER_CHAT,
    FeedbackEntry,
    TranscriptCache,
    chunk_text,
)
from src.bot.extractors import (
    extract_chat_sender_label,
    extract_speaker_name,
    extract_voice_target,
)
from src.bot.formatting import format_telegram_html
from src.bot.keyboards import (
    build_chat_summary_reply_keyboard,
    build_feedback_keyboard,
    build_transcript_keyboard,
)
from src.exceptions import FileTooLargeError, ProviderChainExhaustedError
from src.services.stt.manager import STTManager
from src.services.summarizer.manager import SummarizerManager
from src.services.summarizer.prompts import (
    CHAT_HISTORY_SPEAKER_PREFIX,
    sanitize_sensitive_text,
)
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
    "/summary_200",
    "/ask",
    "/vibe",
    "/personality",
    "/reset_chat",
    "/feedbacks",
    "/stats",
    "/users",
    "/set_admin",
)

_PERSONALITY_KEYWORDS = (
    "شخصیت",
    "روانشناسی",
    "کاراکتر",
    "وایب",
    "اخلاق",
    "تیپ شخصیتی",
    "personality",
    "vibe",
    "character",
    "roast",
)

_OUTPUT_TYPE_LABELS = {
    "voice_summary": "📝 خلاصه متنی ویس",
    "tts_voice": "🔊 نسخه صوتی خلاصه (TTS)",
    "transcript": "📜 متن کامل ویس",
    "chat_summary": "📊 خلاصه پیام‌های گروه/چنل",
    "chat_qa": "💬 چت تعاملی و شخصیت‌شناسی",
}

_PERSIAN_DIGIT_MAP = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")


def _extract_requested_count(raw_text: str) -> int | None:
    """Extract an optional message count (clamped to `MAX_CHAT_SUMMARY_MESSAGES`) from command or text."""
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
        or bool(re.match(r"^خلاصه\s+[۰-۹0-9]+\s+پیام", text))
    )


def _get_chat_title(message: Any) -> str:
    chat = getattr(message, "chat", None)
    if chat is None:
        return "چت خصوصی"
    title = getattr(chat, "title", None)
    if title:
        return str(title).strip()
    first = getattr(chat, "first_name", None) or getattr(chat, "username", None)
    return str(first).strip() if first else f"چت {getattr(chat, 'id', 'private')}"


def _strip_bot_mention_and_command(
    raw_text: str, bot_username: str | None = None
) -> str:
    """Remove leading /ask, /vibe, /personality command or @bot_username mention from query."""
    text = (raw_text or "").strip()
    clean_username = (bot_username or "vaez_summary_bot").lstrip("@").strip()
    if clean_username:
        text = re.sub(
            rf"@{re.escape(clean_username)}\b", "", text, flags=re.IGNORECASE
        )
    text = re.sub(
        r"^/(?:ask|vibe|personality)(?:@\S+)?\s*", "", text, flags=re.IGNORECASE
    )
    return " ".join(text.split())


def _mentions_bot(
    raw_text: str,
    bot_username: str | None = None,
    bot_first_name: str | None = None,
) -> bool:
    lowered = (raw_text or "").strip().lower()
    if not lowered:
        return False
    clean_username = (bot_username or "vaez_summary_bot").lstrip("@").strip().lower()
    if clean_username and clean_username in lowered:
        return True
    clean_first = (bot_first_name or "voice-summary").strip().lower()
    if clean_first and len(clean_first) >= 4 and clean_first in lowered:
        return True
    return False


def _is_explicit_summary_text(
    raw_text: str,
    bot_username: str | None = None,
    bot_first_name: str | None = None,
) -> bool:
    """Return True when a voice reply text explicitly requests voice summarization."""
    text = (raw_text or "").strip().lower()
    if not text:
        return False

    first_token = text.split()[0]
    if any(
        first_token == cmd or first_token.startswith(f"{cmd}@")
        for cmd in ("/summary", "/summarize", "/خلاصه")
    ):
        return True

    if _mentions_bot(text, bot_username=bot_username, bot_first_name=bot_first_name):
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


def should_trigger_chat_interaction(
    message: Any,
    bot_id: int | None = None,
    bot_username: str | None = None,
    bot_first_name: str | None = None,
) -> bool:
    """Return True if message should trigger interactive per-chat Q&A / fun conversation."""
    sender = getattr(message, "from_user", None)
    if getattr(sender, "is_bot", False):
        return False

    text = (getattr(message, "text", None) or getattr(message, "caption", None) or "").strip()
    if not text or text.startswith("/") or _is_bottom_keyboard_selection(text):
        return False

    chat = getattr(message, "chat", None)
    chat_type = getattr(chat, "type", "private")
    if chat_type == "private":
        return True

    replied = getattr(message, "reply_to_message", None)
    if replied is not None:
        replied_user = getattr(replied, "from_user", None)
        if replied_user is not None and getattr(replied_user, "is_bot", False):
            replied_id = getattr(replied_user, "id", None)
            replied_username = (getattr(replied_user, "username", None) or "").lower()
            target_username = (bot_username or "vaez_summary_bot").lstrip("@").lower()
            if (bot_id is not None and replied_id == bot_id) or (
                target_username and replied_username == target_username
            ):
                return True

    return _mentions_bot(
        text, bot_username=bot_username, bot_first_name=bot_first_name
    )


def _is_explicit_chat_summary_phrase(raw_text: str) -> bool:
    """Return True only when user explicitly asks for the group/channel recap (not a person Q&A)."""
    lowered = " ".join((raw_text or "").strip().lower().split())
    if not lowered or "?" in lowered or "؟" in lowered:
        return False
    return lowered in {
        "خلاصه چت",
        "خلاصه گروه",
        "خلاصه چنل",
        "خلاصه کانال",
        "خلاصه پیام",
        "خلاصه پیام ها",
        "خلاصه پیام‌ها",
        "خلاصه پیامهای گروه",
        "خلاصه پیام‌های گروه",
        "خلاصه گفتگو",
    }


def _should_record_text(text: str) -> bool:
    if not text or text.startswith("/") or _is_bottom_keyboard_selection(text):
        return False
    normalized = re.sub(r"[?؟!.،,\-_]+", " ", text.lower())
    normalized = " ".join(normalized.split())
    if normalized in _EXACT_TRIGGER_PHRASES:
        return False
    return True


def record_incoming_chat_message(message: Any, cache: TranscriptCache) -> None:
    """Record non-command text/caption messages strictly in that chat's isolated rolling store."""
    sender_user = getattr(message, "from_user", None)
    if getattr(sender_user, "is_bot", False):
        return

    chat = getattr(message, "chat", None)
    chat_id = getattr(chat, "id", None)
    if chat_id is None:
        return

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
                    sender=extract_chat_sender_label(replied),
                    text=replied_text,
                )

    text = (getattr(message, "text", None) or getattr(message, "caption", None) or "").strip()
    if not _should_record_text(text):
        return

    speaker = extract_chat_sender_label(message)
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


def _build_voice_summary_markup(
    cache: TranscriptCache, cache_id: str, entry: Any
) -> InlineKeyboardMarkup | None:
    """Build current inline keyboard state for a voice summary (unused action + feedback)."""
    show_feedback = False
    if entry.fb_id:
        fb_entry = cache.get_feedback_target(entry.fb_id)
        show_feedback = fb_entry is not None and fb_entry.rating is None

    return build_transcript_keyboard(
        cache_id=cache_id,
        language=entry.language,
        show_tts=not entry.tts_used,
        show_transcript=not entry.transcript_used,
        show_feedback=show_feedback,
        fb_id=entry.fb_id,
    )


_TOKEN_EXHAUSTED_MESSAGE = (
    "😅 فعلاً توکنام تموم شده! به ادمینم بگین توکن رایگان بریزه 🙏"
)
_QUOTA_ERROR_MARKERS = (
    "429",
    "402",
    "quota",
    "resource_exhausted",
    "rate",
    "limit",
    "token",
    "credit",
    "insufficient",
    "exhausted",
    "too many requests",
)


def _is_token_or_quota_error(exc: Exception) -> bool:
    """Return True when all providers failed due to token/quota/rate-limit exhaustion."""
    if isinstance(exc, ProviderChainExhaustedError):
        joined = " | ".join(exc.errors).lower()
        return bool(exc.errors) and any(m in joined for m in _QUOTA_ERROR_MARKERS)
    return any(m in str(exc).lower() for m in _QUOTA_ERROR_MARKERS)


def _format_wait_duration(seconds: int) -> str:
    mins, secs = divmod(max(1, int(seconds)), 60)
    if mins and secs:
        return f"{mins} دقیقه و {secs} ثانیه"
    if mins:
        return f"{mins} دقیقه"
    return f"{secs} ثانیه"


def _enforce_chat_rate_limit(chat_id: int | str, cache: TranscriptCache) -> str | None:
    """Return None if `chat_id` is within rate limit, else a clear Persian cooldown notice."""
    allowed, retry_after = cache.check_and_consume_chat_rate_limit(chat_id)
    if allowed:
        return None
    wait_str = _format_wait_duration(retry_after)
    return (
        "⛔ **شما به محدودیت پیام برخوردید!**\n"
        f"(سقف مجاز در این چت: **{cache.rate_limit_max_requests} پیام در هر "
        f"{max(1, cache.rate_limit_window_seconds // 60)} دقیقه**)\n"
        f"⏳ زمان باقی‌مانده: **{wait_str}**"
    )


def build_help_text(max_messages: int = MAX_CHAT_SUMMARY_MESSAGES) -> str:
    """Return comprehensive, structured Persian usage guide for /help."""
    return (
        "📖 **راهنمای کامل ربات هوشمند @vaez_summary_bot**\n\n"
        "۱. 🎙️ **خلاصه‌سازی ویس و فایل صوتی:**\n"
        "   • **چت خصوصی (پیوی):** ویس را مستقیم برای ربات بفرست تا متنی + صوتی خلاصه‌اش کند.\n"
        "   • **توی گروه:** روی هر ویسی ریپلای بزن و یکی از این‌ها را بفرست:\n"
        "     `خلاصه`  |  `/summary`  |  `@vaez_summary_bot`\n"
        "   • زیر خلاصه دکمه‌های **«🔊 شنیدن خلاصه صوتی»** و **«📜 متن کامل ویس»** هست.\n\n"
        f"۲. 💬 **خلاصه‌سازی پیام‌های گروه و چنل (تا {max_messages} پیام آخر):**\n"
        "   • دستور `/chat_summary` را بزن و از دکمه‌های پایین صفحه (`۱۰` تا `۲۰۰` پیام) انتخاب کن.\n"
        f"   • یا مستقیم عدد بفرست، مثلاً: `/chat_summary 250` یا `/summary_100`.\n\n"
        f"۳. 😎 **چت و سؤال درباره حرف‌های بچه‌ها (با حافظه {max_messages} پیام اخیرِ همین چت):**\n"
        "   • **با منشن آیدی ربات:** `@vaez_summary_bot سحر چی گفته؟`\n"
        "   • **با دستور `/ask`:** `/ask مهدی راجع به جلسه چی گفت؟`\n"
        "   • **با ریپلای:** روی پیام ربات یا یکی از بچه‌ها ریپلای بزن و با منشن `@vaez_summary_bot` بپرس «منظورش چیه؟»\n"
        "   • ⚠️ **نکته مهم:** چت کردن با ربات و خلاصه‌سازی گروه فقط شامل **پیام‌های متنی** می‌شود و ربات هیچ خبری از محتوای ویس‌ها، عکس‌ها، ویدیوها و سایر فایل‌های چندرسانه‌ای ندارد (برای خلاصه ویس باید روی خود ویس ریپلای بزنید).\n\n"
        "۴. 🎭 **تحلیل تیپ شخصیت، وایب و رُست طنز (`/vibe`):**\n"
        "   • `/vibe` ← تحلیل باحال و کل‌کل طنز با بچه‌های فعال گروه\n"
        "   • `/vibe مهدی` یا `@vaez_summary_bot مهدی رو رست کن` ← تحلیل یا **رُست طنزآمیز و پُررو** روی یک نفر\n"
        "   • ⚠️ ربات فقط وقتی از کسی **دیتای متنی واقعی** تو چت داشته باشد رُست باحال می‌زند و اگر پیامی ازش نباشد الکی چرت‌وپرت و شوخی بی‌مزه نمی‌بافد!\n\n"
        "۵. 🛡️ **سهمیه و حریم خصوصی:**\n"
        "   • تاریخچه و سشن هر گروه/چت کاملاً از بقیه چت‌ها جداست.\n"
        "   • برای جلوگیری از شلوغی، در هر گروه/چت حداکثر **۵ درخواست هوشمند در هر ۵ دقیقه** مجاز است.\n"
        "   • پاک کردن حافظه گفتگوی تعاملی همین چت: `/reset_chat`"
    )


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

    chat_obj = getattr(message, "chat", None)
    voice_chat_id = getattr(chat_obj, "id", None)
    if voice_chat_id is not None:
        rate_msg = _enforce_chat_rate_limit(voice_chat_id, cache)
        if rate_msg is not None:
            cache.release_voice(voice_key)
            await message.reply(format_telegram_html(rate_msg))
            return

    status_msg = await message.reply(f"⏳ در حال خلاصه‌سازی ویس «{html.escape(speaker_name, quote=False)}»...")
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
            chat_id=chat_id,
        )
        fb_id = cache.create_feedback_target(
            output_type="voice_summary",
            chat_id=chat_id if chat_id is not None else "private",
            chat_title=_get_chat_title(message),
            preview=summary_result.summary,
            language=summary_result.language,
            linked_transcript_cache_id=cache_id,
        )
        entry = cache.get(cache_id)
        if entry is not None:
            entry.fb_id = fb_id

        keyboard = build_transcript_keyboard(
            cache_id=cache_id,
            language=summary_result.language,
            show_tts=True,
            show_transcript=True,
            show_feedback=True,
            fb_id=fb_id,
        )

        await status_msg.edit_text(
            format_telegram_html(summary_result.summary),
            reply_markup=keyboard,
        )
        cache.mark_voice_completed(voice_key)

    except FileTooLargeError:
        cache.release_voice(voice_key)
        await status_msg.edit_text(
            f"⚠️ حجم فایل صوتی بیش از سقف مجاز ({max_file_size_mb} مگابایت) است."
        )
    except ProviderChainExhaustedError as exc:
        cache.release_voice(voice_key)
        logger.error("All providers failed during voice processing: %s", exc)
        err_reply = (
            _TOKEN_EXHAUSTED_MESSAGE
            if _is_token_or_quota_error(exc)
            else "❌ خطا در پردازش ویس: تمامی سرویس‌های هوش مصنوعی در حال حاضر با محدودیت مواجه شدند. لطفاً چند لحظه دیگر دوباره تلاش کنید."
        )
        await status_msg.edit_text(err_reply)
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
    """Summarize the last `requested_count` messages for `chat_id` (no TTS, with feedback)."""
    clamped_count = max(1, min(int(requested_count), MAX_CHAT_SUMMARY_MESSAGES))
    entries = cache.get_recent_chat_messages(chat_id, limit=clamped_count)
    if not entries:
        empty_msg = (
            "📭 هنوز پیامی از این گفتگو در حافظه ربات ثبت نشده است.\n\n"
            "💡 **چرا پیام‌های قبلی دیده نشد؟**\n"
            f"۱. تلگرام به ربات‌ها اجازه دانلود پیام‌های قبل از عضویت را نمی‌دهد و ربات از این لحظه به بعد پیام‌ها را (تا **{MAX_CHAT_SUMMARY_MESSAGES} پیام**) ذخیره می‌کند.\n"
            "۲. برای اینکه ربات در گروه پیام‌های عادی بچه‌ها را ببیند، حتماً باید در گروه **Admin (ادمین)** باشد (یا در `@BotFather` گزینه `Group Privacy` روی `Turn off` گذاشته شود)."
        )
        await _delete_and_send_output(
            origin_message=origin_message,
            status_msg=status_updater,
            text=format_telegram_html(empty_msg),
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
        fb_id = cache.create_feedback_target(
            output_type="chat_summary",
            chat_id=chat_id,
            chat_title=_get_chat_title(origin_message or status_updater),
            preview=summary_result.summary,
            language=summary_result.language,
        )
        fb_keyboard = build_feedback_keyboard(fb_id, language=summary_result.language)
        await _delete_and_send_output(
            origin_message=origin_message,
            status_msg=status_updater,
            text=format_telegram_html(summary_result.summary),
            reply_markup=fb_keyboard,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to summarize chat history: %s", exc)
        err_reply = (
            _TOKEN_EXHAUSTED_MESSAGE
            if _is_token_or_quota_error(exc)
            else "❌ متأسفانه در خلاصه‌سازی پیام‌های گفتگو خطایی رخ داد. لطفاً چند لحظه دیگر دوباره تلاش کنید."
        )
        await _delete_and_send_output(
            origin_message=origin_message,
            status_msg=status_updater,
            text=err_reply,
            reply_markup=None,
        )


async def handle_chat_interaction(
    message: Any,
    summarizer_manager: SummarizerManager,
    cache: TranscriptCache,
    bot_username: str | None = None,
    mode: str = "chat",
    reply_context: str | None = None,
) -> None:
    """Answer fun questions, analyze member personalities, or chat using ONLY `chat_id`'s isolated session."""
    chat = getattr(message, "chat", None)
    chat_id = getattr(chat, "id", "private")
    raw_text = (getattr(message, "text", None) or getattr(message, "caption", None) or "").strip()
    question = _strip_bot_mention_and_command(raw_text, bot_username=bot_username)

    if mode == "personality" and not question:
        question = "شخصیت و وایب اعضای فعال این گفتگو رو با لحن بامزه و فان تحلیل کن!"
    elif not question:
        await message.reply(
            format_telegram_html(
                "😎 پایه‌ام! درباره صحبت‌های بچه‌های همین چت هر سوالی داری بپرس:\n\n"
                "• مثلاً: `@vaez_summary_bot سحر چی گفته؟`\n"
                "• یا: `@vaez_summary_bot تیپ شخصیت مهدی طبق این چتا چیه؟`\n"
                "• یا با `/vibe` شخصیت بچه‌های گروه رو تحلیل کن!"
            )
        )
        return

    effective_question = (
        f"(در پاسخ به پیام «{reply_context}»)\n{question}" if reply_context else question
    )
    rate_msg = _enforce_chat_rate_limit(chat_id, cache)
    if rate_msg is not None:
        await message.reply(format_telegram_html(rate_msg))
        return

    detected_mode = mode
    if detected_mode == "chat" and any(kw in question.lower() for kw in _PERSONALITY_KEYWORDS):
        detected_mode = "personality"

    asker_name = extract_chat_sender_label(message)
    recent_messages = cache.get_recent_chat_messages(chat_id, limit=MAX_CHAT_SUMMARY_MESSAGES)
    chat_history = "\n".join(f"[{m.sender}]: {m.text}" for m in recent_messages)
    session_turns = cache.get_chat_session_turns(chat_id)

    try:
        result = await summarizer_manager.chat(
            chat_history=chat_history,
            session_turns=session_turns,
            question=effective_question,
            asker_name=asker_name,
            mode=detected_mode,
            language="fa",
        )
        safe_reply = sanitize_sensitive_text(result.summary)
        cache.add_chat_turn(
            chat_id=chat_id,
            user_name=asker_name,
            question=question,
            answer=safe_reply,
        )
        fb_id = cache.create_feedback_target(
            output_type="chat_qa",
            chat_id=chat_id,
            chat_title=_get_chat_title(message),
            preview=safe_reply,
            language=result.language,
        )
        fb_keyboard = build_feedback_keyboard(fb_id, language=result.language)
        await message.reply(
            format_telegram_html(safe_reply),
            reply_markup=fb_keyboard,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Chat interaction failed for chat_id=%s: %s", chat_id, exc)
        err_reply = (
            _TOKEN_EXHAUSTED_MESSAGE
            if _is_token_or_quota_error(exc)
            else "❌ الان مغزم یکم شلوغ شد، چند لحظه دیگه دوباره سوالتو بپرس!"
        )
        await message.reply(err_reply)


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

    prev_prompt_id = cache.pop_prompt_message_id(chat_id)
    bot = getattr(message, "bot", None)
    if prev_prompt_id is not None and bot is not None and hasattr(bot, "delete_message"):
        try:
            await bot.delete_message(chat_id=chat_id, message_id=prev_prompt_id)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not delete prompt message %s: %s", prev_prompt_id, exc)

    rate_msg = _enforce_chat_rate_limit(chat_id, cache)
    if rate_msg is not None:
        await message.reply(format_telegram_html(rate_msg), reply_markup=ReplyKeyboardRemove())
        return

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

    msg = getattr(callback, "message", None)
    if msg is None:
        await callback.answer()
        return

    chat = getattr(msg, "chat", None)
    chat_id = getattr(chat, "id", "private")
    rate_msg = _enforce_chat_rate_limit(chat_id, cache)
    if rate_msg is not None:
        await callback.answer("⏳ سقف استفاده این چت موقتاً پر شده!", show_alert=True)
        await msg.reply(format_telegram_html(rate_msg))
        return

    await callback.answer(f"⏳ خلاصه‌سازی {count} پیام آخر...")
    await _run_chat_history_summary(
        chat_id=chat_id,
        requested_count=count,
        status_updater=msg,
        summarizer_manager=summarizer_manager,
        cache=cache,
        origin_message=msg,
    )


async def _update_message_buttons(
    msg: Any, cache: TranscriptCache, cache_id: str, entry: Any
) -> None:
    """Update the inline keyboard on a voice summary message cleanly."""
    if msg is None or not hasattr(msg, "edit_reply_markup"):
        return
    updated_markup = _build_voice_summary_markup(cache, cache_id, entry)
    try:
        await msg.edit_reply_markup(reply_markup=updated_markup)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not update inline reply markup for %s: %s", cache_id, exc)


async def handle_transcript_callback(
    callback: Any, cache: TranscriptCache
) -> None:
    """Send the cached full transcript once (with feedback buttons) and disable transcript button."""
    data = getattr(callback, "data", "") or ""
    cache_id = data.split(":", 1)[1] if ":" in data else ""
    entry = cache.get(cache_id)

    if entry is None:
        await callback.answer(
            "⏳ متن کامل این ویس از حافظه موقت منقضی شده است. لطفاً دوباره ویس را ارسال کنید.",
            show_alert=True,
        )
        return

    msg = getattr(callback, "message", None)
    msg_chat_id = getattr(getattr(msg, "chat", None), "id", None) if msg is not None else None
    if entry.chat_id is not None and msg_chat_id is not None and entry.chat_id != str(msg_chat_id):
        await callback.answer("⛔ این دکمه متعلق به چت دیگری است.", show_alert=True)
        return

    if entry.transcript_used:
        await callback.answer("📜 متن کامل این ویس قبلاً ارسال شده است.", show_alert=False)
        return

    entry.transcript_used = True
    await _update_message_buttons(msg, cache, cache_id, entry)
    await callback.answer()

    header = f"📜 متن کامل صحبت‌های {entry.speaker_name}:\n\n"
    chunks = chunk_text(header + entry.text, max_len=4000)
    if msg is not None:
        chat = getattr(msg, "chat", None)
        chat_id = getattr(chat, "id", "private")
        fb_id = cache.create_feedback_target(
            output_type="transcript",
            chat_id=chat_id,
            chat_title=_get_chat_title(msg),
            preview=header + entry.text,
            language=entry.language,
        )
        fb_markup = build_feedback_keyboard(fb_id, language=entry.language)

        for idx, chunk in enumerate(chunks):
            safe_chunk = html.escape(chunk, quote=False)
            is_last = idx == len(chunks) - 1
            if is_last:
                await msg.reply(safe_chunk, reply_markup=fb_markup)
            else:
                await msg.reply(safe_chunk)


async def handle_tts_callback(
    callback: Any, cache: TranscriptCache, tts_manager: TTSManager
) -> None:
    """Synthesize and send the voice summary (with feedback buttons) and disable TTS button."""
    data = getattr(callback, "data", "") or ""
    cache_id = data.split(":", 1)[1] if ":" in data else ""
    entry = cache.get(cache_id)

    if entry is None:
        await callback.answer(
            "⏳ خلاصه این ویس از حافظه موقت منقضی شده است. لطفاً دوباره ویس را ارسال کنید.",
            show_alert=True,
        )
        return

    msg = getattr(callback, "message", None)
    msg_chat_id = getattr(getattr(msg, "chat", None), "id", None) if msg is not None else None
    if entry.chat_id is not None and msg_chat_id is not None and entry.chat_id != str(msg_chat_id):
        await callback.answer("⛔ این دکمه متعلق به چت دیگری است.", show_alert=True)
        return

    if entry.tts_used:
        await callback.answer("🔊 نسخه صوتی خلاصه قبلاً ارسال شده است.", show_alert=False)
        return

    entry.tts_used = True
    await _update_message_buttons(msg, cache, cache_id, entry)
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
        ext = "wav" if tts_result.audio_bytes.startswith(b"RIFF") else "mp3"
        audio_file = BufferedInputFile(
            tts_result.audio_bytes, filename=f"summary_{cache_id}.{ext}"
        )
        chat = getattr(msg, "chat", None)
        chat_id = getattr(chat, "id", "private")
        tts_fb_id = cache.create_feedback_target(
            output_type="tts_voice",
            chat_id=chat_id,
            chat_title=_get_chat_title(msg),
            preview=entry.summary,
            language=entry.language,
        )
        tts_fb_markup = build_feedback_keyboard(tts_fb_id, language=entry.language)
        await msg.reply_voice(
            voice=audio_file,
            caption=html.escape(caption, quote=False),
            reply_markup=tts_fb_markup,
        )
    except Exception as exc:  # noqa: BLE001
        entry.tts_used = False
        await _update_message_buttons(msg, cache, cache_id, entry)
        logger.error("Failed to generate TTS audio summary: %s", exc)
        err_reply = (
            _TOKEN_EXHAUSTED_MESSAGE
            if _is_token_or_quota_error(exc)
            else "❌ متأسفانه در تولید فایل صوتی خلاصه خطایی رخ داد."
        )
        await msg.reply(err_reply)


def _format_admin_feedback_notification(entry: FeedbackEntry) -> str:
    rating_label = "👍 عالی / مثبت" if entry.rating == "up" else "👎 ضعیف / نیاز به بهبود"
    type_label = _OUTPUT_TYPE_LABELS.get(entry.output_type, entry.output_type)
    username_str = f" (@{entry.voter_username})" if entry.voter_username else ""
    md_text = (
        "📊 **بازخورد جدید دریافت شد!**\n\n"
        f"• **امتیاز:** {rating_label}\n"
        f"• **نوع خروجی:** {type_label}\n"
        f"• **چت / گروه:** {entry.chat_title} (`{entry.chat_id}`)\n"
        f"• **کاربر:** {entry.voter_name}{username_str}\n"
        f"• **پیش‌نمایش خروجی:**\n`{entry.preview}`"
    )
    return format_telegram_html(md_text)


def track_update_activity(
    event: Any,
    cache: TranscriptCache,
    bot_identity: dict[str, Any] | None = None,
) -> None:
    """Record user and chat activity from any Message or CallbackQuery update."""
    sender = getattr(event, "from_user", None)
    if getattr(sender, "is_bot", False):
        return

    user_id = getattr(sender, "id", None)
    user_name = (
        (
            getattr(sender, "full_name", None)
            or getattr(sender, "first_name", None)
            or "کاربر"
        ).strip()
        if sender is not None
        else ""
    )
    username = getattr(sender, "username", None) if sender is not None else None

    msg = getattr(event, "message", None) if hasattr(event, "data") else event
    chat = getattr(msg, "chat", None)
    chat_id = getattr(chat, "id", None)
    chat_type = str(getattr(chat, "type", "private") or "private")
    chat_title = _get_chat_title(msg) if msg is not None else ""

    if hasattr(event, "data"):
        is_direct = True
    else:
        text = (
            getattr(event, "text", None) or getattr(event, "caption", None) or ""
        ).strip()
        ident = bot_identity or {}
        is_direct = bool(
            chat_type == "private"
            or text.startswith("/")
            or _is_bottom_keyboard_selection(text)
            or _is_explicit_chat_summary_phrase(text)
            or _is_explicit_summary_text(
                text,
                bot_username=ident.get("username"),
                bot_first_name=ident.get("first_name"),
            )
            or should_trigger_chat_interaction(
                event,
                bot_id=ident.get("id"),
                bot_username=ident.get("username"),
                bot_first_name=ident.get("first_name"),
            )
        )

    cache.record_activity(
        user_id=user_id,
        user_name=user_name,
        username=username,
        chat_id=chat_id,
        chat_title=chat_title,
        chat_type=chat_type,
        is_direct=is_direct,
    )


def format_stats_report(cache: TranscriptCache) -> str:
    """Build Persian HTML-ready Markdown report of active users, chats, feedbacks, and admin."""
    ustats = cache.get_active_users_stats()
    fstats = cache.get_feedback_stats()
    admin_id = cache.get_admin_chat_id()
    total_fb = fstats["total"]
    pct_fb = round((fstats["up"] / total_fb) * 100) if total_fb else 0

    lines = [
        "📈 **آمار کاربران فعال و وضعیت سرویس:**",
        "",
        f"• ⚡ کاربران فعال **۲۴ ساعت اخیر:** **{ustats['active_24h']}** نفر (تعامل مستقیم با ربات: **{ustats['direct_24h']}**)",
        f"• 🗓️ کاربران فعال **۷ روز اخیر:** **{ustats['active_7d']}** نفر",
        f"• 📆 کاربران فعال **۳۰ روز اخیر:** **{ustats['active_30d']}** نفر",
        f"• 👥 کل کاربران یکتای دیده‌شده: **{ustats['total_users']}** نفر (کاربران مستقیم: **{ustats['direct_users']}**)",
        "",
        f"• 💬 کل چت‌ها و گروه‌ها: **{ustats['total_chats']}** ({ustats['group_chats']} گروه/چنل | {ustats['private_chats']} پیوی | فعال ۷ روز اخیر: **{ustats['active_chats_7d']}**)",
        f"• 👍 بازخوردها: **{total_fb}** (👍 {fstats['up']} | 👎 {fstats['down']} | رضایت **{pct_fb}%**)",
        f"• 🛡️ ادمین فعلی: `{admin_id}`" if admin_id is not None else "• ⚠️ ادمین فعلی: هنوز تنظیم نشده (`/set_admin`)",
    ]

    recent_users = ustats.get("recent_users") or []
    if recent_users:
        lines.append("\n🕒 **آخرین کاربران فعال:**")
        for idx, u in enumerate(recent_users[:8], start=1):
            handle = f" (@{u['username']})" if u.get("username") else ""
            lines.append(
                f"{idx}. **{u['name']}**{handle} — تعامل مستقیم: `{u['direct_interactions']}`"
            )

    return format_telegram_html("\n".join(lines))


async def handle_stats_command(message: Any, cache: TranscriptCache) -> None:
    """Respond with active user & service analytics for the configured admin."""
    admin_id = cache.get_admin_chat_id()
    user_id = getattr(getattr(message, "from_user", None), "id", None)
    if admin_id is None:
        await message.reply(
            format_telegram_html(
                "⚠️ هنوز ادمین سرویس تنظیم نشده است!\n"
                "برای ثبت خودتان به‌عنوان ادمین و دیدن آمار + دریافت بازخوردها، دستور `/set_admin` را بزنید."
            )
        )
        return
    if user_id != admin_id:
        await message.reply("⛔ مشاهده آمار کاربران فعال سرویس فقط برای ادمین ربات مجاز است.")
        return

    await message.reply(format_stats_report(cache))


async def handle_feedback_callback(
    callback: Any, cache: TranscriptCache
) -> None:
    """Record thumbs up/down feedback once, make feedback buttons vanish, and notify admin."""
    data = getattr(callback, "data", "") or ""
    parts = data.split(":")
    if len(parts) < 3:
        await callback.answer("داده بازخورد نامعتبر است.")
        return

    fb_id = parts[1]
    rating = parts[2]
    voter = getattr(callback, "from_user", None)
    voter_id = getattr(voter, "id", None)
    voter_name = (
        getattr(voter, "full_name", None)
        or getattr(voter, "first_name", None)
        or "کاربر"
    )
    voter_username = getattr(voter, "username", None)

    entry, already_voted = cache.record_feedback(
        fb_id=fb_id,
        rating=rating,
        voter_id=voter_id,
        voter_name=voter_name,
        voter_username=voter_username,
    )

    msg = getattr(callback, "message", None)
    if entry is None:
        if msg is not None and hasattr(msg, "edit_reply_markup"):
            try:
                await msg.edit_reply_markup(reply_markup=None)
            except Exception:  # noqa: BLE001
                pass
        await callback.answer("ممنون! بازخوردت ثبت شد 🙏")
        return

    # Remove feedback buttons (preserving unused TTS/Transcript buttons on voice summary)
    if msg is not None and hasattr(msg, "edit_reply_markup"):
        new_markup: InlineKeyboardMarkup | None = None
        if entry.linked_transcript_cache_id:
            linked = cache.get(entry.linked_transcript_cache_id)
            if linked is not None:
                new_markup = build_transcript_keyboard(
                    cache_id=entry.linked_transcript_cache_id,
                    language=linked.language,
                    show_tts=not linked.tts_used,
                    show_transcript=not linked.transcript_used,
                    show_feedback=False,
                    fb_id=fb_id,
                )
        try:
            await msg.edit_reply_markup(reply_markup=new_markup)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not remove feedback markup for %s: %s", fb_id, exc)

    if already_voted:
        await callback.answer("بازخورد این پیام قبلاً ثبت شده است 🙏")
        return

    ack_text = (
        "👍 مرسی! خوشحالم که راضی بودی"
        if entry.rating == "up"
        else "📝 ممنون! بازخوردت ثبت شد تا بهتر بشم"
    )
    await callback.answer(ack_text)

    admin_chat_id = cache.get_admin_chat_id()
    bot = getattr(msg, "bot", None) if msg is not None else getattr(callback, "bot", None)
    if admin_chat_id is not None and bot is not None and hasattr(bot, "send_message"):
        try:
            await bot.send_message(
                chat_id=admin_chat_id,
                text=_format_admin_feedback_notification(entry),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to deliver feedback notification to admin %s: %s", admin_chat_id, exc)


def create_bot_router(
    stt_manager: STTManager,
    summarizer_manager: SummarizerManager,
    cache: TranscriptCache,
    tts_manager: TTSManager | None = None,
    max_file_size_mb: int = 20,
) -> Router:
    """Create and wire the aiogram Router with all commands, chat Q&A, voice handlers, and callbacks."""
    router = Router(name="voice_summary_router")
    bot_identity: dict[str, Any] = {"id": None, "username": None, "first_name": None}

    async def _ensure_bot_identity(message: Message) -> None:
        if bot_identity["username"] is None and message.bot is not None:
            me = await message.bot.get_me()
            bot_identity["id"] = getattr(me, "id", None)
            bot_identity["username"] = getattr(me, "username", None)
            bot_identity["first_name"] = getattr(me, "first_name", None)

    @router.message.outer_middleware()
    async def _track_message_mw(handler: Any, event: Message, data: dict[str, Any]) -> Any:
        try:
            track_update_activity(event, cache, bot_identity)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Message activity tracking skipped: %s", exc)
        return await handler(event, data)

    @router.callback_query.outer_middleware()
    async def _track_callback_mw(handler: Any, event: CallbackQuery, data: dict[str, Any]) -> Any:
        try:
            track_update_activity(event, cache, bot_identity)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Callback activity tracking skipped: %s", exc)
        return await handler(event, data)

    @router.message(Command("start"))
    async def cmd_start(message: Message) -> None:
        welcome = (
            "👋 سلام! من ربات هوشمند خلاصه‌ساز ویس، پیام‌های گروه و هم‌صحبت فان شما هستم.\n\n"
            "✨ **کارهای اصلی من:**\n"
            "• 🎙️ خلاصه خودمونی ویس‌ها (پیوی یا ریپلای در گروه با `/summary` و `خلاصه`) + نسخه صوتی\n"
            f"• 💬 خلاصه ۱۰ تا {MAX_CHAT_SUMMARY_MESSAGES} پیام آخر گروه و چنل با `/chat_summary`\n"
            "• 😎 چت و سؤال درباره بچه‌ها با `@vaez_summary_bot` یا `/ask`\n"
            "• 🎭 تحلیل شخصیت و وایب فان بچه‌های گروه با `/vibe`\n\n"
            "📖 برای دیدن **راهنمای کامل و مثال‌ها** دستور `/help` را بزنید!"
        )
        await message.reply(format_telegram_html(welcome))

    @router.message(Command("help"))
    async def cmd_help(message: Message) -> None:
        await message.reply(format_telegram_html(build_help_text()))

    @router.message(Command("set_admin"))
    async def cmd_set_admin(message: Message) -> None:
        current_admin = cache.get_admin_chat_id()
        user_id = getattr(getattr(message, "from_user", None), "id", None)
        chat_id = getattr(getattr(message, "chat", None), "id", None)
        if user_id is None or chat_id is None:
            return
        if current_admin is not None and current_admin != user_id:
            await message.reply("⛔ ادمین دریافت‌کننده بازخوردها قبلاً توسط مالک ربات ثبت شده است.")
            return
        target_chat = user_id if getattr(message.chat, "type", "private") == "private" else user_id
        cache.set_admin_chat_id(target_chat)
        await message.reply(
            format_telegram_html(
                f"✅ **تنظیم شد!** از این به بعد خروجی تمام بازخوردهای 👍/👎 بلافاصله برای شما (`{target_chat}`) ارسال می‌شود.\n"
                "• با `/stats` تعداد کاربران فعال و وضعیت سرویس را ببینید.\n"
                "• با `/feedbacks` آمار کل بازخوردها را ببینید."
            )
        )

    @router.message(Command("stats", "users"))
    async def cmd_stats(message: Message) -> None:
        await handle_stats_command(message, cache)

    @router.message(Command("feedbacks"))
    async def cmd_feedbacks(message: Message) -> None:
        admin_id = cache.get_admin_chat_id()
        user_id = getattr(getattr(message, "from_user", None), "id", None)
        if admin_id is None:
            await message.reply(
                format_telegram_html(
                    "⚠️ هنوز ادمین سرویس تنظیم نشده است! لطفاً ابتدا `/set_admin` را بزنید."
                )
            )
            return
        if user_id != admin_id:
            await message.reply("⛔ مشاهده گزارش کامل بازخوردها فقط برای ادمین ربات مجاز است.")
            return
        stats = cache.get_feedback_stats()
        ustats = cache.get_active_users_stats()
        total = stats["total"]
        up = stats["up"]
        down = stats["down"]
        pct = round((up / total) * 100) if total else 0
        lines = [
            "📊 **گزارش کلی بازخوردها و سرویس:**",
            f"• 👥 کاربران فعال ۲۴ساعت: **{ustats['active_24h']}** نفر | کل کاربران: **{ustats['total_users']}** نفر (`/stats`)",
            f"• کل بازخوردها: **{total}** | 👍 عالی: **{up}** | 👎 ضعیف: **{down}** | رضایت: **{pct}%**",
        ]
        recent: list[FeedbackEntry] = stats["recent"]
        if recent:
            lines.append("\n🕒 **آخرین بازخوردها:**")
            for item in reversed(recent[-8:]):
                icon = "👍" if item.rating == "up" else "👎"
                tlabel = _OUTPUT_TYPE_LABELS.get(item.output_type, item.output_type)
                lines.append(
                    f"• {icon} {tlabel} | **{item.voter_name}** در **{item.chat_title}**: `{item.preview[:60]}`"
                )
        else:
            lines.append("\nهنوز بازخوردی ثبت نشده است.")
        await message.reply(format_telegram_html("\n".join(lines)))

    @router.message(Command("vibe", "personality"))
    async def cmd_vibe(message: Message) -> None:
        await _ensure_bot_identity(message)
        await handle_chat_interaction(
            message=message,
            summarizer_manager=summarizer_manager,
            cache=cache,
            bot_username=bot_identity["username"],
            mode="personality",
        )

    @router.message(Command("ask"))
    async def cmd_ask(message: Message) -> None:
        await _ensure_bot_identity(message)
        await handle_chat_interaction(
            message=message,
            summarizer_manager=summarizer_manager,
            cache=cache,
            bot_username=bot_identity["username"],
            mode="chat",
        )

    @router.message(Command("reset_chat"))
    async def cmd_reset_chat(message: Message) -> None:
        chat_id = getattr(getattr(message, "chat", None), "id", "private")
        cache.clear_chat_session(chat_id)
        await message.reply("🧹 حافظه گفتگوی تعاملی این سشن ریست شد! هر وقت خواستید بحث جدید را شروع کنید.")

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
            "summary_200",
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
    async def on_reply_to_message(message: Message) -> None:
        text = (getattr(message, "text", None) or getattr(message, "caption", None) or "").strip()
        if _is_bottom_keyboard_selection(text):
            await handle_chat_summary_command(
                message=message,
                summarizer_manager=summarizer_manager,
                cache=cache,
            )
            return

        await _ensure_bot_identity(message)
        replied_msg = getattr(message, "reply_to_message", None)
        replied_user = getattr(replied_msg, "from_user", None)
        replied_is_self = bool(
            replied_user is not None
            and getattr(replied_user, "is_bot", False)
            and (
                (bot_identity["id"] is not None and getattr(replied_user, "id", None) == bot_identity["id"])
                or (
                    bot_identity["username"]
                    and (getattr(replied_user, "username", None) or "").lower()
                    == str(bot_identity["username"]).lower()
                )
            )
        )

        if replied_is_self and should_trigger_chat_interaction(
            message,
            bot_id=bot_identity["id"],
            bot_username=bot_identity["username"],
            bot_first_name=bot_identity["first_name"],
        ):
            await handle_chat_interaction(
                message=message,
                summarizer_manager=summarizer_manager,
                cache=cache,
                bot_username=bot_identity["username"],
            )
            return

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
            return

        if should_trigger_chat_interaction(
            message,
            bot_id=bot_identity["id"],
            bot_username=bot_identity["username"],
            bot_first_name=bot_identity["first_name"],
        ):
            record_incoming_chat_message(replied_msg, cache)
            replied_sender_label = extract_chat_sender_label(replied_msg)
            replied_body = (
                getattr(replied_msg, "text", None) or getattr(replied_msg, "caption", None) or ""
            ).strip()
            reply_ctx = (
                f"{replied_sender_label}: {replied_body[:200]}"
                if replied_body
                else None
            )
            await handle_chat_interaction(
                message=message,
                summarizer_manager=summarizer_manager,
                cache=cache,
                bot_username=bot_identity["username"],
                reply_context=reply_ctx,
            )
            return

        record_incoming_chat_message(message, cache)

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
        if _is_bottom_keyboard_selection(text) or _is_explicit_chat_summary_phrase(text):
            await handle_chat_summary_command(
                message=message,
                summarizer_manager=summarizer_manager,
                cache=cache,
            )
            return

        await _ensure_bot_identity(message)
        if should_trigger_chat_interaction(
            message,
            bot_id=bot_identity["id"],
            bot_username=bot_identity["username"],
            bot_first_name=bot_identity["first_name"],
        ):
            await handle_chat_interaction(
                message=message,
                summarizer_manager=summarizer_manager,
                cache=cache,
                bot_username=bot_identity["username"],
            )
            return

        record_incoming_chat_message(message, cache)

    @router.callback_query(F.data.startswith("fb:"))
    async def on_feedback_button(callback: CallbackQuery) -> None:
        await handle_feedback_callback(callback, cache)

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
