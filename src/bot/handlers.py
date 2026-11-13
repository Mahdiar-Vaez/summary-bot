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
from src.bot.formatting import format_telegram_html, split_telegram_html
from src.bot.keyboards import (
    build_chat_summary_count_keyboard,
    build_chat_summary_reply_keyboard,
    build_feedback_keyboard,
    build_transcript_keyboard,
    build_vibe_target_keyboard,
)
from src.exceptions import FileTooLargeError, ProviderChainExhaustedError
from src.services.http_utils import redact_secrets
from src.services.stt.base import normalize_language
from src.services.stt.manager import STTManager
from src.services.summarizer.manager import SummarizerManager
from src.services.summarizer.prompts import (
    CHAT_HISTORY_SPEAKER_PREFIX,
    _trim_history_to_recent_chars,
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
    return bool(
        re.fullmatch(r"(?:[📊🔥]\s*)?خلاصه\s+[۰-۹0-9]+\s+پیام(?:\s+(?:اخیر|آخر))?", text)
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


_INVISIBLE_LEADING_CHARS = "\u200e\u200f\u200b\u200c\ufeff \t\r\n"
_RELAXED_CMD_RE = re.compile(
    r"^(?:/(?P<cmd>[A-Za-z0-9_\u0621-\u064A\u067E\u0686\u0698\u06A9\u06AF\u06CC]+)(?:@(?P<mention>[A-Za-z0-9_]+))?"
    r"|(?P<cmd_post>[A-Za-z0-9_]+)(?:@(?P<mention_post>[A-Za-z0-9_]+))?/)"
    r"(?=$|[\s\u200c\u200e\u200f:،؛؟,.?!\-])"
)
_INTERACTIVE_CHAT_COMMANDS = frozenset({"ask", "vibe", "personality"})
_CHAT_SUMMARY_COMMANDS = frozenset(
    {
        "chat_summary",
        "messages",
        "history",
        "summary_10",
        "summary_30",
        "summary_50",
        "summary_100",
        "summary_150",
        "summary_200",
    }
)
_SUMMARY_COMMANDS = frozenset({"summary", "summarize", "خلاصه"})


def _is_disallowed_bot_sender(message: Any) -> bool:
    """Return True for third-party bots, while allowing anonymous group admins / channels with sender_chat."""
    sender = getattr(message, "from_user", None)
    if not getattr(sender, "is_bot", False):
        return False
    return getattr(message, "sender_chat", None) is None


def _match_relaxed_command(
    raw_text: str,
    commands: frozenset[str] | tuple[str, ...],
    bot_username: str | None = None,
) -> str | None:
    """Match slash commands tolerating case, RTL marks, ZWNJ, trailing slash (`ask/`), and punctuation."""
    cleaned = (raw_text or "").lstrip(_INVISIBLE_LEADING_CHARS)
    match = _RELAXED_CMD_RE.match(cleaned)
    if not match:
        return None
    raw_cmd = match.group("cmd") or match.group("cmd_post") or ""
    cmd = raw_cmd.lower()
    if cmd not in commands:
        return None
    mention = match.group("mention") or match.group("mention_post")
    if mention:
        target = (bot_username or "vaez_summary_bot").lstrip("@").strip().lower()
        if not target or mention.lower() != target:
            return None
    return cmd


def _strip_bot_mention_and_command(
    raw_text: str, bot_username: str | None = None
) -> str:
    """Remove leading /ask, /vibe, /personality (or RTL `ask/`, `vibe/`) or @bot_username mention."""
    text = (raw_text or "").lstrip(_INVISIBLE_LEADING_CHARS).strip()
    clean_username = (bot_username or "vaez_summary_bot").lstrip("@").strip()
    if clean_username:
        text = re.sub(
            rf"@{re.escape(clean_username)}\b", "", text, flags=re.IGNORECASE
        )
    text = re.sub(
        r"^(?:/(?:ask|vibe|personality)(?:@[A-Za-z0-9_]+)?|(?:ask|vibe|personality)(?:@[A-Za-z0-9_]+)?/)[\s\u200c\u200e\u200f:،؛؟,.?!\-]*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return " ".join(text.split())


def _mentions_bot(
    raw_text: str,
    bot_username: str | None = None,
    bot_first_name: str | None = None,
) -> bool:
    """Return True if raw_text explicitly mentions the bot username or display name on a word boundary."""
    lowered = (raw_text or "").strip().lower()
    if not lowered:
        return False
    clean_username = (bot_username or "vaez_summary_bot").lstrip("@").strip().lower()
    if clean_username and re.search(rf"(?<![\w])@?{re.escape(clean_username)}(?![\w])", lowered):
        return True
    clean_first = (bot_first_name or "voice-summary").strip().lower()
    if (
        clean_first
        and len(clean_first) >= 4
        and re.search(rf"(?<![\w\u0600-\u06FF]){re.escape(clean_first)}(?![\w\u0600-\u06FF])", lowered)
    ):
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
    if _is_disallowed_bot_sender(message):
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
    if _is_disallowed_bot_sender(message):
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
    if _is_disallowed_bot_sender(message):
        return False

    text = (getattr(message, "text", None) or getattr(message, "caption", None) or "").strip()
    if not text or text.lstrip(_INVISIBLE_LEADING_CHARS).startswith("/") or _is_bottom_keyboard_selection(text):
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


def record_incoming_chat_message(
    message: Any,
    cache: TranscriptCache,
    *,
    deduplicate_recent: bool = False,
) -> None:
    """Record non-command text/caption messages strictly in that chat's isolated rolling store."""
    if _is_disallowed_bot_sender(message):
        return

    chat = getattr(message, "chat", None)
    chat_id = getattr(chat, "id", None)
    if chat_id is None:
        return

    replied = getattr(message, "reply_to_message", None)
    if replied is not None and not _is_disallowed_bot_sender(replied):
        replied_text = (
            getattr(replied, "text", None) or getattr(replied, "caption", None) or ""
        ).strip()
        if _should_record_text(replied_text):
            cache.add_chat_message(
                chat_id=chat_id,
                sender=extract_chat_sender_label(replied),
                text=replied_text,
                deduplicate_recent=True,
            )

    text = (getattr(message, "text", None) or getattr(message, "caption", None) or "").strip()
    if not _should_record_text(text):
        return

    speaker = extract_chat_sender_label(message)
    cache.add_chat_message(
        chat_id=chat_id,
        sender=speaker,
        text=text,
        deduplicate_recent=deduplicate_recent,
    )


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
_AUTH_OR_INVALID_KEY_RE = re.compile(
    r"(?:\b(?:401|403|unauthorized|forbidden)\b|invalid[_\s-]?api[_\s-]?key|"
    r"api[_\s-]?key[_\s-]?(?:invalid|not valid)|incorrect api key|authentication failed)",
    re.IGNORECASE,
)
_QUOTA_ERROR_RE = re.compile(
    r"(?:\b(?:429|402)\b|resource_exhausted|too many requests|"
    r"rate[_\s-]?limit(?:ed| exceeded)?|quota[_\s-]?(?:exceeded|exhausted|limit)?|"
    r"insufficient[_\s-]?(?:quota|credit|credits|balance|funds)|"
    r"out of credits|credit[s]?\s+(?:exhausted|depleted|insufficient)|"
    r"token[s]?\s+(?:quota|limit)\s+(?:exceeded|exhausted)|usage[_\s-]?limit)",
    re.IGNORECASE,
)


def _is_single_quota_error(text: str) -> bool:
    return bool(_QUOTA_ERROR_RE.search(text) and not _AUTH_OR_INVALID_KEY_RE.search(text))


def _is_token_or_quota_error(exc: Exception) -> bool:
    """Return True when failures indicate token/quota/rate-limit exhaustion rather than auth or malformed input."""
    if isinstance(exc, ProviderChainExhaustedError):
        return bool(exc.errors) and any(_is_single_quota_error(err) for err in exc.errors)
    return _is_single_quota_error(str(exc))


def _format_wait_duration(seconds: int) -> str:
    mins, secs = divmod(max(1, int(seconds)), 60)
    if mins and secs:
        return f"{mins} دقیقه و {secs} ثانیه"
    if mins:
        return f"{mins} دقیقه"
    return f"{secs} ثانیه"


def _format_window_label(window_seconds: int) -> str:
    sec = max(1, int(window_seconds))
    if sec < 60:
        return f"{sec} ثانیه"
    mins, rem = divmod(sec, 60)
    if rem == 0:
        return f"{mins} دقیقه"
    return f"{mins} دقیقه و {rem} ثانیه"


def _enforce_chat_rate_limit(chat_id: int | str, cache: TranscriptCache) -> str | None:
    """Return None if `chat_id` is within rate limit, else a clear Persian cooldown notice."""
    allowed, retry_after = cache.check_and_consume_chat_rate_limit(chat_id)
    if allowed:
        return None
    wait_str = _format_wait_duration(retry_after)
    window_label = _format_window_label(cache.rate_limit_window_seconds)
    return (
        "⛔ **شما به محدودیت پیام برخوردید!**\n"
        f"(سقف مجاز در این چت: **{cache.rate_limit_max_requests} پیام در هر "
        f"{window_label}**)\n"
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
        "۴. 🎭 **تحلیل محترمانه سبک گفتگو و حال‌وهوا (`/vibe`):**\n"
        "   • `/vibe` از منوی ربات ← نمایش لیست آیدی/نام اعضای فعال برای انتخاب یک نفر (یا همه)\n"
        "   • `/vibe @username` یا `/vibe مهدی` یا ریپلای روی پیام فرد با `/vibe` ← بررسی محترمانه و دقیق سبک پیام‌های همان شخص\n"
        "   • ⚠️ ربات فقط بر اساس **پیام‌های متنی واقعی** ثبت‌شده در همین چت پاسخ می‌دهد و هرگز مطالب غیرواقعی یا حدس بی‌اساس نمی‌سازد.\n\n"
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

    completed = False
    tmp_path: Path | None = None

    try:
        chat_obj = getattr(message, "chat", None)
        voice_chat_id = getattr(chat_obj, "id", None)
        if voice_chat_id is not None:
            rate_msg = _enforce_chat_rate_limit(voice_chat_id, cache)
            if rate_msg is not None:
                await _safe_reply(
                    message,
                    format_telegram_html(rate_msg),
                    bot=bot,
                    chat_id=voice_chat_id,
                )
                return

        status_msg = await _safe_reply(
            message,
            f"⏳ در حال خلاصه‌سازی ویس «{html.escape(speaker_name, quote=False)}»...",
            bot=bot,
            chat_id=voice_chat_id,
        )

        try:
            file_info = await bot.get_file(media.file_id)
            remote_size = getattr(file_info, "file_size", 0) or 0
            if remote_size > max_bytes:
                raise FileTooLargeError(
                    f"Telegram file_info.file_size={remote_size} exceeds limit={max_bytes}"
                )
            file_path_remote = getattr(file_info, "file_path", None)
            if not file_path_remote:
                raise ValueError("Telegram get_file returned empty file_path")
            suffix = Path(file_path_remote or "voice.ogg").suffix or ".ogg"

            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_file:
                tmp_path = Path(tmp_file.name)

            await bot.download_file(file_path_remote, destination=tmp_path)
            if tmp_path.stat().st_size > max_bytes:
                raise FileTooLargeError("Downloaded audio file exceeds max_file_size_mb")

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
            safe_summary = sanitize_sensitive_text(summary_result.summary)

            cache_id = cache.put(
                speaker_name=speaker_name,
                text=stt_result.text,
                summary=safe_summary,
                language=summary_result.language,
                chat_id=chat_id,
            )
            fb_id = cache.create_feedback_target(
                output_type="voice_summary",
                chat_id=chat_id if chat_id is not None else "private",
                chat_title=_get_chat_title(message),
                preview=safe_summary,
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

            await _edit_or_send_chunks(
                status_msg=status_msg,
                origin_message=message,
                raw_text=safe_summary,
                reply_markup=keyboard,
            )
            cache.mark_voice_completed(voice_key)
            completed = True

        except FileTooLargeError:
            await status_msg.edit_text(
                f"⚠️ حجم فایل صوتی بیش از سقف مجاز ({max_file_size_mb} مگابایت) است."
            )
        except ProviderChainExhaustedError as exc:
            logger.error(
                "All providers failed during voice processing: %s",
                redact_secrets(str(exc)),
            )
            err_reply = (
                _TOKEN_EXHAUSTED_MESSAGE
                if _is_token_or_quota_error(exc)
                else "❌ خطا در پردازش ویس: تمامی سرویس‌های هوش مصنوعی در حال حاضر با محدودیت مواجه شدند. لطفاً چند لحظه دیگر دوباره تلاش کنید."
            )
            await status_msg.edit_text(err_reply)
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "Unexpected error processing voice message: %s",
                redact_secrets(str(exc)),
            )
            await status_msg.edit_text(
                "❌ متأسفانه خطایی هنگام پردازش ویس رخ داد. لطفاً دوباره امتحان کنید."
            )
    finally:
        if not completed:
            cache.release_voice(voice_key)
        if tmp_path is not None and tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError as unlink_err:
                logger.warning("Failed to delete temp file %s: %s", tmp_path, unlink_err)


async def _safe_reply(
    target: Any,
    text: str,
    reply_markup: Any = None,
    *,
    bot: Any = None,
    chat_id: int | str | None = None,
) -> Any:
    """Reply to target message, falling back to bot.send_message only when the reply target vanished."""
    resolved_bot = bot or getattr(target, "bot", None)
    resolved_chat_id = (
        chat_id if chat_id is not None else getattr(getattr(target, "chat", None), "id", None)
    )
    if target is not None and hasattr(target, "reply"):
        try:
            if reply_markup is None:
                return await target.reply(text)
            return await target.reply(text, reply_markup=reply_markup)
        except Exception as exc:  # noqa: BLE001
            err_lower = str(exc).lower()
            target_missing = (
                "message to be replied not found" in err_lower
                or "message to reply not found" in err_lower
                or "replied message not found" in err_lower
                or "message not found" in err_lower
            )
            if (
                target_missing
                and resolved_bot is not None
                and resolved_chat_id is not None
                and hasattr(resolved_bot, "send_message")
            ):
                return await resolved_bot.send_message(
                    chat_id=resolved_chat_id,
                    text=text,
                    reply_markup=reply_markup,
                )
            raise
    if resolved_bot is not None and resolved_chat_id is not None and hasattr(resolved_bot, "send_message"):
        return await resolved_bot.send_message(
            chat_id=resolved_chat_id,
            text=text,
            reply_markup=reply_markup,
        )
    raise RuntimeError("No reply target or bot.send_message fallback available")


async def _edit_or_send_chunks(
    status_msg: Any,
    origin_message: Any,
    raw_text: str,
    reply_markup: Any = None,
    *,
    escape_only: bool = False,
) -> None:
    """Render raw_text into Telegram-safe HTML chunks <=4096 chars and deliver via edit + replies."""
    chunks = split_telegram_html(raw_text, max_len=3950, escape_only=escape_only)
    if not chunks:
        return
    first_markup = reply_markup if len(chunks) == 1 else None
    if status_msg is not None and hasattr(status_msg, "edit_text"):
        await status_msg.edit_text(chunks[0], reply_markup=first_markup)
    else:
        await _safe_reply(origin_message, chunks[0], reply_markup=first_markup)

    for idx in range(1, len(chunks)):
        chunk_markup = reply_markup if idx == len(chunks) - 1 else None
        target = origin_message or status_msg
        await _safe_reply(
            target,
            chunks[idx],
            reply_markup=chunk_markup,
            bot=getattr(status_msg, "bot", None),
            chat_id=getattr(getattr(status_msg, "chat", None), "id", None),
        )


async def _delete_and_send_output(
    origin_message: Any,
    status_msg: Any,
    text: str,
    reply_markup: Any = None,
) -> None:
    """Delete distinct status message and send output, or edit in-place when status_msg is origin_message."""
    if status_msg is not None and origin_message is not None and status_msg is origin_message:
        if hasattr(status_msg, "edit_text"):
            try:
                await status_msg.edit_text(text, reply_markup=reply_markup)
                return
            except Exception as exc:  # noqa: BLE001
                logger.debug("In-place edit_text failed (%s); falling back to send_message", exc)
        bot_obj = getattr(status_msg, "bot", None)
        chat_id = getattr(getattr(status_msg, "chat", None), "id", None)
        if bot_obj is not None and chat_id is not None and hasattr(bot_obj, "send_message"):
            await bot_obj.send_message(chat_id=chat_id, text=text, reply_markup=reply_markup)
        return

    if status_msg is not None and hasattr(status_msg, "delete"):
        try:
            await status_msg.delete()
            if origin_message is not None:
                await _safe_reply(
                    origin_message,
                    text,
                    reply_markup=reply_markup,
                    bot=getattr(status_msg, "bot", None),
                )
                return
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not delete status message (%s), falling back to edit_text/send_message", exc)

    if status_msg is not None and hasattr(status_msg, "edit_text"):
        try:
            await status_msg.edit_text(text, reply_markup=reply_markup)
            return
        except Exception as exc:  # noqa: BLE001
            logger.debug("Fallback edit_text failed (%s); sending new message", exc)

    await _safe_reply(
        origin_message or status_msg,
        text,
        reply_markup=reply_markup,
        bot=getattr(status_msg, "bot", None),
        chat_id=getattr(getattr(status_msg, "chat", None), "id", None),
    )


async def _delete_and_send_chunks(
    origin_message: Any,
    status_msg: Any,
    raw_text: str,
    reply_markup: Any = None,
    *,
    escape_only: bool = False,
) -> None:
    """Delete temporary status message and deliver multi-chunk formatted HTML safely <=4096 chars."""
    chunks = split_telegram_html(raw_text, max_len=3950, escape_only=escape_only)
    if not chunks:
        return
    await _delete_and_send_output(
        origin_message=origin_message,
        status_msg=status_msg,
        text=chunks[0],
        reply_markup=(reply_markup if len(chunks) == 1 else None),
    )
    for idx in range(1, len(chunks)):
        await _safe_reply(
            origin_message or status_msg,
            chunks[idx],
            reply_markup=(reply_markup if idx == len(chunks) - 1 else None),
            bot=getattr(status_msg, "bot", None),
            chat_id=getattr(getattr(status_msg, "chat", None), "id", None),
        )


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
    combined_transcript = _trim_history_to_recent_chars("\n".join(formatted_lines))
    detected_lang = normalize_language(None, combined_transcript)

    try:
        summary_result = await summarizer_manager.summarize(
            transcript=combined_transcript,
            speaker_name=f"{CHAT_HISTORY_SPEAKER_PREFIX}{actual_count}",
            language=detected_lang,
        )
        safe_summary = sanitize_sensitive_text(summary_result.summary)
        fb_id = cache.create_feedback_target(
            output_type="chat_summary",
            chat_id=chat_id,
            chat_title=_get_chat_title(origin_message or status_updater),
            preview=safe_summary,
            language=summary_result.language,
        )
        fb_keyboard = build_feedback_keyboard(fb_id, language=summary_result.language)
        await _delete_and_send_chunks(
            origin_message=origin_message,
            status_msg=status_updater,
            raw_text=safe_summary,
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


async def _execute_chat_qa_prompt(
    message: Any,
    chat_id: int | str,
    effective_question: str,
    turn_question: str,
    asker_name: str,
    detected_mode: str,
    summarizer_manager: SummarizerManager,
    cache: TranscriptCache,
    *,
    status_updater: Any = None,
) -> None:
    """Run LLM chat/personality inference on `chat_id`'s isolated state and deliver safe HTML output."""
    recent_messages = cache.get_recent_chat_messages(chat_id, limit=MAX_CHAT_SUMMARY_MESSAGES)
    chat_history = _trim_history_to_recent_chars(
        "\n".join(f"[{m.sender}]: {m.text}" for m in recent_messages)
    )
    session_turns = cache.get_chat_session_turns(chat_id)
    detected_lang = normalize_language(None, effective_question)

    logger.info(
        "Executing chat QA: chat_id=%s chat_type=%s mode=%s history_msgs=%d",
        chat_id,
        getattr(getattr(message, "chat", None), "type", "unknown"),
        detected_mode,
        len(recent_messages),
    )

    try:
        result = await summarizer_manager.chat(
            chat_history=chat_history,
            session_turns=session_turns,
            question=effective_question,
            asker_name=asker_name,
            mode=detected_mode,
            language=detected_lang,
        )
        safe_reply = sanitize_sensitive_text(result.summary)
        fb_id = cache.create_feedback_target(
            output_type="chat_qa",
            chat_id=chat_id,
            chat_title=_get_chat_title(message),
            preview=safe_reply,
            language=result.language,
        )
        fb_keyboard = build_feedback_keyboard(fb_id, language=result.language)
        if status_updater is not None:
            await _delete_and_send_chunks(
                origin_message=message,
                status_msg=status_updater,
                raw_text=safe_reply,
                reply_markup=fb_keyboard,
            )
        else:
            chunks = split_telegram_html(safe_reply, max_len=3950)
            for idx, chunk in enumerate(chunks):
                markup = fb_keyboard if idx == len(chunks) - 1 else None
                await _safe_reply(message, chunk, reply_markup=markup)
        cache.add_chat_turn(
            chat_id=chat_id,
            user_name=asker_name,
            question=sanitize_sensitive_text(turn_question),
            answer=safe_reply,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Chat interaction failed for chat_id=%s: %s",
            chat_id,
            redact_secrets(str(exc)),
        )
        err_reply = (
            _TOKEN_EXHAUSTED_MESSAGE
            if _is_token_or_quota_error(exc)
            else "❌ الان مغزم یکم شلوغ شد، چند لحظه دیگه دوباره سوالتو بپرس!"
        )
        try:
            if status_updater is not None:
                await _delete_and_send_output(
                    origin_message=message,
                    status_msg=status_updater,
                    text=err_reply,
                    reply_markup=None,
                )
            else:
                await _safe_reply(message, err_reply)
        except Exception as reply_exc:  # noqa: BLE001
            logger.warning("Could not deliver fallback error reply: %s", reply_exc)


async def handle_chat_interaction(
    message: Any,
    summarizer_manager: SummarizerManager,
    cache: TranscriptCache,
    bot_username: str | None = None,
    mode: str = "chat",
    reply_context: str | None = None,
) -> None:
    """Answer fun questions, show /vibe target picker, or chat using ONLY `chat_id`'s isolated session."""
    chat = getattr(message, "chat", None)
    chat_id = getattr(chat, "id", "private")
    raw_text = (getattr(message, "text", None) or getattr(message, "caption", None) or "").strip()
    question = _strip_bot_mention_and_command(raw_text, bot_username=bot_username)

    if mode == "personality" and not question:
        if reply_context:
            question = "لطفاً شخصیت، حال‌وهوا و سبک گفتگوی فرستنده همین پیام ریپلای‌شده را به‌صورت محترمانه و بر اساس پیام‌های واقعی خودش در این چت بررسی کن."
        else:
            senders = cache.get_recent_chat_senders(chat_id, max_senders=8)
            if not senders:
                await _safe_reply(
                    message,
                    format_telegram_html(
                        "📭 هنوز پیامی از اعضای این گفتگو ثبت نشده است که بتوانم تحلیلش کنم.\n\n"
                        "• پس از ردوبدل شدن چند پیام در چت، دوباره `/vibe` را بزنید تا لیست یوزرنیم‌ها نمایش داده شود.\n"
                        "• یا مستقیم بنویسید `/vibe @username` یا روی پیام فرد موردنظر ریپلای کرده و `/vibe` بفرستید."
                    ),
                )
                return
            picker_id = cache.create_vibe_picker(chat_id, senders)
            await _safe_reply(
                message,
                format_telegram_html(
                    "🎭 **تحلیل سبک گفتگو برای کدام کاربر انجام شود؟**\n"
                    "از فهرست زیر یوزرنیم یا نام کاربر موردنظر را انتخاب کنید (یا `/vibe @username` بنویسید / روی پیام او `/vibe` ریپلای کنید):"
                ),
                reply_markup=build_vibe_target_keyboard(picker_id, senders),
            )
            return
    elif not question and reply_context:
        question = "لطفاً درباره این پیام با توجه به بافت گفتگو یک توضیح محترمانه و کوتاه ارائه بده."
    elif not question:
        await _safe_reply(
            message,
            format_telegram_html(
                "👋 در خدمتم! درباره پیام‌های همین گفتگو هر سؤالی دارید بفرمایید:\n\n"
                "• مثلاً: `/ask سحر چی گفته؟` یا `@vaez_summary_bot سحر چی گفته؟`\n"
                "• یا روی یک پیام ریپلای کنید و `/ask` را بفرستید.\n"
                "• یا با `/vibe` یوزرنیم یکی از اعضا را برای تحلیل سبک گفتگو انتخاب کنید."
            ),
        )
        return

    effective_question = (
        f"(در پاسخ به پیام «{reply_context}»)\n{question}" if reply_context else question
    )
    rate_msg = _enforce_chat_rate_limit(chat_id, cache)
    if rate_msg is not None:
        await _safe_reply(message, format_telegram_html(rate_msg))
        return

    detected_mode = mode
    if detected_mode == "chat" and any(kw in question.lower() for kw in _PERSONALITY_KEYWORDS):
        detected_mode = "personality"

    asker_name = extract_chat_sender_label(message)
    await _execute_chat_qa_prompt(
        message=message,
        chat_id=chat_id,
        effective_question=effective_question,
        turn_question=question,
        asker_name=asker_name,
        detected_mode=detected_mode,
        summarizer_manager=summarizer_manager,
        cache=cache,
    )


async def handle_vibe_callback(
    callback: Any,
    summarizer_manager: SummarizerManager,
    cache: TranscriptCache,
) -> None:
    """Handle click on `vibe:<picker_id>:<idx|all>` member selector buttons."""
    data = getattr(callback, "data", "") or ""
    parts = data.split(":", 2)
    picker_id = parts[1] if len(parts) > 1 else ""
    choice = parts[2] if len(parts) > 2 else ""

    msg = getattr(callback, "message", None)
    if msg is None:
        await callback.answer("⛔ این پیام دیگر در دسترس نیست.", show_alert=True)
        return

    picker = cache.get_vibe_picker(picker_id)
    if picker is None:
        await callback.answer("⏳ این لیست انتخاب منقضی شده؛ دوباره /vibe رو بزن!", show_alert=True)
        return

    expected_chat_id, targets = picker
    chat = getattr(msg, "chat", None)
    chat_id = getattr(chat, "id", "private")
    if str(chat_id) != str(expected_chat_id):
        await callback.answer("⛔ این دکمه متعلق به چت دیگری است.", show_alert=True)
        return

    if choice == "all":
        question = "شخصیت، حال‌وهوا و سبک گفتگوی اعضای فعال این گفتگو را به‌صورت محترمانه، دقیق و کوتاه تحلیل کن."
        status_label = "👥 در حال تحلیل محترمانه اعضای فعال..."
    else:
        try:
            idx = int(choice)
            target_sender = targets[idx]
        except (ValueError, IndexError):
            await callback.answer("⛔ گزینه انتخاب‌شده نامعتبر است.", show_alert=True)
            return
        question = (
            f"لطفاً شخصیت و سبک گفتگوی «{target_sender}» را به‌صورت محترمانه و فقط بر اساس پیام‌های واقعی خودش در همین چت تحلیل کن."
        )
        status_label = f"🎭 در حال تحلیل سبک گفتگوی {target_sender}..."

    rate_msg = _enforce_chat_rate_limit(chat_id, cache)
    if rate_msg is not None:
        await callback.answer("⏳ سقف استفاده این چت موقتاً پر شده!", show_alert=True)
        await _safe_reply(msg, format_telegram_html(rate_msg))
        return

    await callback.answer(status_label)
    clicker = getattr(callback, "from_user", None)
    asker_name = (
        getattr(clicker, "full_name", None)
        or getattr(clicker, "first_name", None)
        or "کاربر"
    )
    if hasattr(msg, "edit_text"):
        try:
            await msg.edit_text(html.escape(status_label, quote=False))
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not edit vibe picker status message: %s", exc)

    await _execute_chat_qa_prompt(
        message=msg,
        chat_id=chat_id,
        effective_question=question,
        turn_question=question,
        asker_name=asker_name,
        detected_mode="personality",
        summarizer_manager=summarizer_manager,
        cache=cache,
        status_updater=msg,
    )


async def handle_chat_summary_command(
    message: Any,
    summarizer_manager: SummarizerManager,
    cache: TranscriptCache,
    requested_count: int | None = None,
) -> None:
    """Show the bottom input ReplyKeyboardMarkup (or inline buttons in channels) or summarize & delete temp messages."""
    chat = getattr(message, "chat", None)
    chat_id = getattr(chat, "id", "private")
    chat_type = str(getattr(chat, "type", "private") or "private")
    text = getattr(message, "text", None) or getattr(message, "caption", None) or ""
    count = requested_count if requested_count is not None else _extract_requested_count(text)

    if count is None:
        if chat_type == "channel":
            prompt_msg = await _safe_reply(
                message,
                "👇 تعداد پیام‌های اخیر چنل برای خلاصه‌سازی را انتخاب کنید:",
                reply_markup=build_chat_summary_count_keyboard(),
            )
        else:
            prompt_msg = await _safe_reply(
                message,
                "👇 از دکمه‌های پایین صفحه (قسمت تایپ) تعداد پیام‌ها را برای خلاصه‌سازی انتخاب کنید:",
                reply_markup=build_chat_summary_reply_keyboard(),
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

    remove_markup = None if chat_type == "channel" else ReplyKeyboardRemove()
    if not cache.get_recent_chat_messages(chat_id, limit=1):
        await _safe_reply(
            message,
            "📭 هنوز پیامی از این گفتگو در حافظه ثبت نشده است!\n"
            "از الان پیام‌های متنی همین گفتگو (به‌صورت کاملاً جدا و ایزوله) ذخیره می‌شوند و بعد از ردوبدل شدن چند پیام می‌توانید `/chat_summary` بزنید.",
            reply_markup=remove_markup,
        )
        return

    rate_msg = _enforce_chat_rate_limit(chat_id, cache)
    if rate_msg is not None:
        await _safe_reply(message, format_telegram_html(rate_msg), reply_markup=remove_markup)
        return

    status_msg = await _safe_reply(
        message,
        f"⏳ در حال خلاصه‌سازی {min(count, MAX_CHAT_SUMMARY_MESSAGES)} پیام آخر...",
        reply_markup=remove_markup,
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
    if msg is None or not hasattr(msg, "reply"):
        await callback.answer("⛔ این پیام دیگر در دسترس نیست.", show_alert=True)
        return

    chat = getattr(msg, "chat", None)
    chat_id = getattr(chat, "id", "private")
    if not cache.get_recent_chat_messages(chat_id, limit=1):
        await callback.answer("📭 هنوز پیامی برای خلاصه‌سازی ثبت نشده است!", show_alert=True)
        return

    rate_msg = _enforce_chat_rate_limit(chat_id, cache)
    if rate_msg is not None:
        await callback.answer("⏳ سقف استفاده این چت موقتاً پر شده!", show_alert=True)
        await _safe_reply(msg, format_telegram_html(rate_msg))
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
    if msg is None or not hasattr(msg, "reply"):
        await callback.answer("⛔ این پیام دیگر در دسترس نیست.", show_alert=True)
        return

    msg_chat_id = getattr(getattr(msg, "chat", None), "id", None)
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
    full_text = header + entry.text
    chunks = split_telegram_html(full_text, max_len=3950, escape_only=True)
    chat = getattr(msg, "chat", None)
    chat_id = getattr(chat, "id", "private")
    fb_id = cache.create_feedback_target(
        output_type="transcript",
        chat_id=chat_id,
        chat_title=_get_chat_title(msg),
        preview=full_text,
        language=entry.language,
    )
    fb_markup = build_feedback_keyboard(fb_id, language=entry.language)

    try:
        for idx, safe_chunk in enumerate(chunks):
            is_last = idx == len(chunks) - 1
            await _safe_reply(msg, safe_chunk, reply_markup=(fb_markup if is_last else None))
    except Exception as exc:  # noqa: BLE001
        entry.transcript_used = False
        await _update_message_buttons(msg, cache, cache_id, entry)
        logger.error("Failed to send transcript chunks: %s", redact_secrets(str(exc)))


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
    if msg is None or not hasattr(msg, "reply"):
        await callback.answer("⛔ این پیام دیگر در دسترس نیست.", show_alert=True)
        return

    msg_chat_id = getattr(getattr(msg, "chat", None), "id", None)
    if entry.chat_id is not None and msg_chat_id is not None and entry.chat_id != str(msg_chat_id):
        await callback.answer("⛔ این دکمه متعلق به چت دیگری است.", show_alert=True)
        return

    if entry.tts_used:
        await callback.answer("🔊 نسخه صوتی خلاصه قبلاً ارسال شده است.", show_alert=False)
        return

    attempts = int(getattr(entry, "tts_attempts", 0))
    if attempts >= 3:
        await callback.answer("⏳ حداکثر تلاش برای تولید نسخه صوتی این ویس انجام شد.", show_alert=True)
        return

    entry.tts_used = True
    await _update_message_buttons(msg, cache, cache_id, entry)
    await callback.answer("🔊 در حال تولید فایل صوتی خلاصه...")

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
        filename = f"summary_{cache_id}.{ext}"
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
        safe_caption = html.escape(caption, quote=False)
        try:
            await msg.reply_voice(
                voice=BufferedInputFile(tts_result.audio_bytes, filename=filename),
                caption=safe_caption,
                reply_markup=tts_fb_markup,
            )
        except Exception as voice_exc:  # noqa: BLE001
            if hasattr(msg, "reply_audio"):
                logger.info("reply_voice failed (%s); falling back to reply_audio", voice_exc)
                await msg.reply_audio(
                    audio=BufferedInputFile(tts_result.audio_bytes, filename=filename),
                    caption=safe_caption,
                    reply_markup=tts_fb_markup,
                )
            else:
                raise voice_exc
    except Exception as exc:  # noqa: BLE001
        entry.tts_used = False
        setattr(entry, "tts_attempts", attempts + 1)
        await _update_message_buttons(msg, cache, cache_id, entry)
        logger.error("Failed to generate TTS audio summary: %s", redact_secrets(str(exc)))
        err_reply = (
            _TOKEN_EXHAUSTED_MESSAGE
            if _is_token_or_quota_error(exc)
            else "❌ متأسفانه در تولید فایل صوتی خلاصه خطایی رخ داد."
        )
        try:
            await _safe_reply(msg, err_reply)
        except Exception:  # noqa: BLE001
            pass


def _format_admin_feedback_notification(entry: FeedbackEntry) -> str:
    """Format admin feedback notification using strict HTML escaping so user text cannot cross-nest tags."""
    rating_label = "👍 عالی / مثبت" if entry.rating == "up" else "👎 ضعیف / نیاز به بهبود"
    type_label = _OUTPUT_TYPE_LABELS.get(entry.output_type, entry.output_type)
    esc_title = html.escape(str(entry.chat_title or ""), quote=False)
    esc_chat_id = html.escape(str(entry.chat_id), quote=False)
    esc_voter = html.escape(str(entry.voter_name or ""), quote=False)
    username_str = (
        f" (@{html.escape(entry.voter_username, quote=False)})"
        if entry.voter_username
        else ""
    )
    safe_preview = sanitize_sensitive_text(str(entry.preview or ""))
    if len(safe_preview) > 2500:
        safe_preview = safe_preview[:2497] + "..."
    esc_preview = html.escape(safe_preview, quote=False)
    return (
        "📊 <b>بازخورد جدید دریافت شد!</b>\n\n"
        f"• <b>امتیاز:</b> {rating_label}\n"
        f"• <b>نوع خروجی:</b> {html.escape(str(type_label), quote=False)}\n"
        f"• <b>چت / گروه:</b> {esc_title} (<code>{esc_chat_id}</code>)\n"
        f"• <b>کاربر:</b> {esc_voter}{username_str}\n"
        f"• <b>پیش‌نمایش خروجی:</b>\n<code>{esc_preview}</code>"
    )


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


def _is_human_sender(message: Any) -> bool:
    """Return True if the message came directly from a real human user (not a bot or anonymous sender_chat)."""
    sender = getattr(message, "from_user", None)
    if sender is None or getattr(sender, "is_bot", False):
        return False
    if getattr(message, "sender_chat", None) is not None:
        return False
    return getattr(sender, "id", None) is not None


def format_stats_report(cache: TranscriptCache) -> str:
    """Build Persian HTML report of active users, chats, feedbacks, and admin with escaped user strings."""
    ustats = cache.get_active_users_stats()
    fstats = cache.get_feedback_stats()
    admin_id = cache.get_admin_chat_id()
    total_fb = fstats["total"]
    pct_fb = round((fstats["up"] / total_fb) * 100) if total_fb else 0

    lines = [
        "📈 <b>آمار کاربران فعال و وضعیت سرویس:</b>",
        "",
        f"• ⚡ کاربران فعال <b>۲۴ ساعت اخیر:</b> <b>{ustats['active_24h']}</b> نفر (تعامل مستقیم با ربات: <b>{ustats['direct_24h']}</b>)",
        f"• 🗓️ کاربران فعال <b>۷ روز اخیر:</b> <b>{ustats['active_7d']}</b> نفر",
        f"• 📆 کاربران فعال <b>۳۰ روز اخیر:</b> <b>{ustats['active_30d']}</b> نفر",
        f"• 👥 کل کاربران یکتای دیده‌شده: <b>{ustats['total_users']}</b> نفر (کاربران مستقیم: <b>{ustats['direct_users']}</b>)",
        "",
        f"• 💬 کل چت‌ها و گروه‌ها: <b>{ustats['total_chats']}</b> ({ustats['group_chats']} گروه/چنل | {ustats['private_chats']} پیوی | فعال ۷ روز اخیر: <b>{ustats['active_chats_7d']}</b>)",
        f"• 👍 بازخوردها: <b>{total_fb}</b> (👍 {fstats['up']} | 👎 {fstats['down']} | رضایت <b>{pct_fb}%</b>)",
        (
            f"• 🛡️ ادمین فعلی: <code>{html.escape(str(admin_id), quote=False)}</code>"
            if admin_id is not None
            else "• ⚠️ ادمین فعلی: هنوز تنظیم نشده (<code>/set_admin</code>)"
        ),
    ]

    recent_users = ustats.get("recent_users") or []
    if recent_users:
        lines.append("\n🕒 <b>آخرین کاربران فعال:</b>")
        for idx, u in enumerate(recent_users[:8], start=1):
            esc_name = html.escape(str(u.get("name") or "کاربر"), quote=False)
            raw_username = u.get("username")
            handle = f" (@{html.escape(str(raw_username), quote=False)})" if raw_username else ""
            lines.append(
                f"{idx}. <b>{esc_name}</b>{handle} — تعامل مستقیم: <code>{u['direct_interactions']}</code>"
            )

    return "\n".join(lines)


async def handle_stats_command(message: Any, cache: TranscriptCache) -> None:
    """Respond with active user & service analytics for the configured human admin."""
    if not _is_human_sender(message):
        return
    chat_type = getattr(getattr(message, "chat", None), "type", "private")
    if chat_type not in (None, "private"):
        await _safe_reply(message, "🔒 مشاهده آمار سرویس فقط در چت خصوصی (پیوی) ربات مجاز است.")
        return

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
            try:
                me = await (
                    message.bot.me()
                    if hasattr(message.bot, "me")
                    else message.bot.get_me()
                )
                bot_identity["id"] = getattr(me, "id", None)
                bot_identity["username"] = getattr(me, "username", None)
                bot_identity["first_name"] = getattr(me, "first_name", None)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Could not resolve bot identity: %s", exc)

    def _extract_and_record_reply_context(message: Any) -> str | None:
        replied_msg = getattr(message, "reply_to_message", None)
        if replied_msg is None:
            return None
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
        if replied_is_self:
            return None
        record_incoming_chat_message(replied_msg, cache, deduplicate_recent=True)
        replied_body = (
            getattr(replied_msg, "text", None) or getattr(replied_msg, "caption", None) or ""
        ).strip()
        if not replied_body:
            return None
        replied_sender_label = extract_chat_sender_label(replied_msg)
        return f"{replied_sender_label}: {replied_body[:200]}"

    async def _send_start_message(message: Message) -> None:
        welcome = (
            "👋 سلام! من ربات هوشمند خلاصه‌ساز ویس، پیام‌های گروه و هم‌صحبت فان شما هستم.\n\n"
            "✨ **کارهای اصلی من:**\n"
            "• 🎙️ خلاصه خودمونی ویس‌ها (پیوی یا ریپلای در گروه با `/summary` و `خلاصه`) + نسخه صوتی\n"
            f"• 💬 خلاصه ۱۰ تا {MAX_CHAT_SUMMARY_MESSAGES} پیام آخر گروه و چنل با `/chat_summary`\n"
            "• 😎 چت و سؤال درباره بچه‌ها با `@vaez_summary_bot` یا `/ask`\n"
            "• 🎭 تحلیل شخصیت و وایب فان بچه‌های گروه با `/vibe`\n\n"
            "📖 برای دیدن **راهنمای کامل و مثال‌ها** دستور `/help` را بزنید!"
        )
        await _safe_reply(message, format_telegram_html(welcome))

    async def _send_help_message(message: Message) -> None:
        await _safe_reply(message, format_telegram_html(build_help_text()))

    async def _try_dispatch_relaxed_command(message: Message, raw_text: str) -> bool:
        all_cmds = (
            _INTERACTIVE_CHAT_COMMANDS
            | _CHAT_SUMMARY_COMMANDS
            | _SUMMARY_COMMANDS
            | frozenset({"reset_chat", "help", "start"})
        )
        matched = _match_relaxed_command(
            raw_text,
            commands=all_cmds,
            bot_username=bot_identity["username"],
        )
        if matched is None:
            return False
        if matched == "start":
            await _send_start_message(message)
            return True
        if matched == "help":
            await _send_help_message(message)
            return True
        if matched == "reset_chat":
            chat_id = getattr(getattr(message, "chat", None), "id", "private")
            cache.clear_chat_session(chat_id)
            await _safe_reply(
                message,
                "🧹 حافظه گفتگوی تعاملی این سشن ریست شد! هر وقت خواستید بحث جدید را شروع کنید.",
            )
            return True
        if matched in _INTERACTIVE_CHAT_COMMANDS:
            mode = "personality" if matched in {"vibe", "personality"} else "chat"
            await handle_chat_interaction(
                message=message,
                summarizer_manager=summarizer_manager,
                cache=cache,
                bot_username=bot_identity["username"],
                mode=mode,
                reply_context=_extract_and_record_reply_context(message),
            )
            return True
        if matched in _SUMMARY_COMMANDS and extract_voice_target(message) is not None:
            await process_voice_request(
                message=message,
                bot=message.bot,
                stt_manager=stt_manager,
                summarizer_manager=summarizer_manager,
                cache=cache,
                max_file_size_mb=max_file_size_mb,
            )
            return True
        if matched in _CHAT_SUMMARY_COMMANDS or matched in _SUMMARY_COMMANDS:
            await handle_chat_summary_command(
                message=message,
                summarizer_manager=summarizer_manager,
                cache=cache,
            )
            return True
        return False

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

    @router.message(Command("start", ignore_case=True))
    async def cmd_start(message: Message) -> None:
        await _send_start_message(message)

    @router.message(Command("help", ignore_case=True))
    async def cmd_help(message: Message) -> None:
        await _send_help_message(message)

    @router.message(Command("set_admin"))
    async def cmd_set_admin(message: Message) -> None:
        if not _is_human_sender(message):
            return
        chat = getattr(message, "chat", None)
        if getattr(chat, "type", None) != "private":
            await message.reply(
                "🔒 دستور /set_admin فقط در چت خصوصی (پیوی) ربات مجاز است تا کسی در گروه دسترسی ادمین نگیرد."
            )
            return
        current_admin = cache.get_admin_chat_id()
        user_id = getattr(getattr(message, "from_user", None), "id", None)
        if user_id is None:
            return
        if current_admin is not None and current_admin != user_id:
            await message.reply("⛔ ادمین دریافت‌کننده بازخوردها قبلاً توسط مالک ربات ثبت شده است.")
            return
        cache.set_admin_chat_id(user_id)
        await message.reply(
            format_telegram_html(
                f"✅ **تنظیم شد!** از این به بعد خروجی تمام بازخوردهای 👍/👎 بلافاصله برای شما (`{user_id}`) ارسال می‌شود.\n"
                "• با `/stats` تعداد کاربران فعال و وضعیت سرویس را ببینید.\n"
                "• با `/feedbacks` آمار کل بازخوردها را ببینید."
            )
        )

    @router.message(Command("stats", "users"))
    async def cmd_stats(message: Message) -> None:
        await handle_stats_command(message, cache)

    @router.message(Command("feedbacks"))
    async def cmd_feedbacks(message: Message) -> None:
        if not _is_human_sender(message):
            return
        chat_type = getattr(getattr(message, "chat", None), "type", "private")
        if chat_type not in (None, "private"):
            await message.reply("🔒 گزارش بازخوردها فقط در چت خصوصی (پیوی) ربات نمایش داده می‌شود.")
            return
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
            "📊 <b>گزارش کلی بازخوردها و سرویس:</b>",
            f"• 👥 کاربران فعال ۲۴ساعت: <b>{ustats['active_24h']}</b> نفر | کل کاربران: <b>{ustats['total_users']}</b> نفر (<code>/stats</code>)",
            f"• کل بازخوردها: <b>{total}</b> | 👍 عالی: <b>{up}</b> | 👎 ضعیف: <b>{down}</b> | رضایت: <b>{pct}%</b>",
        ]
        recent: list[FeedbackEntry] = stats["recent"]
        if recent:
            lines.append("\n🕒 <b>آخرین بازخوردها:</b>")
            for item in reversed(recent[-8:]):
                icon = "👍" if item.rating == "up" else "👎"
                tlabel = html.escape(str(_OUTPUT_TYPE_LABELS.get(item.output_type, item.output_type)), quote=False)
                esc_voter = html.escape(str(item.voter_name or ""), quote=False)
                esc_chat = html.escape(str(item.chat_title or ""), quote=False)
                esc_preview = html.escape(sanitize_sensitive_text(str(item.preview or ""))[:60], quote=False)
                lines.append(
                    f"• {icon} {tlabel} | <b>{esc_voter}</b> در <b>{esc_chat}</b>: <code>{esc_preview}</code>"
                )
        else:
            lines.append("\nهنوز بازخوردی ثبت نشده است.")
        await message.reply("\n".join(lines))

    @router.message(Command("vibe", "personality", ignore_case=True))
    async def cmd_vibe(message: Message) -> None:
        await _ensure_bot_identity(message)
        await handle_chat_interaction(
            message=message,
            summarizer_manager=summarizer_manager,
            cache=cache,
            bot_username=bot_identity["username"],
            mode="personality",
            reply_context=_extract_and_record_reply_context(message),
        )

    @router.message(Command("ask", ignore_case=True))
    async def cmd_ask(message: Message) -> None:
        await _ensure_bot_identity(message)
        await handle_chat_interaction(
            message=message,
            summarizer_manager=summarizer_manager,
            cache=cache,
            bot_username=bot_identity["username"],
            mode="chat",
            reply_context=_extract_and_record_reply_context(message),
        )

    @router.message(Command("reset_chat", ignore_case=True))
    async def cmd_reset_chat(message: Message) -> None:
        chat_id = getattr(getattr(message, "chat", None), "id", "private")
        cache.clear_chat_session(chat_id)
        await _safe_reply(
            message,
            "🧹 حافظه گفتگوی تعاملی این سشن ریست شد! هر وقت خواستید بحث جدید را شروع کنید.",
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
            "summary_200",
            ignore_case=True,
        )
    )
    async def cmd_chat_summary(message: Message) -> None:
        await handle_chat_summary_command(
            message=message,
            summarizer_manager=summarizer_manager,
            cache=cache,
        )

    @router.message(Command("summary", "summarize", "خلاصه", ignore_case=True))
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
            await handle_chat_interaction(
                message=message,
                summarizer_manager=summarizer_manager,
                cache=cache,
                bot_username=bot_identity["username"],
                reply_context=_extract_and_record_reply_context(message),
            )
            return

        if await _try_dispatch_relaxed_command(message, text):
            return

        record_incoming_chat_message(message, cache)

    @router.channel_post()
    async def on_channel_post(message: Message) -> None:
        text = (getattr(message, "text", None) or getattr(message, "caption", None) or "").strip()
        await _ensure_bot_identity(message)
        if await _try_dispatch_relaxed_command(message, text):
            return
        if (
            getattr(message, "voice", None) is not None
            or getattr(message, "audio", None) is not None
        ) and _is_explicit_summary_text(
            text,
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
        if _is_bottom_keyboard_selection(text) or _is_explicit_chat_summary_phrase(text):
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
        if await _try_dispatch_relaxed_command(message, text):
            return

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
            if getattr(getattr(message, "chat", None), "type", None) == "private":
                record_incoming_chat_message(message, cache, deduplicate_recent=True)
            return

        record_incoming_chat_message(message, cache)

    @router.callback_query(F.data.startswith("fb:"))
    async def on_feedback_button(callback: CallbackQuery) -> None:
        await handle_feedback_callback(callback, cache)

    @router.callback_query(F.data.startswith("chatsum:"))
    async def on_chat_summary_button(callback: CallbackQuery) -> None:
        await handle_chat_summary_callback(callback, summarizer_manager, cache)

    @router.callback_query(F.data.startswith("vibe:"))
    async def on_vibe_target_button(callback: CallbackQuery) -> None:
        await handle_vibe_callback(callback, summarizer_manager, cache)

    @router.callback_query(F.data.startswith("transcript:"))
    async def on_transcript_button(callback: CallbackQuery) -> None:
        await handle_transcript_callback(callback, cache)

    @router.callback_query(F.data.startswith("tts:"))
    async def on_tts_button(callback: CallbackQuery) -> None:
        if tts_manager is not None:
            await handle_tts_callback(callback, cache, tts_manager)
        else:
            await callback.answer("🔇 سرویس تبدیل متن به صوت فعال نیست.", show_alert=True)

    return router
