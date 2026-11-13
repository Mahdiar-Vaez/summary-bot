"""Inline and bottom-input keyboards for Telegram voice, chat summaries, and feedback."""

from __future__ import annotations

import re

from aiogram.types import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    ReplyKeyboardMarkup,
)

_HANDLE_IN_SENDER_RE = re.compile(r"\(@([A-Za-z0-9_]+)\)\s*$")


def build_feedback_row(fb_id: str, language: str = "fa") -> list[InlineKeyboardButton]:
    """Create thumbs-up / thumbs-down inline feedback buttons for a single output message."""
    if language == "fa":
        up_label = "👍 عالی بود"
        down_label = "👎 نیاز به بهبود"
    else:
        up_label = "👍 Helpful"
        down_label = "👎 Needs work"

    return [
        InlineKeyboardButton(text=up_label, callback_data=f"fb:{fb_id}:up"),
        InlineKeyboardButton(text=down_label, callback_data=f"fb:{fb_id}:down"),
    ]


def build_feedback_keyboard(fb_id: str, language: str = "fa") -> InlineKeyboardMarkup:
    """Create a standalone inline feedback keyboard for text, voice TTS, or Q&A outputs."""
    return InlineKeyboardMarkup(inline_keyboard=[build_feedback_row(fb_id, language)])


def build_transcript_keyboard(
    cache_id: str,
    language: str = "fa",
    *,
    show_tts: bool = True,
    show_transcript: bool = True,
    show_feedback: bool = False,
    fb_id: str | None = None,
) -> InlineKeyboardMarkup | None:
    """Create inline buttons for voice summary outputs (audio, transcript, optional feedback).

    Returns None when all buttons have already been clicked and disabled.
    """
    if language == "fa":
        tts_label = "🔊 شنیدن خلاصه صوتی"
        transcript_label = "📜 مشاهده متن کامل ویس"
    else:
        tts_label = "🔊 Listen to Audio Summary"
        transcript_label = "📜 View Full Transcript"

    rows: list[list[InlineKeyboardButton]] = []
    if show_tts:
        rows.append(
            [
                InlineKeyboardButton(
                    text=tts_label,
                    callback_data=f"tts:{cache_id}",
                )
            ]
        )
    if show_transcript:
        rows.append(
            [
                InlineKeyboardButton(
                    text=transcript_label,
                    callback_data=f"transcript:{cache_id}",
                )
            ]
        )
    if show_feedback and fb_id:
        rows.append(build_feedback_row(fb_id, language=language))

    if not rows:
        return None

    return InlineKeyboardMarkup(inline_keyboard=rows)


def build_chat_summary_reply_keyboard() -> ReplyKeyboardMarkup:
    """Create a bottom input-bar ReplyKeyboardMarkup for selecting 10..200 recent messages."""
    return ReplyKeyboardMarkup(
        keyboard=[
            [
                KeyboardButton(text="📊 خلاصه ۱۰ پیام آخر"),
                KeyboardButton(text="📊 خلاصه ۳۰ پیام آخر"),
            ],
            [
                KeyboardButton(text="📊 خلاصه ۵۰ پیام آخر"),
                KeyboardButton(text="📊 خلاصه ۱۰۰ پیام آخر"),
            ],
            [
                KeyboardButton(text="📊 خلاصه ۱۵۰ پیام آخر"),
                KeyboardButton(text="🔥 خلاصه ۲۰۰ پیام آخر"),
            ],
        ],
        resize_keyboard=True,
        one_time_keyboard=True,
        selective=True,
        input_field_placeholder="تعداد پیام‌ها را از پایین انتخاب کنید...",
    )


def build_chat_summary_count_keyboard() -> InlineKeyboardMarkup:
    """Fallback inline keyboard for selecting 10..150 recent messages."""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="🔟 ۱۰ پیام آخر", callback_data="chatsum:10"),
                InlineKeyboardButton(text="3️⃣0️⃣ ۳۰ پیام آخر", callback_data="chatsum:30"),
            ],
            [
                InlineKeyboardButton(text="5️⃣0️⃣ ۵۰ پیام آخر", callback_data="chatsum:50"),
                InlineKeyboardButton(text="💯 ۱۰۰ پیام آخر", callback_data="chatsum:100"),
            ],
            [
                InlineKeyboardButton(
                    text="🔥 ۱۵۰ پیام آخر (حداکثر)", callback_data="chatsum:150"
                ),
            ],
        ]
    )


def _format_vibe_button_label(sender: str) -> str:
    """Format a sender label putting `@username` first when present for dropdown readability."""
    clean = " ".join((sender or "").split())
    match = _HANDLE_IN_SENDER_RE.search(clean)
    if match:
        handle = match.group(1)
        display_part = clean[: match.start()].strip()
        if display_part:
            short_name = display_part if len(display_part) <= 14 else f"{display_part[:13]}…"
            label = f"👤 @{handle} ({short_name})"
        else:
            label = f"👤 @{handle}"
    else:
        label = f"👤 {clean}"
    return label if len(label) <= 34 else f"{label[:33]}…"


def build_vibe_target_keyboard(
    picker_id: str, targets: list[str]
) -> InlineKeyboardMarkup:
    """Create inline buttons to select a member (with @username) or all active members for /vibe."""
    rows: list[list[InlineKeyboardButton]] = []
    current_row: list[InlineKeyboardButton] = []
    for idx, sender in enumerate(targets):
        btn = InlineKeyboardButton(
            text=_format_vibe_button_label(sender),
            callback_data=f"vibe:{picker_id}:{idx}",
        )
        current_row.append(btn)
        if len(current_row) == 2:
            rows.append(current_row)
            current_row = []
    if current_row:
        rows.append(current_row)

    rows.append(
        [
            InlineKeyboardButton(
                text="🔥 تحلیل و رُست همه بچه‌های فعال",
                callback_data=f"vibe:{picker_id}:all",
            )
        ]
    )
    return InlineKeyboardMarkup(inline_keyboard=rows)
