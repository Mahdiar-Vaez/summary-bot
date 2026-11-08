"""Inline keyboards for Telegram voice and chat history summary messages."""

from __future__ import annotations

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup


def build_transcript_keyboard(
    cache_id: str,
    language: str = "fa",
    *,
    show_tts: bool = True,
    show_transcript: bool = True,
) -> InlineKeyboardMarkup | None:
    """Create inline buttons to listen to the audio summary or view the full transcript.

    Returns None when both buttons have already been clicked and disabled.
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

    if not rows:
        return None

    return InlineKeyboardMarkup(inline_keyboard=rows)


def build_chat_summary_count_keyboard() -> InlineKeyboardMarkup:
    """Create an inline keyboard allowing the user to pick how many recent messages (10..150) to summarize."""
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
