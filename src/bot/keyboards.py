"""Inline keyboards for Telegram voice summary messages."""

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
