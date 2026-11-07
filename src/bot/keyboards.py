"""Inline keyboards for Telegram voice summary messages."""

from __future__ import annotations

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup


def build_transcript_keyboard(cache_id: str, language: str = "fa") -> InlineKeyboardMarkup:
    """Create inline buttons to listen to the audio summary or view the full transcript."""
    if language == "fa":
        tts_label = "🔊 شنیدن خلاصه صوتی"
        transcript_label = "📜 مشاهده متن کامل ویس"
    else:
        tts_label = "🔊 Listen to Audio Summary"
        transcript_label = "📜 View Full Transcript"

    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=tts_label,
                    callback_data=f"tts:{cache_id}",
                )
            ],
            [
                InlineKeyboardButton(
                    text=transcript_label,
                    callback_data=f"transcript:{cache_id}",
                )
            ],
        ]
    )
