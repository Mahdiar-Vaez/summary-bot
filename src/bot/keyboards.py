"""Inline keyboards for Telegram voice summary messages."""

from __future__ import annotations

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup


def build_transcript_keyboard(cache_id: str, language: str = "fa") -> InlineKeyboardMarkup:
    """Create inline button to view the full verbatim voice transcript."""
    label = (
        "📜 مشاهده متن کامل ویس"
        if language == "fa"
        else "📜 View Full Transcript"
    )
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=label,
                    callback_data=f"transcript:{cache_id}",
                )
            ]
        ]
    )
