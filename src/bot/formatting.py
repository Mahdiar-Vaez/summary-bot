"""Markdown-to-Telegram-HTML converter for clean, error-free rich text rendering in Telegram."""

from __future__ import annotations

import html
import re


def format_telegram_html(text: str) -> str:
    """Convert Markdown formatting (**bold**, __bold__, `code`, # headers) into safe Telegram HTML."""
    if not text:
        return ""

    escaped = html.escape(text, quote=False)

    # Convert inline `code`
    escaped = re.sub(r"`([^`\n]+)`", r"<code>\1</code>", escaped)

    # Convert **bold** and __bold__
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)
    escaped = re.sub(r"__(.+?)__", r"<b>\1</b>", escaped)

    # Convert markdown headings (# Title) at start of lines
    escaped = re.sub(r"(?m)^#{1,4}\s+(.+)$", r"<b>\1</b>", escaped)

    # Remove any leftover stray double-asterisks
    escaped = escaped.replace("**", "")

    # Ensure the top 📌 header line is bolded for visual hierarchy
    lines = escaped.splitlines()
    if lines and "📌" in lines[0] and "<b>" not in lines[0]:
        lines[0] = f"<b>{lines[0].strip()}</b>"

    return "\n".join(lines)
