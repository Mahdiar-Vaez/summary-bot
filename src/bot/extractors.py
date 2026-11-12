"""Speaker name and voice/audio media extraction from Telegram messages."""

from __future__ import annotations

from typing import Any


def _extract_speaker_from_message(msg: Any) -> str:
    """Extract the original speaker's display name from forward_origin or from_user."""
    origin = getattr(msg, "forward_origin", None)
    if origin is not None:
        sender_user = getattr(origin, "sender_user", None)
        if sender_user is not None:
            name = getattr(sender_user, "full_name", None) or getattr(
                sender_user, "first_name", None
            )
            if name:
                return str(name).strip()

        sender_user_name = getattr(origin, "sender_user_name", None)
        if sender_user_name:
            return str(sender_user_name).strip()

        sender_chat = getattr(origin, "sender_chat", None)
        if sender_chat is not None:
            title = getattr(sender_chat, "title", None)
            if title:
                return str(title).strip()

        chat = getattr(origin, "chat", None)
        if chat is not None:
            title = getattr(chat, "title", None)
            if title:
                return str(title).strip()

    from_user = getattr(msg, "from_user", None)
    if from_user is not None:
        name = getattr(from_user, "full_name", None) or getattr(
            from_user, "first_name", None
        )
        if name:
            return str(name).strip()

    sender_chat = getattr(msg, "sender_chat", None)
    if sender_chat is not None:
        title = getattr(sender_chat, "title", None)
        if title:
            return str(title).strip()

    chat = getattr(msg, "chat", None)
    if chat is not None and getattr(chat, "type", "") == "channel":
        title = getattr(chat, "title", None)
        if title:
            return str(title).strip()

    return "گوینده"


extract_speaker_name = _extract_speaker_from_message


def extract_chat_sender_label(msg: Any) -> str:
    """Return sender display name with optional @username for unambiguous per-member chat Q&A."""
    name = _extract_speaker_from_message(msg)
    from_user = getattr(msg, "from_user", None)
    username = getattr(from_user, "username", None) if from_user is not None else None
    if username:
        clean_handle = str(username).lstrip("@").strip()
        if clean_handle and clean_handle.lower() not in name.lower():
            return f"{name} (@{clean_handle})"
    return name


def extract_voice_target(message: Any) -> tuple[Any, str] | None:
    """Return (media_obj, speaker_name) from message or its reply_to_message, or None."""
    replied = getattr(message, "reply_to_message", None)
    if replied is not None:
        media = getattr(replied, "voice", None) or getattr(replied, "audio", None)
        if media is not None:
            return media, _extract_speaker_from_message(replied)

    media = getattr(message, "voice", None) or getattr(message, "audio", None)
    if media is not None:
        return media, _extract_speaker_from_message(message)

    return None
