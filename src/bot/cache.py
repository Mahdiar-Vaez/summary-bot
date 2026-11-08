"""Bounded TTL cache for storing full voice transcripts, summaries, and persistent chat history."""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections import OrderedDict, deque
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

MAX_CHAT_HISTORY_PER_CHAT = 150


@dataclass(slots=True)
class TranscriptEntry:
    """Cached transcript and summary entry with speaker name, language, and button state."""

    speaker_name: str
    text: str
    summary: str
    language: str
    created_at: float
    tts_used: bool = False
    transcript_used: bool = False


@dataclass(slots=True)
class ChatMessageEntry:
    """Single recorded text/caption/voice message in a group, channel, or chat."""

    sender: str
    text: str
    created_at: float


class TranscriptCache:
    """Bounded LRU + TTL cache for voice message transcripts, summaries, deduplication, and chat history."""

    def __init__(
        self,
        ttl_seconds: int = 3600,
        max_items: int = 500,
        max_messages_per_chat: int = MAX_CHAT_HISTORY_PER_CHAT,
        persistence_path: Path | str | None = None,
    ) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_items = max_items
        self.max_messages_per_chat = max_messages_per_chat
        self.persistence_path = Path(persistence_path) if persistence_path else None
        self._store: OrderedDict[str, TranscriptEntry] = OrderedDict()
        self._inflight: set[str] = set()
        self._completed_voices: OrderedDict[str, float] = OrderedDict()
        self._chat_history: OrderedDict[str, deque[ChatMessageEntry]] = OrderedDict()
        self._prompt_message_ids: dict[str, int] = {}
        self._load_chat_history()

    def _load_chat_history(self) -> None:
        if self.persistence_path is None or not self.persistence_path.exists():
            return
        try:
            raw = json.loads(self.persistence_path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                return
            now = time.monotonic()
            for chat_key, items in raw.items():
                if not isinstance(items, list):
                    continue
                q: deque[ChatMessageEntry] = deque(maxlen=self.max_messages_per_chat)
                for item in items[-self.max_messages_per_chat :]:
                    if isinstance(item, dict) and item.get("text"):
                        q.append(
                            ChatMessageEntry(
                                sender=str(item.get("sender") or "کاربر"),
                                text=str(item["text"]),
                                created_at=now,
                            )
                        )
                if q:
                    self._chat_history[str(chat_key)] = q
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load persisted chat history: %s", exc)

    def _save_chat_history(self) -> None:
        if self.persistence_path is None:
            return
        try:
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                chat_key: [
                    {"sender": entry.sender, "text": entry.text}
                    for entry in entries
                ]
                for chat_key, entries in self._chat_history.items()
            }
            self.persistence_path.write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8"
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to save chat history: %s", exc)

    def set_prompt_message_id(self, chat_id: int | str, message_id: int | None) -> None:
        if message_id is not None:
            self._prompt_message_ids[str(chat_id)] = int(message_id)

    def pop_prompt_message_id(self, chat_id: int | str) -> int | None:
        return self._prompt_message_ids.pop(str(chat_id), None)

    def _evict_expired(self) -> None:
        now = time.monotonic()
        expired_keys = [
            k
            for k, entry in self._store.items()
            if (now - entry.created_at) > self.ttl_seconds
        ]
        for k in expired_keys:
            self._store.pop(k, None)

        expired_voices = [
            vk
            for vk, ts in self._completed_voices.items()
            if (now - ts) > self.ttl_seconds
        ]
        for vk in expired_voices:
            self._completed_voices.pop(vk, None)

    def add_chat_message(self, chat_id: int | str, sender: str, text: str) -> None:
        """Append a message to the chat's rolling buffer (capped at max_messages_per_chat = 150)."""
        cleaned = (text or "").strip()
        if not cleaned:
            return
        key = str(chat_id)
        if key not in self._chat_history:
            while len(self._chat_history) >= self.max_items:
                self._chat_history.popitem(last=False)
            self._chat_history[key] = deque(maxlen=self.max_messages_per_chat)
        else:
            self._chat_history.move_to_end(key)

        q = self._chat_history[key]
        if q and q[-1].text == cleaned and q[-1].sender == (sender or "کاربر").strip():
            return

        q.append(
            ChatMessageEntry(
                sender=(sender or "کاربر").strip(),
                text=cleaned,
                created_at=time.monotonic(),
            )
        )
        self._save_chat_history()

    def get_recent_chat_messages(
        self, chat_id: int | str, limit: int = MAX_CHAT_HISTORY_PER_CHAT
    ) -> list[ChatMessageEntry]:
        """Return up to `limit` (clamped to 1..150) most recent messages for `chat_id`."""
        clamped = max(1, min(int(limit), self.max_messages_per_chat))
        key = str(chat_id)
        history = self._chat_history.get(key)
        if not history:
            return []
        items = list(history)
        return items[-clamped:]

    def try_acquire_voice(self, voice_key: str) -> bool:
        """Return True and lock voice_key if not currently in-flight or already summarized."""
        if not voice_key:
            return True
        self._evict_expired()
        if voice_key in self._inflight or voice_key in self._completed_voices:
            return False
        self._inflight.add(voice_key)
        return True

    def mark_voice_completed(self, voice_key: str) -> None:
        """Mark voice_key as completed and remove from in-flight set."""
        if not voice_key:
            return
        self._inflight.discard(voice_key)
        while len(self._completed_voices) >= self.max_items:
            self._completed_voices.popitem(last=False)
        self._completed_voices[voice_key] = time.monotonic()

    def release_voice(self, voice_key: str) -> None:
        """Release an in-flight lock without marking completed (e.g. on failure so retry is allowed)."""
        if not voice_key:
            return
        self._inflight.discard(voice_key)

    def put(
        self,
        speaker_name: str,
        text: str,
        summary: str = "",
        language: str = "fa",
    ) -> str:
        """Store transcript and summary and return a compact 12-char cache ID."""
        self._evict_expired()
        while len(self._store) >= self.max_items:
            self._store.popitem(last=False)

        cache_id = uuid.uuid4().hex[:12]
        self._store[cache_id] = TranscriptEntry(
            speaker_name=speaker_name,
            text=text,
            summary=summary or text,
            language=language,
            created_at=time.monotonic(),
        )
        return cache_id

    def get(self, cache_id: str) -> TranscriptEntry | None:
        """Retrieve a transcript entry if present and not expired."""
        self._evict_expired()
        entry = self._store.get(cache_id)
        if entry is not None:
            self._store.move_to_end(cache_id)
        return entry


def chunk_text(text: str, max_len: int = 4000) -> list[str]:
    """Split text into safe Telegram message chunks (<= max_len characters)."""
    if not text:
        return [""]
    return [text[i : i + max_len] for i in range(0, len(text), max_len)]
