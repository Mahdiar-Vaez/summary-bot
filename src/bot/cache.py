"""Bounded TTL cache for transcripts, isolated per-chat history/sessions, and feedback persistence."""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections import OrderedDict, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MAX_CHAT_HISTORY_PER_CHAT = 250
MAX_SESSION_TURNS_PER_CHAT = 20
MAX_PERSISTED_FEEDBACKS = 500


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
    fb_id: str | None = None


@dataclass(slots=True)
class ChatMessageEntry:
    """Single recorded text/caption/voice message in a group, channel, or chat."""

    sender: str
    text: str
    created_at: float


@dataclass(slots=True)
class FeedbackEntry:
    """Represents a bot output that can receive a one-time thumbs up/down feedback."""

    fb_id: str
    output_type: str
    chat_id: str
    chat_title: str
    preview: str
    created_at: str
    language: str = "fa"
    rating: str | None = None
    voter_id: int | None = None
    voter_name: str | None = None
    voter_username: str | None = None
    voted_at: str | None = None
    linked_transcript_cache_id: str | None = None


class TranscriptCache:
    """Bounded LRU + TTL cache for transcripts, isolated per-chat sessions, and feedback."""

    def __init__(
        self,
        ttl_seconds: int = 3600,
        max_items: int = 500,
        max_messages_per_chat: int = MAX_CHAT_HISTORY_PER_CHAT,
        persistence_path: Path | str | None = None,
        feedback_path: Path | str | None = None,
        admin_config_path: Path | str | None = None,
        admin_chat_id: int | None = None,
    ) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_items = max_items
        self.max_messages_per_chat = max_messages_per_chat
        self.persistence_path = Path(persistence_path) if persistence_path else None
        self.feedback_path = Path(feedback_path) if feedback_path else None
        self.admin_config_path = Path(admin_config_path) if admin_config_path else None
        self._admin_chat_id: int | None = admin_chat_id

        self._store: OrderedDict[str, TranscriptEntry] = OrderedDict()
        self._inflight: set[str] = set()
        self._completed_voices: OrderedDict[str, float] = OrderedDict()
        self._chat_history: OrderedDict[str, deque[ChatMessageEntry]] = OrderedDict()
        self._chat_sessions: OrderedDict[str, deque[tuple[str, str, str]]] = OrderedDict()
        self._prompt_message_ids: dict[str, int] = {}
        self._feedbacks: OrderedDict[str, FeedbackEntry] = OrderedDict()

        self._load_chat_history()
        self._load_feedbacks()
        self._load_admin_config()

    def _load_json_safe(self, path: Path | None) -> Any:
        if path is None or not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load %s: %s", path, exc)
            return None

    def _save_json_safe(self, path: Path | None, payload: Any) -> None:
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to save %s: %s", path, exc)

    def _load_chat_history(self) -> None:
        raw = self._load_json_safe(self.persistence_path)
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

    def _save_chat_history(self) -> None:
        payload = {
            chat_key: [{"sender": entry.sender, "text": entry.text} for entry in entries]
            for chat_key, entries in self._chat_history.items()
        }
        self._save_json_safe(self.persistence_path, payload)

    def _load_feedbacks(self) -> None:
        raw = self._load_json_safe(self.feedback_path)
        if not isinstance(raw, list):
            return
        for item in raw[-MAX_PERSISTED_FEEDBACKS:]:
            if not isinstance(item, dict) or not item.get("fb_id"):
                continue
            try:
                entry = FeedbackEntry(**item)
                self._feedbacks[entry.fb_id] = entry
            except TypeError:
                continue

    def _save_feedbacks(self) -> None:
        payload = [asdict(entry) for entry in list(self._feedbacks.values())[-MAX_PERSISTED_FEEDBACKS:]]
        self._save_json_safe(self.feedback_path, payload)

    def _load_admin_config(self) -> None:
        if self._admin_chat_id is not None:
            return
        raw = self._load_json_safe(self.admin_config_path)
        if isinstance(raw, dict) and raw.get("admin_chat_id"):
            try:
                self._admin_chat_id = int(raw["admin_chat_id"])
            except (ValueError, TypeError):
                pass

    def get_admin_chat_id(self) -> int | None:
        """Return configured admin Telegram chat ID for live feedback delivery."""
        return self._admin_chat_id

    def set_admin_chat_id(self, chat_id: int) -> None:
        """Persist the owner's Telegram chat ID for feedback notifications."""
        self._admin_chat_id = int(chat_id)
        self._save_json_safe(self.admin_config_path, {"admin_chat_id": self._admin_chat_id})

    def create_feedback_target(
        self,
        output_type: str,
        chat_id: int | str,
        chat_title: str,
        preview: str,
        language: str = "fa",
        linked_transcript_cache_id: str | None = None,
    ) -> str:
        """Create and register a new feedback target for a bot output message."""
        while len(self._feedbacks) >= MAX_PERSISTED_FEEDBACKS:
            self._feedbacks.popitem(last=False)

        fb_id = uuid.uuid4().hex[:10]
        clean_preview = " ".join((preview or "").split())[:180]
        entry = FeedbackEntry(
            fb_id=fb_id,
            output_type=output_type,
            chat_id=str(chat_id),
            chat_title=(chat_title or str(chat_id)).strip(),
            preview=clean_preview,
            created_at=datetime.now(timezone.utc).isoformat(),
            language=language,
            linked_transcript_cache_id=linked_transcript_cache_id,
        )
        self._feedbacks[fb_id] = entry
        return fb_id

    def get_feedback_target(self, fb_id: str) -> FeedbackEntry | None:
        """Return the feedback entry by ID if present."""
        return self._feedbacks.get(fb_id)

    def record_feedback(
        self,
        fb_id: str,
        rating: str,
        voter_id: int | None = None,
        voter_name: str = "",
        voter_username: str | None = None,
    ) -> tuple[FeedbackEntry | None, bool]:
        """Record a rating ('up' or 'down') once. Returns (entry, already_voted)."""
        entry = self._feedbacks.get(fb_id)
        if entry is None:
            return None, False
        if entry.rating is not None:
            return entry, True

        entry.rating = "up" if rating == "up" else "down"
        entry.voter_id = voter_id
        entry.voter_name = (voter_name or "کاربر").strip()
        entry.voter_username = voter_username.lstrip("@") if voter_username else None
        entry.voted_at = datetime.now(timezone.utc).isoformat()
        self._save_feedbacks()
        return entry, False

    def get_feedback_stats(self) -> dict[str, Any]:
        """Return aggregated feedback statistics and recent rated items."""
        rated = [entry for entry in self._feedbacks.values() if entry.rating is not None]
        up_count = sum(1 for entry in rated if entry.rating == "up")
        down_count = sum(1 for entry in rated if entry.rating == "down")
        return {
            "total": len(rated),
            "up": up_count,
            "down": down_count,
            "recent": rated[-10:],
        }

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
        """Append a message strictly to `chat_id`'s isolated rolling buffer (>= 200 msgs)."""
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
        """Return up to `limit` (clamped to `max_messages_per_chat`) most recent messages strictly for `chat_id`."""
        clamped = max(1, min(int(limit), self.max_messages_per_chat))
        key = str(chat_id)
        history = self._chat_history.get(key)
        if not history:
            return []
        items = list(history)
        return items[-clamped:]

    def add_chat_turn(
        self, chat_id: int | str, user_name: str, question: str, answer: str
    ) -> None:
        """Store a conversational Q&A turn strictly inside `chat_id`'s isolated session."""
        key = str(chat_id)
        if key not in self._chat_sessions:
            while len(self._chat_sessions) >= self.max_items:
                self._chat_sessions.popitem(last=False)
            self._chat_sessions[key] = deque(maxlen=MAX_SESSION_TURNS_PER_CHAT)
        else:
            self._chat_sessions.move_to_end(key)

        self._chat_sessions[key].append((user_name, question, answer))

    def get_chat_session_turns(
        self, chat_id: int | str
    ) -> list[tuple[str, str, str]]:
        """Return isolated Q&A turns strictly for `chat_id`."""
        turns = self._chat_sessions.get(str(chat_id))
        return list(turns) if turns else []

    def clear_chat_session(self, chat_id: int | str) -> None:
        """Clear ONLY `chat_id`'s interactive Q&A session turns."""
        self._chat_sessions.pop(str(chat_id), None)

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
        """Release an in-flight lock without marking completed."""
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
